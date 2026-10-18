# Adapted from AllTheAIs streaming/optimizer.py; see SOURCE_ATTRIBUTION.md.
"""FP32 AdamW updates for the projection-level LoRA parameters.

``LoRAProjection.grad_A`` and ``grad_B`` are the optimizer's gradient
buffers.  They are deliberately separate from ``A.grad`` and ``B.grad``:
callers copy or add each backward result to these buffers.  Calling
``zero_grad`` starts a new accumulation window; omitting it between backward
passes accumulates gradients.  ``step`` consumes the buffers exactly as they
stand and only clears them when ``clear_grad=True`` is explicitly requested.

The optimizer never receives a base-weight tensor, so its only mutable model
state is the FP32 LoRA A/B tensors and their FP32 Adam moments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import torch

from .lora import LoRAProjection


class LoRAOptimizer:
    """AdamW optimizer for LoRA projection parameters only.

    The update is the non-AMSGrad, non-maximizing AdamW update used by
    :class:`torch.optim.AdamW`, including first- and second-moment bias
    correction and decoupled weight decay.  Every tensor involved in an
    update must be FP32, finite, on the same device, and have the shape
    declared by its :class:`LoRAProjection`.

    Gradient-buffer semantics are explicit:

    * :meth:`zero_grad` clears both optimizer buffers in place and discards
      any autograd-owned ``A.grad``/``B.grad`` tensors.
    * Multiple backward results may be accumulated with ``add_`` before
      :meth:`step`.
    * :meth:`step` does not clear gradients unless ``clear_grad=True``.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        validation_mode: str = 'scalar',
    ) -> None:
        if validation_mode not in ('scalar', 'batched'):
            raise ValueError('unsupported_capability: unknown optimizer validation mode')
        self.validation_mode = validation_mode
        self.lr = self._validated_scalar("lr", lr, minimum=0.0)
        self.beta1 = self._validated_beta("beta1", beta1)
        self.beta2 = self._validated_beta("beta2", beta2)
        self.eps = self._validated_scalar("eps", eps, minimum=0.0)
        self.weight_decay = self._validated_scalar(
            "weight_decay", weight_decay, minimum=0.0
        )
        self.step_count = 0

    @staticmethod
    def _validated_scalar(name: str, value: object, *, minimum: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if result < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {result}")
        return result

    @classmethod
    def _validated_beta(cls, name: str, value: object) -> float:
        result = cls._validated_scalar(name, value, minimum=0.0)
        if result >= 1.0:
            raise ValueError(f"{name} must be < 1.0, got {result}")
        return result

    @staticmethod
    def _validate_tensor(
        adapter_name: str,
        tensor_name: str,
        tensor: torch.Tensor,
        expected_shape: tuple[int, ...],
        expected_device: torch.device,
        *,
        check_finite: bool,
    ) -> None:
        label = f"adapter {adapter_name!r} {tensor_name}"
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{label} must be FP32, got {tensor.dtype}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{label} has shape {tuple(tensor.shape)}, expected {expected_shape}"
            )
        if tensor.device != expected_device:
            raise ValueError(
                f"{label} is on {tensor.device}, expected {expected_device}"
            )
        if check_finite and not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(f"{label} contains NaN or Inf")

    @classmethod
    def _validate_projection(
        cls,
        adapter_name: str,
        proj: LoRAProjection,
        *,
        check_finite: bool,
    ) -> None:
        if not isinstance(proj, LoRAProjection):
            raise TypeError(
                f"adapter {adapter_name!r} must be a LoRAProjection, "
                f"got {type(proj).__name__}"
            )
        if proj.rank <= 0 or proj.in_features <= 0 or proj.out_features <= 0:
            raise ValueError(
                f"adapter {adapter_name!r} dimensions must be positive; got "
                f"rank={proj.rank}, in_features={proj.in_features}, "
                f"out_features={proj.out_features}"
            )

        a_shape = (proj.rank, proj.in_features)
        b_shape = (proj.out_features, proj.rank)
        if not isinstance(proj.A, torch.Tensor):
            raise TypeError(f"adapter {adapter_name!r} A must be a torch.Tensor")
        device = proj.A.device
        for tensor_name, tensor, shape in (
            ("A", proj.A, a_shape),
            ("grad_A", proj.grad_A, a_shape),
            ("m_A", proj.m_A, a_shape),
            ("v_A", proj.v_A, a_shape),
            ("B", proj.B, b_shape),
            ("grad_B", proj.grad_B, b_shape),
            ("m_B", proj.m_B, b_shape),
            ("v_B", proj.v_B, b_shape),
        ):
            cls._validate_tensor(
                adapter_name,
                tensor_name,
                tensor,
                shape,
                device,
                check_finite=check_finite,
            )

        if check_finite:
            for tensor_name, tensor in (("v_A", proj.v_A), ("v_B", proj.v_B)):
                if bool((tensor < 0).any().item()):
                    raise ValueError(
                        f"adapter {adapter_name!r} {tensor_name} must be non-negative"
                    )

    @classmethod
    def _validate_adapters(
        cls,
        adapters: dict[str, LoRAProjection],
        *,
        check_finite: bool,
        allow_empty: bool,
    ) -> None:
        if not isinstance(adapters, dict):
            raise TypeError("adapters must be a dict[str, LoRAProjection]")
        if not adapters and not allow_empty:
            raise ValueError("adapters must contain at least one LoRA projection")

        seen: set[int] = set()
        for adapter_name, proj in adapters.items():
            if id(proj) in seen:
                raise ValueError("the same LoRAProjection cannot be updated more than once")
            seen.add(id(proj))
            cls._validate_projection(
                str(adapter_name), proj, check_finite=check_finite
            )

    @staticmethod
    def _validate_finite_batched(adapters: dict[str, LoRAProjection]) -> None:
        # Reduce each tensor with the original predicates and order, enqueue
        # all flags, then read one small flag vector per device. No parameter,
        # gradient or moment changes before every flag has been checked.
        # Groups permit existing callers with projections on several devices.
        checks = []
        groups = {}
        for name, proj in adapters.items():
            for field in ('A', 'grad_A', 'm_A', 'v_A', 'B', 'grad_B', 'm_B', 'v_B'):
                tensor = getattr(proj, field)
                checks.append((tensor.device, torch.isfinite(tensor).all(),
                               FloatingPointError, f"adapter {str(name)!r} {field} contains NaN or Inf"))
            for field in ('v_A', 'v_B'):
                tensor = getattr(proj, field)
                checks.append((tensor.device, ~(tensor < 0).any(), ValueError,
                               f"adapter {str(name)!r} {field} must be non-negative"))
        for index, (device, flag, _, _) in enumerate(checks):
            groups.setdefault(device, []).append((index, flag))
        values = {}
        for group in groups.values():
            flags = torch.stack([flag for _, flag in group]).cpu().tolist()
            values.update((index, good) for (index, _), good in zip(group, flags))
        for index, (_, _, error, message) in enumerate(checks):
            if not values[index]:
                raise error(message)

    def zero_grad(self, adapters: dict[str, LoRAProjection]) -> None:
        """Start a new gradient-accumulation window.

        The dedicated FP32 gradient buffers are zeroed in place, preserving
        their identity for code that holds references to them.  Autograd-owned
        ``A.grad`` and ``B.grad`` values are set to ``None`` because this
        optimizer does not consume them directly.
        """

        # Do not require finite values here: zero_grad is also the recovery
        # operation for a non-finite gradient buffer.
        self._validate_adapters(adapters, check_finite=False, allow_empty=True)
        for proj in adapters.values():
            proj.grad_A.zero_()
            proj.grad_B.zero_()
            proj.A.grad = None
            proj.B.grad = None

    def step(
        self,
        adapters: dict[str, LoRAProjection],
        *,
        clear_grad: bool = False,
    ) -> None:
        """Apply one bias-corrected FP32 AdamW update.

        ``grad_A`` and ``grad_B`` are interpreted as the complete gradient for
        this logical optimizer step (whether produced by one backward pass or
        accumulated across several).  All adapters are validated before any
        tensor or counter is mutated.  A successful call advances
        :attr:`step_count` exactly once.  Set ``clear_grad=True`` to clear the
        consumed buffers after the update; the default preserves them.
        """

        if not isinstance(clear_grad, bool):
            raise TypeError("clear_grad must be a bool")
        if self.validation_mode == 'scalar':
            self._validate_adapters(adapters, check_finite=True, allow_empty=False)
        else:
            self._validate_adapters(adapters, check_finite=False, allow_empty=False)
            self._validate_finite_batched(adapters)

        next_step = self.step_count + 1
        bias_correction1 = 1.0 - self.beta1**next_step
        bias_correction2 = 1.0 - self.beta2**next_step
        step_size = self.lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)

        with torch.no_grad():
            for proj in adapters.values():
                self._update_tensor(
                    proj.A,
                    proj.grad_A,
                    proj.m_A,
                    proj.v_A,
                    step_size,
                    bias_correction2_sqrt,
                )
                self._update_tensor(
                    proj.B,
                    proj.grad_B,
                    proj.m_B,
                    proj.v_B,
                    step_size,
                    bias_correction2_sqrt,
                )

        self.step_count = next_step
        if clear_grad:
            self.zero_grad(adapters)

    def state_dict(self) -> dict[str, float | int]:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError(
                f"state must be a Mapping, got {type(state).__name__}"
            )
        expected_keys = {"lr", "beta1", "beta2", "eps", "weight_decay", "step_count"}
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            extra = sorted(set(state) - expected_keys)
            raise ValueError(
                f"state key mismatch: missing {missing}, extra {extra}"
            )

        lr = self._validated_scalar("lr", state["lr"], minimum=0.0)
        beta1 = self._validated_beta("beta1", state["beta1"])
        beta2 = self._validated_beta("beta2", state["beta2"])
        eps = self._validated_scalar("eps", state["eps"], minimum=0.0)
        weight_decay = self._validated_scalar(
            "weight_decay", state["weight_decay"], minimum=0.0
        )

        step_count = state["step_count"]
        if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
            raise TypeError(
                f"step_count must be a non-negative int, got {type(step_count).__name__}"
            )

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = step_count

    def _update_tensor(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        # Keep the operation order aligned with torch.optim.AdamW's
        # non-foreach, non-fused implementation for equivalent FP32 rounding.
        if self.weight_decay != 0.0:
            param.mul_(1.0 - self.lr * self.weight_decay)
        exp_avg.lerp_(grad, 1.0 - self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(
            grad, grad, value=1.0 - self.beta2
        )
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)

    def get_gradient_norms(
        self, adapters: dict[str, LoRAProjection]
    ) -> dict[str, float]:
        """Return each optimizer gradient-buffer norm."""

        return {
            key: value
            for name, proj in adapters.items()
            for key, value in (
                (f"{name}_A", proj.grad_A.norm().item()),
                (f"{name}_B", proj.grad_B.norm().item()),
            )
        }

    def get_param_norms(
        self, adapters: dict[str, LoRAProjection]
    ) -> dict[str, float]:
        """Return each FP32 LoRA parameter norm."""

        return {
            key: value
            for name, proj in adapters.items()
            for key, value in (
                (f"{name}_A", proj.A.norm().item()),
                (f"{name}_B", proj.B.norm().item()),
            )
        }
