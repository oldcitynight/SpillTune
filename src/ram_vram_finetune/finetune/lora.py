# Adapted from AllTheAIs streaming/lora.py; see SOURCE_ATTRIBUTION.md.
"""Projection-level LoRA containers and target discovery.

Target names are supplied explicitly by the model adapter and describe its
real module paths; attention kind never infers target names.  A LoRA projection always owns independent FP32 master tensors::

    y = linear(x, W) + linear(linear(x, A), B) * (alpha / rank)

``A`` has shape ``[rank, in_features]`` and ``B`` has shape
``[out_features, rank]``.  Base weights are detached by :meth:`forward`; they
are never made trainable by this container.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, cast

import torch
import torch.nn.functional as F

def _validate_targets(targets):
    if isinstance(targets, (str, bytes)) or not isinstance(targets, (list, tuple)) or not targets:
        raise ValueError("LoRA requires an explicit nonempty ordered target mapping")
    if any(not isinstance(t, str) or not t or any(not part or not all(c.isalnum() or c == '_' for c in part) for part in t.split('.')) for t in targets):
        raise ValueError("invalid canonical LoRA target path")
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate LoRA target paths")
    return tuple(targets)


def _canonical_target(targets, name):
    if name in targets:
        return name
    matches = [target for target in targets if target.rsplit(".", 1)[-1] == name]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Invalid or ambiguous target {name!r}; expected {targets}")


class LoRAAdapterMap(dict[str, "LoRAProjection"]):
    """Dictionary whose stored keys are canonical module paths.

    Leaf-name lookup (for example ``adapters['q_proj']``) is retained for old
    callers, but iteration, serialization, equality, and ``keys()`` expose
    only the real projection paths.  This prevents aliases from becoming
    duplicate serialized modules.
    """

    def __init__(self, block_type: str, targets, *args, **kwargs) -> None:
        self.block_type = block_type
        self.targets = _validate_targets(targets)
        super().__init__(*args, **kwargs)

    def _lookup_key(self, key: object) -> object:
        if not isinstance(key, str) or dict.__contains__(self, key):
            return key
        try:
            return _canonical_target(self.targets, key)
        except ValueError:
            return key

    def __getitem__(self, key: str) -> "LoRAProjection":
        return dict.__getitem__(self, cast(str, self._lookup_key(key)))

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, self._lookup_key(key))

    def get(self, key: str, default=None):
        return dict.get(self, cast(str, self._lookup_key(key)), default)


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int],
    device: torch.device,
    trainable: bool,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be FP32, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if trainable and not tensor.requires_grad:
        tensor.requires_grad_(True)
    if not trainable and tensor.requires_grad:
        raise ValueError(f"{name} is an optimizer/gradient buffer and must not require gradients")


@dataclass
class LoRAProjection:
    """One projection-level LoRA adapter with FP32 master state."""

    name: str
    rank: int
    alpha: float
    in_features: int
    out_features: int
    A: torch.Tensor
    B: torch.Tensor
    grad_A: torch.Tensor
    grad_B: torch.Tensor
    m_A: torch.Tensor
    v_A: torch.Tensor
    m_B: torch.Tensor
    v_B: torch.Tensor
    target_path: Optional[str] = None
    layer: Optional[int] = None
    block_type: Optional[str] = None
    architecture: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {self.rank!r}")
        self.alpha = float(self.alpha)
        if not math.isfinite(self.alpha):
            raise ValueError(f"alpha must be finite, got {self.alpha!r}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.in_features, self.out_features)
        ):
            raise ValueError(
                f"projection dimensions must be positive, got "
                f"out={self.out_features}, in={self.in_features}"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("LoRA projection name must not be empty")
        if not isinstance(self.A, torch.Tensor) or not isinstance(self.B, torch.Tensor):
            raise TypeError("LoRA A and B master parameters must be torch.Tensor instances")

        self.target_path = self.target_path or self.name
        if not isinstance(self.target_path, str):
            raise TypeError("LoRA target_path must be a string")
        _validate_targets([self.target_path])
        if self.layer is not None and (
            isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0
        ):
            raise ValueError(f"layer must be a non-negative integer, got {self.layer!r}")

        # Validate the supplied state, then take private copies.  Always
        # cloning prevents two separately constructed projections from
        # accidentally sharing a master parameter or optimizer buffer.
        device = self.A.device
        _validate_tensor(
            self.A,
            name=f"{self.target_path}.A",
            shape=(self.rank, self.in_features),
            device=device,
            trainable=True,
        )
        _validate_tensor(
            self.B,
            name=f"{self.target_path}.B",
            shape=(self.out_features, self.rank),
            device=device,
            trainable=True,
        )
        for suffix, tensor, shape in (
            ("grad_A", self.grad_A, (self.rank, self.in_features)),
            ("grad_B", self.grad_B, (self.out_features, self.rank)),
            ("m_A", self.m_A, (self.rank, self.in_features)),
            ("v_A", self.v_A, (self.rank, self.in_features)),
            ("m_B", self.m_B, (self.out_features, self.rank)),
            ("v_B", self.v_B, (self.out_features, self.rank)),
        ):
            _validate_tensor(
                tensor,
                name=f"{self.target_path}.{suffix}",
                shape=shape,
                device=device,
                trainable=False,
            )
        self.A = self.A.detach().clone().requires_grad_(True)
        self.B = self.B.detach().clone().requires_grad_(True)
        self.grad_A = self.grad_A.detach().clone()
        self.grad_B = self.grad_B.detach().clone()
        self.m_A = self.m_A.detach().clone()
        self.v_A = self.v_A.detach().clone()
        self.m_B = self.m_B.detach().clone()
        self.v_B = self.v_B.detach().clone()

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def requires_grad_A(self) -> bool:
        return self.A.requires_grad

    @property
    def requires_grad_B(self) -> bool:
        return self.B.requires_grad

    @property
    def base_weight_shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor, base_bias=None) -> torch.Tensor:
        """Apply the frozen base projection and this LoRA update."""

        if base_weight.ndim != 2 or tuple(base_weight.shape) != self.base_weight_shape:
            raise ValueError(
                f"{self.target_path} base weight shape mismatch: expected "
                f"{self.base_weight_shape}, got {tuple(base_weight.shape)}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"{self.target_path} input width mismatch: expected "
                f"{self.in_features}, got {x.shape[-1]}"
            )
        base_out = F.linear(x, base_weight.detach(), None if base_bias is None else base_bias.detach())
        lora_out = F.linear(F.linear(x.float(), self.A), self.B)
        return (base_out.float() + lora_out * self.scale).to(base_out.dtype)

    def zero_grad(self) -> None:
        self.grad_A.zero_()
        self.grad_B.zero_()
        if self.A.grad is not None:
            self.A.grad.zero_()
        if self.B.grad is not None:
            self.B.grad.zero_()


def _shape_from_source(source: object, target_path: str) -> tuple[int, int]:
    if isinstance(source, torch.nn.Module):
        if not hasattr(source, "weight"):
            raise TypeError(f"Module for {target_path} has no weight tensor")
        source = source.weight
    if isinstance(source, torch.Tensor):
        shape = tuple(source.shape)
    elif isinstance(source, (tuple, list, torch.Size)):
        shape = tuple(source)
    else:
        raise TypeError(
            f"Weight source for {target_path} must be a tensor, a module with "
            f"a weight, or a legacy (out, in) shape; got {type(source).__name__}"
        )
    if len(shape) != 2 or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape
    ):
        raise ValueError(
            f"Weight for {target_path} must have a positive 2-D [out, in] shape, got {shape}"
        )
    return int(shape[0]), int(shape[1])


def _resolve_from_mapping(
    sources: Mapping[str, object], target_path: str, block_idx: int
) -> object:
    leaf = target_path.rsplit(".", 1)[-1]

    def without_weight(name: str) -> str:
        return name[:-7] if name.endswith(".weight") else name

    normalised = {key: without_weight(key) for key in sources if isinstance(key, str)}
    exact = [key for key, name in normalised.items() if name == target_path]
    if not exact:
        layer_suffix = f"layers.{block_idx}.{target_path}"
        exact = [
            key
            for key, name in normalised.items()
            if name == layer_suffix or name.endswith("." + layer_suffix)
        ]
    if not exact:
        exact = [key for key, name in normalised.items() if name == leaf]
    if not exact:
        suffix = "." + target_path
        exact = [key for key, name in normalised.items() if name.endswith(suffix)]
    if len(exact) != 1:
        if not exact:
            raise ValueError(
                f"Missing required LoRA target {target_path!r} in block {block_idx}"
            )
        raise ValueError(
            f"Ambiguous weight sources for {target_path!r} in block {block_idx}: {exact}"
        )
    return sources[exact[0]]


def _resolve_from_module(
    module: torch.nn.Module, target_path: str, block_idx: int
) -> object:
    try:
        return module.get_submodule(target_path)
    except AttributeError:
        pass

    named = list(module.named_modules())
    layer_suffix = f"layers.{block_idx}.{target_path}"
    matches = [
        child
        for name, child in named
        if name == layer_suffix or name.endswith("." + layer_suffix)
    ]
    if not matches:
        matches = [
            child
            for name, child in named
            if name == target_path or name.endswith("." + target_path)
        ]
    if len(matches) != 1:
        if not matches:
            raise ValueError(
                f"Missing required LoRA target {target_path!r} in block {block_idx}"
            )
        raise ValueError(
            f"Ambiguous module matches for {target_path!r} in block {block_idx}"
        )
    return matches[0]


def create_lora_for_block(
    block_idx: int,
    block_type: str,
    weight_shapes: Mapping[str, object] | torch.nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    device: torch.device | str | None = None,
    *,
    targets: list[str] | tuple[str, ...] | None = None,
    architecture: Optional[str] = None,
) -> LoRAAdapterMap:
    """Create every required projection adapter for one decoder block.

    ``weight_shapes`` may be the actual decoder-layer module, a state/module
    mapping, or (for compatibility) a mapping of explicit two-dimensional
    shapes.  Actual tensors/modules are preferred and their ``weight.shape`` is
    the sole source of adapter dimensions.  Missing targets are never skipped.
    """

    if isinstance(block_idx, bool) or not isinstance(block_idx, int) or block_idx < 0:
        raise ValueError(f"block_idx must be a non-negative integer, got {block_idx!r}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"rank must be a positive integer, got {rank!r}")
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha!r}")
    if not isinstance(weight_shapes, (Mapping, torch.nn.Module)):
        raise TypeError("weight_shapes must be a mapping or torch.nn.Module")

    resolved_device = torch.device("cpu" if device is None else device)
    targets = _validate_targets(targets)
    adapters = LoRAAdapterMap(block_type, targets)

    for target_path in targets:
        if isinstance(weight_shapes, torch.nn.Module):
            source = _resolve_from_module(weight_shapes, target_path, block_idx)
        else:
            source = _resolve_from_mapping(weight_shapes, target_path, block_idx)
        out_features, in_features = _shape_from_source(source, target_path)

        # Every projection receives fresh storage, including optimizer state.
        A = torch.randn(
            rank, in_features, dtype=torch.float32, device=resolved_device
        ).mul_(0.02)
        B = torch.zeros(
            out_features, rank, dtype=torch.float32, device=resolved_device
        )
        A.requires_grad_(True)
        B.requires_grad_(True)
        adapters[target_path] = LoRAProjection(
            name=target_path,
            target_path=target_path,
            layer=block_idx,
            block_type=block_type,
            architecture=architecture,
            rank=rank,
            alpha=alpha,
            in_features=in_features,
            out_features=out_features,
            A=A,
            B=B,
            grad_A=torch.zeros_like(A),
            grad_B=torch.zeros_like(B),
            m_A=torch.zeros_like(A),
            v_A=torch.zeros_like(A),
            m_B=torch.zeros_like(B),
            v_B=torch.zeros_like(B),
        )

    return adapters


def get_lora_tensors_for_autograd(
    adapters: Mapping[str, LoRAProjection],
) -> list[torch.Tensor]:
    """Return ``[A0, B0, A1, B1, ...]`` in canonical name order."""

    result: list[torch.Tensor] = []
    for proj in sorted(adapters.values(), key=lambda item: item.target_path or item.name):
        result.extend((proj.A, proj.B))
    return result


def accumulate_gradients(
    adapters: Mapping[str, LoRAProjection],
    grad_list: list[Optional[torch.Tensor]] | tuple[Optional[torch.Tensor], ...],
    start_idx: int = 0,
) -> None:
    """Accumulate autograd results into the dedicated FP32 gradient buffers."""

    idx = start_idx
    for proj in sorted(adapters.values(), key=lambda item: item.target_path or item.name):
        grad_A = grad_list[idx] if idx < len(grad_list) else None
        grad_B = grad_list[idx + 1] if idx + 1 < len(grad_list) else None
        if grad_A is not None:
            if tuple(grad_A.shape) != tuple(proj.grad_A.shape):
                raise ValueError(
                    f"Gradient shape mismatch for {proj.target_path}.A: "
                    f"expected {tuple(proj.grad_A.shape)}, got {tuple(grad_A.shape)}"
                )
            proj.grad_A.add_(grad_A.detach().to(dtype=torch.float32))
        if grad_B is not None:
            if tuple(grad_B.shape) != tuple(proj.grad_B.shape):
                raise ValueError(
                    f"Gradient shape mismatch for {proj.target_path}.B: "
                    f"expected {tuple(proj.grad_B.shape)}, got {tuple(grad_B.shape)}"
                )
            proj.grad_B.add_(grad_B.detach().to(dtype=torch.float32))
        idx += 2
