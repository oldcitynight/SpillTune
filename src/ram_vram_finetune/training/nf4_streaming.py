"""Synchronous CPU simulation of leased NF4 block streaming and recomputation.

This module is a small CPU-only lifecycle foundation.  It does not implement
CUDA transfers, asynchronous execution, decoder-block adapters, or a training
runtime.  Canonical NF4Record objects remain owned by the caller; each lease
gets separate packed and nested-state CPU tensor copies.
"""
from __future__ import annotations

import math
import threading
from functools import partial
from collections.abc import Mapping
from numbers import Real
from types import FunctionType, MappingProxyType
from typing import Any, Callable

import torch
from torch import nn

from ..quantization.nf4_resident import nf4_lora_linear_cpu
from ..quantization.nf4_state import NF4Record


_VALID_PHASES = frozenset(("forward", "backward"))
_VALID_FINISH_OUTCOMES = frozenset(("complete", "abort"))


def _validated_records(records: Mapping[str, NF4Record]) -> tuple[dict[str, NF4Record], tuple[str, ...]]:
    if not isinstance(records, Mapping) or not records:
        raise ValueError("records must be a non-empty mapping of names to NF4Record values")
    copied: dict[str, NF4Record] = {}
    for name, record in records.items():
        if type(name) is not str or not name:
            raise ValueError("NF4 record names must be non-empty strings")
        if not isinstance(record, NF4Record):
            raise TypeError(f"record {name!r} must be an NF4Record")
        copied[name] = record
    names = tuple(copied)
    return {name: copied[name] for name in names}, names


def _tensor_storage_bytes(named_tensors: list[tuple[str, torch.Tensor]]) -> int:
    """Count unique tensor storages, including nested quantization state."""
    seen: set[tuple[str, int | None, int, int]] = set()
    total = 0
    for _name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("NF4 lease contents must be tensors")
        if tensor.device.type != "cpu":
            raise ValueError("NF4 lease contents must be CPU tensors")
        storage = tensor.untyped_storage()
        nbytes = storage.nbytes()
        identity = (tensor.device.type, tensor.device.index, storage.data_ptr(), nbytes)
        if identity not in seen:
            seen.add(identity)
            total += nbytes
    return total


class NF4Lease:
    """One generation of owned packed and nested NF4 state tensors."""

    __slots__ = ("_pool", "generation", "phase", "staged_bytes", "_owned")

    def __init__(
        self,
        pool: "NF4LeasePool",
        generation: int,
        phase: str,
        owned: dict[str, tuple[torch.Tensor, dict[str, Any]]],
        staged_bytes: int,
    ) -> None:
        self._pool = pool
        self.generation = generation
        self.phase = phase
        self.staged_bytes = staged_bytes
        self._owned: dict[str, tuple[torch.Tensor, dict[str, Any]]] | None = owned

    @property
    def is_active(self) -> bool:
        return self._pool._lease_is_active(self)

    def parts(self, name: str) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return this lease's owned tensors while this generation is active."""
        return self._pool._lease_parts(self, name)

    def complete(self) -> None:
        self._pool._finish(self, "complete")

    def abort(self) -> None:
        self._pool._finish(self, "abort")

    def _invalidate(self) -> None:
        self._owned = None


class NF4LeasePool:
    """Single-slot, synchronous CPU lease pool for a block's NF4 records.

    ``completion_hook(lease, outcome)`` is invoked before a lease is released.
    Outcomes are ``"complete"``, ``"abort"``, and ``"recovery"``.  If the
    hook fails, the active slot is poisoned and remains unavailable until
    ``recover_after_barrier_failure()`` explicitly retries the hook.
    """

    def __init__(
        self,
        records: Mapping[str, NF4Record],
        *,
        completion_hook: Callable[[NF4Lease, str], Any] | None = None,
    ) -> None:
        copied, names = _validated_records(records)
        if completion_hook is not None and not callable(completion_hook):
            raise TypeError("completion_hook must be callable or None")
        self._records = MappingProxyType(copied)
        self._names = names
        self._completion_hook = completion_hook
        self._lock = threading.RLock()
        self._generation = 0
        self._active_lease: NF4Lease | None = None
        self._acquiring = False
        self._finishing = False
        self._poisoned = False
        self._recovery_thread_id: int | None = None
        self._events: list[dict[str, Any]] = []
        self._counts = {"acquires": 0, "completes": 0, "aborts": 0, "recoveries": 0}
        self._current_staged_bytes = 0
        self._peak_staged_bytes = 0

    @property
    def record_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return tuple(MappingProxyType(dict(event)) for event in self._events)

    @property
    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                **self._counts,
                "active": int(self._active_lease is not None),
                "current_staged_bytes": self._current_staged_bytes,
                "peak_staged_bytes": self._peak_staged_bytes,
                "poisoned": self._poisoned,
            }

    def acquire(self, phase: str) -> NF4Lease:
        if type(phase) is not str or phase not in _VALID_PHASES:
            raise ValueError("phase must be 'forward' or 'backward'")
        with self._lock:
            if self._poisoned:
                raise RuntimeError("NF4 lease pool is poisoned; explicit recovery is required")
            if self._active_lease is not None or self._acquiring or self._finishing:
                raise RuntimeError("NF4 lease pool already has an active operation")
            self._acquiring = True
            next_generation = self._generation + 1

        owned: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
        try:
            for name in self._names:
                # export_parts returns independently owned copies and never
                # exposes the canonical record's packed or nested tensors.
                packed, parts = self._records[name].export_parts()
                tensors = [(f"{name}/packed", packed)]
                tensors.extend((f"{name}/state/{key}", value) for key, value in parts.items() if isinstance(value, torch.Tensor))
                if not all(isinstance(value, torch.Tensor) for value in parts.values()):
                    raise ValueError(f"NF4 record {name!r} contains non-tensor state parts")
                for tensor_name, tensor in tensors:
                    if tensor.device.type != "cpu":
                        raise ValueError(f"{tensor_name} is not a CPU tensor")
                owned[name] = (packed, parts)
            staged_bytes = _tensor_storage_bytes(
                [
                    (f"{name}/packed", packed)
                    for name, (packed, _parts) in owned.items()
                ]
                + [
                    (f"{name}/state/{key}", value)
                    for name, (_packed, parts) in owned.items()
                    for key, value in parts.items()
                    if isinstance(value, torch.Tensor)
                ]
            )
        except BaseException as exc:
            with self._lock:
                self._acquiring = False
                self._events.append({"event": "acquire_failed", "phase": phase, "generation": next_generation})
            # Drop any partial tensor copies before propagating the original error.
            owned.clear()
            raise exc

        with self._lock:
            self._acquiring = False
            # The reservation above keeps the slot exclusive while copies are made.
            self._generation = next_generation
            lease = NF4Lease(self, next_generation, phase, owned, staged_bytes)
            self._active_lease = lease
            self._current_staged_bytes = staged_bytes
            self._peak_staged_bytes = max(self._peak_staged_bytes, staged_bytes)
            self._counts["acquires"] += 1
            self._events.append(
                {"event": "acquire", "phase": phase, "generation": next_generation, "staged_bytes": staged_bytes}
            )
            return lease

    def recover_after_barrier_failure(self) -> None:
        """Explicitly retry the completion hook before releasing a poisoned lease."""
        with self._lock:
            if not self._poisoned or self._active_lease is None:
                raise RuntimeError("NF4 lease pool has no poisoned lease to recover")
            if self._finishing:
                raise RuntimeError("NF4 lease pool is already running its completion hook")
            lease = self._active_lease
            self._finishing = True
            self._recovery_thread_id = threading.get_ident()
        try:
            with torch.no_grad():
                self._run_completion_hook(lease, "recovery")
        except BaseException:
            with self._lock:
                self._finishing = False
                self._recovery_thread_id = None
                self._events.append(
                    {"event": "recovery_failed", "phase": lease.phase, "generation": lease.generation}
                )
            raise
        with self._lock:
            if self._active_lease is not lease:
                self._finishing = False
                self._recovery_thread_id = None
                raise RuntimeError("NF4 lease generation changed during recovery")
            lease._invalidate()
            self._active_lease = None
            self._current_staged_bytes = 0
            self._poisoned = False
            self._finishing = False
            self._recovery_thread_id = None
            self._counts["recoveries"] += 1
            self._events.append({"event": "recovery", "phase": lease.phase, "generation": lease.generation})

    def _run_completion_hook(self, lease: NF4Lease, outcome: str) -> None:
        if self._completion_hook is not None:
            self._completion_hook(lease, outcome)

    def _lease_is_active(self, lease: NF4Lease) -> bool:
        with self._lock:
            return self._active_lease is lease and lease._owned is not None

    def _lease_parts(self, lease: NF4Lease, name: str) -> tuple[torch.Tensor, dict[str, Any]]:
        with self._lock:
            if self._active_lease is not lease or lease._owned is None:
                raise RuntimeError("NF4 lease is stale or has already been released")
            recovery_hook_thread = (
                self._finishing and self._recovery_thread_id == threading.get_ident()
            )
            if self._poisoned and not recovery_hook_thread:
                raise RuntimeError("NF4 lease pool is poisoned; lease parts are unavailable")
            if self._finishing and self._active_lease is not lease:
                raise RuntimeError("NF4 lease generation is no longer active")
            try:
                return lease._owned[name]
            except KeyError as exc:
                raise KeyError(f"unknown NF4 lease record {name!r}") from exc

    def _finish(self, lease: NF4Lease, outcome: str) -> None:
        if outcome not in _VALID_FINISH_OUTCOMES:
            raise ValueError("invalid NF4 lease completion outcome")
        with self._lock:
            if self._active_lease is not lease or lease._owned is None:
                raise RuntimeError("NF4 lease is stale or has already been released")
            if self._poisoned:
                raise RuntimeError("NF4 lease pool is poisoned; explicit recovery is required")
            if self._finishing:
                raise RuntimeError("NF4 lease completion is already in progress")
            self._finishing = True
        try:
            with torch.no_grad():
                self._run_completion_hook(lease, outcome)
        except BaseException:
            with self._lock:
                self._poisoned = True
                self._finishing = False
                self._events.append(
                    {"event": "barrier_failed", "phase": lease.phase, "generation": lease.generation, "outcome": outcome}
                )
            raise
        with self._lock:
            if self._active_lease is not lease:
                self._finishing = False
                raise RuntimeError("NF4 lease generation changed during completion")
            lease._invalidate()
            self._active_lease = None
            self._current_staged_bytes = 0
            self._finishing = False
            self._counts["completes" if outcome == "complete" else "aborts"] += 1
            self._events.append({"event": outcome, "phase": lease.phase, "generation": lease.generation})


class _LoRAFactors(nn.Module):
    def _apply(self, fn, recurse=True):
        raise RuntimeError("NF4 streaming factors are fixed to CPU/FP32; module migration is unsupported")

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.zeros((rank, in_features), dtype=torch.float32, device="cpu"))
        self.B = nn.Parameter(torch.zeros((out_features, rank), dtype=torch.float32, device="cpu"))


def _contains_trainable_tensor(value: Any, seen: set[int] | None = None) -> bool:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        return bool(value.requires_grad)
    if isinstance(value, nn.Module):
        return any(parameter.requires_grad for parameter in value.parameters())
    if isinstance(value, Mapping):
        return any(_contains_trainable_tensor(item, seen) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_trainable_tensor(item, seen) for item in value)
    return False


def _reject_hidden_trainable_captures(block_fn: Callable[..., Any]) -> None:
    """Allow auditable functions and reject trainable state hidden from autograd."""
    target = block_fn
    values: list[Any] = []
    while isinstance(target, partial):
        values.extend(target.args)
        values.extend((target.keywords or {}).values())
        target = target.func
    if not isinstance(target, FunctionType):
        raise TypeError(
            "block_fn must be a plain Python function or functools.partial of one; "
            "callable instances and bound methods are unsupported because their "
            "captured trainable state cannot be verified"
        )

    values.extend(target.__defaults__ or ())
    values.extend((target.__kwdefaults__ or {}).values())
    for cell in target.__closure__ or ():
        try:
            values.append(cell.cell_contents)
        except ValueError:
            continue
    namespace = target.__globals__
    values.extend(namespace[name] for name in target.__code__.co_names if name in namespace)
    if any(_contains_trainable_tensor(value) for value in values):
        raise ValueError("block_fn must not capture trainable tensors; use project() and block inputs")


def _validated_cpu_device(device: str | torch.device) -> torch.device:
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be 'cpu' or torch.device('cpu')")
    try:
        result = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device {device!r}") from exc
    if result.type != "cpu" or result.index is not None:
        raise ValueError("NF4StreamingBlock supports CPU only")
    return torch.device("cpu")


class NF4StreamingBlock(nn.Module):
    """Recompute one generic CPU block under fresh forward/backward leases.

    ``block_fn(x, project)`` must call ``project(name, hidden)`` for every
    registered projection.  It must not capture trainable tensors; the block's
    A/B Parameters are passed explicitly through the custom autograd function.
    This primitive is a CPU lifecycle simulation, not a CUDA streamer or a
    whole-decoder-block qualification.
    """

    def _apply(self, fn, recurse=True):
        # Fail before nn.Module invokes ``fn`` on a Parameter.  This CPU oracle
        # has no valid module migration or dtype conversion path.
        raise RuntimeError("NF4StreamingBlock is fixed to CPU/FP32; module migration is unsupported")

    def __init__(
        self,
        records: Mapping[str, NF4Record],
        block_fn: Callable[[torch.Tensor, Callable[[str, torch.Tensor], torch.Tensor]], torch.Tensor],
        *,
        rank: int = 2,
        alpha: float = 4.0,
        device: str | torch.device = "cpu",
        completion_hook: Callable[[NF4Lease, str], Any] | None = None,
    ) -> None:
        cpu_device = _validated_cpu_device(device)  # Fail closed before tensor allocation.
        super().__init__()
        copied_records, names = _validated_records(records)
        if not callable(block_fn):
            raise TypeError("block_fn must be callable")
        _reject_hidden_trainable_captures(block_fn)
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("alpha must be a finite real number")
        alpha_value = float(alpha)
        if not math.isfinite(alpha_value):
            raise ValueError("alpha must be finite")
        self.rank = rank
        self.alpha = alpha_value
        self._scale = self.alpha / self.rank
        self._names = names
        self._records = MappingProxyType(copied_records)
        self._projection_modules = nn.ModuleList(
            [
                _LoRAFactors(
                    in_features=copied_records[name].shape[1],
                    out_features=copied_records[name].shape[0],
                    rank=rank,
                )
                for name in names
            ]
        )
        self.projections = MappingProxyType(
            {name: self._projection_modules[index] for index, name in enumerate(names)}
        )
        object.__setattr__(self, "_block_fn", block_fn)
        self.pool = NF4LeasePool(copied_records, completion_hook=completion_hook)
        # Keep the selected CPU device visible for diagnostics, without moving records.
        self.device = cpu_device

    def _ordered_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for name in self._names
            for parameter in (self.projections[name].A, self.projections[name].B)
        )

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("block inputs must be a torch.Tensor")
        if inputs.device.type != "cpu":
            raise ValueError("NF4StreamingBlock inputs must be on CPU")
        if inputs.dtype is not torch.bfloat16:
            raise TypeError("NF4StreamingBlock inputs must have dtype torch.bfloat16")
        if inputs.ndim < 1:
            raise ValueError("block inputs must have at least one dimension")

    def _validate_parameters(self, parameters: tuple[torch.Tensor, ...]) -> None:
        expected = self._ordered_parameters()
        if len(parameters) != len(expected) or any(actual is not wanted for actual, wanted in zip(parameters, expected)):
            raise RuntimeError("streaming projection parameter order changed")
        for parameter in expected:
            if parameter.device.type != "cpu" or parameter.dtype is not torch.float32 or not parameter.requires_grad:
                raise ValueError("streaming A/B factors must be trainable FP32 CPU Parameters")

    def _run_block(self, inputs: torch.Tensor, lease: NF4Lease) -> torch.Tensor:
        used: set[str] = set()

        def project(name: str, hidden: torch.Tensor) -> torch.Tensor:
            if type(name) is not str or name not in self.projections:
                raise KeyError(f"unknown NF4 streaming projection {name!r}")
            used.add(name)
            packed, parts = lease.parts(name)
            factors = self.projections[name]
            return nf4_lora_linear_cpu(hidden, packed, parts, factors.A, factors.B, self._scale)

        output = self._block_fn(inputs, project)
        if used != set(self._names):
            missing = sorted(set(self._names).difference(used))
            raise RuntimeError(f"block_fn did not use every registered projection: {missing}")
        if not isinstance(output, torch.Tensor):
            raise TypeError("block_fn must return one torch.Tensor")
        if output.device.type != "cpu" or output.dtype is not torch.bfloat16:
            raise ValueError("block_fn output must be a BF16 CPU tensor")
        return output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_input(inputs)
        parameters = self._ordered_parameters()
        self._validate_parameters(parameters)
        return _NF4StreamingAutograd.apply(inputs, self, *parameters)


class _NF4StreamingAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, block: NF4StreamingBlock, *parameters: torch.Tensor):
        block._validate_input(inputs)
        block._validate_parameters(parameters)
        ctx.block = block
        # Keep identity references to the exact Function inputs without saving
        # them as tensors; saved_tensors remains limited to the block input.
        ctx.parameters = tuple(parameters)
        ctx.parameter_versions = tuple(parameter._version for parameter in parameters)
        ctx.save_for_backward(inputs)  # The block input is the only saved tensor.
        lease: NF4Lease | None = None
        try:
            with torch.no_grad():
                lease = block.pool.acquire("forward")
                output = block._run_block(inputs, lease)
                lease.complete()
            return output
        except BaseException as error:
            if lease is not None:
                _abort_preserving_error(lease, error)
            raise

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError("NF4StreamingBlock does not support higher-order gradients")
        block: NF4StreamingBlock = ctx.block
        (saved_input,) = ctx.saved_tensors
        parameters = ctx.parameters
        current_parameters = block._ordered_parameters()
        if len(current_parameters) != len(parameters) or any(
            current is not original for current, original in zip(current_parameters, parameters)
        ):
            raise RuntimeError("streaming A/B Parameters were replaced between forward and backward")
        if tuple(parameter._version for parameter in parameters) != ctx.parameter_versions:
            raise RuntimeError("streaming A/B factors changed in-place between forward and backward")
        need_input = bool(ctx.needs_input_grad[0])
        if any(not bool(needed) for needed in ctx.needs_input_grad[2:]):
            raise RuntimeError("all streaming A/B Parameters must receive gradients")
        lease = block.pool.acquire("backward")
        try:
            grad_input, grad_parameters = _recompute_vjp(
                block, saved_input, parameters, grad_output, lease, need_input
            )
            with torch.no_grad():
                lease.complete()
            return (grad_input, None, *grad_parameters)
        except BaseException as error:
            _abort_preserving_error(lease, error)
            raise


def _abort_preserving_error(lease: NF4Lease, original: BaseException) -> None:
    """Release an active lease after an operation failure, preserving both errors."""
    if not lease.is_active or lease._pool.stats["poisoned"]:
        return
    try:
        lease.abort()
    except BaseException as cleanup_error:
        original.add_note(f"NF4 lease abort also failed: {cleanup_error!r}")
        raise original from cleanup_error


def _recompute_vjp(
    block: NF4StreamingBlock,
    saved_input: torch.Tensor,
    parameters: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    lease: NF4Lease,
    need_input: bool,
) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
    recompute_input = saved_input.detach().requires_grad_(need_input)
    output: torch.Tensor | None = None
    try:
        with torch.enable_grad():
            output = block._run_block(recompute_input, lease)
            if not output.requires_grad:
                raise RuntimeError("streaming block output is disconnected from all trainable inputs")
            targets: tuple[torch.Tensor, ...] = ((recompute_input,) if need_input else ()) + parameters
            gradients = torch.autograd.grad(
                output,
                targets,
                grad_outputs=grad_output,
                allow_unused=False,
                create_graph=False,
                retain_graph=False,
            )
            if need_input:
                grad_input = gradients[0]
                grad_parameters = tuple(gradients[1:])
            else:
                grad_input = None
                grad_parameters = tuple(gradients)
            if len(grad_parameters) != len(parameters) or any(gradient is None for gradient in grad_parameters):
                raise RuntimeError("one or more streaming A/B Parameters are disconnected from the block output")
            # Release references to the recompute graph before the caller's lease barrier.
            del output, gradients, targets
            return grad_input, grad_parameters
    except BaseException:
        # Tracebacks retain frame locals.  Drop graph-bearing references before
        # the caller runs the lease abort barrier.
        output = None
        gradients = None
        targets = None
        recompute_input = None
        raise
