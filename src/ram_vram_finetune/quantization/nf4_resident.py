"""Resident CPU NF4 + DQ + BF16 LoRA primitives.

This module provides model-independent linear and block building blocks. It
keeps a private CPU snapshot of each record's official packed state and uses the
shared backend for the frozen base operation and its input VJP.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from numbers import Real
from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.checkpoint import set_checkpoint_early_stop

from . import nf4_backend
from .nf4_state import NF4Record

_FIXED_MODULE_MESSAGE = (
    "NF4 resident modules are fixed to CPU with FP32 parameters; "
    "module conversions via .to(), .cuda(), .half(), or related methods are unsupported"
)


def _cpu_device(device: str | torch.device) -> torch.device:
    """Validate a requested device before constructing any tensors or modules."""
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be 'cpu' or torch.device('cpu')")
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"unsupported device {device!r}; NF4 resident modules are CPU-only") from exc
    if parsed.type != "cpu" or parsed.index is not None:
        raise ValueError(f"unsupported device {parsed}; NF4 resident modules are CPU-only")
    return torch.device("cpu")


def _positive_rank(rank: int) -> int:
    if type(rank) is not int or rank <= 0:
        raise ValueError("rank must be a positive integer")
    return rank


def _finite_alpha(alpha: Real, *, name: str = "alpha") -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(alpha)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _cpu_tensor(value: Any, name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU, got {value.device}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    return value


def _validate_adapter_dimensions(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: Real,
    base_shape: tuple[int, int],
) -> float:
    _cpu_tensor(x, "x", dtype=torch.bfloat16)
    _cpu_tensor(A, "A", dtype=torch.float32)
    _cpu_tensor(B, "B", dtype=torch.float32)
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be two-dimensional matrices")
    if any(extent <= 0 for extent in A.shape) or any(extent <= 0 for extent in B.shape):
        raise ValueError("A and B dimensions must be positive")
    if A.shape[0] != B.shape[1]:
        raise ValueError(f"A rank {A.shape[0]} does not match B rank {B.shape[1]}")
    if x.shape[-1] != A.shape[1]:
        raise ValueError(f"x feature size {x.shape[-1]} does not match A input size {A.shape[1]}")
    if base_shape[1] != A.shape[1]:
        raise ValueError(f"NF4 input size {base_shape[1]} does not match A input size {A.shape[1]}")
    if base_shape[0] != B.shape[0]:
        raise ValueError(f"NF4 output size {base_shape[0]} does not match B output size {B.shape[0]}")
    return _finite_alpha(scale, name="scale")


def nf4_lora_linear_cpu(
    x: torch.Tensor,
    packed: torch.Tensor,
    parts: Mapping[str, Any],
    A: torch.Tensor,
    B: torch.Tensor,
    scale: Real,
) -> torch.Tensor:
    """Apply an official frozen NF4 base linear plus a trainable FP32 LoRA path."""
    # Reject wrong-device or wrong-dtype inputs before asking the backend to
    # inspect or dispatch any NF4 operation.
    _cpu_tensor(x, "x", dtype=torch.bfloat16)
    _cpu_tensor(A, "A", dtype=torch.float32)
    _cpu_tensor(B, "B", dtype=torch.float32)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be two-dimensional matrices")
    # Validate adapter-only dimensions before backend inspection. The supplied
    # output/input pair here is provisional; the canonical base shape below is
    # checked against it after complete state validation.
    scalar = _validate_adapter_dimensions(
        x,
        A,
        B,
        scale,
        (B.shape[0], A.shape[1]),
    )
    # inspect_state validates the full official NF4/DQ state and gives us the
    # logical matrix dimensions before either branch is dispatched.
    logical = nf4_backend.inspect_state(packed, parts)
    try:
        base_shape = tuple(logical["shape"])
    except (KeyError, TypeError) as exc:
        raise ValueError("NF4 backend returned malformed logical shape metadata") from exc
    if len(base_shape) != 2 or any(type(n) is not int or n <= 0 for n in base_shape):
        raise ValueError("NF4 backend returned malformed logical shape metadata")
    if base_shape[1] != A.shape[1]:
        raise ValueError(f"NF4 input size {base_shape[1]} does not match A input size {A.shape[1]}")
    if base_shape[0] != B.shape[0]:
        raise ValueError(f"NF4 output size {base_shape[0]} does not match B output size {B.shape[0]}")

    base = nf4_backend.linear_cpu(x, packed, parts)
    adapter = F.linear(F.linear(x.float(), A), B)
    return (base.float() + adapter * scalar).to(dtype=torch.bfloat16)


class NF4ResidentLinear(nn.Module):
    """One CPU-resident official NF4 record with deterministic FP32 LoRA factors."""

    def __init__(
        self,
        record: NF4Record,
        *,
        rank: int = 2,
        alpha: float = 4.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        _cpu_device(device)  # Fail closed before exporting state or allocating Parameters.
        rank = _positive_rank(rank)
        alpha_value = _finite_alpha(alpha)
        if not isinstance(record, NF4Record):
            raise TypeError("record must be an NF4Record")
        shape = record.shape
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(type(extent) is not int or extent <= 0 for extent in shape)
        ):
            raise ValueError("record must describe a positive two-dimensional NF4 weight")

        packed, parts = record.export_parts()
        self.record = record
        self.rank = rank
        self.alpha = alpha_value
        self.scale = alpha_value / rank
        self._packed = packed
        self._parts = parts
        self.A = nn.Parameter(torch.zeros((rank, shape[1]), dtype=torch.float32, device="cpu"))
        self.B = nn.Parameter(torch.zeros((shape[0], rank), dtype=torch.float32, device="cpu"))

    def _apply(self, fn, recurse: bool = True):
        # nn.Module.to(), cuda(), half(), and related conversions all route
        # through _apply. Reject before the callback can allocate or move a
        # single parameter, keeping packed state and LoRA parameters CPU/FP32.
        raise RuntimeError(_FIXED_MODULE_MESSAGE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nf4_lora_linear_cpu(x, self._packed, self._parts, self.A, self.B, self.scale)


class NF4ResidentBlock(nn.Module):
    """Checkpoint a whole caller-defined block around resident NF4 projections."""

    def __init__(
        self,
        records: Mapping[str, NF4Record],
        block_fn: Callable[[torch.Tensor, Callable[[str, torch.Tensor], torch.Tensor]], torch.Tensor],
        *,
        rank: int = 2,
        alpha: float = 4.0,
        checkpoint: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        _cpu_device(device)  # Fail closed before constructing any projection tensors.
        rank = _positive_rank(rank)
        alpha_value = _finite_alpha(alpha)
        if not isinstance(records, Mapping) or not records:
            raise ValueError("records must be a non-empty mapping of names to NF4Record values")
        if not callable(block_fn):
            raise TypeError("block_fn must be callable")
        if type(checkpoint) is not bool:
            raise TypeError("checkpoint must be a bool")

        names = list(records.keys())
        if any(type(name) is not str or not name for name in names):
            raise ValueError("projection names must be non-empty strings")
        if any(not isinstance(record, NF4Record) for record in records.values()):
            raise TypeError("every projection value must be an NF4Record")

        modules = nn.ModuleList()
        by_name: dict[str, NF4ResidentLinear] = {}
        for name, record in records.items():
            projection = NF4ResidentLinear(
                record,
                rank=rank,
                alpha=alpha_value,
                device="cpu",
            )
            modules.append(projection)
            by_name[name] = projection

        self._projection_modules = modules
        # ModuleList gives each child a stable numeric registration name while
        # this public view preserves every caller key, including names with dots.
        self.projections = MappingProxyType(by_name)
        self.block_fn = block_fn
        self.checkpoint = checkpoint
        self.rank = rank
        self.alpha = alpha_value
        self.scale = alpha_value / rank

    def _apply(self, fn, recurse: bool = True):
        # Guard at the block boundary before nn.Module can recurse into its
        # child projections and invoke a device/dtype conversion callback.
        raise RuntimeError(_FIXED_MODULE_MESSAGE)

    def _project(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if type(name) is not str:
            raise TypeError("project name must be a string")
        try:
            projection = self.projections[name]
        except KeyError as exc:
            raise KeyError(f"unknown NF4 projection name {name!r}") from exc
        _cpu_tensor(x, f"input for projection {name!r}", dtype=torch.bfloat16)
        return projection(x)

    @staticmethod
    def _validate_output(output: Any) -> torch.Tensor:
        _cpu_tensor(output, "block_fn output", dtype=torch.bfloat16)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _cpu_tensor(x, "x", dtype=torch.bfloat16)
        if x.ndim < 1:
            raise ValueError("x must have at least one dimension")
        if not self.checkpoint:
            return self._validate_output(self.block_fn(x, self._project))
        # Disable non-reentrant checkpoint's early-stop optimization so backward
        # recomputes the complete same block boundary used by the resident arm.
        with set_checkpoint_early_stop(False):
            return self._validate_output(
                torch_checkpoint(
                    self.block_fn,
                    x,
                    self._project,
                    use_reentrant=False,
                )
            )
