"""Controlled whole-block NF4 recomputation for native Qwen3.5 CPU layers.

The wrapper intentionally accepts only the pinned Transformers native decoder
layer.  Streaming is a synchronous CPU lease simulation; it makes no claim
about CUDA transfers or asynchronous execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType, MethodType
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.checkpoint import set_checkpoint_early_stop

from ..quantization.nf4_state import NF4Record
from .nf4_streaming import NF4Lease, NF4LeasePool

_FIXED_MODULE_MESSAGE = (
    "Controlled NF4 decoder modules are fixed to CPU with BF16 frozen state "
    "and FP32 adapter parameters; module conversions are unsupported"
)


def _reject_native_layer_apply(_module: nn.Module, _fn: Any, recurse: bool = True):
    """Keep direct conversion of the exposed native layer CPU-safe as well."""
    raise RuntimeError(_FIXED_MODULE_MESSAGE)


def _copy_records(records: Mapping[str, NF4Record]) -> dict[str, NF4Record]:
    if not isinstance(records, Mapping) or not records:
        raise ValueError("records must be a non-empty mapping of canonical names to NF4Record values")
    copied: dict[str, NF4Record] = {}
    for name, record in records.items():
        if type(name) is not str or not name:
            raise ValueError("NF4 record names must be non-empty strings")
        if not isinstance(record, NF4Record):
            raise TypeError(f"record {name!r} must be an NF4Record")
        if name in copied:
            raise ValueError(f"duplicate NF4 record name {name!r}")
        copied[name] = record
    return {name: copied[name] for name in sorted(copied)}


def _copy_projections(projections: Mapping[str, nn.Module]) -> dict[str, nn.Module]:
    if not isinstance(projections, Mapping) or not projections:
        raise ValueError("projections must be a non-empty mapping")
    copied: dict[str, nn.Module] = {}
    for name, projection in projections.items():
        if type(name) is not str or not name:
            raise ValueError("projection names must be non-empty strings")
        if name in copied:
            raise ValueError(f"duplicate projection name {name!r}")
        copied[name] = projection
    return {name: copied[name] for name in sorted(copied)}


def _cpu_tensor(value: Any, name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return value


def _snapshot_auxiliary(
    hidden_states: torch.Tensor,
    position_embeddings: Any,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
    """Validate per-call native inputs and detach owned CPU snapshots."""
    hidden = _cpu_tensor(hidden_states, "hidden_states", dtype=torch.bfloat16)
    if hidden.ndim < 2:
        raise ValueError("hidden_states must include batch and sequence dimensions")
    if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
        raise TypeError("position_embeddings must be a (cos, sin) pair")
    snapshots: list[torch.Tensor] = []
    for name, value in zip(("position_embeddings.cos", "position_embeddings.sin"), position_embeddings):
        tensor = _cpu_tensor(value, name)
        if not tensor.is_floating_point() or tensor.requires_grad:
            raise ValueError(f"{name} must be a non-gradient floating CPU tensor")
        snapshots.append(tensor.detach().clone(memory_format=torch.contiguous_format))
    if snapshots[0].shape != snapshots[1].shape or snapshots[0].dtype != snapshots[1].dtype:
        raise ValueError("position embedding cos/sin shapes and dtypes must match")

    mask_snapshot = None
    if attention_mask is not None:
        mask = _cpu_tensor(attention_mask, "attention_mask")
        if mask.requires_grad or not (mask.dtype is torch.bool or mask.is_floating_point()):
            raise ValueError("attention_mask must be a non-gradient CPU boolean or floating tensor")
        mask_snapshot = mask.detach().clone(memory_format=torch.contiguous_format)

    ids_snapshot = None
    if position_ids is not None:
        ids = _cpu_tensor(position_ids, "position_ids", dtype=torch.long)
        if ids.requires_grad:
            raise ValueError("position_ids must not require gradients")
        ids_snapshot = ids.detach().clone(memory_format=torch.contiguous_format)
    return (snapshots[0], snapshots[1]), mask_snapshot, ids_snapshot


def _validate_native_output(output: Any) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError("Qwen3_5DecoderLayer must return exactly one hidden-state Tensor")
    if output.device.type != "cpu" or output.dtype is not torch.bfloat16:
        raise ValueError("native decoder output must be a BF16 CPU Tensor")
    return output


class ControlledNF4Decoder(nn.Module):
    """Resident-checkpoint or lease-stream one exact native Qwen3.5 layer."""

    def __init__(
        self,
        layer: nn.Module,
        records: Mapping[str, NF4Record],
        projections: Mapping[str, nn.Module],
        *,
        mode: str,
    ) -> None:
        # Reject non-CPU state before registering/converting modules or creating
        # any tensor.  Native layer class checks are lazy to avoid import cycles.
        if type(mode) is not str or mode not in {"resident", "streaming"}:
            raise ValueError("mode must be 'resident' or 'streaming'")
        if not isinstance(layer, nn.Module):
            raise TypeError("layer must be a native torch.nn.Module")
        for tensor in (*layer.parameters(), *layer.buffers()):
            if tensor.device.type != "cpu":
                raise ValueError("ControlledNF4Decoder is fixed to CPU")

        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        if type(layer) is not Qwen3_5DecoderLayer:
            raise TypeError("layer must be exactly Transformers 5.13.0 Qwen3_5DecoderLayer")
        from ..adapters.qwen35.nf4_cpu import _NF4CPUProjection

        copied_records = _copy_records(records)
        copied_projections = _copy_projections(projections)
        if any(type(projection) is not _NF4CPUProjection for projection in copied_projections.values()):
            raise TypeError("every projection must be exactly _NF4CPUProjection")
        if len({id(projection) for projection in copied_projections.values()}) != len(copied_projections):
            raise ValueError("a projection module cannot occur under multiple mapping keys")
        record_names: list[str] = []
        for key, projection in copied_projections.items():
            record_name = projection.record_name
            if type(record_name) is not str or record_name not in copied_records:
                raise ValueError(f"projection {key!r} names no supplied canonical record")
            projection_record = getattr(projection, "record", copied_records[record_name])
            if projection_record is not copied_records[record_name]:
                raise ValueError(f"projection {key!r} is bound to a different NF4Record")
            if projection.mode != mode:
                raise ValueError(f"projection {key!r} mode does not match the controlled decoder")
            if projection.record_fingerprint != copied_records[record_name].fingerprint():
                raise ValueError(f"projection {key!r} fingerprint does not match its canonical NF4 record")
            record_names.append(record_name)
        if len(record_names) != len(set(record_names)) or set(record_names) != set(copied_records):
            raise ValueError("records must correspond one-to-one with all selected projections")

        # The native layer is intentionally exposed for model inspection. Guard
        # its own conversion entry point too, before _apply can reach a raw
        # parameter or a child projection callback.
        layer._apply = MethodType(_reject_native_layer_apply, layer)
        super().__init__()
        self.layer = layer
        self.mode = mode
        self.records = MappingProxyType(copied_records)
        self.projections = MappingProxyType(copied_projections)
        self._projection_order = tuple(copied_projections[key] for key in copied_projections)
        self._record_for_projection = MappingProxyType(
            {projection.record_name: projection for projection in self._projection_order}
        )
        self.pool = NF4LeasePool(self.records) if mode == "streaming" else None
        self._expected_parameter_ids: tuple[int, ...] = ()
        self._validate_parameter_inventory()

    def _apply(self, fn, recurse: bool = True):
        # Reject before nn.Module invokes fn on any child parameter/buffer.
        raise RuntimeError(_FIXED_MODULE_MESSAGE)

    def _adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        result: list[nn.Parameter] = []
        for projection in self._projection_order:
            a = projection.A
            b = projection.B
            if a is None and b is None:
                continue
            if not isinstance(a, nn.Parameter) or not isinstance(b, nn.Parameter):
                raise ValueError("rank-positive NF4 projections must expose both A and B Parameters")
            result.extend((a, b))
        if len({id(parameter) for parameter in result}) != len(result):
            raise ValueError("adapter Parameters must be distinct")
        return tuple(result)

    def _validate_parameter_inventory(self) -> tuple[nn.Parameter, ...]:
        explicit = self._adapter_parameters()
        if any(not parameter.requires_grad for parameter in explicit):
            raise ValueError("every explicit A/B Parameter must require gradients")
        if any(parameter.device.type != "cpu" or parameter.dtype is not torch.float32 for parameter in explicit):
            raise ValueError("A/B Parameters must remain FP32 on CPU")
        trainable = tuple(parameter for parameter in self.layer.parameters() if parameter.requires_grad)
        if len(trainable) != len(explicit) or {id(p) for p in trainable} != {id(p) for p in explicit}:
            raise ValueError("native layer trainable inventory must be exactly the selected A/B Parameters")
        current = tuple(id(parameter) for parameter in explicit)
        if self._expected_parameter_ids and current != self._expected_parameter_ids:
            raise RuntimeError("native layer A/B Parameter identity changed")
        if not self._expected_parameter_ids:
            self._expected_parameter_ids = current
        return explicit

    def _invoke_native(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        result = self.layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        return _validate_native_output(result)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if past_key_values is not None:
            raise ValueError("past_key_values/cache are unsupported; pass None")
        unknown = set(kwargs) - {"use_cache"}
        if unknown:
            raise TypeError(f"unsupported native decoder kwargs: {sorted(unknown)}")
        use_cache = kwargs.get("use_cache", None)
        if use_cache not in (None, False):
            raise ValueError("use_cache must be False or None")
        position_snapshot, mask_snapshot, ids_snapshot = _snapshot_auxiliary(
            hidden_states, position_embeddings, attention_mask, position_ids
        )
        parameters = self._validate_parameter_inventory()

        if self.mode == "resident":
            def recompute(hidden: torch.Tensor) -> torch.Tensor:
                return self._invoke_native(hidden, position_snapshot, mask_snapshot, ids_snapshot)

            with set_checkpoint_early_stop(False):
                output = torch_checkpoint(recompute, hidden_states, use_reentrant=False)
            return _validate_native_output(output)

        if self.pool is None:  # defensive invariant; no fallback to resident
            raise RuntimeError("streaming decoder is missing its NF4LeasePool")
        return _StreamingNativeDecoderFunction.apply(
            hidden_states,
            self,
            position_snapshot,
            mask_snapshot,
            ids_snapshot,
            *parameters,
        )

    def _bind_all(self, lease: NF4Lease) -> None:
        bound: list[nn.Module] = []
        try:
            for projection in self._projection_order:
                projection._set_lease(lease)
                bound.append(projection)
        except BaseException:
            for projection in reversed(bound):
                projection._set_lease(None)
            raise

    def _unbind_all(self) -> None:
        errors: list[BaseException] = []
        for projection in reversed(self._projection_order):
            try:
                projection._set_lease(None)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("failed to unbind every streaming NF4 projection") from errors[0]

    def _lease_scope(self, phase: str):
        if self.pool is None:
            raise RuntimeError("streaming decoder is missing its NF4LeasePool")
        lease = self.pool.acquire(phase)
        try:
            self._bind_all(lease)
        except BaseException:
            if lease.is_active and not self.pool.stats["poisoned"]:
                lease.abort()
            raise
        return lease

    def _release_scope(self, lease: NF4Lease, *, success: bool, primary_error: BaseException | None) -> None:
        cleanup_error: BaseException | None = None
        try:
            self._unbind_all()
        except BaseException as exc:
            cleanup_error = exc
        if lease.is_active and self.pool is not None and not self.pool.stats["poisoned"]:
            try:
                if success and cleanup_error is None:
                    lease.complete()
                else:
                    lease.abort()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


class _StreamingNativeDecoderFunction(torch.autograd.Function):
    """One forward lease plus one fresh backward lease for an exact block."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        controller: ControlledNF4Decoder,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        *parameters: nn.Parameter,
    ) -> torch.Tensor:
        current = controller._validate_parameter_inventory()
        if len(current) != len(parameters) or any(a is not b for a, b in zip(current, parameters)):
            raise RuntimeError("streaming autograd A/B inputs do not match the native layer inventory")
        ctx.controller = controller
        ctx.parameters = tuple(parameters)
        ctx.parameter_versions = tuple(parameter._version for parameter in parameters)
        ctx.position_embeddings = position_embeddings
        ctx.attention_mask = attention_mask
        ctx.position_ids = position_ids
        ctx.save_for_backward(hidden_states)
        lease = controller._lease_scope("forward")
        primary: BaseException | None = None
        try:
            with torch.no_grad():
                output = controller._invoke_native(
                    hidden_states, position_embeddings, attention_mask, position_ids
                )
        except BaseException as exc:
            primary = exc
            controller._release_scope(lease, success=False, primary_error=primary)
            raise
        controller._release_scope(lease, success=True, primary_error=None)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("higher-order gradients are unsupported by streaming NF4 decoder recomputation")
        controller: ControlledNF4Decoder = ctx.controller
        original_parameters: tuple[nn.Parameter, ...] = ctx.parameters
        current = controller._validate_parameter_inventory()
        if len(current) != len(original_parameters) or any(a is not b for a, b in zip(current, original_parameters)):
            raise RuntimeError("native layer A/B Parameter identity changed between forward and backward")
        if tuple(parameter._version for parameter in original_parameters) != ctx.parameter_versions:
            raise RuntimeError("native layer A/B Parameter version changed between forward and backward")
        if len(ctx.needs_input_grad) < 1 or any(not needs for needs in ctx.needs_input_grad[5:]):
            raise RuntimeError("every explicit A/B input must require gradients")
        if grad_output.device.type != "cpu" or grad_output.dtype is not torch.bfloat16:
            raise ValueError("streaming decoder grad_output must be BF16 on CPU")

        saved_hidden = ctx.saved_tensors[0]
        hidden = saved_hidden.detach().requires_grad_(True)
        lease = controller._lease_scope("backward")
        primary: BaseException | None = None
        try:
            with torch.enable_grad():
                output = controller._invoke_native(
                    hidden,
                    ctx.position_embeddings,
                    ctx.attention_mask,
                    ctx.position_ids,
                )
                targets = (hidden, *original_parameters)
                gradients = torch.autograd.grad(
                    output,
                    targets,
                    grad_output,
                    allow_unused=False,
                    create_graph=False,
                    retain_graph=False,
                )
            grad_input, *parameter_gradients = gradients
            if grad_input is None or any(gradient is None for gradient in parameter_gradients):
                raise RuntimeError("native decoder input or adapter gradient is disconnected")
            if tuple(parameter._version for parameter in original_parameters) != ctx.parameter_versions:
                raise RuntimeError("native layer A/B Parameter changed during backward recomputation")
        except BaseException as exc:
            primary = exc
            controller._release_scope(lease, success=False, primary_error=primary)
            raise
        # Release graph references before the completion barrier returns the
        # lease's owned NF4 copies; gradients themselves are returned normally.
        del output, targets
        controller._release_scope(lease, success=True, primary_error=None)
        return (grad_input, None, None, None, None, *parameter_gradients)
