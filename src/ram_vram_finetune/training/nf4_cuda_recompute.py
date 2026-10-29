"""Controlled whole-block CUDA recomputation for native Qwen3.5 decoder layers.

Only the exact pinned Transformers decoder class and SpillTune CUDA
projections are supported. Streaming uses explicit A/B autograd inputs and a
fresh preflight-bound NF4 lease for forward and backward; arbitrary callback modules
and hidden trainable captures are not accepted.
"""
from __future__ import annotations

from .. import diagnostic_timer as diag

from collections.abc import Mapping
from types import MappingProxyType, MethodType
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.checkpoint import set_checkpoint_early_stop

from ..quantization.nf4_state import NF4Record
from .nf4_cuda_pool import CUDANF4Lease, CUDANF4LeasePool

_FIXED_MODULE_MESSAGE = (
    "Controlled NF4 CUDA decoder modules are fixed to CUDA:0 with BF16 frozen "
    "state and FP32 adapter parameters; module conversions are unsupported"
)


def _reject_native_layer_apply(_module: nn.Module, _fn: Any, recurse: bool = True):
    """Guard direct conversion of the exposed native layer before any callback."""
    raise RuntimeError(_FIXED_MODULE_MESSAGE)


def _copy_records(records: Mapping[str, NF4Record]) -> dict[str, NF4Record]:
    if not isinstance(records, Mapping) or not records:
        raise ValueError("records must be a non-empty mapping of canonical names to NF4Record values")
    copied: dict[str, NF4Record] = {}
    for name, record in records.items():
        if type(name) is not str or not name:
            raise ValueError("NF4 record names must be non-empty strings")
        if not isinstance(record, NF4Record):
            raise TypeError(f"record {name!r} must be a canonical NF4Record")
        copied[name] = record
    return {name: copied[name] for name in sorted(copied)}


def _copy_projections(projections: Mapping[str, nn.Module]) -> dict[str, nn.Module]:
    if not isinstance(projections, Mapping) or not projections:
        raise ValueError("projections must be a non-empty mapping")
    copied: dict[str, nn.Module] = {}
    for name, projection in projections.items():
        if type(name) is not str or not name:
            raise ValueError("projection names must be non-empty strings")
        copied[name] = projection
    return {name: copied[name] for name in sorted(copied)}


def _cuda0_tensor(value: Any, name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.device.type != "cuda" or value.device.index != 0:
        raise ValueError(f"{name} must be on cuda:0")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return value


def _snapshot_auxiliary(
    hidden_states: torch.Tensor,
    position_embeddings: Any,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
    """Validate and clone every non-gradient native input for this invocation."""
    hidden = _cuda0_tensor(hidden_states, "hidden_states", dtype=torch.bfloat16)
    if hidden.ndim < 2:
        raise ValueError("hidden_states must include batch and sequence dimensions")
    if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
        raise TypeError("position_embeddings must be a (cos, sin) pair")
    snapshots: list[torch.Tensor] = []
    for name, value in zip(("position_embeddings.cos", "position_embeddings.sin"), position_embeddings):
        tensor = _cuda0_tensor(value, name)
        if not tensor.is_floating_point() or tensor.requires_grad:
            raise ValueError(f"{name} must be a non-gradient floating CUDA tensor")
        snapshots.append(tensor.detach().clone(memory_format=torch.contiguous_format))
    if snapshots[0].shape != snapshots[1].shape or snapshots[0].dtype != snapshots[1].dtype:
        raise ValueError("position embedding cos/sin shapes and dtypes must match")

    mask_snapshot = None
    if attention_mask is not None:
        mask = _cuda0_tensor(attention_mask, "attention_mask")
        if mask.requires_grad or not (mask.dtype is torch.bool or mask.is_floating_point()):
            raise ValueError("attention_mask must be a non-gradient CUDA boolean or floating tensor")
        mask_snapshot = mask.detach().clone(memory_format=torch.contiguous_format)

    ids_snapshot = None
    if position_ids is not None:
        ids = _cuda0_tensor(position_ids, "position_ids", dtype=torch.long)
        if ids.requires_grad:
            raise ValueError("position_ids must not require gradients")
        ids_snapshot = ids.detach().clone(memory_format=torch.contiguous_format)
    return (snapshots[0], snapshots[1]), mask_snapshot, ids_snapshot


def _validate_native_output(output: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError("Qwen3_5DecoderLayer must return exactly one hidden-state Tensor")
    if output.device.type != "cuda" or output.device.index != 0 or output.dtype is not torch.bfloat16:
        raise ValueError("native decoder output must be a BF16 CUDA:0 Tensor")
    if output.shape != hidden_states.shape:
        raise ValueError("native decoder output shape changed")
    return output


class ControlledNF4CUDADecoder(nn.Module):
    """Resident-checkpoint or lease-stream one exact native Qwen3.5 layer."""

    def __init__(
        self,
        layer: nn.Module,
        records: Mapping[str, NF4Record],
        projections: Mapping[str, nn.Module],
        *,
        mode: str,
        backend_run: Any = None,
        copy_stream: Any = None,
    ) -> None:
        if type(mode) is not str or mode not in {"resident", "streaming"}:
            raise ValueError("mode must be 'resident' or 'streaming'")
        if not isinstance(layer, nn.Module):
            raise TypeError("layer must be a native torch.nn.Module")
        for tensor in (*layer.parameters(), *layer.buffers()):
            if tensor.device.type != "cuda" or tensor.device.index != 0:
                raise ValueError("ControlledNF4CUDADecoder requires all native state on CUDA:0")

        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        if type(layer) is not Qwen3_5DecoderLayer:
            raise TypeError("layer must be exactly Transformers 5.13.0 Qwen3_5DecoderLayer")
        from ..adapters.qwen35.nf4_cuda import _NF4CUDAProjection

        copied_records = _copy_records(records)
        copied_projections = _copy_projections(projections)
        if any(type(projection) is not _NF4CUDAProjection for projection in copied_projections.values()):
            raise TypeError("every projection must be exactly _NF4CUDAProjection")
        if len({id(projection) for projection in copied_projections.values()}) != len(copied_projections):
            raise ValueError("a projection module cannot occur under multiple mapping keys")
        record_names: list[str] = []
        for key, projection in copied_projections.items():
            record_name = projection.record_name
            if type(record_name) is not str or record_name not in copied_records:
                raise ValueError(f"projection {key!r} names no supplied canonical record")
            if projection.mode != mode:
                raise ValueError(f"projection {key!r} mode does not match the controlled decoder")
            if projection.record_fingerprint != copied_records[record_name].fingerprint():
                raise ValueError(f"projection {key!r} fingerprint does not match its canonical NF4 record")
            record_names.append(record_name)
        if len(record_names) != len(set(record_names)) or set(record_names) != set(copied_records):
            raise ValueError("records must correspond one-to-one with all selected projections")

        # The native layer remains public for inspection. Guard its conversion
        # entry point before _apply can invoke any child Tensor conversion.
        layer._apply = MethodType(_reject_native_layer_apply, layer)
        super().__init__()
        self.layer = layer
        self._diagnostic = diag.active_timer() is not None
        self.mode = mode
        self.records = MappingProxyType(copied_records)
        self.projections = MappingProxyType(copied_projections)
        self._projection_order = tuple(copied_projections[key] for key in copied_projections)
        self._record_for_projection = MappingProxyType(
            {projection.record_name: projection for projection in self._projection_order}
        )
        self.pool = CUDANF4LeasePool(self.records, backend_run=backend_run, preflight_only=True, copy_stream=copy_stream) if mode == "streaming" else None
        if self.pool is not None:self.pool.preflight()
        self._expected_parameter_ids: tuple[int, ...] = ()
        self._runtime_parameters = self._validate_parameter_inventory()
        for name in ('_invoke_native','_lease_scope','_release_scope'):
            setattr(self,name,diag.bind(getattr(self,name)))

    def _apply(self, fn, recurse: bool = True):
        # Reject wrapper conversion before nn.Module invokes fn on any child.
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
        if any(
            parameter.device.type != "cuda"
            or parameter.device.index != 0
            or parameter.dtype is not torch.float32
            for parameter in explicit
        ):
            raise ValueError("A/B Parameters must remain FP32 on CUDA:0")
        trainable = tuple(parameter for parameter in self.layer.parameters() if parameter.requires_grad)
        if len(trainable) != len(explicit) or {id(p) for p in trainable} != {id(p) for p in explicit}:
            raise ValueError("native layer trainable inventory must be exactly the selected A/B Parameters")
        current = tuple(id(parameter) for parameter in explicit)
        if self._expected_parameter_ids and current != self._expected_parameter_ids:
            raise RuntimeError("native layer A/B Parameter identity changed")
        if not self._expected_parameter_ids:
            self._expected_parameter_ids = current
        return explicit

    def _current_adapter_parameters(self):
        # Explicit autograd inputs must still be the same A/B objects. Do not
        # enumerate the native module tree, devices or trainable flags at FW/BW.
        current=self._adapter_parameters()
        if len(current)!=len(self._runtime_parameters) or any(a is not b for a,b in zip(current,self._runtime_parameters)):
            raise RuntimeError('native layer A/B Parameter identity changed')
        return current

    @diag.timed('layer.native_graph',tag=diag.record_tag)
    def _invoke_native(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self._diagnostic:
            result = self.layer(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, position_ids=position_ids, past_key_values=None, use_cache=False)
        else:
            with diag.forward_device_span():
                result = self.layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
        return _validate_native_output(result, hidden_states)

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
        if kwargs.get("use_cache", None) not in (None, False):
            raise ValueError("use_cache must be False or None")
        position_snapshot, mask_snapshot, ids_snapshot = _snapshot_auxiliary(
            hidden_states, position_embeddings, attention_mask, position_ids
        )
        parameters = self._current_adapter_parameters() if self.mode=="resident" else self._runtime_parameters

        if self.mode == "resident":
            def recompute(hidden: torch.Tensor) -> torch.Tensor:
                return self._invoke_native(hidden, position_snapshot, mask_snapshot, ids_snapshot)

            with set_checkpoint_early_stop(False):
                output = torch_checkpoint(recompute, hidden_states, use_reentrant=False)
            return _validate_native_output(output, hidden_states)

        if self.pool is None:
            raise RuntimeError("streaming decoder is missing its CUDANF4LeasePool")
        return _StreamingNativeCUDADecoderFunction.apply(
            hidden_states,
            self,
            position_snapshot,
            mask_snapshot,
            ids_snapshot,
            *parameters,
        )

    def _bind_all(self, lease: CUDANF4Lease) -> None:
        bound: list[nn.Module] = []
        try:
            for projection in self._projection_order:
                projection._set_lease(lease)
                bound.append(projection)
        except BaseException:
            for projection in reversed(bound):
                try:
                    projection._set_lease(None)
                except BaseException:
                    pass
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

    @diag.timed('layer.bind_acquire',tag=diag.record_tag)
    def _lease_scope(self, phase: str) -> CUDANF4Lease:
        if self.pool is None:
            raise RuntimeError("streaming decoder is missing its CUDANF4LeasePool")
        lease = self.pool.acquire(phase)
        try:
            self._bind_all(lease)
        except BaseException as exc:
            if lease.is_active and not self.pool.poisoned:
                try:
                    lease.abort()
                except BaseException as cleanup_error:
                    self._note(exc, f"streaming lease abort failed: {cleanup_error}")
            raise
        return lease

    @staticmethod
    def _note(error: BaseException, message: str) -> None:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            add_note(message)

    @diag.timed('layer.unbind_release',tag=diag.record_tag)
    def _release_scope(
        self,
        lease: CUDANF4Lease,
        *,
        success: bool,
        primary_error: BaseException | None,
    ) -> None:
        cleanup_errors: list[BaseException] = []
        try:
            self._unbind_all()
        except BaseException as exc:
            cleanup_errors.append(exc)
        if lease.is_active and self.pool is not None and not self.pool.poisoned:
            try:
                if success and not cleanup_errors:
                    lease.complete()
                else:
                    lease.abort()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if primary_error is not None:
                for error in cleanup_errors:
                    self._note(primary_error, f"streaming lease cleanup failed: {type(error).__name__}: {error}")
            else:
                raise RuntimeError("streaming NF4 lease cleanup failed") from cleanup_errors[0]


class _StreamingNativeCUDADecoderFunction(torch.autograd.Function):
    """One forward lease plus one fresh backward lease; payload checked at startup."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        controller: ControlledNF4CUDADecoder,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        *parameters: nn.Parameter,
    ) -> torch.Tensor:
        current = controller._current_adapter_parameters()
        if len(current) != len(parameters) or any(a is not b for a, b in zip(current, parameters)):
            raise RuntimeError("streaming autograd A/B inputs do not match the native layer inventory")
        ctx.controller = controller
        ctx.parameters = tuple(parameters)
        ctx.parameter_versions = tuple(parameter._version for parameter in parameters)
        # These are per-invocation owned snapshots, not mutable controller state.
        ctx.position_embeddings = position_embeddings
        ctx.attention_mask = attention_mask
        ctx.position_ids = position_ids
        ctx.save_for_backward(hidden_states)
        lease = controller._lease_scope("forward")
        try:
            with torch.no_grad():
                output = controller._invoke_native(
                    hidden_states, position_embeddings, attention_mask, position_ids
                )
        except BaseException as exc:
            controller._release_scope(lease, success=False, primary_error=exc)
            raise
        controller._release_scope(lease, success=True, primary_error=None)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("higher-order gradients are unsupported by streaming NF4 CUDA recomputation")
        controller: ControlledNF4CUDADecoder = ctx.controller
        original_parameters: tuple[nn.Parameter, ...] = ctx.parameters
        current = controller._current_adapter_parameters()
        if len(current) != len(original_parameters) or any(a is not b for a, b in zip(current, original_parameters)):
            raise RuntimeError("native layer A/B Parameter identity changed between forward and backward")
        if tuple(parameter._version for parameter in original_parameters) != ctx.parameter_versions:
            raise RuntimeError("native layer A/B Parameter version changed between forward and backward")
        if len(ctx.needs_input_grad) < 1 or any(not needs for needs in ctx.needs_input_grad[5:]):
            raise RuntimeError("every explicit A/B input must require gradients")
        if grad_output.device.type != "cuda" or grad_output.device.index != 0 or grad_output.dtype is not torch.bfloat16:
            raise ValueError("streaming decoder grad_output must be BF16 on CUDA:0")

        hidden = ctx.saved_tensors[0].detach().requires_grad_(True)
        lease = controller._lease_scope("backward")
        try:
            if not controller._diagnostic:
                with torch.enable_grad():
                    output = controller._invoke_native(hidden, ctx.position_embeddings, ctx.attention_mask, ctx.position_ids)
                    targets = (hidden, *original_parameters)
                    gradients = torch.autograd.grad(output, targets, grad_output, allow_unused=False, create_graph=False, retain_graph=False)
            else:
                with torch.enable_grad(), diag.span("layer.backward_recompute"), diag.device_span("layer.backward_recompute_vjp.stream_interval"):
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
            controller._release_scope(lease, success=False, primary_error=exc)
            raise
        del output, targets
        controller._release_scope(lease, success=True, primary_error=None)
        return (grad_input, None, None, None, None, *parameter_gradients)


__all__ = ("ControlledNF4CUDADecoder",)
