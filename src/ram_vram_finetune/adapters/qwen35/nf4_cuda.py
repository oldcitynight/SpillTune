"""Qwen3.5 NF4 + DQ + BF16 CUDA adapter with resident and streaming modes.

Streaming stages one decoder layer's packed state from pinned CPU memory for
forward and backward recomputation. The adapter never downloads model weights
or silently falls back to a different device or quantization path.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType, MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_EXPECTED_ARTIFACT = "13570939377d60c2b8f099faafc4ff332f4d6eb1053f7238b5e475b630655584"
_PREFIX = "model.language_model."
_EXPECTED_LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "full_attention")
_FIXED_CUDA_MESSAGE = "NF4 Qwen3.5 CUDA modules are fixed to CUDA:0; module conversions are unsupported"
_PROFILE_UNSET = object()

def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _finite_alpha(alpha: Real) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a finite real number")
    value = float(alpha)
    if not math.isfinite(value):
        raise ValueError("alpha must be finite")
    return value


def _cuda0_device(device: str | torch.device) -> torch.device:
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be the explicit 'cuda:0' device")
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("NF4Qwen35CUDA accepts CUDA:0 only") from exc
    if parsed.type != "cuda" or parsed.index != 0:
        raise ValueError("NF4Qwen35CUDA accepts CUDA:0 only")
    return torch.device("cuda:0")


def _validate_tiny_profile(config: Mapping[str, Any], artifact_identity: str) -> None:
    """Reject any artifact outside the exactly preregistered tiny profile."""
    if artifact_identity != _EXPECTED_ARTIFACT:
        raise ValueError("CUDA model phase requires the frozen tiny NF4 artifact identity")
    if not isinstance(config, Mapping):
        raise ValueError("CUDA model phase requires the resolved tiny text config")
    expected = {
        "model_type": "qwen3_5_text",
        "hidden_size": 16,
        "intermediate_size": 32,
        "vocab_size": 32,
        "num_hidden_layers": 4,
        "layer_types": list(_EXPECTED_LAYER_TYPES),
        "tie_word_embeddings": True,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("CUDA model phase accepts only the preregistered tiny Qwen3.5 profile")


def _record_type():
    from ...quantization.nf4_state import NF4Record

    return NF4Record


def _reject_fixed_cuda_apply(_module, _fn, recurse: bool = True):
    raise RuntimeError(_FIXED_CUDA_MESSAGE)


def _lock_module_tree(root: nn.Module) -> None:
    """Reject .to/.cuda/.half on each exposed subtree before tensor callbacks."""
    for submodule in root.modules():
        submodule._apply = MethodType(_reject_fixed_cuda_apply, submodule)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if name not in parent._modules:
        raise ValueError(f"native Qwen projection path is not a registered module: {path!r}")
    parent._modules[name] = replacement


class _NF4CUDAProjection(nn.Module):
    """One official packed NF4 matrix and optional FP32 CUDA LoRA factors."""

    def __init__(
        self,
        record,
        record_name: str,
        *,
        mode: str,
        rank: int,
        alpha: float,
        device: torch.device,
        resident_state=None,
    ) -> None:
        super().__init__()
        if not isinstance(record, _record_type()):
            raise TypeError("record must be a canonical NF4Record")
        if type(record_name) is not str or not record_name:
            raise ValueError("record_name must be a canonical non-empty key")
        if mode not in ("resident", "streaming"):
            raise ValueError("projection mode must be 'resident' or 'streaming'")
        if type(rank) is not int or rank < 0:
            raise ValueError("projection rank must be a non-negative integer")
        if device != torch.device("cuda:0"):
            raise ValueError("NF4 CUDA projection requires CUDA:0")
        self.mode = mode
        self.record_name = record_name
        self.record_fingerprint = record.fingerprint()
        shape = tuple(record.shape)
        if len(shape) != 2 or any(type(extent) is not int or extent <= 0 for extent in shape):
            raise ValueError("NF4 CUDA projection requires a positive two-dimensional record")
        self.rank = rank
        self.alpha = _finite_alpha(alpha)
        self.scale = self.alpha / rank if rank else 0.0
        self.device = device
        self._lease = None
        self._resident_state = None
        from ... import diagnostic_timer as diag
        from ...quantization.nf4_backend_cuda import linear_cuda
        self._linear_cuda = diag.bind(linear_cuda)
        if mode == "resident":
            if resident_state is None or getattr(resident_state, "source_fingerprint", None) != self.record_fingerprint:
                raise ValueError("resident CUDA state does not match its canonical NF4 record")
            if tuple(getattr(resident_state, "shape", ())) != tuple(record.shape):
                raise ValueError("resident CUDA state shape does not match its NF4 record")
            self._resident_state = resident_state
        elif resident_state is not None:
            raise ValueError("streaming projection cannot retain a resident CUDA state")
        if rank:
            self.A = nn.Parameter(torch.zeros((rank, shape[1]), dtype=torch.float32, device=device))
            self.B = nn.Parameter(torch.zeros((shape[0], rank), dtype=torch.float32, device=device))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
        self._apply = MethodType(_reject_fixed_cuda_apply, self)

    def _set_lease(self, lease_or_none) -> None:
        if self.mode != "streaming":
            if lease_or_none is not None:
                raise RuntimeError("resident NF4 CUDA projection cannot bind a streaming lease")
            self._lease = None
            return
        if lease_or_none is not None and self._lease is not None:
            raise RuntimeError("NF4 CUDA projection already has an active lease")
        self._lease = lease_or_none

    def _active_state(self):
        if self.mode == "resident":
            return self._resident_state
        lease = self._lease
        if lease is None:
            raise RuntimeError("streaming NF4 CUDA projection requires an active block lease")
        state = lease.parts(self.record_name)
        return state

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or inputs.device != self.device:
            raise ValueError("NF4 CUDA projection inputs must be on CUDA:0")
        if inputs.dtype is not torch.bfloat16 or inputs.ndim < 1:
            raise ValueError("NF4 CUDA projection inputs must be BF16 with the recorded input width")
        state = self._active_state()
        if inputs.shape[-1] != state.shape[1]:
            raise ValueError("NF4 CUDA projection inputs must match the recorded input width")
        # Resident startup verification synchronized all copies once. Streaming
        # acquire queued the final same-stream ready event for this generation.
        base = self._linear_cuda(inputs, state)
        if not self.rank:
            return base
        if self.A.device != self.device or self.B.device != self.device:
            raise RuntimeError("NF4 CUDA LoRA factors escaped CUDA:0")
        return (base.float() + F.linear(F.linear(inputs.float(), self.A), self.B) * self.scale).to(
            dtype=torch.bfloat16
        )


class NF4Qwen35CUDA:
    """Native Qwen3.5 CUDA runtime for a tiny batch or explicit source-bound profile."""

    def __init__(
        self,
        artifact,
        *,
        mode: str = "resident",
        runtime_profile=None,
        rank: Any = _PROFILE_UNSET,
        alpha: Any = _PROFILE_UNSET,
        seed: Any = _PROFILE_UNSET,
        device: str | torch.device = "cuda:0",
    ) -> None:
        if mode not in ("resident", "streaming"):
            raise ValueError("mode must be 'resident' or 'streaming'")
        execution_arm = (
            "spilltune_streaming" if mode == "streaming" else "shared_recompute_resident_secondary"
        )
        self.device = _cuda0_device(device)
        from ... import diagnostic_timer as diag
        self._full_chain_timer_enabled = diag.active_timer() is not None
        if runtime_profile is None:
            rank = 2 if rank is _PROFILE_UNSET else rank
            alpha = 4.0 if alpha is _PROFILE_UNSET else alpha
            seed = 20261019 if seed is _PROFILE_UNSET else seed
            if type(rank) is not int or rank != 2:
                raise ValueError("tiny CUDA qualification fixes LoRA rank at 2")
            alpha_value = _finite_alpha(alpha)
            if alpha_value != 4.0:
                raise ValueError("tiny CUDA qualification fixes LoRA alpha at 4")
            if type(seed) is not int or seed != 20261019:
                raise ValueError("tiny CUDA qualification fixes the adapter seed")
        else:
            if any(value is _PROFILE_UNSET for value in (rank, alpha, seed)):
                raise ValueError("runtime_profile requires explicitly supplied rank, alpha, and seed")
            if type(rank) is not int or rank <= 0:
                raise ValueError("runtime-profile LoRA rank must be a positive integer")
            alpha_value = _finite_alpha(alpha)
            if type(seed) is not int or seed < 0:
                raise ValueError("runtime-profile seed must be a non-negative integer")

        from .nf4_cpu import (
            NF4Qwen35CPU,
            _TARGETS_BY_LAYER,
            _native_source_binding,
            _validate_artifact,
        )

        config_data, source, record_names, tensor_names, selected = _validate_artifact(artifact)
        self.artifact_identity = artifact.fingerprint()
        self.source_identity = source["payload_sha256"]
        validated_profile = None
        if runtime_profile is None:
            _validate_tiny_profile(config_data, self.artifact_identity)
        else:
            from ...quantization.nf4_runtime_profile import validate_runtime_profile

            validated_profile = validate_runtime_profile(
                runtime_profile, config_data, self.artifact_identity, self.source_identity
            )
            if rank != validated_profile["rank"]:
                raise ValueError("explicit rank differs from the source-bound runtime profile")
            if alpha_value != validated_profile["alpha"]:
                raise ValueError("explicit alpha differs from the source-bound runtime profile")
            if seed != validated_profile["seed"]:
                raise ValueError("explicit seed differs from the source-bound runtime profile")
        self.mode = mode
        self.execution_arm = (
            "spilltune_streaming" if mode == "streaming" else "shared_recompute_resident_secondary"
        )
        self.rank = rank
        self.alpha = alpha_value
        self.seed = seed
        self.runtime_profile = validated_profile

        try:
            import transformers
            from transformers.models.qwen3_5 import configuration_qwen3_5, modeling_qwen3_5
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextModel
        except Exception as exc:
            raise RuntimeError("dependency: native Transformers Qwen3.5 text classes are unavailable") from exc
        native_binding = _native_source_binding(modeling_qwen3_5, configuration_qwen3_5, transformers)
        config = Qwen3_5TextConfig(**config_data)
        config._attn_implementation = "eager"
        config.use_cache = False

        from ...quantization.nf4_backend_cuda import (
            PinnedNF4Record,
            backend_info_cuda,
            stage_cuda,
            RunBackend,
        )
        self._backend_run = RunBackend()
        backend_binding = backend_info_cuda(backend_run=self._backend_run)
        if (
            not isinstance(backend_binding, Mapping)
            or backend_binding.get("runtime") != "cuda"
            or backend_binding.get("device") != "cuda:0"
            or backend_binding.get("cpu_only") is not False
        ):
            raise RuntimeError("CUDA NF4 backend returned invalid runtime identity")
        backend_binding = json.loads(_canonical_json(dict(backend_binding)).decode("utf-8"))

        with torch.device("meta"):
            model = Qwen3_5TextModel(config)
        if type(model) is not Qwen3_5TextModel:
            raise ValueError("unsupported native Qwen text model class")
        if len(model.layers) != config.num_hidden_layers:
            raise ValueError("native Qwen decoder layer count mismatch")
        # Reject alternate/fused kernel classes while the native model still
        # resides entirely on meta, before transferring any model state.
        for layer in model.layers:
            NF4Qwen35CPU._select_cpu_kernels(layer, modeling_qwen3_5)
        # Artifact raw tensors move directly to the native meta model on
        # CUDA:0; no CPU model is constructed or moved with Module.to().
        raw_state = {
            key.removeprefix(_PREFIX): artifact.tensor(key).to(self.device)
            for key in tensor_names
        }
        expected_selected_state = {key.removeprefix(_PREFIX) for key in selected}
        incompat = model.load_state_dict(raw_state, strict=False, assign=True)
        if set(incompat.missing_keys) != expected_selected_state or incompat.unexpected_keys:
            raise ValueError("native Qwen state inventory does not match NF4 artifact")
        with torch.device(self.device):
            model.rotary_emb = modeling_qwen3_5.Qwen3_5TextRotaryEmbedding(config=config)
        model.eval().requires_grad_(False)

        resident_copy_stream = torch.cuda.Stream(device=self.device) if mode == "resident" else None
        resident_states: dict[str, Any] = {}
        all_by_layer: dict[int, dict[str, Any]] = {index: {} for index in range(config.num_hidden_layers)}
        all_projections_by_layer: dict[int, dict[str, _NF4CUDAProjection]] = {
            index: {} for index in range(config.num_hidden_layers)
        }
        rank_targets: dict[str, _NF4CUDAProjection] = {}
        record_fingerprints: dict[str, str] = {}
        resident_transfer_digests: dict[str, str] = {}
        record_type = _record_type()
        for full_name in record_names:
            if not full_name.startswith(_PREFIX + "layers.") or not full_name.endswith(".weight"):
                raise ValueError(f"NF4 record is outside the decoder matrix scope: {full_name!r}")
            path = full_name.removeprefix(_PREFIX).removesuffix(".weight")
            try:
                parts = path.split(".")
                if len(parts) < 3 or parts[0] != "layers":
                    raise ValueError("invalid decoder path")
                index = int(parts[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid decoder record name {full_name!r}") from exc
            if index not in all_by_layer:
                raise ValueError(f"NF4 record layer index is out of range: {full_name!r}")
            record = artifact.record(full_name)
            if not isinstance(record, record_type):
                raise ValueError(f"NF4 record {full_name!r} has an invalid type")
            module = model.get_submodule(path)
            if type(module) is not nn.Linear or tuple(module.weight.shape) != tuple(record.shape) or module.bias is not None:
                raise ValueError(f"NF4 record/native linear mismatch at {full_name!r}")
            layer_targets = _TARGETS_BY_LAYER[config.layer_types[index]]
            is_target = any(path == f"layers.{index}.{target}" for target in layer_targets)
            staged = None
            if mode == "resident":
                pinned = PinnedNF4Record(record, backend_run=self._backend_run)
                staged = stage_cuda(pinned, device=self.device, stream=resident_copy_stream)
                if staged.source_fingerprint != record.fingerprint() or tuple(staged.shape) != tuple(record.shape):
                    raise ValueError("staged resident CUDA state differs from its canonical NF4 record")
                # One full byte/layout transfer check before any training.
                staged.verify_source()
                resident_states[full_name] = staged
                resident_transfer_digests[full_name] = staged.transfer_digest
            projection = _NF4CUDAProjection(
                record,
                full_name,
                mode=mode,
                rank=rank if is_target else 0,
                alpha=alpha_value,
                device=self.device,
                resident_state=staged,
            )
            _replace_submodule(model, path, projection)
            all_by_layer[index][full_name] = record
            all_projections_by_layer[index][full_name] = projection
            record_fingerprints[full_name] = record.fingerprint()
            if is_target:
                target_path = path.removeprefix(f"layers.{index}.")
                rank_targets[f"{index}:{target_path}"] = projection

        expected_targets = {
            f"{index}:{target}"
            for index, kind in enumerate(config.layer_types)
            for target in _TARGETS_BY_LAYER[kind]
        }
        if set(rank_targets) != expected_targets:
            raise ValueError("NF4 artifact is missing an eligible native LoRA target")
        if validated_profile is None:
            if len(expected_targets) != 6:
                raise ValueError("tiny CUDA qualification requires exactly six LoRA targets")
        elif validated_profile["targets"] != sorted(expected_targets):
            raise ValueError("runtime profile target inventory differs from native Qwen config")
        from ...finetune.initialization import canonical_fingerprint, make_canonical_lora

        init_state = make_canonical_lora(rank_targets, seed=seed, b_std=0.0)
        self.initialization_identity = canonical_fingerprint(init_state)
        with torch.no_grad():
            for key in sorted(rank_targets):
                rank_targets[key].A.copy_(init_state[key]["A"].to(self.device))
                rank_targets[key].B.copy_(init_state[key]["B"].to(self.device))

        model.layers = self._wrap_decoder_layers(model, all_by_layer, all_projections_by_layer, mode)
        model.eval()
        trainable = {
            id(parameter)
            for projection in rank_targets.values()
            for parameter in (projection.A, projection.B)
        }
        for name, parameter in model.named_parameters():
            if parameter.device != self.device:
                raise ValueError(f"native CUDA parameter {name!r} is not on CUDA:0")
            if parameter.requires_grad and id(parameter) not in trainable:
                raise ValueError(f"unexpected trainable native Qwen parameter {name!r}")
        for projection in rank_targets.values():
            if not projection.A.requires_grad or not projection.B.requires_grad:
                raise ValueError("native Qwen CUDA adapter factors must remain trainable")

        _lock_module_tree(model)
        self._config = config
        self._model = model
        self._projections = MappingProxyType(dict(sorted(rank_targets.items())))
        expected_parameter_count = 12 if validated_profile is None else 2 * len(expected_targets)
        if len(self.named_adapter_parameters()) != expected_parameter_count:
            raise RuntimeError("CUDA adapter parameter inventory does not match its explicit profile")
        self._head = model.embed_tokens.weight
        self._embedding = model.embed_tokens.weight
        if self._head is not self._embedding:
            raise RuntimeError("CUDA Qwen text head and embedding must be the same physical parameter")
        self._native_binding = native_binding
        self._backend_binding = backend_binding
        self._record_fingerprints = MappingProxyType(dict(sorted(record_fingerprints.items())))
        self.record_fingerprints = self._record_fingerprints
        self.record_state_identity = hashlib.sha256(_canonical_json(dict(self._record_fingerprints))).hexdigest()
        self._resident_copy_stream = resident_copy_stream
        self.resident_states = MappingProxyType(dict(sorted(resident_states.items())))
        self.resident_transfer_digests = MappingProxyType(dict(sorted(resident_transfer_digests.items())))
        self.pools = tuple(layer.pool for layer in model.layers) if mode == "streaming" else ()

        self._checkpoint_binding = {
            "schema": "rvft.nf4_qwen35_cuda_checkpoint_binding.v1" if validated_profile is None else "rvft.nf4_qwen35_cuda_checkpoint_binding.v2",
            "execution_arm": self.execution_arm,
            "artifact_identity": self.artifact_identity,
            "source_identity": self.source_identity,
            "record_state_identity": self.record_state_identity,
            "config_sha256": source["config_sha256"],
            "initialization_identity": self.initialization_identity,
            "rank": rank,
            "alpha": alpha_value,
            "targets": sorted(expected_targets),
            "backend": backend_binding,
            "integrity_policy": "startup_full_no_training_content_checks_v1",
            "transformers": native_binding,
            "torch_version": str(torch.__version__),
            "execution_device": "cuda:0",
            "frozen_dtype": "bfloat16_except_artifact_fp32_linear_attention_state_roles",
            "source_fp32_roles": [
                "layers.*.linear_attn.A_log",
                "layers.*.linear_attn.norm.weight",
            ],
            "artifact_fp32_roles": [
                "layers.*.linear_attn.A_log",
                "layers.*.linear_attn.norm.weight",
                "layers.*.linear_attn.dt_bias",
            ],
            "adapter_dtype": "float32",
            "compute_dtype": "bfloat16",
            "checkpoint_policy": "native_qwen35_full_decoder_nonreentrant_early_stop_false",
            "attention_backend": "eager",
            "linear_attention_math": "explicit_native_torch_formulas_cuda",
            "linear_attention_kernels": {
                "causal_conv1d_fn": None,
                "causal_conv1d_update": "torch_causal_conv1d_update",
                "chunk_gated_delta_rule": "torch_chunk_gated_delta_rule",
                "recurrent_gated_delta_rule": "torch_recurrent_gated_delta_rule",
            },
        }
        if validated_profile is not None:
            self._checkpoint_binding["runtime_profile"] = json.loads(
                _canonical_json(validated_profile).decode("utf-8")
            )

        self._checkpoint_binding["full_chain_timer_enabled"] = self._full_chain_timer_enabled
        self._prepared_training_rows = None
        self._backend_run.ready()

    def _wrap_decoder_layers(self, model, all_by_layer, all_projections_by_layer, mode):
        """One model-owned copy stream; decoder/pool lifetime barriers stay local."""
        from ...training.nf4_cuda_recompute import ControlledNF4CUDADecoder
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        self._copy_stream = torch.cuda.Stream(device=self.device) if mode == "streaming" else None
        wrapped_layers = []
        for index, layer in enumerate(model.layers):
            wrapped = ControlledNF4CUDADecoder(
                layer, all_by_layer[index], all_projections_by_layer[index],
                mode=mode, backend_run=self._backend_run, copy_stream=self._copy_stream,
            )
            if type(wrapped.layer) is not Qwen3_5DecoderLayer:
                raise ValueError("controlled CUDA recompute did not retain the exact native decoder")
            wrapped_layers.append(wrapped)
        return nn.ModuleList(wrapped_layers)

    def begin_compute(self):
        """Enter the shared one-use compute lifecycle after restore/preflight."""
        for pool in self.pools:pool.check_compute_start()
        self._backend_run.begin()
        for pool in self.pools:pool.begin_compute()

    def finish_compute(self):
        for pool in self.pools:pool.assert_idle()
        for pool in self.pools:pool.finish_compute()
        self._backend_run.finish()

    def prepare_preflight_training(self, rows, *, binding, order):
        from ...training.nf4_preflight import PreparedTrainingRows
        if self._prepared_training_rows is not None:raise RuntimeError('new training job requires new model preflight')
        snapshot=PreparedTrainingRows(rows,binding={**binding,'model':self.checkpoint_binding()},vocab_size=self._config.vocab_size,order=order)
        snapshot.claim(self)
        self.begin_compute()
        self._prepared_training_rows=snapshot
        return self._loss_prepared

    def finish_preflight_training(self):
        if self._prepared_training_rows is not None:
            self._prepared_training_rows.close(self)
            self.finish_compute()

    def _loss_prepared(self,row_id,*,normalization,token_chunk):
        if self._prepared_training_rows is None:raise RuntimeError('startup row preflight required')
        (ids,labels,_),chunks=self._prepared_training_rows.row_with_chunks(self,row_id,token_chunk=token_chunk)
        ids=torch.tensor([ids],dtype=torch.int64,device='cpu')
        labels=torch.tensor([labels],dtype=torch.int64,device='cpu')
        cuda_ids=ids.to(self.device);cuda_labels=labels.to(self.device)
        leaf=self._model.embed_tokens(cuda_ids).detach().requires_grad_(True)
        output=self._model(inputs_embeds=leaf,attention_mask=None,use_cache=False)
        from ...quantization.native_terminal import chunked_causal_loss
        loss=chunked_causal_loss(output.last_hidden_state,self._head,cuda_labels,
            normalization=normalization,token_chunk=token_chunk,supervised_chunks=chunks)
        return loss,leaf

    @property
    def full_chain_timer_enabled(self):
        return self._full_chain_timer_enabled

    @property
    def config(self):
        import copy
        return copy.deepcopy(self._config)

    def close(self):
        for pool in self.pools:
            if pool.stats['active'] or pool.stats['poisoned']:raise RuntimeError('cannot close live/poisoned pool')
        self.finish_preflight_training()
        self.finish_compute()
        self._backend_run.close()

    def named_adapter_parameters(self) -> dict[str, nn.Parameter]:
        result: dict[str, nn.Parameter] = {}
        for key, projection in self._projections.items():
            result[f"{key}/A"] = projection.A
            result[f"{key}/B"] = projection.B
        return result

    def checkpoint_binding(self) -> dict[str, Any]:
        return json.loads(_canonical_json(self._checkpoint_binding).decode("utf-8"))

    def loss(self,*args,**kwargs):
        raise RuntimeError('raw CUDA loss refuses unprepared training rows; use run_updates startup preflight')

    def next_token_logits(
        self,
        ids: torch.Tensor,
        *,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Recompute the full prefix without cache and return final-position logits."""
        if past_key_values is not None:
            raise ValueError("past_key_values/cache are unsupported; pass None")
        if use_cache is not False:
            raise ValueError("use_cache must be False")
        if not isinstance(ids, torch.Tensor) or ids.device.type != "cpu" or ids.dtype is not torch.int64:
            raise TypeError("ids must be a CPU int64 tensor")
        if ids.ndim != 2 or ids.shape[0] <= 0 or ids.shape[1] <= 0:
            raise ValueError("ids must have shape [batch>0, sequence>0]")
        if bool(((ids < 0) | (ids >= self._config.vocab_size)).any()):
            raise ValueError("input token id is outside the configured vocabulary")
        device_ids = ids.to(self.device)
        with torch.no_grad():
            embeddings = self._model.embed_tokens(device_ids)
            output = self._model(inputs_embeds=embeddings, use_cache=False)
            logits = F.linear(output.last_hidden_state[:, -1, :], self._head)
        expected_shape = (ids.shape[0], self._config.vocab_size)
        if logits.device != self.device or logits.shape != expected_shape or not logits.is_floating_point():
            raise RuntimeError("native CUDA generation returned invalid final-position logits")
        return logits
