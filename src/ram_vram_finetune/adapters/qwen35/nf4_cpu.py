"""CPU-only native Qwen3.5 NF4 + DQ + BF16 model adapter.

This module consumes a bounded, canonical NF4ModelArtifact. It does not load,
download, or convert a pretrained model except through the explicit capped
build_qwen35_nf4_artifact entry point.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from types import MappingProxyType, MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ...quantization.nf4_resident import nf4_lora_linear_cpu
from .adapter import SUFFIXES, inspect_model
from ..targets import ATTENTION_TARGETS

_PREFIX = "model.language_model."
_MAX_SOURCE_PAYLOAD_BYTES = 16 * 1024 * 1024
_SOURCE_SCHEMA = "rvft.nf4_source.v1"
_SOURCE_KIND = "bounded_cpu_bf16_qwen35_text"
_FP32_SOURCE_SUFFIXES = (".linear_attn.A_log", ".linear_attn.norm.weight")
_FP32_ARTIFACT_SUFFIXES = _FP32_SOURCE_SUFFIXES + (".linear_attn.dt_bias",)
_FIXED_CPU_MESSAGE = (
    "NF4 Qwen3.5 CPU modules are fixed to CPU with role-specific frozen dtypes and "
    "FP32 adapter parameters; module conversions are unsupported"
)
_TARGETS_BY_LAYER = {kind: tuple(targets) for kind, targets in ATTENTION_TARGETS.items()}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _finite_alpha(alpha: Real, *, name: str = "alpha") -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(alpha)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _cpu_device(device: str | torch.device) -> torch.device:
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be 'cpu' or torch.device('cpu')")
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("NF4Qwen35CPU accepts CPU only") from exc
    if parsed.type != "cpu" or parsed.index is not None:
        raise ValueError("NF4Qwen35CPU accepts CPU only")
    return torch.device("cpu")


def _require_digest(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid {name}")
    return value


def _record_type():
    from ...quantization.nf4_state import NF4Record

    return NF4Record


def _artifact_type():
    from ...quantization.nf4_model_state import NF4ModelArtifact

    return NF4ModelArtifact


def _source_inventory(report: Mapping[str, Any]) -> tuple[dict[str, tuple[int, ...]], int]:
    entries = report.get("entries")
    required = report.get("required")
    if not isinstance(entries, Mapping) or not isinstance(required, list) or not required:
        raise ValueError("invalid Qwen source header inventory")
    item_sizes = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4,
                  "U32": 4, "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
                  "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1}
    total_bytes = 0
    for entry in entries.values():
        if not isinstance(entry, Mapping) or "spec" in entry:
            raise ValueError("source must be original safetensors, not a quantized archive")
        shape, dtype = entry.get("shape"), entry.get("dtype")
        if (not isinstance(shape, list) or not shape
                or any(type(extent) is not int or extent <= 0 for extent in shape)
                or dtype not in item_sizes):
            raise ValueError("source header has an unsupported dtype or shape")
        declared = entry.get("stored_bytes")
        size = math.prod(shape) * item_sizes[dtype] if declared is None else declared
        if type(size) is not int or size <= 0:
            raise ValueError("invalid source payload byte count")
        total_bytes += size
    if any(type(key) is not str for key in required):
        raise ValueError("invalid required Qwen source tensor name")
    if len(set(required)) != len(required):
        raise ValueError("duplicate required Qwen source tensor name")
    source_fp32_keys = {key for key in required if key.endswith(_FP32_SOURCE_SUFFIXES)}
    shapes: dict[str, tuple[int, ...]] = {}
    for key in required:
        if type(key) is not str or key not in entries:
            raise ValueError("invalid required Qwen source tensor name")
        entry = entries[key]
        shape, dtype = entry.get("shape"), entry.get("dtype")
        expected_dtype = "F32" if key in source_fp32_keys else "BF16"
        if dtype != expected_dtype:
            raise ValueError(f"required Qwen source tensor {key!r} must use {expected_dtype} storage")
        shapes[key] = tuple(shape)
    return shapes, total_bytes


def _hash_source_payload(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"rvft.nf4_source_payload.v1\0")
    for key in sorted(tensors):
        tensor = tensors[key].detach().contiguous()
        descriptor = _canonical_json(
            {"key": key, "dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}
        )
        raw = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def build_qwen35_nf4_artifact(source_path: str | Path):
    """Build one tiny/bounded Qwen text artifact; cap is checked before payload reads."""
    from ...quantization.nf4_model_state import NF4ModelArtifact
    from ...quantization.nf4_state import NF4Record
    from ...quantization.storage import WeightLoader

    source_root = Path(source_path)
    if (source_root / "rvft_quantization.json").exists():
        raise ValueError("source must be original safetensors, not an RVFT quantized archive")
    report = inspect_model(source_root)  # Raw safetensors headers and JSON config only.
    shapes, payload_bytes = _source_inventory(report)
    if payload_bytes > _MAX_SOURCE_PAYLOAD_BYTES:
        raise ValueError(
            f"source payload exceeds the fixed {_MAX_SOURCE_PAYLOAD_BYTES}-byte CPU model phase cap"
        )
    if report.get("architecture") != "qwen35":
        raise ValueError("unsupported architecture: expected Qwen3.5")
    config = report.get("text_config")
    if not isinstance(config, Mapping) or config.get("tie_word_embeddings") is not True:
        raise ValueError("unsupported capability: only tied Qwen3.5 text sources are accepted")
    expected_aliases = {"lm_head.weight": _PREFIX + "embed_tokens.weight"}
    aliases = report.get("source_aliases")
    if aliases != expected_aliases:
        raise ValueError("unsupported capability: canonical tied text embedding/head is required")
    role_keys = {key for key in report["required"] if key.endswith(_FP32_ARTIFACT_SUFFIXES)}
    selected = set(report.get("quantization_matrix_keys", ()))
    if not selected or not selected.issubset(shapes) or selected.intersection(role_keys):
        raise ValueError("unsupported quantization: no eligible BF16 decoder matrices")

    # Hash the original required payload bytes and dtypes before artifact
    # normalization; A_log and the gated linear-attention norm are source
    # FP32, while dt_bias is source BF16. Hashing preserves those original
    # dtypes before the artifact's explicit-state normalization.
    source_tensors: dict[str, torch.Tensor] = {}
    loader = WeightLoader(source_path, report, "none")
    for key in report["required"]:
        raw_record = loader.read(key)
        value = raw_record.parts["value"]
        expected_dtype = torch.float32 if key.endswith(_FP32_SOURCE_SUFFIXES) else torch.bfloat16
        if value.dtype is not expected_dtype or tuple(value.shape) != shapes[key]:
            raise ValueError(f"source tensor {key!r} failed role-specific dtype/header verification")
        source_tensors[key] = value.detach().contiguous().clone()
    loader.finish()

    config_sha = hashlib.sha256(_canonical_json(dict(config))).hexdigest()
    payload_sha = _hash_source_payload(source_tensors)
    source = {
        "schema": _SOURCE_SCHEMA,
        "kind": _SOURCE_KIND,
        "config_sha256": config_sha,
        "payload_sha256": payload_sha,
        "text_tensor_count": len(source_tensors),
        "text_parameter_count": sum(math.prod(t.shape) for t in source_tensors.values()),
        "excluded_tensors": {
            "vision": len(report.get("excluded_nontext_tensors", {}).get("vision", ())),
            "mtp": len(report.get("excluded_nontext_tensors", {}).get("mtp", ())),
        },
    }
    tensors: dict[str, torch.Tensor] = {}
    records: dict[str, Any] = {}
    for key in report["required"]:
        original = source_tensors[key]
        if key in selected:
            records[key] = NF4Record.from_weight(original)
        elif key.endswith(_FP32_ARTIFACT_SUFFIXES):
            # Preserve original FP32 A_log and gated norm. The canonical
            # artifact stores dt_bias in FP32 after hashing its BF16 source.
            tensors[key] = original.to(dtype=torch.float32)
        else:
            tensors[key] = original
    if set(records).intersection(tensors) or set(records).union(tensors) != set(report["required"]):
        raise RuntimeError("internal Qwen artifact tensor inventory mismatch")
    del source_tensors
    return NF4ModelArtifact.from_parts(
        config=dict(config),
        records=records,
        tensors=tensors,
        aliases=aliases,
        source=source,
    )


def _reject_fixed_cpu_apply(_module, _fn, recurse: bool = True):
    raise RuntimeError(_FIXED_CPU_MESSAGE)


def _lock_module_tree(root: nn.Module) -> None:
    """Guard every exposed native subtree entry point before tensor callbacks."""
    for submodule in root.modules():
        submodule._apply = MethodType(_reject_fixed_cpu_apply, submodule)


class _NF4CPUProjection(nn.Module):
    """One frozen NF4 matrix and optional FP32 LoRA factors, resident or leased."""

    def __init__(
        self,
        record,
        record_name: str,
        *,
        mode: str,
        rank: int = 0,
        alpha: float = 4.0,
    ) -> None:
        super().__init__()
        record_type = _record_type()
        if not isinstance(record, record_type):
            raise TypeError("record must be an NF4Record")
        if type(record_name) is not str or not record_name:
            raise ValueError("record_name must be a canonical non-empty source key")
        if mode not in ("resident", "streaming"):
            raise ValueError("projection mode must be 'resident' or 'streaming'")
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        alpha_value = _finite_alpha(alpha)
        shape = record.shape
        if len(shape) != 2 or any(type(extent) is not int or extent <= 0 for extent in shape):
            raise ValueError("record must describe a positive two-dimensional weight")
        self.mode = mode
        self.record_name = record_name
        self.record_fingerprint = record.fingerprint()
        self.rank = rank
        self.alpha = alpha_value
        self.scale = alpha_value / rank if rank else 0.0
        self._record = record
        self._lease = None
        if mode == "resident":
            self._packed, self._parts = record.export_parts()
        else:
            self._packed = None
            self._parts = None
        if rank:
            self.A = nn.Parameter(torch.zeros((rank, shape[1]), dtype=torch.float32, device="cpu"))
            self.B = nn.Parameter(torch.zeros((shape[0], rank), dtype=torch.float32, device="cpu"))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
        self._apply = MethodType(_reject_fixed_cpu_apply, self)

    def _apply(self, fn, recurse: bool = True):
        raise RuntimeError(_FIXED_CPU_MESSAGE)

    def _set_lease(self, lease_or_none) -> None:
        if self.mode != "streaming":
            if lease_or_none is not None:
                raise RuntimeError("resident NF4 projection cannot bind a streaming lease")
            self._lease = None
            return
        if lease_or_none is not None and self._lease is not None:
            raise RuntimeError("NF4 streaming projection already has an active lease")
        self._lease = lease_or_none

    def _active_parts(self):
        if self.mode == "resident":
            return self._packed, self._parts
        lease = self._lease
        if lease is None or not bool(getattr(lease, "is_active", False)):
            raise RuntimeError("streaming NF4 projection requires an active controlled-block lease")
        return lease.parts(self.record_name)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or inputs.device.type != "cpu":
            raise ValueError("NF4 Qwen projection inputs must be CPU tensors")
        if inputs.dtype is not torch.bfloat16:
            raise TypeError("NF4 Qwen projection inputs must be BF16")
        packed, parts = self._active_parts()
        if self.rank == 0:
            from ...quantization.nf4_backend import linear_cpu

            return linear_cpu(inputs, packed, parts)
        return nf4_lora_linear_cpu(inputs, packed, parts, self.A, self.B, self.scale)


def _replace_submodule(root: nn.Module, path: str, value: nn.Module) -> None:
    if type(path) is not str or not path or "." not in path:
        raise ValueError("invalid native Qwen projection path")
    parent_path, leaf = path.rsplit(".", 1)
    parent = root.get_submodule(parent_path)
    if leaf not in parent._modules:
        raise ValueError(f"native Qwen module has no expected projection {path!r}")
    setattr(parent, leaf, value)


def _expected_text_keys(config: Mapping[str, Any]) -> set[str]:
    count = config.get("num_hidden_layers")
    layer_types = config.get("layer_types")
    if type(count) is not int or count <= 0:
        raise ValueError("invalid Qwen text layer count")
    if not isinstance(layer_types, list) or len(layer_types) != count:
        raise ValueError("Qwen text config requires an explicit layer schedule")
    if set(layer_types) - set(ATTENTION_TARGETS):
        raise ValueError("unsupported Qwen layer schedule")
    required = {_PREFIX + "embed_tokens.weight", _PREFIX + "norm.weight"}
    for index, kind in enumerate(layer_types):
        required.update(
            _PREFIX + f"layers.{index}." + suffix
            for suffix in SUFFIXES
            if not (suffix.startswith("self_attn.") and kind != "full_attention")
            and not (suffix.startswith("linear_attn.") and kind != "linear_attention")
        )
    return required


def _validate_artifact(artifact):
    artifact_type = _artifact_type()
    record_type = _record_type()
    if not isinstance(artifact, artifact_type):
        raise TypeError("artifact must be an NF4ModelArtifact")
    config = artifact.config
    source = artifact.source
    aliases = artifact.aliases
    if not isinstance(config, Mapping) or config.get("model_type") != "qwen3_5_text":
        raise ValueError("artifact must contain the resolved Qwen3.5 text config")
    if config.get("tie_word_embeddings") is not True:
        raise ValueError("artifact must contain tied Qwen3.5 text embeddings")
    if config.get("attention_dropout", 0) != 0 or config.get("attention_bias", False) is not False:
        raise ValueError("unsupported Qwen attention dropout/bias policy")
    if config.get("hidden_act", "silu") != "silu" or config.get("rms_norm_eps", 1e-6) != 1e-6:
        raise ValueError("unsupported Qwen activation or RMSNorm policy")
    if not isinstance(source, Mapping) or set(source) != {
        "schema", "kind", "config_sha256", "payload_sha256",
        "text_tensor_count", "text_parameter_count", "excluded_tensors",
    }:
        raise ValueError("artifact source metadata does not satisfy the Qwen source schema")
    if source.get("schema") != _SOURCE_SCHEMA or source.get("kind") != _SOURCE_KIND:
        raise ValueError("unsupported Qwen source identity schema/kind")
    if _require_digest(source.get("config_sha256"), "source config digest") != hashlib.sha256(
        _canonical_json(dict(config))
    ).hexdigest():
        raise ValueError("artifact config does not match its source identity")
    _require_digest(source.get("payload_sha256"), "source payload digest")
    if type(source.get("text_tensor_count")) is not int or type(source.get("text_parameter_count")) is not int:
        raise ValueError("invalid Qwen source counts")
    excluded = source.get("excluded_tensors")
    if (
        not isinstance(excluded, Mapping)
        or set(excluded) != {"vision", "mtp"}
        or any(type(value) is not int or value < 0 for value in excluded.values())
    ):
        raise ValueError("invalid excluded vision/MTP source counts")
    expected = _expected_text_keys(config)
    if aliases != {"lm_head.weight": _PREFIX + "embed_tokens.weight"}:
        raise ValueError("artifact must bind the direct canonical text embedding/head alias")
    record_names = tuple(artifact.record_names)
    tensor_names = tuple(artifact.tensor_names)
    if len(record_names) + len(tensor_names) != len(set(record_names).union(tensor_names)):
        raise ValueError("artifact raw/quantized tensor inventories overlap")
    if set(record_names).union(tensor_names) != expected:
        raise ValueError("artifact text tensor inventory does not match native Qwen config")
    record_set = set(record_names)
    raw_metadata = {key: artifact.tensor_metadata(key) for key in tensor_names}
    selected = {
        key for key in expected
        if key.startswith(_PREFIX + "layers.")
        and len(artifact.record(key).shape if key in record_set else raw_metadata[key]["shape"]) == 2
        and math.prod(artifact.record(key).shape if key in record_set else raw_metadata[key]["shape"]) >= 128
    }
    if set(record_names) != selected:
        raise ValueError("artifact NF4 record inventory is not the exact eligible decoder matrix set")
    parameter_count = 0
    for key in sorted(expected):
        if key in record_names:
            record = artifact.record(key)
            if not isinstance(record, record_type):
                raise ValueError(f"artifact record {key!r} has an invalid type")
            shape = record.shape
        else:
            metadata = raw_metadata[key]
            if metadata["device"].type != "cpu" or metadata["requires_grad"]:
                raise ValueError(f"artifact raw tensor {key!r} is not frozen CPU storage")
            expected_dtype = torch.float32 if key.endswith(_FP32_ARTIFACT_SUFFIXES) else torch.bfloat16
            if metadata["dtype"] is not expected_dtype:
                raise ValueError(f"artifact raw tensor {key!r} has an invalid dtype")
            shape = metadata["shape"]
            if not artifact.tensor_all_finite(key):
                raise ValueError(f"artifact raw tensor {key!r} contains non-finite values")
        if len(shape) == 0 or any(type(extent) is not int or extent <= 0 for extent in shape):
            raise ValueError(f"artifact tensor {key!r} has an invalid shape")
        parameter_count += math.prod(shape)
    if source["text_tensor_count"] != len(expected) or source["text_parameter_count"] != parameter_count:
        raise ValueError("artifact source counts do not match its canonical text inventory")
    hidden = config.get("hidden_size")
    vocab = config.get("vocab_size")
    if type(hidden) is not int or type(vocab) is not int or hidden <= 0 or vocab <= 0:
        raise ValueError("invalid Qwen embedding dimensions")
    if artifact.tensor_metadata(_PREFIX + "embed_tokens.weight")["shape"] != (vocab, hidden):
        raise ValueError("artifact tied text embedding shape does not match config")
    if artifact.tensor_metadata(_PREFIX + "norm.weight")["shape"] != (hidden,):
        raise ValueError("artifact final norm shape does not match config")
    return dict(config), dict(source), tuple(sorted(record_names)), tuple(sorted(tensor_names)), selected


def _native_source_binding(modeling_module, config_module, transformers_module) -> dict[str, Any]:
    expected_version = "5.13.0"
    version = getattr(transformers_module, "__version__", None)
    if version != expected_version:
        raise RuntimeError(f"NF4 Qwen CPU requires Transformers {expected_version}, got {version!r}")
    mask_path = Path(transformers_module.__file__).resolve().parent / "masking_utils.py"
    paths = {
        "modeling_qwen3_5": Path(modeling_module.__file__),
        "configuration_qwen3_5": Path(config_module.__file__),
        "masking_utils": mask_path,
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing critical Transformers source file: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"version": version, "critical_source_sha256": hashes}


class NF4Qwen35CPU:
    """Native Transformers Qwen3.5 text model over one immutable NF4 artifact."""

    def __init__(
        self,
        artifact,
        *,
        mode: str = "resident",
        rank: int = 2,
        alpha: float = 4.0,
        seed: int = 20261019,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = _cpu_device(device)  # Must reject before model/parameter allocation.
        if mode not in ("resident", "streaming"):
            raise ValueError("mode must be 'resident' or 'streaming'")
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be a positive integer")
        alpha_value = _finite_alpha(alpha)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        config_data, source, record_names, tensor_names, selected = _validate_artifact(artifact)
        self.mode = mode
        self.rank = rank
        self.alpha = alpha_value
        self.artifact_identity = artifact.fingerprint()
        self.source_identity = source["payload_sha256"]

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
        model = None
        with torch.device("meta"):
            model = Qwen3_5TextModel(config)
        raw_state = {
            key.removeprefix(_PREFIX): artifact.tensor(key)
            for key in tensor_names
        }
        expected_selected_state = {key.removeprefix(_PREFIX) for key in selected}
        incompat = model.load_state_dict(raw_state, strict=False, assign=True)
        if set(incompat.missing_keys) != expected_selected_state or incompat.unexpected_keys:
            raise ValueError("native Qwen state inventory does not match NF4 artifact")
        # Rotary frequencies are nonpersistent computation buffers, not checkpoint weights.
        model.rotary_emb = modeling_qwen3_5.Qwen3_5TextRotaryEmbedding(config=config)
        model.eval().requires_grad_(False)
        if type(model) is not Qwen3_5TextModel:
            raise ValueError("unsupported native Qwen text model class")
        if len(model.layers) != config.num_hidden_layers:
            raise ValueError("native Qwen decoder layer count mismatch")

        rank_targets: dict[str, _NF4CPUProjection] = {}
        all_by_layer: dict[int, dict[str, Any]] = {
            index: {} for index in range(config.num_hidden_layers)
        }
        all_projections_by_layer: dict[int, dict[str, _NF4CPUProjection]] = {
            index: {} for index in range(config.num_hidden_layers)
        }
        record_type = _record_type()
        target_names = []
        for full_name in record_names:
            if not full_name.startswith(_PREFIX + "layers.") or not full_name.endswith(".weight"):
                raise ValueError(f"NF4 record is outside the decoder matrix scope: {full_name!r}")
            local = full_name.removeprefix(_PREFIX)
            path = local.removesuffix(".weight")
            try:
                path_parts = path.split(".")
                if len(path_parts) < 3 or path_parts[0] != "layers":
                    raise ValueError("invalid layer path")
                index = int(path_parts[1])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid decoder record name {full_name!r}") from exc
            if index not in all_by_layer:
                raise ValueError(f"NF4 record layer index is out of range: {full_name!r}")
            record = artifact.record(full_name)
            if not isinstance(record, record_type):
                raise ValueError(f"NF4 record {full_name!r} has an invalid type")
            module = model.get_submodule(path)
            if type(module) is not nn.Linear or tuple(module.weight.shape) != record.shape or module.bias is not None:
                raise ValueError(f"NF4 record/native linear mismatch at {full_name!r}")
            kind = config.layer_types[index]
            layer_targets = _TARGETS_BY_LAYER[kind]
            is_target = any(path == f"layers.{index}.{target}" for target in layer_targets)
            projection = _NF4CPUProjection(
                record,
                full_name,
                mode=mode,
                rank=rank if is_target else 0,
                alpha=alpha_value,
            )
            _replace_submodule(model, path, projection)
            all_by_layer[index][full_name] = record
            all_projections_by_layer[index][full_name] = projection
            if is_target:
                target_path = path.removeprefix("layers." + str(index) + ".")
                key = str(index) + ":" + target_path
                target_names.append(key)
                rank_targets[key] = projection

        expected_targets = {
            f"{index}:{target}"
            for index, kind in enumerate(config.layer_types)
            for target in _TARGETS_BY_LAYER[kind]
        }
        if set(rank_targets) != expected_targets:
            raise ValueError("NF4 artifact is missing an eligible native LoRA target")
        from ...finetune.initialization import canonical_fingerprint, make_canonical_lora

        init_state = make_canonical_lora(rank_targets, seed=seed, b_std=0.0)
        self.initialization_identity = canonical_fingerprint(init_state)
        # Use the exact same global sorted stream as the established initializer.
        for key in sorted(rank_targets):
            init_state_item = init_state[key]
            with torch.no_grad():
                rank_targets[key].A.copy_(init_state_item["A"])
                rank_targets[key].B.copy_(init_state_item["B"])

        from ...training.nf4_model_recompute import ControlledNF4Decoder

        wrapped_layers = []
        for index, layer in enumerate(model.layers):
            self._select_cpu_kernels(layer, modeling_qwen3_5)
            wrapped = ControlledNF4Decoder(
                layer,
                all_by_layer[index],
                all_projections_by_layer[index],
                mode=mode,
            )
            if type(wrapped.layer) is not Qwen3_5DecoderLayer:
                raise ValueError("controlled recompute did not retain the exact native decoder")
            wrapped_layers.append(wrapped)
        model.layers = nn.ModuleList(wrapped_layers)
        model.eval()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if not any(parameter is p for projection in rank_targets.values() for p in (projection.A, projection.B)):
                    raise ValueError(f"unexpected trainable native Qwen parameter {name!r}")
        for projection in rank_targets.values():
            if not projection.A.requires_grad or not projection.B.requires_grad:
                raise ValueError("native Qwen adapter factors must remain trainable")
        # Every exposed native module guards its own conversion entry point.
        # Guarding only the root would leave model.embed_tokens.to(), raw norm,
        # convolution, and inner-attention conversions able to visit tensors.
        _lock_module_tree(model)
        self.config = config
        self.model = model
        self.projections = MappingProxyType(dict(sorted(rank_targets.items())))
        self.head = model.embed_tokens.weight
        self.embedding = model.embed_tokens.weight
        self._native_binding = native_binding
        self._backend_binding = __import__(
            "ram_vram_finetune.quantization.nf4_backend", fromlist=["backend_info"]
        ).backend_info()
        if mode == "streaming":
            self.pools = tuple(layer.pool for layer in model.layers)
        else:
            self.pools = ()
        self.initialization_identity = self.initialization_identity
        self._checkpoint_binding = {
            "schema": "rvft.nf4_qwen35_checkpoint_binding.v1",
            "artifact_identity": self.artifact_identity,
            "source_identity": self.source_identity,
            "config_sha256": source["config_sha256"],
            "initialization_identity": self.initialization_identity,
            "rank": rank,
            "alpha": alpha_value,
            "targets": sorted(expected_targets),
            "backend": self._backend_binding,
            "transformers": self._native_binding,
            "torch_version": str(torch.__version__),
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
            "linear_attention_kernels": {
                "causal_conv1d_fn": None,
                "causal_conv1d_update": "torch_causal_conv1d_update",
                "chunk_gated_delta_rule": "torch_chunk_gated_delta_rule",
                "recurrent_gated_delta_rule": "torch_recurrent_gated_delta_rule",
            },
        }

    @staticmethod
    def _select_cpu_kernels(layer: nn.Module, modeling_module) -> None:
        expected_layer = getattr(modeling_module, "Qwen3_5DecoderLayer", None)
        if expected_layer is None or type(layer) is not expected_layer:
            raise ValueError("unsupported non-native Qwen decoder layer")
        core = getattr(layer, "linear_attn", None)
        if core is None:
            expected_attention = getattr(modeling_module, "Qwen3_5Attention", None)
            if expected_attention is None or type(getattr(layer, "self_attn", None)) is not expected_attention:
                raise ValueError("unsupported custom/fused Qwen full-attention implementation")
            return
        expected_core = getattr(modeling_module, "Qwen3_5GatedDeltaNet", None)
        if expected_core is None or type(core) is not expected_core:
            raise ValueError("unsupported custom or fused Qwen linear-attention kernel class")
        expected_norm = getattr(modeling_module, "Qwen3_5RMSNormGated", None)
        if expected_norm is None or type(core.norm) is not expected_norm:
            raise ValueError("unsupported fused/custom gated RMSNorm implementation")
        required = (
            "torch_causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        )
        if any(not callable(getattr(modeling_module, name, None)) for name in required):
            raise RuntimeError("native Transformers CPU linear-attention kernels are unavailable")
        core.causal_conv1d_fn = None
        core.causal_conv1d_update = modeling_module.torch_causal_conv1d_update
        core.chunk_gated_delta_rule = modeling_module.torch_chunk_gated_delta_rule
        core.recurrent_gated_delta_rule = modeling_module.torch_recurrent_gated_delta_rule

    def named_adapter_parameters(self) -> dict[str, nn.Parameter]:
        result: dict[str, nn.Parameter] = {}
        for key, projection in self.projections.items():
            result[f"{key}/A"] = projection.A
            result[f"{key}/B"] = projection.B
        return result

    def checkpoint_binding(self) -> dict[str, Any]:
        return json.loads(_canonical_json(self._checkpoint_binding).decode("utf-8"))

    def loss(
        self,
        ids: torch.Tensor,
        labels: torch.Tensor,
        normalization: float | None = None,
        *,
        token_chunk: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("ids and labels must be CPU int64 tensors")
        if ids.device.type != "cpu" or labels.device.type != "cpu":
            raise ValueError("ids and labels must be on CPU")
        if ids.dtype is not torch.int64 or labels.dtype is not torch.int64:
            raise TypeError("ids and labels must have dtype torch.int64")
        if ids.ndim != 2 or labels.shape != ids.shape or ids.shape[0] <= 0 or ids.shape[1] < 2:
            raise ValueError("ids and labels must have matching [batch, sequence>=2] shapes")
        if type(token_chunk) is not int or token_chunk < 1:
            raise ValueError("token_chunk must be a positive integer")
        vocab = self.config.vocab_size
        if bool(((ids < 0) | (ids >= vocab)).any()):
            raise ValueError("input token id is outside the configured vocabulary")
        shifted = labels[:, 1:]
        bad = (shifted != -100) & ((shifted < 0) | (shifted >= vocab))
        if bool(bad.any()):
            raise ValueError("non-ignored shifted label is outside the configured vocabulary")
        if normalization is None:
            normalizer = int((shifted != -100).sum().item())
        else:
            if isinstance(normalization, bool) or not isinstance(normalization, Real):
                raise TypeError("normalization must be a positive finite real")
            normalizer = float(normalization)
        if not math.isfinite(float(normalizer)) or normalizer <= 0:
            raise ValueError("normalization must be positive and finite")
        leaf = self.model.embed_tokens(ids).detach().requires_grad_(True)
        output = self.model(inputs_embeds=leaf, use_cache=False)
        hidden = output.last_hidden_state
        from ...quantization.native_terminal import chunked_causal_loss

        loss = chunked_causal_loss(
            hidden,
            self.head,
            labels,
            normalization=normalizer,
            token_chunk=token_chunk,
        )
        if loss.dtype is not torch.float32 or loss.ndim != 0:
            raise RuntimeError("native chunked causal loss returned an invalid scalar")
        return loss, leaf

    def close(self) -> None:
        self.pools = ()
        self.projections = MappingProxyType({})
        self.model = None
        self.head = None
        self.embedding = None
