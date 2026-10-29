"""Public update loop for Qwen3.5 NF4 streaming LoRA.

This library API accepts already-tokenized CPU rows and a canonical NF4 model
artifact. It performs ordinary AdamW updates and writes a final adapter-only
safetensors file. The formal R112 plans, supervisors, scoring, and benchmarking
remain outside this library entry point.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


_ADAPTER_FORMAT = "spilltune-nf4-lora-adapter-v1"
_ADAPTER_SCHEMA = "rvft.nf4_lora_adapter.v1"
_ADAPTER_BINDING_KEYS = (
    "artifact_identity", "source_identity", "record_state_identity",
    "config_sha256", "initialization_identity", "rank", "alpha", "targets",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _normalize_rows(rows_by_id: Mapping[str, Any]) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], int]]:
    if not isinstance(rows_by_id, Mapping) or not rows_by_id:
        raise ValueError("rows_by_id must be a non-empty mapping")
    result: dict[str, tuple[tuple[int, ...], tuple[int, ...], int]] = {}
    for row_id, raw in rows_by_id.items():
        if type(row_id) is not str or not row_id:
            raise ValueError("training row IDs must be non-empty strings")
        if isinstance(raw, Mapping):
            ids, labels = raw.get("input_ids"), raw.get("labels")
            declared_count = raw.get("supervised_tokens")
        elif isinstance(raw, (tuple, list)) and len(raw) in (2, 3):
            ids, labels = raw[:2]
            declared_count = raw[2] if len(raw) == 3 else None
        else:
            raise ValueError(f"row {row_id!r} must contain input_ids and labels")
        if not isinstance(ids, (tuple, list)) or not isinstance(labels, (tuple, list)):
            raise TypeError(f"row {row_id!r} input_ids and labels must be sequences")
        ids_tuple, labels_tuple = tuple(ids), tuple(labels)
        if len(ids_tuple) < 2 or len(ids_tuple) != len(labels_tuple):
            raise ValueError(f"row {row_id!r} must have aligned sequences of length at least two")
        if any(type(token) is not int or token < 0 for token in ids_tuple):
            raise ValueError(f"row {row_id!r} contains an invalid input token")
        if any(type(token) is not int or (token != -100 and token < 0) for token in labels_tuple):
            raise ValueError(f"row {row_id!r} contains an invalid label token")
        count = sum(token != -100 for token in labels_tuple[1:])
        if count <= 0 or (declared_count is not None and (type(declared_count) is not int or declared_count != count)):
            raise ValueError(f"row {row_id!r} has an invalid supervised-token count")
        result[row_id] = (ids_tuple, labels_tuple, count)
    return result


def run_nf4_updates(
    model: Any,
    rows_by_id: Mapping[str, Any],
    order_ids: Sequence[str],
    *,
    updates: int,
    accumulation: int = 1,
    learning_rate: float = 2e-4,
) -> dict[str, Any]:
    """Train an already constructed streaming model for a fixed number of updates.

    Rows must be tokenized CPU integer sequences. Each optimizer update sums
    shifted-token cross-entropy over one accumulation window, matching the
    model's startup preflight contract. The caller owns final model cleanup.
    """
    if type(updates) is not int or updates <= 0:
        raise ValueError("updates must be a positive integer")
    if type(accumulation) is not int or accumulation <= 0:
        raise ValueError("accumulation must be a positive integer")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(order_ids, Sequence) or isinstance(order_ids, (str, bytes)):
        raise TypeError("order_ids must be an ordered sequence")
    order = tuple(order_ids)
    if len(order) < updates * accumulation:
        raise ValueError("order_ids does not cover all requested microbatches")
    rows = _normalize_rows(rows_by_id)
    if any(type(row_id) is not str or row_id not in rows for row_id in order):
        raise ValueError("order_ids contains an unknown row ID")

    active_order = order[:updates * accumulation]
    row_digest = _digest([
        {"id": row_id, "input_ids": rows[row_id][0], "labels": rows[row_id][1]}
        for row_id in active_order
    ])
    order_digest = _digest(list(order))
    binding = {
        "target_updates": updates,
        "accumulation": accumulation,
        "training_rows_sha256": row_digest,
        "training_order_sha256": order_digest,
    }
    prepare = getattr(model, "prepare_preflight_training", None)
    if not callable(prepare):
        raise TypeError("model must expose prepare_preflight_training")
    loss_fn = prepare(rows, binding=binding, order=order)
    if not callable(loss_fn):
        raise TypeError("prepared model must return a loss function")

    named = model.named_adapter_parameters()
    if not isinstance(named, Mapping) or not named:
        raise ValueError("model has no trainable adapter parameters")
    parameters = list(named.values())
    if any(not isinstance(p, torch.nn.Parameter) or not p.requires_grad for p in parameters):
        raise ValueError("adapter mapping must contain trainable Parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=float(learning_rate), betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=False, fused=False,
    )

    losses: list[float] = []
    for update in range(updates):
        window = order[update * accumulation:(update + 1) * accumulation]
        token_total = sum(rows[row_id][2] for row_id in window)
        if token_total <= 0:
            raise ValueError("an accumulation window has no supervised tokens")
        optimizer.zero_grad(set_to_none=True)
        update_loss: torch.Tensor | None = None
        for row_id in window:
            loss, _input_leaf = loss_fn(
                row_id, normalization=token_total, token_chunk=2
            )
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("prepared model returned a non-scalar loss")
            update_loss = loss.detach() if update_loss is None else update_loss + loss.detach()
            loss.backward()
        assert update_loss is not None
        logged_loss = float(update_loss.item())
        if not math.isfinite(logged_loss):
            raise FloatingPointError("non-finite update loss; optimizer step was not applied")
        optimizer.step()
        losses.append(logged_loss)
    return {
        "updates": updates,
        "microbatches": updates * accumulation,
        "losses": losses,
        "training_rows_sha256": row_digest,
        "training_order_sha256": order_digest,
        "model_binding": model.checkpoint_binding(),
    }


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _adapter_binding(model: Any) -> dict[str, Any]:
    getter = getattr(model, "checkpoint_binding", None)
    if not callable(getter):
        raise TypeError("model must expose checkpoint_binding")
    raw = getter()
    if not isinstance(raw, Mapping):
        raise TypeError("model checkpoint binding must be a mapping")
    bound_profile = raw.get("runtime_profile")
    model_profile = getattr(model, "runtime_profile", None)
    if (not isinstance(bound_profile, Mapping) or not isinstance(model_profile, Mapping)
            or _canonical(dict(bound_profile)) != _canonical(dict(model_profile))):
        raise ValueError("model runtime profile differs from its checkpoint binding")
    missing = set(_ADAPTER_BINDING_KEYS) - set(raw)
    if missing:
        raise ValueError(f"model checkpoint binding is missing {sorted(missing)}")
    result = {key: raw[key] for key in _ADAPTER_BINDING_KEYS}
    for key in ("artifact_identity", "source_identity", "record_state_identity", "config_sha256", "initialization_identity"):
        _require_sha256(result[key], key)
    if type(result["rank"]) is not int or result["rank"] <= 0:
        raise ValueError("model binding has an invalid LoRA rank")
    if isinstance(result["alpha"], bool) or type(result["alpha"]) not in (int, float) or not math.isfinite(result["alpha"]):
        raise ValueError("model binding has an invalid LoRA alpha")
    targets = result["targets"]
    if (not isinstance(targets, list) or not targets or any(type(item) is not str or not item for item in targets)
            or targets != sorted(set(targets))):
        raise ValueError("model binding has an invalid target inventory")
    return result


def _adapter_parameters(model: Any, profile: Mapping[str, Any]) -> tuple[dict[str, torch.nn.Parameter], dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    getter = getattr(model, "named_adapter_parameters", None)
    if not callable(getter):
        raise TypeError("model must expose named_adapter_parameters")
    named = getter()
    if not isinstance(named, Mapping) or not named:
        raise ValueError("model has no trainable adapter parameters")
    expected = {f"{target}/{suffix}" for target in profile["targets"] for suffix in ("A", "B")}
    if set(named) != expected:
        raise ValueError("model adapter parameter inventory differs from the runtime profile")
    parameters: dict[str, torch.nn.Parameter] = {}
    values: dict[str, torch.Tensor] = {}
    specs: dict[str, dict[str, Any]] = {}
    by_target: dict[str, dict[str, torch.Tensor]] = {}
    for name, parameter in named.items():
        if type(name) is not str or not name or not isinstance(parameter, torch.nn.Parameter):
            raise TypeError("adapter parameters must be named torch.nn.Parameter values")
        if not parameter.requires_grad or parameter.dtype is not torch.float32:
            raise ValueError(f"adapter parameter {name!r} must be trainable FP32")
        target, separator, role = name.rpartition("/")
        if not separator or role not in ("A", "B") or target not in profile["targets"]:
            raise ValueError(f"invalid adapter parameter name {name!r}")
        if parameter.ndim != 2:
            raise ValueError(f"adapter parameter {name!r} must be a matrix")
        value = parameter.detach().to(device="cpu").contiguous()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"adapter parameter {name!r} contains non-finite values")
        parameters[name] = parameter
        values[name] = value
        specs[name] = {"shape": list(value.shape), "dtype": "float32"}
        by_target.setdefault(target, {})[role] = value
    for target, factors in by_target.items():
        a, b = factors["A"], factors["B"]
        if a.shape[0] != profile["rank"] or b.shape[1] != profile["rank"]:
            raise ValueError(f"adapter parameter rank mismatch for {target!r}")
    return parameters, values, specs


def _validate_training_metadata(training: Mapping[str, Any]) -> dict[str, Any]:
    required = {"updates", "accumulation", "training_rows_sha256", "training_order_sha256"}
    if not isinstance(training, Mapping) or set(training) != required:
        raise ValueError("training metadata must contain updates, accumulation, and both data hashes")
    if type(training["updates"]) is not int or training["updates"] <= 0:
        raise ValueError("training metadata has an invalid update count")
    if type(training["accumulation"]) is not int or training["accumulation"] <= 0:
        raise ValueError("training metadata has an invalid accumulation count")
    return {
        "updates": training["updates"],
        "accumulation": training["accumulation"],
        "training_rows_sha256": _require_sha256(training["training_rows_sha256"], "training rows digest"),
        "training_order_sha256": _require_sha256(training["training_order_sha256"], "training order digest"),
    }


def save_nf4_adapter(
    path: str | os.PathLike[str],
    model: Any,
    *,
    training: Mapping[str, Any],
) -> Path:
    """Save a final adapter with its full profile and an atomic no-replace write."""
    from safetensors.torch import save_file

    binding = _adapter_binding(model)
    profile = getattr(model, "runtime_profile", None)
    if not isinstance(profile, Mapping):
        raise ValueError("a source-bound runtime profile is required to save an NF4 adapter")
    profile_value = json.loads(_canonical(dict(profile)).decode("utf-8"))
    for key, value in (("artifact_identity", binding["artifact_identity"]),
                       ("source_identity", binding["source_identity"]),
                       ("rank", binding["rank"]), ("alpha", binding["alpha"]),
                       ("targets", binding["targets"])):
        if profile_value.get(key) != value:
            raise ValueError(f"runtime profile does not match model binding field {key!r}")
    _parameters, values, specs = _adapter_parameters(model, profile_value)
    metadata = {
        "schema": _ADAPTER_SCHEMA,
        "model_binding": binding,
        "runtime_profile": profile_value,
        "training": _validate_training_metadata(training),
        "parameters": specs,
    }
    metadata_json = _canonical(metadata).decode("utf-8")
    header = {
        "format": _ADAPTER_FORMAT,
        "schema": _ADAPTER_SCHEMA,
        "metadata_json": metadata_json,
        "metadata_sha256": hashlib.sha256(metadata_json.encode("utf-8")).hexdigest(),
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(values, str(temporary), metadata=header)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails if another writer won the
        # destination. Unlike exists()+replace(), it cannot overwrite a peer.
        os.link(temporary, target, follow_symlinks=False)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def read_nf4_adapter_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate the portable metadata needed to reconstruct an adapter model."""
    from safetensors import safe_open

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError("NF4 adapter must be a regular safetensors file")
    try:
        with safe_open(str(target), framework="pt", device="cpu") as archive:
            header = archive.metadata()
    except Exception as exc:
        raise ValueError("invalid NF4 adapter safetensors file") from exc
    if not isinstance(header, dict) or set(header) != {"format", "schema", "metadata_json", "metadata_sha256"}:
        raise ValueError("invalid NF4 adapter header metadata")
    if header["format"] != _ADAPTER_FORMAT or header["schema"] != _ADAPTER_SCHEMA:
        raise ValueError("unsupported NF4 adapter format/schema")
    metadata_json = header["metadata_json"]
    if type(metadata_json) is not str or _require_sha256(header["metadata_sha256"], "adapter metadata digest") != hashlib.sha256(metadata_json.encode("utf-8")).hexdigest():
        raise ValueError("NF4 adapter metadata digest mismatch")
    try:
        metadata = json.loads(metadata_json, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid NF4 adapter metadata JSON") from exc
    if _canonical(metadata).decode("utf-8") != metadata_json:
        raise ValueError("NF4 adapter metadata is not canonical JSON")
    if not isinstance(metadata, dict) or set(metadata) != {"schema", "model_binding", "runtime_profile", "training", "parameters"}:
        raise ValueError("invalid NF4 adapter metadata schema")
    if metadata["schema"] != _ADAPTER_SCHEMA:
        raise ValueError("unsupported NF4 adapter metadata schema")
    binding, profile, specs = metadata["model_binding"], metadata["runtime_profile"], metadata["parameters"]
    if not isinstance(binding, dict) or set(binding) != set(_ADAPTER_BINDING_KEYS):
        raise ValueError("invalid NF4 adapter model binding")
    if not isinstance(profile, dict) or not isinstance(specs, dict) or not specs:
        raise ValueError("invalid NF4 adapter profile/parameter inventory")
    _validate_training_metadata(metadata["training"])
    if any(profile.get(key) != binding.get(key) for key in ("artifact_identity", "source_identity", "rank", "alpha", "targets")):
        raise ValueError("NF4 adapter profile and model binding disagree")
    if not isinstance(profile.get("profile_sha256"), str):
        raise ValueError("NF4 adapter is missing the full source-bound runtime profile")
    profile_payload = dict(profile)
    profile_digest = profile_payload.pop("profile_sha256")
    if _require_sha256(profile_digest, "runtime profile digest") != hashlib.sha256(_canonical(profile_payload)).hexdigest():
        raise ValueError("NF4 adapter runtime profile digest mismatch")
    for name, spec in specs.items():
        if (type(name) is not str or not name or not isinstance(spec, dict) or set(spec) != {"shape", "dtype"}
                or not isinstance(spec["shape"], list) or any(type(size) is not int or size <= 0 for size in spec["shape"])
                or spec["dtype"] != "float32"):
            raise ValueError("invalid NF4 adapter tensor specification")
    return metadata


def load_nf4_adapter(model: Any, path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load an adapter file into a matching constructed model after identity/shape checks."""
    from safetensors.torch import load_file

    metadata = read_nf4_adapter_metadata(path)
    binding = _adapter_binding(model)
    if metadata["model_binding"] != binding:
        raise ValueError("NF4 adapter belongs to a different model/artifact/profile binding")
    model_profile = getattr(model, "runtime_profile", None)
    if not isinstance(model_profile, Mapping) or metadata["runtime_profile"] != dict(model_profile):
        raise ValueError("NF4 adapter runtime profile does not match the constructed model")
    parameters, _unused_values, specs = _adapter_parameters(model, metadata["runtime_profile"])
    if metadata["parameters"] != specs:
        raise ValueError("NF4 adapter tensor shapes do not match the constructed model")
    try:
        values = load_file(str(path), device="cpu")
    except Exception as exc:
        raise ValueError("invalid NF4 adapter tensor payload") from exc
    if not isinstance(values, dict) or set(values) != set(parameters):
        raise ValueError("NF4 adapter tensor inventory mismatch")
    for name, value in values.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype is not torch.float32 or list(value.shape) != specs[name]["shape"]
                or not torch.isfinite(value).all()):
            raise ValueError(f"invalid NF4 adapter tensor {name!r}")
    with torch.no_grad():
        for name in sorted(parameters):
            destination = parameters[name]
            destination.copy_(values[name].to(device=destination.device, dtype=destination.dtype))
    return metadata


def train_qwen35_nf4_streaming(
    artifact: Any,
    rows_by_id: Mapping[str, Any],
    order_ids: Sequence[str],
    *,
    output_path: str | os.PathLike[str],
    updates: int,
    accumulation: int = 1,
    rank: int = 8,
    alpha: float = 16.0,
    seed: int = 20261019,
    learning_rate: float = 2e-4,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Construct and train the public Qwen3.5 NF4 streaming adapter path.

    `artifact` is a canonical NF4ModelArtifact. It can be loaded with
    `load_nf4_model` or created with `build_profiled_qwen35_nf4_artifact`.
    This API writes a final adapter-only file; it does not resume training or
    claim GPU qualification for arbitrary hardware.
    """
    from ..adapters.qwen35.nf4_cuda import NF4Qwen35CUDA
    from ..quantization.nf4_runtime_profile import make_runtime_profile

    if device != "cuda:0":
        raise ValueError("the public Qwen3.5 NF4 streaming path requires explicit cuda:0")
    profile = make_runtime_profile(
        artifact.config, artifact.fingerprint(), artifact.source["payload_sha256"],
        rank=rank, alpha=alpha, seed=seed,
    )
    model = NF4Qwen35CUDA(
        artifact, mode="streaming", runtime_profile=profile,
        rank=rank, alpha=alpha, seed=seed, device=device,
    )
    primary: BaseException | None = None
    try:
        result = run_nf4_updates(
            model, rows_by_id, order_ids, updates=updates,
            accumulation=accumulation, learning_rate=learning_rate,
        )
        output = save_nf4_adapter(
            output_path, model,
            training={
                "updates": result["updates"],
                "accumulation": result["microbatches"] // result["updates"],
                "training_rows_sha256": result["training_rows_sha256"],
                "training_order_sha256": result["training_order_sha256"],
            },
        )
        return {**result, "profile": profile, "adapter_path": str(output)}
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            model.close()
        except BaseException as close_error:
            if primary is None:
                raise
            primary.add_note(f"NF4 model cleanup also failed: {close_error!r}")
