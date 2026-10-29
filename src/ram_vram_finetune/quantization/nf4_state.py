"""Canonical CPU NF4 + double-quantization records and safe persistence.

This module deliberately stores the official bitsandbytes packed QuantState
parts.  It does not implement a quantization codec and never uses pickle.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from . import nf4_backend

FORMAT = "rvft.nf4_record.v1"
STATE_SCHEMA = "rvft.nf4_state.v1"
EXPECTED_BACKEND_VERSION = "0.50.2"

_REQUIRED_LOGICAL = {
    "shape",
    "dtype",
    "blocksize",
    "secondary_blocksize",
    "quant_type",
    "nested",
    "backend_version",
}
_REQUIRED_METADATA = {
    "format",
    "state_schema",
    "backend_info",
    "logical",
    "state_metadata",
    "state_tensor_keys",
    "fingerprint",
}


def _json_value(value: Any) -> Any:
    """Return a detached JSON value, rejecting objects and non-finite floats."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not (float("-inf") < value < float("inf")):
            raise ValueError("non-finite NF4 metadata")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("NF4 metadata keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported NF4 metadata value: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _copy_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    return value.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)


def _copy_parts(parts: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(parts, Mapping) or not parts:
        raise ValueError("quant_state_parts must be a non-empty mapping")
    result: dict[str, Any] = {}
    for key, value in parts.items():
        if type(key) is not str or not key or key in result:
            raise ValueError("invalid quant_state_parts key")
        if isinstance(value, torch.Tensor):
            result[key] = _copy_tensor(value, name=f"quant_state_parts[{key!r}]")
        else:
            result[key] = _json_value(value)
    return result


def _backend_info() -> dict[str, Any]:
    try:
        value = nf4_backend.backend_info()
    except Exception as exc:
        raise ValueError("NF4 backend information unavailable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("invalid NF4 backend information")
    info = _json_value(value)
    if info.get("version") != EXPECTED_BACKEND_VERSION:
        raise ValueError("unsupported NF4 backend version")
    source_digest = info.get("source_digest")
    if type(source_digest) is not str or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
        raise ValueError("invalid NF4 backend source identity")
    return info


def _normalize_logical(value: Mapping[str, Any], backend_info: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid NF4 logical state metadata")
    logical = _json_value(value)
    if not _REQUIRED_LOGICAL.issubset(logical):
        raise ValueError("incomplete NF4 logical state metadata")
    shape = logical["shape"]
    if not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("NF4 record shape must be positive and two-dimensional")
    logical["dtype"] = str(logical["dtype"]).removeprefix("torch.")
    if (
        logical["dtype"] != "bfloat16"
        or type(logical["blocksize"]) is not int
        or logical["blocksize"] != 64
        or type(logical["secondary_blocksize"]) is not int
        or logical["secondary_blocksize"] != 256
        or logical["quant_type"] != "nf4"
        or logical["nested"] is not True
        or type(logical["backend_version"]) is not str
        or logical["backend_version"] != backend_info["version"]
    ):
        raise ValueError("NF4/DQ state does not match the supported CPU schema")
    return logical


def _inspect(packed: torch.Tensor, parts: Mapping[str, Any], backend_info: Mapping[str, Any]) -> dict[str, Any]:
    # The backend is allowed to consume/mutate a QuantState dictionary.  Give it
    # disposable copies and retain only the separate canonical snapshots.
    inspect_packed = _copy_tensor(packed, name="packed")
    inspect_parts = _copy_parts(parts)
    try:
        logical = nf4_backend.inspect_state(inspect_packed, inspect_parts)
    except Exception as exc:
        raise ValueError("invalid or unsupported NF4 quantization state") from exc
    return _normalize_logical(logical, backend_info)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    # Viewing storage as uint8 preserves the exact bytes for every tensor dtype,
    # including bfloat16, without a dtype conversion.
    return tensor.detach().contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()


class NF4Record:
    """Owned, canonical CPU storage for one official NF4/DQ packed weight."""

    __slots__ = ("_packed", "_parts", "_logical", "_backend")

    def __init__(self, packed: torch.Tensor, parts: Mapping[str, Any]) -> None:
        backend = _backend_info()
        packed_copy = _copy_tensor(packed, name="packed")
        if packed_copy.dtype is not torch.uint8 or packed_copy.numel() == 0:
            raise ValueError("packed NF4 weight must be non-empty uint8 CPU storage")
        parts_copy = _copy_parts(parts)
        logical = _inspect(packed_copy, parts_copy, backend)
        self._packed = packed_copy
        self._parts = parts_copy
        self._logical = logical
        self._backend = backend

    @classmethod
    def from_weight(cls, weight: torch.Tensor) -> "NF4Record":
        if not isinstance(weight, torch.Tensor):
            raise ValueError("weight must be a tensor")
        if weight.device.type != "cpu" or weight.dtype is not torch.bfloat16 or weight.ndim != 2:
            raise ValueError("weight must be a two-dimensional BF16 CPU tensor")
        if weight.requires_grad:
            raise ValueError("NF4 base weights must be frozen")
        try:
            packed, parts = nf4_backend.quantize_cpu(weight)
        except Exception as exc:
            raise ValueError("official CPU NF4/DQ quantization failed") from exc
        return cls(packed, parts)

    @classmethod
    def from_parts(cls, packed: torch.Tensor, quant_state_parts: Mapping[str, Any]) -> "NF4Record":
        """Create an independently owned record from official packed parts."""
        return cls(packed, quant_state_parts)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._logical["shape"])

    def export_parts(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return fresh CPU copies; no returned tensor aliases canonical storage."""
        packed = _copy_tensor(self._packed, name="packed")
        parts = _copy_parts(self._parts)
        return packed, parts

    def fingerprint(self) -> str:
        """Stable SHA-256 identity for bytes, metadata, schema, and backend."""
        digest = hashlib.sha256()
        digest.update(b"rvft.nf4.fingerprint.v1\0")
        identity = {
            "format": FORMAT,
            "state_schema": STATE_SCHEMA,
            "backend_info": self._backend,
            "logical": self._logical,
            "state_metadata": {
                key: value for key, value in self._parts.items() if not isinstance(value, torch.Tensor)
            },
        }
        encoded = _canonical_json(identity)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        tensors = {"packed": self._packed}
        tensors.update({f"state/{key}": value for key, value in self._parts.items() if isinstance(value, torch.Tensor)})
        for key in sorted(tensors):
            tensor = tensors[key]
            descriptor = _canonical_json({
                "key": key,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
            })
            raw = _tensor_bytes(tensor)
            digest.update(len(descriptor).to_bytes(8, "big"))
            digest.update(descriptor)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(value), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_nf4_record(path: str | os.PathLike[str], record: NF4Record) -> Path:
    """Atomically publish a complete safetensors record directory, never overwrite."""
    if not isinstance(record, NF4Record):
        raise TypeError("record must be an NF4Record")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / ("." + target.name + ".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage: Path | None = None
    try:
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        stage = Path(tempfile.mkdtemp(prefix=".nf4-stage-", dir=target.parent))
        tensors: dict[str, torch.Tensor] = {"packed": record._packed}
        state_tensor_keys = sorted(key for key, value in record._parts.items() if isinstance(value, torch.Tensor))
        tensors.update({f"state/{key}": record._parts[key] for key in state_tensor_keys})
        # Save_file serializes tensor data only; no pickle or executable metadata.
        save_file(tensors, str(stage / "state.safetensors"))
        state_metadata = {
            key: value for key, value in record._parts.items() if not isinstance(value, torch.Tensor)
        }
        metadata = {
            "format": FORMAT,
            "state_schema": STATE_SCHEMA,
            "backend_info": record._backend,
            "logical": record._logical,
            "state_metadata": state_metadata,
            "state_tensor_keys": state_tensor_keys,
            "fingerprint": record.fingerprint(),
        }
        _write_json(stage / "metadata.json", metadata)
        manifest = {
            name: _sha256(stage / name)
            for name in ("state.safetensors", "metadata.json")
        }
        _write_json(stage / "manifest.json", manifest)
        with (stage / "COMPLETE").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("complete\n")
            handle.flush()
            os.fsync(handle.fileno())
        for item in stage.iterdir():
            if item.is_file():
                with item.open("rb") as handle:
                    os.fsync(handle.fileno())
        _fsync_directory(stage)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.rename(stage, target)
        stage = None
        _fsync_directory(target.parent)
        return target
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        os.close(lock_fd)
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in NF4 record")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_pairs,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("invalid JSON constant")),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid NF4 record JSON") from exc


def load_nf4_record(path: str | os.PathLike[str]) -> NF4Record:
    """Validate a complete artifact and current schema/backend before returning it."""
    target = Path(path)
    if target.is_symlink() or not target.is_dir():
        raise ValueError("NF4 record must be a real directory")
    expected_files = {"COMPLETE", "manifest.json", "metadata.json", "state.safetensors"}
    try:
        entries = list(target.iterdir())
    except OSError as exc:
        raise ValueError("cannot read NF4 record directory") from exc
    if any(entry.is_symlink() or not entry.is_file() for entry in entries) or {entry.name for entry in entries} != expected_files:
        raise ValueError("invalid NF4 record directory inventory")
    if (target / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
        raise ValueError("NF4 record is incomplete")
    manifest = _read_json(target / "manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {"state.safetensors", "metadata.json"}:
        raise ValueError("invalid NF4 record manifest")
    for name, expected_digest in manifest.items():
        if type(expected_digest) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
            raise ValueError("invalid NF4 record manifest digest")
        if _sha256(target / name) != expected_digest:
            raise ValueError("NF4 record file hash mismatch")
    metadata = _read_json(target / "metadata.json")
    if not isinstance(metadata, dict) or set(metadata) != _REQUIRED_METADATA:
        raise ValueError("invalid NF4 record metadata schema")
    current_backend = _backend_info()
    if metadata.get("format") != FORMAT or metadata.get("state_schema") != STATE_SCHEMA:
        raise ValueError("unsupported NF4 record format/schema")
    if metadata.get("backend_info") != current_backend:
        raise ValueError("NF4 record backend identity mismatch")
    try:
        loaded = load_file(str(target / "state.safetensors"), device="cpu")
    except Exception as exc:
        raise ValueError("invalid NF4 safetensors state") from exc
    declared_state_keys = metadata.get("state_tensor_keys")
    if (
        not isinstance(declared_state_keys, list)
        or any(type(key) is not str or not key for key in declared_state_keys)
        or declared_state_keys != sorted(set(declared_state_keys))
    ):
        raise ValueError("invalid NF4 state tensor inventory")
    expected_tensor_keys = {"packed", *(f"state/{key}" for key in declared_state_keys)}
    if set(loaded) != expected_tensor_keys:
        raise ValueError("NF4 safetensors tensor inventory mismatch")
    state_metadata = metadata.get("state_metadata")
    if not isinstance(state_metadata, dict) or set(state_metadata) & set(declared_state_keys):
        raise ValueError("invalid NF4 state metadata inventory")
    parts: dict[str, Any] = dict(_json_value(state_metadata))
    parts.update({key: loaded[f"state/{key}"] for key in declared_state_keys})
    record = NF4Record.from_parts(loaded["packed"], parts)
    if record._backend != current_backend or record._logical != metadata.get("logical"):
        raise ValueError("NF4 logical state metadata mismatch")
    if record.fingerprint() != metadata.get("fingerprint"):
        raise ValueError("NF4 record fingerprint mismatch")
    return record
