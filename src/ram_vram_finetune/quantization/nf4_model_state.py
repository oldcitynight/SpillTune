"""Owned, portable model-wide CPU NF4 artifacts and atomic archives."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .nf4_state import NF4Record, load_nf4_record, save_nf4_record

MODEL_FORMAT = "rvft.nf4_model.v1"
MODEL_SCHEMA = "rvft.nf4_model_state.v1"
MANIFEST_FORMAT = "rvft.nf4_model_manifest.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_METADATA_KEYS = {"format", "schema", "config", "source", "aliases", "records", "tensors", "fingerprint"}
_RECORD_FILES = {"COMPLETE", "manifest.json", "metadata.json", "state.safetensors"}


def _json_safe(value: Any) -> Any:
    """Validate and detach JSON primitives; tuples normalize to arrays."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite model artifact metadata")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("model artifact JSON keys must be strings")
            if key in result:
                raise ValueError("duplicate model artifact JSON key")
            result[key] = _json_safe(item)
        return result
    raise ValueError(f"unsupported model artifact metadata type: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _snapshot_json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    result = _json_safe(value)
    if type(result) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return result


def _public_json(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(value).decode("utf-8"))


def _name(value: Any, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ValueError(f"invalid {label}")
    return value


def _name_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a name mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = _name(key, f"{label} name")
        if name in result:
            raise ValueError(f"duplicate {label} name")
        result[name] = item
    return result


def _dtype_name(value: torch.Tensor) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()


def _copy_raw_tensors(tensors: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    staged = _name_mapping(tensors, "tensor")
    storage_owners: dict[int, str] = {}
    result: dict[str, torch.Tensor] = {}
    for name in sorted(staged):
        value = staged[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{name} must be BF16 or FP32 storage")
        if value.requires_grad:
            raise ValueError(f"{name} must be a frozen source tensor")
        storage_pointer = value.untyped_storage().data_ptr()
        if storage_pointer:
            previous = storage_owners.get(storage_pointer)
            if previous is not None:
                raise ValueError(f"raw tensor storage aliases {previous!r} and {name!r}; use aliases metadata")
            storage_owners[storage_pointer] = name
        result[name] = value.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)
    return result


def _normalize_aliases(aliases: Any, canonical_names: set[str]) -> dict[str, str]:
    staged = _name_mapping(aliases, "alias")
    normalized: dict[str, str] = {}
    destinations: set[str] = set()
    for alias, destination in staged.items():
        alias = _name(alias, "alias name")
        destination = _name(destination, "alias destination")
        if alias in canonical_names:
            raise ValueError("alias payload is forbidden; store only its canonical tensor")
        if destination not in canonical_names:
            raise ValueError("alias destination does not exist as canonical payload")
        if destination in destinations:
            raise ValueError("duplicate alias destination")
        destinations.add(destination)
        normalized[alias] = destination
    alias_names = set(normalized)
    if alias_names & set(normalized.values()):
        raise ValueError("alias chains and cycles are forbidden")
    return {key: normalized[key] for key in sorted(normalized)}


def _hex_digest(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {label} SHA-256")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key in model artifact JSON")
        value[key] = item
    return value


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=lambda _item: (_ for _ in ()).throw(ValueError("invalid JSON constant")),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid model artifact JSON") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        directory = Path(current)
        for dirname in dirnames:
            child = directory / dirname
            if child.is_symlink():
                raise ValueError("symlink in model artifact staging tree")
            directories.append(child)
        for filename in filenames:
            item = directory / filename
            if item.is_symlink() or not item.is_file():
                raise ValueError("non-regular file in model artifact staging tree")
            with item.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in reversed(directories):
        _fsync_directory(directory)


def _build_root_manifest(stage: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for item in sorted(stage.rglob("*")):
        if item.is_symlink():
            raise ValueError("symlink in model artifact staging tree")
        if item.is_file():
            relative = item.relative_to(stage).as_posix()
            if relative in ("manifest.json", "COMPLETE"):
                continue
            files[relative] = _file_sha256(item)
        elif not item.is_dir():
            raise ValueError("invalid model artifact staging entry")
    return {"format": MANIFEST_FORMAT, "files": files}


class NF4ModelArtifact:
    """Canonical owned CPU state for NF4 records, raw tensors, and direct aliases."""

    __slots__ = ("_config", "_source", "_aliases", "_records", "_tensors")

    def __init__(self, *, config: Any, records: Any, tensors: Any, aliases: Any, source: Any) -> None:
        self._config = _snapshot_json_object(config, "config")
        self._source = _snapshot_json_object(source, "source")
        incoming_records = _name_mapping(records, "record")
        incoming_tensors = _name_mapping(tensors, "tensor")
        overlap = set(incoming_records) & set(incoming_tensors)
        if overlap:
            raise ValueError("a source tensor cannot appear in both record and raw tensor maps")
        if not incoming_records and not incoming_tensors:
            raise ValueError("model artifact must contain source tensors")

        canonical_names = set(incoming_records) | set(incoming_tensors)
        self._aliases = _normalize_aliases(aliases, canonical_names)

        owned_records: dict[str, NF4Record] = {}
        seen_record_objects: dict[int, str] = {}
        for name in sorted(incoming_records):
            value = incoming_records[name]
            if not isinstance(value, NF4Record):
                raise ValueError(f"{name} must be an NF4Record")
            identity = id(value)
            previous = seen_record_objects.get(identity)
            if previous is not None:
                raise ValueError(f"record object is reused for {previous!r} and {name!r}")
            seen_record_objects[identity] = name
            packed, parts = value.export_parts()
            owned_records[name] = NF4Record.from_parts(packed, parts)
        owned_tensors = _copy_raw_tensors(incoming_tensors)
        self._records = owned_records
        self._tensors = owned_tensors

    @classmethod
    def from_parts(
        cls,
        *,
        config: Any,
        records: Mapping[str, NF4Record],
        tensors: Mapping[str, torch.Tensor],
        aliases: Mapping[str, str],
        source: Any,
    ) -> "NF4ModelArtifact":
        return cls(config=config, records=records, tensors=tensors, aliases=aliases, source=source)

    @property
    def config(self) -> dict[str, Any]:
        return _public_json(self._config)

    @property
    def source(self) -> dict[str, Any]:
        return _public_json(self._source)

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    @property
    def record_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensors))

    def record(self, name: str) -> NF4Record:
        return self._records[name]

    def tensor(self, name: str) -> torch.Tensor:
        value = self._tensors[name]
        return value.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)

    def tensor_metadata(self, name: str):
        """Return immutable scalar/shape metadata without exposing tensor storage."""
        from types import MappingProxyType

        value = self._tensors[name]
        return MappingProxyType({
            "shape": tuple(value.shape),
            "dtype": value.dtype,
            "device": value.device,
            "requires_grad": bool(value.requires_grad),
            "contiguous": bool(value.is_contiguous()),
        })

    def tensor_all_finite(self, name: str, chunk_elements: int = 1 << 20) -> bool:
        """Check every owned CPU tensor element with a bounded temporary mask."""
        if type(chunk_elements) is not int or not 1 <= chunk_elements <= 1 << 20:
            raise ValueError("chunk_elements must be an integer from 1 through 1 Mi elements")
        value = self._tensors[name]
        if value.device.type != "cpu" or value.requires_grad or not value.is_contiguous():
            raise ValueError("finite scan requires an owned contiguous frozen CPU tensor")
        flat = value.view(-1)
        for start in range(0, flat.numel(), chunk_elements):
            if not bool(torch.isfinite(flat[start:start + chunk_elements]).all()):
                return False
        return True

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"rvft.nf4.model.fingerprint.v1\0")
        record_identity = [
            {"name": name, "fingerprint": self._records[name].fingerprint()}
            for name in sorted(self._records)
        ]
        tensor_identity = [
            {"name": name, "dtype": _dtype_name(value), "shape": list(value.shape)}
            for name, value in sorted(self._tensors.items())
        ]
        identity = _canonical_json({
            "format": MODEL_FORMAT,
            "schema": MODEL_SCHEMA,
            "config": self._config,
            "source": self._source,
            "aliases": self._aliases,
            "records": record_identity,
            "tensors": tensor_identity,
        })
        digest.update(len(identity).to_bytes(8, "big"))
        digest.update(identity)
        for name in sorted(self._tensors):
            value = self._tensors[name]
            descriptor = _canonical_json({"name": name, "dtype": _dtype_name(value), "shape": list(value.shape)})
            raw = _tensor_bytes(value)
            digest.update(len(descriptor).to_bytes(8, "big"))
            digest.update(descriptor)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()


def save_nf4_model(path: str | os.PathLike[str], artifact: NF4ModelArtifact) -> Path:
    """Atomically publish a complete no-pickle model artifact directory."""
    if not isinstance(artifact, NF4ModelArtifact):
        raise TypeError("artifact must be an NF4ModelArtifact")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / ("." + target.name + ".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage: Path | None = None
    try:
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        stage = Path(tempfile.mkdtemp(prefix=".nf4-model-stage-", dir=target.parent))
        records_root = stage / "records"
        records_root.mkdir()
        record_inventory: list[dict[str, str]] = []
        for index, name in enumerate(artifact.record_names):
            slot = f"{index:08d}"
            record = artifact._records[name]
            save_nf4_record(records_root / slot, record)
            record_inventory.append({"name": name, "slot": slot, "fingerprint": record.fingerprint()})

        tensor_inventory: list[dict[str, Any]] = []
        raw_tensors: dict[str, torch.Tensor] = {}
        for index, name in enumerate(artifact.tensor_names):
            key = f"tensor/{index:08d}"
            value = artifact._tensors[name]
            raw_tensors[key] = value
            tensor_inventory.append({"name": name, "key": key, "dtype": _dtype_name(value), "shape": list(value.shape)})
        save_file(raw_tensors, str(stage / "tensors.safetensors"))

        metadata = {
            "format": MODEL_FORMAT,
            "schema": MODEL_SCHEMA,
            "config": artifact._config,
            "source": artifact._source,
            "aliases": artifact._aliases,
            "records": record_inventory,
            "tensors": tensor_inventory,
            "fingerprint": artifact.fingerprint(),
        }
        _write_json(stage / "metadata.json", metadata)
        manifest = _build_root_manifest(stage)
        _write_json(stage / "manifest.json", manifest)
        manifest_digest = _file_sha256(stage / "manifest.json")
        with (stage / "COMPLETE").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(f"manifest-sha256:{manifest_digest}\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_tree(stage)
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


def _validated_inventory(metadata: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    records = metadata.get("records")
    tensors = metadata.get("tensors")
    if not isinstance(records, list) or not isinstance(tensors, list):
        raise ValueError("invalid model artifact inventory")
    normalized_records: list[dict[str, Any]] = []
    record_names: list[str] = []
    for index, entry in enumerate(records):
        if not isinstance(entry, dict) or set(entry) != {"name", "slot", "fingerprint"}:
            raise ValueError("invalid model record inventory entry")
        name = _name(entry["name"], "record name")
        slot = f"{index:08d}"
        if entry["slot"] != slot:
            raise ValueError("noncanonical model record slot")
        fingerprint = _hex_digest(entry["fingerprint"], "record fingerprint")
        normalized_records.append({"name": name, "slot": slot, "fingerprint": fingerprint})
        record_names.append(name)
    if record_names != sorted(set(record_names)):
        raise ValueError("model record names are not canonical")

    normalized_tensors: list[dict[str, Any]] = []
    tensor_names: list[str] = []
    for index, entry in enumerate(tensors):
        if not isinstance(entry, dict) or set(entry) != {"name", "key", "dtype", "shape"}:
            raise ValueError("invalid raw tensor inventory entry")
        name = _name(entry["name"], "tensor name")
        key = f"tensor/{index:08d}"
        if entry["key"] != key or entry["dtype"] not in ("bfloat16", "float32"):
            raise ValueError("noncanonical raw tensor descriptor")
        shape = entry["shape"]
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("invalid raw tensor shape")
        normalized_tensors.append({"name": name, "key": key, "dtype": entry["dtype"], "shape": list(shape)})
        tensor_names.append(name)
    if tensor_names != sorted(set(tensor_names)) or set(record_names) & set(tensor_names):
        raise ValueError("model tensor names are noncanonical or overlap")
    if not normalized_records and not normalized_tensors:
        raise ValueError("empty model artifact inventory")

    canonical_names = set(record_names) | set(tensor_names)
    aliases = _normalize_aliases(metadata.get("aliases"), canonical_names)
    if aliases != metadata.get("aliases"):
        raise ValueError("noncanonical alias metadata")
    return normalized_records, normalized_tensors, aliases


def _tree_inventory(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(current)
        for dirname in dirnames:
            child = base / dirname
            if child.is_symlink() or not child.is_dir():
                raise ValueError("invalid directory in model artifact")
            directories.add(child.relative_to(root).as_posix())
        for filename in filenames:
            child = base / filename
            if child.is_symlink() or not child.is_file():
                raise ValueError("invalid file in model artifact")
            files.add(child.relative_to(root).as_posix())
    return files, directories


def load_nf4_model(path: str | os.PathLike[str]) -> NF4ModelArtifact:
    """Validate every archive member, record, alias, and fingerprint before return."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("NF4 model artifact must be a real directory")
    top_entries = {item.name for item in root.iterdir()}
    if top_entries != {"COMPLETE", "manifest.json", "metadata.json", "records", "tensors.safetensors"}:
        raise ValueError("invalid NF4 model directory inventory")
    metadata = _read_json(root / "metadata.json")
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        raise ValueError("invalid NF4 model metadata schema")
    if metadata.get("format") != MODEL_FORMAT or metadata.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported NF4 model format/schema")
    config = _snapshot_json_object(metadata.get("config"), "config")
    source = _snapshot_json_object(metadata.get("source"), "source")
    records, tensor_entries, aliases = _validated_inventory(metadata)
    fingerprint = _hex_digest(metadata.get("fingerprint"), "model fingerprint")

    expected_files = {"COMPLETE", "manifest.json", "metadata.json", "tensors.safetensors"}
    expected_directories = {"records"}
    for entry in records:
        base = f"records/{entry['slot']}"
        expected_directories.add(base)
        expected_files.update(f"{base}/{name}" for name in _RECORD_FILES)
    actual_files, actual_directories = _tree_inventory(root)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("NF4 model file inventory mismatch")

    manifest = _read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {"format", "files"} or manifest["format"] != MANIFEST_FORMAT:
        raise ValueError("invalid NF4 model manifest schema")
    manifest_files = manifest.get("files")
    expected_hashed_files = expected_files - {"COMPLETE", "manifest.json"}
    if not isinstance(manifest_files, dict) or set(manifest_files) != expected_hashed_files:
        raise ValueError("NF4 model manifest inventory mismatch")
    for relative, expected_digest in manifest_files.items():
        expected_digest = _hex_digest(expected_digest, "model manifest entry")
        if _file_sha256(root / relative) != expected_digest:
            raise ValueError("NF4 model file hash mismatch")
    expected_complete = f"manifest-sha256:{_file_sha256(root / 'manifest.json')}\n"
    try:
        complete = (root / "COMPLETE").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("invalid NF4 model completion marker") from exc
    if complete != expected_complete:
        raise ValueError("NF4 model completion marker mismatch")

    try:
        loaded_tensors = load_file(str(root / "tensors.safetensors"), device="cpu")
    except Exception as exc:
        raise ValueError("invalid NF4 model safetensors archive") from exc
    expected_tensor_keys = {entry["key"] for entry in tensor_entries}
    if set(loaded_tensors) != expected_tensor_keys:
        raise ValueError("NF4 model tensor inventory mismatch")
    raw_tensors: dict[str, torch.Tensor] = {}
    for entry in tensor_entries:
        value = loaded_tensors[entry["key"]]
        if value.device.type != "cpu" or _dtype_name(value) != entry["dtype"] or list(value.shape) != entry["shape"]:
            raise ValueError("NF4 model tensor descriptor mismatch")
        raw_tensors[entry["name"]] = value

    model_records: dict[str, NF4Record] = {}
    for entry in records:
        record = load_nf4_record(root / "records" / entry["slot"])
        if record.fingerprint() != entry["fingerprint"]:
            raise ValueError("NF4 record fingerprint does not match model inventory")
        model_records[entry["name"]] = record

    artifact = NF4ModelArtifact.from_parts(
        config=config,
        records=model_records,
        tensors=raw_tensors,
        aliases=aliases,
        source=source,
    )
    if artifact.fingerprint() != fingerprint:
        raise ValueError("NF4 model content fingerprint mismatch")
    return artifact
