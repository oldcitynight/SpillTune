"""CPU-only NF4 + double-quantization backend built on bitsandbytes 0.50.2."""
from __future__ import annotations

import hashlib
import math
import sys
from importlib import util as importlib_util
from collections.abc import Mapping
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


_PINNED_VERSION = "0.50.2"
_PRIMARY_BLOCKSIZE = 64
_SECONDARY_BLOCKSIZE = 256
_PACKED_STATE_KEY = "quant_state.bitsandbytes__nf4"
_EXPECTED_KEYS = {
    "absmax",
    "nested_absmax",
    "nested_quant_map",
    "quant_map",
    _PACKED_STATE_KEY,
}


def _installed_package_location() -> tuple[Path, Path, str]:
    """Resolve the unique installed bitsandbytes package without importing it."""
    spec = importlib_util.find_spec("bitsandbytes")
    if spec is None or not spec.origin or not spec.submodule_search_locations:
        raise RuntimeError("bitsandbytes is not installed as an importable package")
    package_file = Path(spec.origin).resolve()
    if package_file.name != "__init__.py":
        raise RuntimeError("bitsandbytes must resolve to a regular package __init__.py")
    locations = tuple(Path(location).resolve() for location in spec.submodule_search_locations)
    package_root = package_file.parent
    if locations != (package_root,):
        raise RuntimeError("bitsandbytes package search root does not match its module origin")

    loaded = sys.modules.get("bitsandbytes")
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_file is not None and Path(loaded_file).resolve() != package_file:
        raise RuntimeError("bitsandbytes is already loaded from a different package root")

    matching_distributions = []
    for distribution in importlib_metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name != "bitsandbytes":
            continue
        installed_init = Path(
            distribution.locate_file("bitsandbytes/__init__.py")
        ).resolve()
        if installed_init == package_file:
            matching_distributions.append(distribution)
    if len(matching_distributions) != 1:
        raise RuntimeError(
            "bitsandbytes import root must match exactly one installed distribution"
        )
    version = matching_distributions[0].version
    if version != _PINNED_VERSION:
        raise RuntimeError(
            f"unsupported bitsandbytes version {version!r}; expected {_PINNED_VERSION}"
        )
    return package_root, package_file, version


@lru_cache(maxsize=1)
def _installed_backend_facts() -> tuple[Path, str, str]:
    package_root, _, version = _installed_package_location()

    # Cover every installed Python source and the CPU implementation binary.
    # Paths are package-relative, so byte-identical official wheel installs have
    # a portable identity independent of their site-packages location.
    implementation_files = [
        path for path in package_root.rglob("*.py") if path.is_file()
    ]
    cpu_library = package_root / "libbitsandbytes_cpu.so"
    if not cpu_library.is_file():
        raise RuntimeError("installed bitsandbytes CPU native library is missing")
    implementation_files.append(cpu_library)

    digest = hashlib.sha256()
    for source_path in sorted(
        implementation_files,
        key=lambda path: path.relative_to(package_root).as_posix(),
    ):
        resolved_path = source_path.resolve()
        try:
            relative = (
                "bitsandbytes/" + resolved_path.relative_to(package_root).as_posix()
            ).encode("utf-8")
        except ValueError as exc:
            raise RuntimeError(
                "bitsandbytes implementation file escapes its installed package root"
            ) from exc
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(resolved_path.stat().st_size.to_bytes(8, "big"))
        with resolved_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return package_root, version, digest.hexdigest()

def backend_info() -> dict[str, Any]:
    """Describe the pinned implementation without importing Torch or probing devices."""
    _, version, source_digest = _installed_backend_facts()
    return {
        "version": version,
        "scheme": "NF4 + Double Quantization + BF16 compute",
        "scope": "CPU-only experimental foundation",
        "cpu_only": True,
        "source_digest": source_digest,
    }


def _load_backend():
    """Load the exact installed bitsandbytes 0.50.2 package and its CPU API."""
    package_root, version, _ = _installed_backend_facts()
    registered_backends = importlib_metadata.entry_points(group="bitsandbytes.backends")
    if registered_backends:
        raise RuntimeError(
            "refusing external bitsandbytes backend extensions in the CPU-only runtime"
        )

    import torch
    import bitsandbytes
    import bitsandbytes.functional as functional
    from bitsandbytes.utils import unpack_tensor_to_dict

    expected_module_file = (package_root / "__init__.py").resolve()
    module_file = Path(bitsandbytes.__file__).resolve()
    if module_file != expected_module_file:
        raise RuntimeError(
            f"bitsandbytes was imported from outside its verified package root: {module_file}"
        )
    if getattr(bitsandbytes, "__version__", None) != version or version != _PINNED_VERSION:
        raise RuntimeError(
            f"unsupported bitsandbytes module version {getattr(bitsandbytes, '__version__', None)!r}"
        )
    return torch, bitsandbytes, functional, unpack_tensor_to_dict

def _require_tensor(value: Any, name: str, torch, *, dtype=None):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU, got {value.device}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    return value


def _copy_parts(parts: Mapping[str, Any], torch) -> dict[str, Any]:
    if not isinstance(parts, Mapping):
        raise TypeError("quant_state_parts must be a mapping")
    keys = set(parts.keys())
    if keys != _EXPECTED_KEYS:
        missing = sorted(_EXPECTED_KEYS.difference(keys))
        extra = sorted(repr(key) for key in keys.difference(_EXPECTED_KEYS))
        raise ValueError(f"malformed quant_state_parts keys (missing={missing}, extra={extra})")

    copied: dict[str, Any] = {}
    for key, value in parts.items():
        _require_tensor(value, f"quant_state_parts[{key!r}]", torch)
        copied[key] = value.detach().clone(memory_format=torch.contiguous_format)
    return copied


def _decode_state_metadata(parts: dict[str, Any], unpack_tensor_to_dict, torch) -> dict[str, Any]:
    packed_metadata = parts[_PACKED_STATE_KEY]
    if packed_metadata.dtype != torch.uint8:
        raise ValueError("packed quant-state metadata must use uint8 storage")
    if packed_metadata.ndim != 1 or packed_metadata.numel() == 0:
        raise ValueError("packed quant-state metadata must be a nonempty one-dimensional tensor")
    try:
        metadata = unpack_tensor_to_dict(packed_metadata)
    except Exception as exc:
        raise ValueError("packed quant-state metadata is malformed") from exc
    if not isinstance(metadata, dict):
        raise ValueError("packed quant-state metadata must decode to a dictionary")
    return metadata


def _validate_state(packed, parts: Mapping[str, Any]):
    torch, _, functional, unpack_tensor_to_dict = _load_backend()
    _require_tensor(packed, "packed", torch, dtype=torch.uint8)
    if packed.ndim != 2 or packed.shape[1] != 1:
        raise ValueError("packed must use the official uint8 [bytes, 1] layout")

    state_parts = _copy_parts(parts, torch)
    metadata = _decode_state_metadata(state_parts, unpack_tensor_to_dict, torch)
    expected_metadata_keys = {
        "quant_type",
        "blocksize",
        "dtype",
        "shape",
        "nested_blocksize",
        "nested_dtype",
        "nested_offset",
    }
    if set(metadata) != expected_metadata_keys:
        raise ValueError("packed quant-state metadata has missing or unknown fields")
    if metadata["quant_type"] != "nf4":
        raise ValueError("only official NF4 quantization state is supported")
    if type(metadata["blocksize"]) is not int or metadata["blocksize"] != _PRIMARY_BLOCKSIZE:
        raise ValueError("NF4 state must use blocksize 64")
    if type(metadata["nested_blocksize"]) is not int or metadata["nested_blocksize"] != _SECONDARY_BLOCKSIZE:
        raise ValueError("double-quantization state must use secondary blocksize 256")
    if metadata["dtype"] != "bfloat16" or metadata["nested_dtype"] != "float32":
        raise ValueError("NF4/DQ state has an unsupported compute or nested dtype")
    shape = metadata["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(extent) is not int or extent <= 0 for extent in shape)
    ):
        raise ValueError("NF4 state shape must contain two positive integer dimensions")

    element_count = shape[0] * shape[1]
    byte_count = (element_count + 1) // 2
    if packed.numel() != byte_count:
        raise ValueError("packed byte count does not match the NF4 state shape")
    block_count = (element_count + _PRIMARY_BLOCKSIZE - 1) // _PRIMARY_BLOCKSIZE
    second_level_count = (block_count + _SECONDARY_BLOCKSIZE - 1) // _SECONDARY_BLOCKSIZE

    absmax = state_parts["absmax"]
    nested_absmax = state_parts["nested_absmax"]
    nested_map = state_parts["nested_quant_map"]
    nf4_map = state_parts["quant_map"]
    packed_metadata = state_parts[_PACKED_STATE_KEY]
    if absmax.dtype != torch.uint8 or absmax.ndim != 1 or absmax.numel() != block_count:
        raise ValueError("primary NF4 absmax state has an invalid dtype or shape")
    if nested_absmax.dtype != torch.float32 or nested_absmax.ndim != 1 or nested_absmax.numel() != second_level_count:
        raise ValueError("secondary DQ absmax state has an invalid dtype or shape")
    if nested_map.dtype != torch.float32 or nested_map.ndim != 1 or nested_map.numel() != 256:
        raise ValueError("secondary DQ codebook must be a 256-entry float32 vector")
    if nf4_map.dtype != torch.float32 or nf4_map.ndim != 1 or nf4_map.numel() != 16:
        raise ValueError("NF4 codebook must be a 16-entry float32 vector")
    if packed_metadata.dtype != torch.uint8:
        raise ValueError("packed quant-state metadata must use uint8 storage")

    offset = metadata["nested_offset"]
    if type(offset) not in (int, float) or not math.isfinite(offset) or offset < 0:
        raise ValueError("secondary DQ offset must be finite and non-negative")
    if not bool(torch.isfinite(nested_absmax).all().item()):
        raise ValueError("secondary DQ absmax state contains a non-finite value")
    if not bool((nested_absmax >= 0).all().item()):
        raise ValueError("secondary DQ absmax values must be non-negative")

    # Compare both codebooks to the official package-generated tables. This catches
    # tampering while preserving legal all-zero absmax/offset values.
    expected_nf4_map = functional.get_4bit_type("nf4", device=torch.device("cpu"))
    expected_nested_map = functional.create_dynamic_map().to(device=torch.device("cpu"))
    if not torch.equal(nf4_map, expected_nf4_map):
        raise ValueError("NF4 state codebook does not match the official NF4 codebook")
    if not torch.equal(nested_map, expected_nested_map):
        raise ValueError("secondary DQ codebook does not match the official dynamic map")

    # from_dict expands and mutates the passed dict, so only hand it private CPU copies.
    try:
        quant_state = functional.QuantState.from_dict(
            dict(state_parts), device=torch.device("cpu")
        )
    except Exception as exc:
        raise ValueError("bitsandbytes rejected the NF4/DQ state parts") from exc
    if (
        quant_state.quant_type != "nf4"
        or quant_state.blocksize != _PRIMARY_BLOCKSIZE
        or quant_state.dtype != torch.bfloat16
        or tuple(quant_state.shape) != tuple(shape)
        or not quant_state.nested
        or quant_state.state2.blocksize != _SECONDARY_BLOCKSIZE
        or quant_state.state2.dtype != torch.float32
    ):
        raise ValueError("decoded QuantState does not satisfy the NF4/DQ/BF16 contract")

    packed_copy = packed.detach().clone(memory_format=torch.contiguous_format)
    return torch, functional, packed_copy, state_parts, quant_state, metadata


def quantize_cpu(weight):
    """Quantize a frozen BF16 matrix with official NF4, DQ block 256, and uint8 storage."""
    torch, _, functional, _ = _load_backend()
    _require_tensor(weight, "weight", torch, dtype=torch.bfloat16)
    if weight.ndim != 2 or any(extent <= 0 for extent in weight.shape):
        raise ValueError("weight must be a nonempty two-dimensional matrix")
    if weight.requires_grad:
        raise ValueError("weight must be frozen with requires_grad=False")
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("weight must contain only finite values")

    packed, quant_state = functional.quantize_4bit(
        weight.detach().contiguous(),
        blocksize=_PRIMARY_BLOCKSIZE,
        compress_statistics=True,
        quant_type="nf4",
        quant_storage=torch.uint8,
    )
    parts = {
        key: value.detach().clone(memory_format=torch.contiguous_format)
        if torch.is_tensor(value)
        else value
        for key, value in quant_state.as_dict(packed=True).items()
    }
    _, _, packed_copy, parts_copy, _, _ = _validate_state(packed, parts)
    return packed_copy, parts_copy


def dequantize_cpu(packed, quant_state_parts):
    """Decode a validated NF4/DQ uint8 tensor through bitsandbytes' official API."""
    torch, functional, packed_copy, _, quant_state, _ = _validate_state(packed, quant_state_parts)
    result = functional.dequantize_4bit(packed_copy, quant_state)
    if result.device.type != "cpu" or result.dtype != torch.bfloat16:
        raise RuntimeError("bitsandbytes returned an unexpected dequantized tensor")
    return result


def linear_cpu(inputs, packed, quant_state_parts):
    """Run the frozen NF4 base linear with bitsandbytes' native input VJP."""
    torch, bitsandbytes, _, _ = _load_backend()
    _require_tensor(inputs, "inputs", torch, dtype=torch.bfloat16)
    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    validated = _validate_state(packed, quant_state_parts)
    _, _, packed_copy, _, quant_state, metadata = validated
    shape = metadata["shape"]
    if inputs.shape[-1] != shape[1]:
        raise ValueError(
            f"input feature size {inputs.shape[-1]} does not match NF4 weight size {shape[1]}"
        )

    result = bitsandbytes.matmul_4bit(inputs, packed_copy, quant_state=quant_state)
    expected_shape = tuple(inputs.shape[:-1]) + (shape[0],)
    if result.device.type != "cpu" or result.dtype != torch.bfloat16 or tuple(result.shape) != expected_shape:
        raise RuntimeError("bitsandbytes returned an unexpected linear output")
    return result


def inspect_state(packed, quant_state_parts) -> dict[str, Any]:
    """Return JSON-safe logical metadata for a validated NF4/DQ state."""
    _, _, _, _, quant_state, _ = _validate_state(packed, quant_state_parts)
    _, version, _ = _installed_backend_facts()
    return {
        "shape": [int(extent) for extent in quant_state.shape],
        "dtype": str(quant_state.dtype).removeprefix("torch."),
        "blocksize": int(quant_state.blocksize),
        "secondary_blocksize": int(quant_state.state2.blocksize),
        "quant_type": str(quant_state.quant_type),
        "nested": bool(quant_state.nested),
        "backend_version": version,
    }
