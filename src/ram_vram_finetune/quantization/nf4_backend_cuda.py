"""Pinned CUDA staging for the canonical CPU NF4/DQ records.

This module intentionally has no top-level Torch or bitsandbytes imports. The
CUDA runtime is resolved and validated only when a CUDA operation is requested.
"""
from __future__ import annotations

from .. import diagnostic_timer as diag

import hashlib
import json
import math
import os
import sys
from functools import lru_cache
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import Any
from types import MappingProxyType


_PINNED_BNB_VERSION = "0.50.2"
_PINNED_CUDA_VERSION = "13.0"
_CUDA_LIBRARY_NAME = "libbitsandbytes_cuda130.so"
_PRIMARY_BLOCKSIZE = 64
_SECONDARY_BLOCKSIZE = 256
_TRANSFER_DIGEST_SCHEMA = b"rvft.nf4_cuda_transfer.v1\0"
_TRANSFER_TENSOR_NAMES = frozenset({
    "packed",
    "absmax",
    "code",
    "offset",
    "state2.absmax",
    "state2.code",
})
_PACKED_STATE_KEY = "quant_state.bitsandbytes__nf4"
_EXPECTED_PART_KEYS = frozenset({
    "absmax",
    "nested_absmax",
    "nested_quant_map",
    "quant_map",
    _PACKED_STATE_KEY,
})
_EXPECTED_META_KEYS = frozenset({
    "quant_type",
    "blocksize",
    "dtype",
    "shape",
    "nested_blocksize",
    "nested_dtype",
    "nested_offset",
})


@diag.timed('backend.resolve_distribution')
def _resolved_package() -> tuple[Path, Path, str]:
    """Resolve the import package and its matching installed distribution."""
    spec = importlib_util.find_spec("bitsandbytes")
    if spec is None or not spec.origin or not spec.submodule_search_locations:
        raise RuntimeError("bitsandbytes is not installed as an importable package")
    package_file = Path(spec.origin).resolve()
    if package_file.name != "__init__.py":
        raise RuntimeError("bitsandbytes must resolve to a package __init__.py")
    locations = tuple(Path(item).resolve() for item in spec.submodule_search_locations)
    package_root = package_file.parent
    if locations != (package_root,):
        raise RuntimeError("bitsandbytes import root is ambiguous or inconsistent")

    loaded = sys.modules.get("bitsandbytes")
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_file is not None and Path(loaded_file).resolve() != package_file:
        raise RuntimeError("bitsandbytes is already loaded from a different package root")

    distributions = []
    for distribution in importlib_metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name != "bitsandbytes":
            continue
        installed_init = Path(distribution.locate_file("bitsandbytes/__init__.py")).resolve()
        if installed_init == package_file:
            distributions.append(distribution)
    if len(distributions) != 1:
        raise RuntimeError("bitsandbytes import root must match exactly one installed distribution")
    version = distributions[0].version
    if version != _PINNED_BNB_VERSION:
        raise RuntimeError(
            f"unsupported bitsandbytes version {version!r}; expected {_PINNED_BNB_VERSION}"
        )
    return package_root, package_file, version


def _framed_hash(files: list[tuple[Path, str]], package_root: Path, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for path, _kind in sorted(files, key=lambda item: item[1]):
        resolved = path.resolve()
        try:
            relative = ("bitsandbytes/" + resolved.relative_to(package_root).as_posix()).encode("utf-8")
        except ValueError as exc:
            raise RuntimeError("bitsandbytes implementation file escapes its package root") from exc
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(resolved.stat().st_size.to_bytes(8, "big"))
        with resolved.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _transfer_metadata(shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "shape": list(shape),
        "dtype": "bfloat16",
        "quant_type": "nf4",
        "blocksize": _PRIMARY_BLOCKSIZE,
        "nested": True,
        "state2_dtype": "float32",
        "state2_blocksize": _SECONDARY_BLOCKSIZE,
    }


def _transfer_tensor_bytes(tensor: Any, *, copy_to_cpu: bool = False) -> bytes:
    value = tensor.detach()
    if copy_to_cpu and value.device.type != "cpu":
        if diag.active_timer() is None:
            value = value.to(device='cpu', non_blocking=False)
        else:
            with diag.span("transfer.D2H.blocking",bytes=value.numel()*value.element_size()):
                with diag.device_span("D2H.stream_interval"):
                    value = value.to(device="cpu", non_blocking=False)
    if value.device.type != "cpu":
        raise ValueError("transfer digest bytes must be materialized on CPU")
    if diag.active_timer() is None:
        return value.contiguous().numpy().tobytes(order='C')
    else:
        with diag.span("transfer.cpu_materialize"):
            return value.contiguous().numpy().tobytes(order="C")


@diag.timed('integrity.digest')
def _transfer_digest(
    shape: tuple[int, int],
    tensors: dict[str, Any],
    *,
    copy_to_cpu: bool = False,
) -> str:
    """Hash canonical QuantState metadata and actual transfer tensor bytes."""
    if set(tensors) != _TRANSFER_TENSOR_NAMES:
        raise ValueError("transfer digest requires the exact canonical NF4/DQ tensor inventory")
    digest = hashlib.sha256()
    digest.update(_TRANSFER_DIGEST_SCHEMA)
    metadata = _canonical_json_bytes(_transfer_metadata(shape))
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    for name in sorted(tensors):
        tensor = tensors[name]
        descriptor = _canonical_json_bytes({
            "name": name,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        })
        raw = _transfer_tensor_bytes(tensor, copy_to_cpu=copy_to_cpu)
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        if diag.active_timer() is None:
            digest.update(raw)
        else:
            with diag.span("integrity.sha256_payload",bytes=len(raw)):
                digest.update(raw)
    return digest.hexdigest()


def _installed_cuda_facts(*, resolved=None) -> tuple[Path, str, str, str, str]:
    # Per-job fresh hashing; no cross-job facts cache.
    package_root, _, version = _resolved_package() if resolved is None else resolved
    python_sources = [path for path in package_root.rglob("*.py") if path.is_file()]
    if not python_sources:
        raise RuntimeError("installed bitsandbytes Python implementation is missing")
    cuda_library = package_root / _CUDA_LIBRARY_NAME
    if not cuda_library.is_file():
        raise RuntimeError(f"required official CUDA 13.0 library is missing: {_CUDA_LIBRARY_NAME}")
    python_files = [(path, path.relative_to(package_root).as_posix()) for path in python_sources]
    combined_files = [*python_files, (cuda_library, cuda_library.relative_to(package_root).as_posix())]
    combined_digest = _framed_hash(combined_files, package_root, b"rvft.bnb.cuda.source.v1\0")
    python_digest = _framed_hash(python_files, package_root, b"rvft.bnb.python.source.v1\0")
    library_digest = hashlib.sha256()
    with cuda_library.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            library_digest.update(chunk)
    return package_root, version, combined_digest, python_digest, library_digest.hexdigest()


class CUDAStageError(RuntimeError):
    """Failed H2D stage that keeps all possibly in-flight resources alive."""

    def __init__(
        self,
        message: str,
        *,
        pinned_owner: "PinnedNF4Record",
        stream: Any,
        retained_owners: tuple[Any, ...] | list[Any] = (),
        copies_scheduled: bool = False,
        ready: Any = None,
        start_event: Any = None,
        end_event: Any = None,
    ) -> None:
        super().__init__(message)
        self.pinned_owner = pinned_owner
        self.stream = stream
        owners = [pinned_owner, *retained_owners]
        seen: set[int] = set()
        self.retained_owners = tuple(
            owner for owner in owners
            if owner is not None and not (id(owner) in seen or seen.add(id(owner)))
        )
        self.copies_scheduled = bool(copies_scheduled)
        self.ready = ready
        self.start_event = start_event
        self.end_event = end_event


class RunBackend:
    """Job-owned capability; full discovery happens exactly once at construction.

    No global backend cache. Identity sentinels are checked only before begin.
    Captured runtime callables are owned by sealed records; training never
    borrows or revalidates this capability. Runtime package installation and private/raw-pointer
    monkeypatching are outside this cooperative-process ownership boundary.
    """
    def __init__(self):
        self._value = _load_cuda_backend()
        torch, bnb, functional, unpack, root, facts = self._value
        self._pid = os.getpid()
        self._phase = 'preflight'
        self._used = False
        self._sentinels = self._identity()
        self.matmul_4bit = bnb.matmul_4bit
        self.QuantState = functional.QuantState
        self._runtime = (torch, self.matmul_4bit, self.QuantState)
        self._matmul_code = getattr(self.matmul_4bit, '__code__', None)
        self.binding = MappingProxyType({'schema':'rvft.nf4.backend_run.v1',
            'version':str(bnb.__version__), 'root':str(root),
            'source_digest':facts[2], 'python_source_digest':facts[3],
            'cuda_library_sha256':facts[4], 'torch_version':str(torch.__version__),
            'cuda_version':str(torch.version.cuda),
            'integrity_policy':'startup_full_no_training_content_checks_v1'})

    def _identity(self):
        torch,bnb,functional,unpack,root,facts=self._value
        names=('torch','bitsandbytes','bitsandbytes.functional','bitsandbytes.cextension','bitsandbytes.utils')
        return (tuple(sys.path),tuple(id(x)for x in sys.meta_path),
            tuple((n,id(sys.modules.get(n)))for n in names),
            getattr(bnb,'__version__',None),getattr(bnb,'__file__',None),
            getattr(torch,'__version__',None),getattr(torch.version,'cuda',None),getattr(torch.version,'hip',None),
            os.environ.get('BNB_CUDA_VERSION'),os.environ.get('BNB_ROCM_VERSION'))

    def borrow(self):
        if self._phase=='closed':raise RuntimeError('backend capability closed; new job preflight required')
        if self._phase not in ('preflight','ready'):
            raise RuntimeError('backend borrow is startup-only; runtime uses captured callables')
        if os.getpid()!=self._pid:
            raise RuntimeError('backend capability closed or belongs to another process')
        if self._identity()!=self._sentinels:
            raise RuntimeError('backend environment/version changed; new job preflight required')
        if self._value[1].matmul_4bit is not self.matmul_4bit or self._value[2].QuantState is not self.QuantState or getattr(self.matmul_4bit,'__code__',None)is not self._matmul_code:
            raise RuntimeError('bound backend callable replaced; new job preflight required')
        return self._value

    def require_preflight(self):
        if self._phase!='preflight':raise RuntimeError('content verification forbidden outside startup preflight')
        self.borrow()

    def ready(self):
        self.require_preflight();self._phase='ready'

    def begin(self):
        self.borrow()
        if self._phase!='ready' or self._used:raise RuntimeError('backend has no unused completed preflight')
        self._used=True;self._phase='computing'

    def finish(self):
        if self._phase=='computing':self._phase='completed'

    def close(self):self._phase='closed'


class PinnedNF4Record:
    """Pinned CPU copies of one canonical NF4/DQ record and its real QuantState."""

    __slots__ = (
        "_source_record", "_packed", "_absmax", "_quant_map", "_nested_absmax",
        "_nested_quant_map", "_offset", "_metadata", "_shape", "_source_fingerprint",
        "_transfer_digest", "_nbytes", "_pinned_bytes", "_dtype", "_quant_type", "_blocksize",
        "_nested_blocksize", "_nested_dtype", "_backend", "_runtime", "_diagnostic", "_sealed",
    )

    def __setattr__(self,name,value):
        if getattr(self,"_sealed",False):raise AttributeError("owned pinned record is sealed")
        object.__setattr__(self,name,value)

    def __init__(self, record: Any, *, backend_run=None) -> None:
        from .nf4_state import NF4Record

        if not isinstance(record, NF4Record):
            raise TypeError("record must be a canonical NF4Record")
        self._backend = backend_run
        if type(self._backend)is not RunBackend:raise TypeError("exact RunBackend required")
        self._backend.require_preflight()
        torch, _bnb, functional, unpack_tensor_to_dict, _root, _facts = self._backend.borrow()
        self._runtime = self._backend._runtime
        self._diagnostic = diag.active_timer() is not None
        packed, parts = record.export_parts()
        if not isinstance(packed, torch.Tensor) or packed.device.type != "cpu":
            raise ValueError("canonical packed storage must be a CPU tensor")
        if packed.dtype is not torch.uint8 or packed.ndim != 2 or packed.shape[1] != 1:
            raise ValueError("canonical packed NF4 storage must use uint8 [bytes, 1] layout")
        if set(parts) != _EXPECTED_PART_KEYS:
            raise ValueError("canonical NF4 state parts have an unexpected inventory")
        if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in parts.values()):
            raise ValueError("canonical NF4 state parts must all be CPU tensors")
        packed_metadata = parts[_PACKED_STATE_KEY]
        if packed_metadata.dtype is not torch.uint8 or packed_metadata.ndim != 1 or packed_metadata.numel() == 0:
            raise ValueError("packed NF4 metadata must be nonempty one-dimensional uint8")
        try:
            metadata = unpack_tensor_to_dict(packed_metadata)
        except Exception as exc:
            raise ValueError("canonical NF4 packed metadata is malformed") from exc
        if not isinstance(metadata, dict) or set(metadata) != _EXPECTED_META_KEYS:
            raise ValueError("canonical NF4 metadata has missing or unknown fields")
        shape_value = metadata["shape"]
        if (
            metadata["quant_type"] != "nf4"
            or type(metadata["blocksize"]) is not int
            or metadata["blocksize"] != _PRIMARY_BLOCKSIZE
            or type(metadata["nested_blocksize"]) is not int
            or metadata["nested_blocksize"] != _SECONDARY_BLOCKSIZE
            or metadata["dtype"] != "bfloat16"
            or metadata["nested_dtype"] != "float32"
            or not isinstance(shape_value, (list, tuple))
            or len(shape_value) != 2
            or any(type(extent) is not int or extent <= 0 for extent in shape_value)
        ):
            raise ValueError("canonical record is not official NF4/DQ/BF16")
        nested_offset = metadata["nested_offset"]
        if type(nested_offset) not in (int, float) or not math.isfinite(float(nested_offset)) or nested_offset < 0:
            raise ValueError("nested DQ offset must be finite and non-negative")
        shape = tuple(shape_value)
        if tuple(record.shape) != shape:
            raise ValueError("record shape and official QuantState shape disagree")
        elements = shape[0] * shape[1]
        primary_count = (elements + _PRIMARY_BLOCKSIZE - 1) // _PRIMARY_BLOCKSIZE
        secondary_count = (primary_count + _SECONDARY_BLOCKSIZE - 1) // _SECONDARY_BLOCKSIZE
        if packed.numel() != (elements + 1) // 2:
            raise ValueError("packed NF4 byte count does not match its shape")
        if parts["absmax"].dtype is not torch.uint8 or parts["absmax"].ndim != 1 or parts["absmax"].numel() != primary_count:
            raise ValueError("primary NF4 absmax tensor has an invalid shape or dtype")
        if parts["nested_absmax"].dtype is not torch.float32 or parts["nested_absmax"].ndim != 1 or parts["nested_absmax"].numel() != secondary_count:
            raise ValueError("secondary DQ absmax tensor has an invalid shape or dtype")
        if parts["quant_map"].dtype is not torch.float32 or parts["quant_map"].ndim != 1 or parts["quant_map"].numel() != 16:
            raise ValueError("NF4 codebook has an invalid shape or dtype")
        if parts["nested_quant_map"].dtype is not torch.float32 or parts["nested_quant_map"].ndim != 1 or parts["nested_quant_map"].numel() != 256:
            raise ValueError("secondary DQ codebook has an invalid shape or dtype")

        try:
            quant_state = functional.QuantState.from_dict(dict(parts), device=torch.device("cpu"))
        except Exception as exc:
            raise ValueError("bitsandbytes rejected the canonical CPU NF4/DQ state") from exc
        if (
            quant_state.quant_type != "nf4"
            or quant_state.blocksize != _PRIMARY_BLOCKSIZE
            or quant_state.dtype != torch.bfloat16
            or tuple(quant_state.shape) != shape
            or not quant_state.nested
            or quant_state.state2.blocksize != _SECONDARY_BLOCKSIZE
            or quant_state.state2.dtype != torch.float32
            or quant_state.offset.device.type != "cpu"
            or quant_state.offset.dtype != torch.float32
            or quant_state.offset.numel() != 1
            or float(quant_state.offset.item()) != float(nested_offset)
        ):
            raise ValueError("decoded official QuantState does not match the canonical record")
        if not bool(torch.isfinite(parts["nested_absmax"]).all().item()) or not bool((parts["nested_absmax"] >= 0).all().item()):
            raise ValueError("nested DQ absmax must be finite and non-negative")

        transfer_values = {
            "packed": packed,
            "absmax": parts["absmax"],
            "quant_map": parts["quant_map"],
            "nested_absmax": parts["nested_absmax"],
            "nested_quant_map": parts["nested_quant_map"],
            "offset": quant_state.offset,
        }
        pinned: dict[str, Any] = {}
        for name, value in transfer_values.items():
            try:
                value_pinned = value.detach().contiguous().pin_memory()
            except Exception as exc:
                raise RuntimeError(f"failed to pin NF4 {name} state for asynchronous CUDA transfer") from exc
            if value_pinned.device.type != "cpu" or not value_pinned.is_pinned():
                raise RuntimeError(f"NF4 {name} state did not receive pinned CPU storage")
            pinned[name] = value_pinned

        self._source_record = record
        self._packed = pinned["packed"]
        self._absmax = pinned["absmax"]
        self._quant_map = pinned["quant_map"]
        self._nested_absmax = pinned["nested_absmax"]
        self._nested_quant_map = pinned["nested_quant_map"]
        self._offset = pinned["offset"]
        self._metadata = {
            "quant_type": "nf4",
            "blocksize": _PRIMARY_BLOCKSIZE,
            "dtype": "bfloat16",
            "shape": list(shape),
            "nested_blocksize": _SECONDARY_BLOCKSIZE,
            "nested_dtype": "float32",
            "nested_offset": float(nested_offset),
        }
        self._shape = shape
        self._source_fingerprint = record.fingerprint()
        self._transfer_digest = _transfer_digest(shape, {
            "packed": self._packed,
            "absmax": self._absmax,
            "code": self._quant_map,
            "offset": self._offset,
            "state2.absmax": self._nested_absmax,
            "state2.code": self._nested_quant_map,
        })
        self._nbytes = sum(value.numel() * value.element_size() for value in pinned.values())
        self._pinned_bytes = self._nbytes
        self._dtype = torch.bfloat16
        self._quant_type = "nf4"
        self._blocksize = _PRIMARY_BLOCKSIZE
        self._nested_blocksize = _SECONDARY_BLOCKSIZE
        self._nested_dtype = torch.float32
        self._sealed = True

    @property
    def packed(self):
        return self._packed.clone()

    @property
    def absmax(self):
        return self._absmax.clone()

    @property
    def quant_map(self):
        return self._quant_map.clone()

    @property
    def nested_absmax(self):
        return self._nested_absmax.clone()

    @property
    def nested_quant_map(self):
        return self._nested_quant_map.clone()

    @property
    def offset(self):
        return self._offset.clone()

    @property
    def metadata(self) -> dict[str, Any]:
        import copy
        return copy.deepcopy(self._metadata)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def source_fingerprint(self) -> str:
        return self._source_fingerprint

    @property
    def transfer_digest(self) -> str:
        """Canonical digest of all pinned H2D bytes and QuantState metadata."""
        return self._transfer_digest

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def pinned_bytes(self) -> int:
        return self._pinned_bytes

    @property
    def dtype(self):
        return self._dtype

    @property
    def quant_type(self) -> str:
        return self._quant_type

    @property
    def blocksize(self) -> int:
        return self._blocksize

    @property
    def nested_blocksize(self) -> int:
        return self._nested_blocksize

    @property
    def nested_dtype(self):
        return self._nested_dtype


class CUDARecord:
    """One staged CUDA NF4 state retaining every H2D source and destination."""

    __slots__ = (
        "_packed", "_absmax", "_quant_map", "_nested_absmax", "_nested_quant_map", "_offset",
        "_quant_state", "ready", "shape", "source_fingerprint", "nbytes", "pinned_owner",
        "transfer_digest", "_start_event", "_end_event", "_metadata", "_backend", "_runtime", "_sealed",
    )

    def __setattr__(self,name,value):
        if getattr(self,"_sealed",False):raise AttributeError("owned CUDA record is sealed")
        object.__setattr__(self,name,value)

    def __init__(
        self,
        *,
        packed: Any,
        absmax: Any,
        quant_map: Any,
        nested_absmax: Any,
        nested_quant_map: Any,
        offset: Any,
        quant_state: Any,
        ready: Any,
        shape: tuple[int, int],
        source_fingerprint: str,
        transfer_digest: str,
        nbytes: int,
        pinned_owner: PinnedNF4Record,
        start_event: Any,
        end_event: Any,
        metadata: dict[str, Any],
    ) -> None:
        self._packed = packed
        self._absmax = absmax
        self._quant_map = quant_map
        self._nested_absmax = nested_absmax
        self._nested_quant_map = nested_quant_map
        self._offset = offset
        self._quant_state = quant_state
        self.ready = ready
        self.shape = shape
        self.source_fingerprint = source_fingerprint
        self.transfer_digest = transfer_digest
        self.nbytes = nbytes
        self.pinned_owner = pinned_owner
        self._start_event = start_event
        self._end_event = end_event
        self._metadata = dict(metadata)
        self._backend = pinned_owner._backend
        self._runtime = pinned_owner._runtime
        self._sealed = True

    @property
    def metadata(self) -> dict[str, Any]:
        import copy
        return copy.deepcopy(self._metadata)

    def wait_on(self, consumer_stream: Any) -> None:
        torch = self._runtime[0]
        _validate_stream(torch, consumer_stream)
        self.ready.wait(consumer_stream)

    def synchronize_ready(self) -> None:
        self.ready.synchronize()

    @property
    def timing_available(self):return self._start_event is not None

    def h2d_elapsed_ms(self) -> float:
        if self._start_event is None:return 0.0
        # Consumer completion already dominates the copy-stream ready event.
        # No extra host wait for optional diagnostic timing.
        return float(self._start_event.elapsed_time(self._end_event))

    def verify_source(self) -> str:
        """Verify staged CUDA bytes still equal their pinned canonical source."""
        self._backend.require_preflight()
        torch, _bnb, _functional, _unpack, _root, _facts = self._backend.borrow()
        self.ready.synchronize()
        _validate_cuda_record(torch, self)
        actual = _transfer_digest(self.shape, {
            "packed": self._packed,
            "absmax": self._absmax,
            "code": self._quant_map,
            "offset": self._offset,
            "state2.absmax": self._nested_absmax,
            "state2.code": self._nested_quant_map,
        }, copy_to_cpu=True)
        if actual != self.transfer_digest or actual != self.pinned_owner.transfer_digest:
            raise RuntimeError("staged CUDA NF4/DQ bytes do not match the pinned transfer digest")
        return actual


    @property
    def packed(self):
        """Defensive export; never a borrowed runtime tensor."""
        return self._packed.clone()

    @property
    def absmax(self):
        """Defensive export; never a borrowed runtime tensor."""
        return self._absmax.clone()

    @property
    def quant_map(self):
        """Defensive export; never a borrowed runtime tensor."""
        return self._quant_map.clone()

    @property
    def nested_absmax(self):
        """Defensive export; never a borrowed runtime tensor."""
        return self._nested_absmax.clone()

    @property
    def nested_quant_map(self):
        """Defensive export; never a borrowed runtime tensor."""
        return self._nested_quant_map.clone()

    @property
    def offset(self):
        """Defensive export; never a borrowed runtime tensor."""
        return self._offset.clone()


@diag.timed('backend.load_context')
def _load_cuda_backend():
    """Load only official BNB 0.50.2 with the CUDA 13.0 native library."""
    package_root, package_file, version = _resolved_package()
    facts = _installed_cuda_facts(resolved=(package_root,package_file,version))
    if facts[0] != package_root or facts[1] != version:
        raise RuntimeError("bitsandbytes package identity changed during CUDA backend resolution")
    if diag.active_timer() is None:
        registered = importlib_metadata.entry_points(group='bitsandbytes.backends')
    else:
        with diag.span("backend.entry_points"):
            registered = importlib_metadata.entry_points(group="bitsandbytes.backends")
    if registered:
        raise RuntimeError("refusing external bitsandbytes backend extensions")

    cuda_override = os.environ.get("BNB_CUDA_VERSION")
    rocm_override = os.environ.get("BNB_ROCM_VERSION")
    if cuda_override not in (None, "130") or rocm_override is not None:
        raise RuntimeError("bitsandbytes CUDA/ROCm version overrides are unsupported")

    import torch

    if getattr(torch.version, "cuda", None) != _PINNED_CUDA_VERSION or getattr(torch.version, "hip", None):
        raise RuntimeError(
            f"CUDA backend requires Torch CUDA {_PINNED_CUDA_VERSION}; found CUDA={getattr(torch.version, 'cuda', None)!r}, HIP={getattr(torch.version, 'hip', None)!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA backend requires an available CUDA device; CPU fallback is forbidden")

    import bitsandbytes
    import bitsandbytes.functional as functional
    import bitsandbytes.cextension as cextension
    from bitsandbytes.utils import unpack_tensor_to_dict

    if Path(bitsandbytes.__file__).resolve() != package_file:
        raise RuntimeError("bitsandbytes imported from a root different from its verified distribution")
    if getattr(bitsandbytes, "__version__", None) != _PINNED_BNB_VERSION:
        raise RuntimeError("loaded bitsandbytes version is not the pinned official version")
    native = getattr(cextension, "lib", None)
    native_path = getattr(getattr(native, "_lib", None), "_name", None)
    if not getattr(native, "compiled_with_cuda", False) or not native_path:
        raise RuntimeError("bitsandbytes loaded its CPU fallback instead of a CUDA library")
    expected_library = (package_root / _CUDA_LIBRARY_NAME).resolve()
    if Path(native_path).resolve() != expected_library:
        raise RuntimeError("bitsandbytes did not load the exact CUDA 13.0 native library")
    return torch, bitsandbytes, functional, unpack_tensor_to_dict, package_root, facts


def backend_info_cuda(*, backend_run=None) -> dict[str, Any]:
    """Return the verified CUDA runtime identity after the caller launches the job."""
    run=RunBackend()if backend_run is None else backend_run
    torch, _bnb, _functional, _unpack, _root, facts = run.borrow()
    device_name = torch.cuda.get_device_name(0)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    return {
        "version": _PINNED_BNB_VERSION,
        "runtime": "cuda",
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_library": _CUDA_LIBRARY_NAME,
        "cuda_library_sha256": facts[4],
        "python_source_digest": facts[3],
        "source_digest": facts[2],
        "device": "cuda:0",
        "device_name": device_name,
        "compute_capability": list(capability),
        "cpu_only": False,
    }


def _explicit_cuda0(torch, device: Any):
    try:
        resolved = torch.device(device)
    except Exception as exc:
        raise ValueError("device must be explicitly cuda:0") from exc
    if resolved.type != "cuda" or resolved.index != 0:
        raise ValueError("only explicit device cuda:0 is supported")
    return resolved


def _validate_stream(torch, stream: Any) -> None:
    if not isinstance(stream, torch.cuda.Stream):
        raise TypeError("an explicit torch.cuda.Stream is required")
    stream_device = stream.device
    if stream_device.type != "cuda" or stream_device.index != 0:
        raise ValueError("CUDA stream must belong to cuda:0")


def _validate_pinned(pinned: Any) -> None:
    if type(pinned) is not PinnedNF4Record:
        raise TypeError("pinned must be an exact PinnedNF4Record")
    if pinned.pinned_bytes != pinned.nbytes or pinned.nbytes <= 0:
        raise ValueError("pinned NF4 state has invalid transfer accounting")
    if any(not value.is_pinned() or value.device.type != "cpu" for value in (
        pinned._packed, pinned._absmax, pinned._quant_map, pinned._nested_absmax,
        pinned._nested_quant_map, pinned._offset,
    )):
        raise ValueError("all NF4 transfer components must remain in pinned CPU storage")


@diag.timed('transfer.stage_cuda')
def stage_cuda(pinned: PinnedNF4Record, *, device: Any, stream: Any) -> CUDARecord:
    """Copy all official packed NF4/DQ components on the supplied copy stream."""
    if type(pinned)is not PinnedNF4Record:raise TypeError("exact owned pinned record required")
    torch, _matmul, QuantState = pinned._runtime
    # All content/layout validation occurred during owned pinned construction.
    resolved_device = _explicit_cuda0(torch, device)
    _validate_stream(torch, stream)
    if stream.device != resolved_device:
        raise ValueError("copy stream and target device must match exactly")

    start_event = None
    end_event = None
    ready = None
    owners: list[Any] = []
    copied: dict[str, Any] = {}
    copies_scheduled = False
    try:
        if pinned._diagnostic:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        ready = torch.cuda.Event(enable_timing=False)
        owners.extend((start_event, end_event, ready))
        with torch.cuda.stream(stream):
            if start_event is not None:start_event.record(stream)
            if not pinned._diagnostic:
                for name in ('packed', 'absmax', 'quant_map', 'nested_absmax', 'nested_quant_map', 'offset'):
                    destination = getattr(pinned, "_"+name).to(device=resolved_device, non_blocking=True)
                    copied[name] = destination
                    owners.append(destination)
                    copies_scheduled = True
            else:
                with diag.span("transfer.H2D.submit",bytes=pinned.nbytes), diag.device_span("H2D.stream_interval",stream):
                    for name in (
                        "packed", "absmax", "quant_map", "nested_absmax", "nested_quant_map", "offset",
                    ):
                        destination = getattr(pinned, "_"+name).to(device=resolved_device, non_blocking=True)
                        copied[name] = destination
                        owners.append(destination)
                        copies_scheduled = True
            if end_event is not None:end_event.record(stream)
            ready.record(stream)

            state2 = QuantState(
                absmax=copied["nested_absmax"],
                blocksize=_SECONDARY_BLOCKSIZE,
                code=copied["nested_quant_map"],
                dtype=torch.float32,
            )
            quant_state = QuantState(
                absmax=copied["absmax"],
                shape=torch.Size(pinned.shape),
                code=copied["quant_map"],
                blocksize=_PRIMARY_BLOCKSIZE,
                quant_type="nf4",
                dtype=torch.bfloat16,
                offset=copied["offset"],
                state2=state2,
            )
            owners.extend((state2, quant_state))
        if quant_state.offset is not copied["offset"] or quant_state.state2.absmax is not copied["nested_absmax"]:
            raise RuntimeError("official CUDA QuantState did not retain staged NF4/DQ tensors")
        return CUDARecord(
            packed=copied["packed"],
            absmax=copied["absmax"],
            quant_map=copied["quant_map"],
            nested_absmax=copied["nested_absmax"],
            nested_quant_map=copied["nested_quant_map"],
            offset=copied["offset"],
            quant_state=quant_state,
            ready=ready,
            shape=pinned.shape,
            source_fingerprint=pinned.source_fingerprint,
            transfer_digest=pinned.transfer_digest,
            nbytes=pinned.nbytes,
            pinned_owner=pinned,
            start_event=start_event,
            end_event=end_event,
            metadata=pinned.metadata,
        )
    except Exception as exc:
        raise CUDAStageError(
            "failed to stage official NF4/DQ state on CUDA",
            pinned_owner=pinned,
            stream=stream,
            retained_owners=owners,
            copies_scheduled=copies_scheduled,
            ready=ready,
            start_event=start_event,
            end_event=end_event,
        ) from exc


def _validate_cuda_record(torch, state: Any) -> None:
    if type(state) is not CUDARecord:
        raise TypeError("state must be an exact CUDARecord")
    components = (
        state._packed, state._absmax, state._quant_map, state._nested_absmax,
        state._nested_quant_map, state._offset,
    )
    if any(value.device.type != "cuda" or value.device.index != 0 for value in components):
        raise ValueError("all NF4/DQ state tensors must be on cuda:0")
    if (
        state._packed.dtype is not torch.uint8
        or state._packed.ndim != 2
        or state._packed.shape[1] != 1
        or state._absmax.dtype is not torch.uint8
        or state._absmax.ndim != 1
        or state._quant_map.dtype is not torch.float32
        or state._quant_map.ndim != 1
        or state._quant_map.numel() != 16
        or state._nested_absmax.dtype is not torch.float32
        or state._nested_absmax.ndim != 1
        or state._nested_quant_map.dtype is not torch.float32
        or state._nested_quant_map.ndim != 1
        or state._nested_quant_map.numel() != 256
        or state._offset.dtype is not torch.float32
        or state._offset.numel() != 1
    ):
        raise ValueError("CUDA NF4/DQ state tensors have an invalid layout")
    quant_state = state._quant_state
    if (
        quant_state.quant_type != "nf4"
        or quant_state.blocksize != _PRIMARY_BLOCKSIZE
        or quant_state.dtype != torch.bfloat16
        or tuple(quant_state.shape) != state.shape
        or not quant_state.nested
        or quant_state.absmax is not state._absmax
        or quant_state.code is not state._quant_map
        or quant_state.offset is not state._offset
        or quant_state.state2.blocksize != _SECONDARY_BLOCKSIZE
        or quant_state.state2.dtype != torch.float32
        or quant_state.state2.absmax is not state._nested_absmax
        or quant_state.state2.code is not state._nested_quant_map
    ):
        raise ValueError("CUDA QuantState does not exactly match its staged NF4/DQ tensors")
    elements = state.shape[0] * state.shape[1]
    if state._packed.numel() != (elements + 1) // 2:
        raise ValueError("packed CUDA byte count does not match the NF4 state shape")


@diag.timed('projection.NF4_linear')
def linear_cuda(inputs: Any, state: CUDARecord):
    """Run BNB's official CUDA 4-bit base linear without cloning or dequantizing."""
    if type(state)is not CUDARecord:raise TypeError("exact owned CUDA record required")
    torch, matmul_4bit, _QuantState = state._runtime
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if inputs.device.type != "cuda" or inputs.device.index != 0:
        raise ValueError("inputs must be on cuda:0")
    if inputs.dtype is not torch.bfloat16:
        raise TypeError("inputs must have dtype torch.bfloat16")
    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    # No content integrity/layout revalidation in training. State is owned and
    # constructed only from the preflight-validated pinned source.
    if inputs.shape[-1] != state.shape[1]:
        raise ValueError(
            f"input feature size {inputs.shape[-1]} does not match NF4 weight size {state.shape[1]}"
        )
    result = matmul_4bit(inputs, state._packed, quant_state=state._quant_state)
    expected_shape = tuple(inputs.shape[:-1]) + (state.shape[0],)
    if (
        not isinstance(result, torch.Tensor)
        or result.device.type != "cuda"
        or result.device.index != 0
        or result.dtype is not torch.bfloat16
        or tuple(result.shape) != expected_shape
    ):
        raise RuntimeError("bitsandbytes returned an unexpected CUDA NF4 linear output")
    return result


__all__ = (
    "CUDARecord",
    "CUDAStageError",
    "PinnedNF4Record",
    "backend_info_cuda",
    "linear_cuda",
    "stage_cuda",
)
