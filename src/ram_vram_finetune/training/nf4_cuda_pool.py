"""Leased, one-block-at-a-time CUDA staging for canonical NF4/DQ records.

The real backend is loaded lazily through the default ops object. Tests can
supply fake ops and exercise lifecycle, poisoning, and recovery without making
any physical CUDA call.
"""
from __future__ import annotations

from .. import diagnostic_timer as diag

import math
import re
import threading
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..quantization.nf4_state import NF4Record


_VALID_PHASES = frozenset(("forward", "backward"))
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class _DefaultCudaOps:
    """Thin lazy adapter around the audited CUDA backend and Torch runtime."""

    def __init__(self, backend_run=None) -> None:
        from ..quantization.nf4_backend_cuda import RunBackend,PinnedNF4Record,stage_cuda
        if type(backend_run)is not RunBackend:raise RuntimeError('explicit preflight RunBackend required; implicit per-record resolution forbidden')
        backend_run.require_preflight()
        import torch

        self._backend_run = backend_run
        self._torch = torch
        self._pinned_type = PinnedNF4Record
        self._stage = diag.bind(stage_cuda)
        self._device = torch.device("cuda:0")

    def pin_record(self, record: NF4Record) -> Any:
        return self._pinned_type(record, backend_run=self._backend_run)

    def new_stream(self, device: Any) -> Any:
        return self._torch.cuda.Stream(device=device)

    def current_stream(self, device: Any) -> Any:
        return self._torch.cuda.current_stream(device=device)

    def stage_cuda(self, pinned: Any, *, device: Any, stream: Any) -> Any:
        return self._stage(pinned, device=device, stream=stream)

    def record_event(self, stream: Any) -> Any:
        event = self._torch.cuda.Event(enable_timing=False)
        event.record(stream)
        return event

    @staticmethod
    def synchronize_event(event: Any) -> None:
        event.synchronize()

    @staticmethod
    def synchronize_stream(stream: Any) -> None:
        stream.synchronize()


class CUDANF4Lease:
    """One generation of staged CUDA states retained through its stream barrier."""

    __slots__ = (
        "_pool", "generation", "phase", "_parts", "_owners", "_consumer_stream",
        "_staged_bytes", "_needs_verify_on_recovery",
    )

    def __init__(self, pool: "CUDANF4LeasePool", generation: int, phase: str) -> None:
        self._pool = pool
        self.generation = generation
        self.phase = phase
        self._parts: dict[str, Any] | None = {}
        self._owners: list[Any] = []
        self._consumer_stream: Any = None
        self._staged_bytes = 0
        self._needs_verify_on_recovery = False

    @property
    def is_active(self) -> bool:
        return self._pool._lease_is_active(self)

    @property
    def staged_bytes(self) -> int:
        return self._staged_bytes

    def parts(self, record_name: str) -> Any:
        return self._pool._lease_parts(self, record_name)

    def complete(self) -> None:
        self._pool._finish(self, "complete")

    def abort(self) -> None:
        self._pool._finish(self, "abort")

    def _invalidate(self) -> None:
        self._parts = None
        self._owners.clear()
        self._consumer_stream = None


class CUDANF4LeasePool:
    """Single-slot CUDA staging pool with explicit stream barriers and recovery.

    ``ops`` is an injectable protocol with ``pin_record``, ``new_stream``,
    ``current_stream``, ``stage_cuda``, ``record_event``,
    ``synchronize_event``, and ``synchronize_stream`` methods. The default
    implementation lazily delegates to the pinned CUDA NF4 backend.

    An owner may supply one copy_stream shared by its sequential decoder pools.
    Standalone pools otherwise create their own stream; all barriers are unchanged.

    Startup preflight checks each record once. Training never verifies payload
    content. Same-copy-stream last-ready dependency and one consumer completion
    barrier retain owned buffers safely. Runtime owner/generation/accounting
    checks are process-local memory safety, not payload correctness checks.
    """

    def __init__(
        self,
        records: Mapping[str, NF4Record],
        *,
        device: Any = "cuda:0",
        ops: Any = None,
        backend_run: Any = None,
        preflight_only: bool = True,
        copy_stream: Any = None,
    ) -> None:
        copied = self._copy_records(records)
        if type(device) is not str or device != "cuda:0":
            raise ValueError("CUDANF4LeasePool supports only explicit device cuda:0")
        self._device = device
        if preflight_only is not True:raise ValueError("R41 permits only startup preflight; legacy per-lease checks retired")
        self._preflight_only=preflight_only
        self._diagnostic=diag.active_timer() is not None
        self._preflight_ready=False
        self._in_preflight=False
        self._compute_started=False
        self._compute_used=False
        self._ops = _DefaultCudaOps(backend_run) if ops is None else ops
        self._check_ops(self._ops)
        self._records = MappingProxyType(copied)
        self._names = tuple(copied)
        self._lock = threading.RLock()
        self._generation = 0
        self._active_lease: CUDANF4Lease | None = None
        self._acquiring = False
        self._finishing = False
        self._poisoned = False
        self._retained_resources: tuple[Any, ...] = ()
        self._recovery_consumer_stream: Any = None
        self._verify_during_recovery = False
        self._pinned: dict[str, Any] = {}
        self._pinned_digests: dict[str, str] = {}
        self._verified_digests: dict[str, str] = {}
        self._counts = {
            "acquires": 0,
            "completes": 0,
            "aborts": 0,
            "recoveries": 0,
        }
        self._h2d_bytes = 0
        self._h2d_ms = 0.0
        self._h2d_timing_samples = 0
        self._validation_wall_seconds = 0.0
        self._validation_attempts = 0
        self._current_staged_bytes = 0
        self._peak_staged_bytes = 0
        self._events: list[dict[str, Any]] = []

        # Pin once and retain the CPU cache for the lifetime of this pool.
        try:
            for name, record in copied.items():
                pinned = self._ops.pin_record(record)
                nbytes = self._nonnegative_int(getattr(pinned, "nbytes", None), f"{name}.nbytes")
                pinned_bytes = self._nonnegative_int(
                    getattr(pinned, "pinned_bytes", None), f"{name}.pinned_bytes"
                )
                if nbytes <= 0 or pinned_bytes < nbytes:
                    raise ValueError(f"pinned NF4 record {name!r} has invalid byte accounting")
                digest = getattr(pinned, "transfer_digest", None)
                if type(digest) is not str or _DIGEST_RE.fullmatch(digest) is None:
                    raise ValueError(f"pinned NF4 record {name!r} has an invalid transfer digest")
                source_fingerprint = getattr(pinned, "source_fingerprint", None)
                if source_fingerprint != record.fingerprint():
                    raise ValueError(f"pinned NF4 record {name!r} has a different canonical source")
                self._pinned[name] = pinned
                self._pinned_digests[name] = digest
        except BaseException:
            # Pinning schedules no device copies. Local pinned owners can be
            # released safely if construction itself fails.
            self._pinned.clear()
            self._pinned_digests.clear()
            raise

        try:
            self._copy_stream = self._ops.new_stream(self._device) if copy_stream is None else copy_stream
        except BaseException:
            self._pinned.clear()
            self._pinned_digests.clear()
            raise
        self._pinned_bytes = sum(
            int(getattr(value, "pinned_bytes")) for value in self._pinned.values()
        )

        # Select bare OFF bodies once; no per-lease timer wrapper/tag/clock.
        for name in ('acquire','_finish','_barrier','_release_lease_locked'):
            setattr(self,name,diag.bind(getattr(self,name)))

    def assert_idle(self) -> None:
        """Minimal local ownership state, without statistics or device sync."""
        with self._lock:
            if self._poisoned or self._active_lease is not None or self._acquiring or self._finishing:
                raise RuntimeError('cannot update/checkpoint with active or poisoned CUDA lease')

    def preflight(self) -> None:
        """One full transfer check before training, never a hot-path fallback."""
        with self._lock:
            if not self._preflight_only or self._preflight_ready or self._compute_used or self._in_preflight:
                raise RuntimeError('pool preflight is absent, reused or already running')
            self._in_preflight=True
        try:
            lease=self.acquire('forward')
            lease.complete()
            if set(self._verified_digests)!=set(self._names):raise RuntimeError('incomplete pool preflight')
            self._preflight_ready=True
        except BaseException as error:
            # Construction must not drop a poisoned in-flight generation.
            if self._poisoned:
                try:self.recover_after_barrier_failure()
                except BaseException as cleanup_error:
                    error.retained_pool=self
                    error.add_note('startup cleanup failed; retained_pool owns all buffers; worker must abort: '+repr(cleanup_error))
            raise
        finally:self._in_preflight=False

    def compute_state(self):
        with self._lock:
            return {'preflight_ready':self._preflight_ready,'compute_started':self._compute_started,
                    'compute_used':self._compute_used,'in_preflight':self._in_preflight,'poisoned':self._poisoned}

    def _compute_error(self, code):
        error=RuntimeError(code)
        error.nf4_pool_state=self.compute_state()
        return error

    def check_compute_start(self):
        with self._lock:
            if (not self._preflight_only or not self._preflight_ready or self._compute_used
                    or self._in_preflight or self._poisoned or self._active_lease is not None
                    or self._acquiring or self._finishing):
                raise self._compute_error('E_NF4_POOL_START: unused successful preflight required')

    def begin_compute(self):
        """Consume one successful preflight for training or inference compute."""
        with self._lock:
            self.check_compute_start()
            self._compute_started=True
            self._compute_used=True

    def finish_compute(self):
        with self._lock:
            if self._poisoned or self._active_lease is not None or self._acquiring or self._finishing:
                raise self._compute_error('E_NF4_POOL_FINISH')
            self._compute_started=False

    def begin_training(self):
        """Compatibility name; this transition prepares no training rows."""
        self.begin_compute()

    @staticmethod
    def _check_ops(ops: Any) -> None:
        required = (
            "pin_record", "new_stream", "current_stream", "stage_cuda",
            "record_event", "synchronize_event", "synchronize_stream",
        )
        missing = [name for name in required if not callable(getattr(ops, name, None))]
        if missing:
            raise TypeError(f"CUDA pool ops are missing callable methods: {', '.join(missing)}")

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    @staticmethod
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

    @staticmethod
    def _freeze(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({key: CUDANF4LeasePool._freeze(item) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(CUDANF4LeasePool._freeze(item) for item in value)
        return value

    def _append_event(self, event: str, **values: Any) -> None:
        if not self._in_preflight and not self._diagnostic:return
        snapshot = {"event": event, **values}
        self._events.append(snapshot)

    @property
    def full_chain_timer_enabled(self):
        return self._diagnostic

    @property
    def record_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def poisoned(self):
        with self._lock:return self._poisoned

    @property
    def transfer_digests(self) -> Mapping[str, str]:
        with self._lock:
            return MappingProxyType(dict(self._verified_digests))

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return tuple(self._freeze(dict(event)) for event in self._events)

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._counts,
                "active": int(self._active_lease is not None),
                "poisoned": self._poisoned,
                "current_staged_bytes": self._current_staged_bytes,
                "peak_staged_bytes": self._peak_staged_bytes,
                "pinned_bytes": self._pinned_bytes,
                "h2d_bytes": self._h2d_bytes,
                "h2d_ms": self._h2d_ms if self._h2d_timing_samples else None,
                "h2d_ms_measured_samples": self._h2d_timing_samples,
                "h2d_ms_scope": "optional diagnostic event intervals; null when disabled, never zero-time inference",
                "statistics_scope": "startup and training diagnostic" if self._diagnostic else "startup only; training telemetry disabled",
                "validation_wall_seconds": self._validation_wall_seconds,
                "validation_attempts": self._validation_attempts,
                "integrity_policy": "startup_full_no_training_content_checks_v1" if self._preflight_only else "legacy_each_lease_diagnostic_only",
                "preflight_ready": self._preflight_ready,
                "transfer_digests": dict(self._verified_digests),
            }

    @diag.timed('pool.acquire',tag=diag.record_tag)
    def acquire(self, phase: str) -> CUDANF4Lease:
        if type(phase) is not str or phase not in _VALID_PHASES:
            raise ValueError("phase must be 'forward' or 'backward'")
        with self._lock:
            if self._preflight_only and not self._in_preflight and not(self._preflight_ready and self._compute_started):
                raise self._compute_error("E_NF4_POOL_NOT_RUNNING: successful startup preflight and begin_compute required")
            if self._poisoned:
                raise RuntimeError("CUDA NF4 lease pool is poisoned; explicit recovery is required")
            if self._active_lease is not None or self._acquiring or self._finishing:
                raise RuntimeError("CUDA NF4 lease pool already has an active operation")
            self._acquiring = True
            self._generation += 1
            lease = CUDANF4Lease(self, self._generation, phase)
            self._active_lease = lease
            self._current_staged_bytes = 0
            try:
                consumer_stream = self._ops.current_stream(self._device)
            except BaseException:
                self._acquiring = False
                lease._invalidate()
                self._active_lease = None
                raise
            lease._consumer_stream = consumer_stream

        stage_failed = False
        try:
            for name in self._names:
                pinned = self._pinned[name]
                try:
                    state = self._ops.stage_cuda(pinned, device=self._device, stream=self._copy_stream)
                except BaseException as exc:
                    stage_failed = True
                    self._retain_partial_stage_error(lease, exc, name)
                    raise
                assert lease._parts is not None
                # Retain the returned state before validating its bindings so
                # an integrity failure cannot drop a possibly in-flight copy.
                lease._parts[name] = state
                lease._owners.append(state)
                state_nbytes = self._nonnegative_int(getattr(state, "nbytes", None), f"staged {name}.nbytes")
                lease._staged_bytes += state_nbytes
                with self._lock:
                    self._current_staged_bytes = lease._staged_bytes
                    if self._in_preflight or self._diagnostic:
                        self._h2d_bytes += state_nbytes
                        self._peak_staged_bytes = max(self._peak_staged_bytes, self._current_staged_bytes)
                self._validate_cuda_record(name, pinned, state)

            lease._needs_verify_on_recovery = True
            # Stage all records first, then queue all ready waits on the one
            # consumer stream before any projection can observe this lease.
            if self._preflight_only:
                # All stages use the same ordered copy stream. Its last ready
                # event dominates every earlier copy; queue one dependency.
                lease.parts(self._names[-1]).wait_on(consumer_stream)
            else:
                for name in self._names:lease.parts(name).wait_on(consumer_stream)
            if not self._preflight_only or self._in_preflight:
                for name in self._names:self._verify_record(name, lease.parts(name))

            with self._lock:
                self._acquiring = False
                if self._in_preflight or self._diagnostic:
                    self._counts["acquires"] += 1
                    self._append_event(
                        "acquire", phase=phase, generation=lease.generation,
                        staged_bytes=lease.staged_bytes,
                        transfer_digests={} if not self._in_preflight else dict(self._verified_digests),
                    )
            return lease
        except BaseException as exc:
            with self._lock:
                self._acquiring = False
            if isinstance(exc, _TransferIntegrityError):
                complete_staging = lease._parts is not None and tuple(lease._parts) == self._names
                self._poison(lease, exc, verify_on_recovery=complete_staging)
                raise

            barrier_errors: list[BaseException] = []
            # All prior H2D copies, and any ready waits already queued, must be
            # quiescent before partial owners can be released.
            try:
                self._ops.synchronize_stream(self._copy_stream)
            except BaseException as barrier_exc:
                barrier_errors.append(barrier_exc)
            consumer_was_used = not stage_failed and bool(lease._parts)
            if consumer_was_used:
                try:
                    self._ops.synchronize_stream(consumer_stream)
                except BaseException as barrier_exc:
                    barrier_errors.append(barrier_exc)
            if barrier_errors:
                complete_staging = lease._parts is not None and tuple(lease._parts) == self._names
                self._poison(lease, exc, verify_on_recovery=complete_staging)
                self._note(exc, "partial acquire cleanup barrier failed; pool remains poisoned")
                raise exc from barrier_errors[0]

            with self._lock:
                self._append_event(
                    "acquire_failed",
                    phase=phase,
                    generation=lease.generation,
                    stage_failed=stage_failed,
                    cleanup_barriers="passed",
                )
                self._release_lease_locked(lease)
            raise

    def _retain_partial_stage_error(self, lease: CUDANF4Lease, exc: BaseException, name: str) -> None:
        retained = getattr(exc, "retained_owners", ())
        if isinstance(retained, (tuple, list)):
            for owner in retained:
                if all(owner is not existing for existing in lease._owners):
                    lease._owners.append(owner)
        if bool(getattr(exc, "copies_scheduled", False)):
            lease._staged_bytes += int(self._pinned[name].nbytes)
            with self._lock:
                self._current_staged_bytes = lease._staged_bytes
                self._peak_staged_bytes = max(self._peak_staged_bytes, self._current_staged_bytes)
        with self._lock:
            self._append_event(
                "partial_stage_failed",
                generation=lease.generation,
                record_name=name,
                copies_scheduled=bool(getattr(exc, "copies_scheduled", False)),
            )

    def _validate_cuda_record(self, name: str, pinned: Any, state: Any) -> None:
        if self._preflight_only and not self._in_preflight:
            # Ownership/accounting only: no payload, digest, metadata hash or D2H.
            if getattr(state,"pinned_owner",None)is not pinned or getattr(state,"nbytes",None)!=pinned.nbytes:
                raise RuntimeError("staged buffer owner/accounting mismatch")
            return
        for attribute in ("nbytes", "source_fingerprint", "pinned_owner", "wait_on", "verify_source"):
            if not hasattr(state, attribute):
                raise TypeError(f"staged NF4 state {name!r} is missing {attribute}")
        nbytes = self._nonnegative_int(state.nbytes, f"staged {name}.nbytes")
        if nbytes != int(pinned.nbytes):
            raise ValueError(f"staged NF4 state {name!r} byte count changed")
        if state.source_fingerprint != pinned.source_fingerprint or state.pinned_owner is not pinned:
            raise ValueError(f"staged NF4 state {name!r} is detached from its pinned canonical source")
        state_digest = getattr(state, "transfer_digest", None)
        if type(state_digest) is not str or _DIGEST_RE.fullmatch(state_digest) is None:
            raise _TransferIntegrityError(f"staged NF4 state {name!r} has an invalid transfer digest")
        if state_digest != self._pinned_digests[name]:
            raise _TransferIntegrityError(f"staged NF4 state {name!r} carries a different transfer digest")

    @diag.timed('integrity.verify_record',tag=diag.record_tag)
    def _verify_record(self, name: str, state: Any) -> str:
        if self._preflight_only and not self._in_preflight:
            raise RuntimeError("training content verification forbidden")
        expected = self._pinned_digests[name]
        started = time.perf_counter()
        with self._lock:
            self._validation_attempts += 1
        try:
            actual = state.verify_source()
        except BaseException as exc:
            raise _TransferIntegrityError(f"staged NF4 transfer verification failed for {name!r}") from exc
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self._validation_wall_seconds += elapsed
        if type(actual) is not str or _DIGEST_RE.fullmatch(actual) is None:
            raise _TransferIntegrityError(f"staged NF4 transfer verification returned an invalid digest for {name!r}")
        if actual != expected:
            raise _TransferIntegrityError(f"staged NF4 transfer digest mismatch for {name!r}")
        previous = self._verified_digests.get(name)
        if previous is not None and previous != actual:
            raise _TransferIntegrityError(f"staged NF4 transfer digest changed across leases for {name!r}")
        self._verified_digests[name] = actual
        return actual

    @staticmethod
    def _note(exc: BaseException, message: str) -> None:
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(message)

    def _poison(
        self,
        lease: CUDANF4Lease,
        cause: BaseException,
        *,
        verify_on_recovery: bool,
    ) -> None:
        with self._lock:
            self._poisoned = True
            self._active_lease = lease
            self._current_staged_bytes = lease._staged_bytes
            self._peak_staged_bytes = max(self._peak_staged_bytes, lease._staged_bytes)
            self._retained_resources = tuple(lease._owners)
            self._recovery_consumer_stream = lease._consumer_stream
            self._verify_during_recovery = bool(verify_on_recovery) and not self._preflight_only
            if self._preflight_only:self._preflight_ready=False
            self._append_event(
                "poisoned",
                phase=lease.phase,
                generation=lease.generation,
                staged_bytes=lease._staged_bytes,
                reason=f"{type(cause).__name__}: {cause}",
            )

    def _lease_is_active(self, lease: CUDANF4Lease) -> bool:
        with self._lock:
            return self._active_lease is lease and lease._parts is not None and lease.generation==self._generation

    def _lease_parts(self, lease: CUDANF4Lease, name: str) -> Any:
        with self._lock:
            if self._poisoned:
                raise RuntimeError("CUDA NF4 lease parts are unavailable while the pool is poisoned")
            if self._active_lease is not lease or lease._parts is None or lease.generation!=self._generation:
                raise RuntimeError("stale or inactive CUDA NF4 lease")
            if type(name) is not str or name not in self._records:
                raise KeyError(f"unknown CUDA NF4 record name: {name!r}")
            if name not in lease._parts:
                raise RuntimeError(f"CUDA NF4 state {name!r} has not been staged in this lease")
            return lease._parts[name]

    @diag.timed('pool.finish',tag=diag.record_tag)
    def _finish(self, lease: CUDANF4Lease, outcome: str) -> None:
        if outcome not in ("complete", "abort"):
            raise ValueError("invalid CUDA lease finish outcome")
        with self._lock:
            if self._poisoned:
                raise RuntimeError("CUDA NF4 lease pool is poisoned; explicit recovery is required")
            if self._active_lease is not lease or lease._parts is None or lease.generation!=self._generation:
                raise RuntimeError("stale or inactive CUDA NF4 lease")
            if self._acquiring or self._finishing:
                raise RuntimeError("CUDA NF4 lease pool is already changing state")
            self._finishing = True

        try:
            streams = [("consumer", lease._consumer_stream)]
            if outcome == "abort":
                streams.insert(0, ("copy", self._copy_stream))
            for label, stream in streams:
                self._barrier(lease, label, stream)
            if outcome == "complete" and not self._preflight_only:
                # Integrity is checked after all consumers have completed and
                # before any staged tensor or pinned owner can be invalidated.
                for name in self._names:
                    self._verify_record(name, lease.parts(name))
        except BaseException as exc:
            with self._lock:
                self._finishing = False
            self._poison(
                lease,
                exc,
                verify_on_recovery=(outcome == "complete" and bool(lease._parts)),
            )
            raise

        timing_error: BaseException | None = None
        elapsed = 0.0
        timing_samples=0
        if self._diagnostic:
            for state in (lease._parts or {}).values():
                if getattr(state,'timing_available',False)is not True:continue
                timing_samples+=1
                timing = getattr(state, "h2d_elapsed_ms", None)
                try:
                    value = float(timing()) if callable(timing) else float(getattr(state, "h2d_ms", 0.0))
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("invalid H2D elapsed time")
                    elapsed += value
                except BaseException as exc:
                    timing_error = timing_error or exc
        with self._lock:
            if self._in_preflight or self._diagnostic:
                self._h2d_ms += elapsed
                self._h2d_timing_samples += timing_samples
                self._counts["completes" if outcome == "complete" else "aborts"] += 1
                self._append_event(
                    outcome,
                    phase=lease.phase,
                    generation=lease.generation,
                    staged_bytes=lease._staged_bytes,
                    h2d_ms=elapsed,
                    transfer_digests={} if self._preflight_only and not self._in_preflight else dict(self._verified_digests),
                )
            self._finishing = False
            self._release_lease_locked(lease)
        if timing_error is not None:
            # Stream safety is already established, so a timing failure cannot
            # poison or retain a completed generation.
            raise RuntimeError("CUDA NF4 lease released but H2D timing could not be read") from timing_error

    @diag.timed('sync.pool_barrier',tag=diag.record_tag)
    def _barrier(self, lease: CUDANF4Lease, label: str, stream: Any) -> None:
        try:
            event = self._ops.record_event(stream)
        except BaseException:
            # Recovery will synchronize this stream directly if event creation
            # failed before there was an event to wait for.
            self._note_stream_for_recovery(lease, stream)
            raise
        lease._owners.append(event)
        try:
            self._ops.synchronize_event(event)
        except BaseException:
            # The event remains in _owners and is also safe to retry indirectly
            # via a stream barrier during explicit recovery.
            self._note_stream_for_recovery(lease, stream)
            raise

    def _note_stream_for_recovery(self, lease: CUDANF4Lease, stream: Any) -> None:
        if stream is lease._consumer_stream:
            self._recovery_consumer_stream = stream

    @diag.timed('pool.release_owners',tag=diag.record_tag)
    def _release_lease_locked(self, lease: CUDANF4Lease) -> None:
        if self._active_lease is lease:
            lease._invalidate()
            self._active_lease = None
        self._retained_resources = ()
        self._recovery_consumer_stream = None
        self._verify_during_recovery = False
        self._current_staged_bytes = 0
        self._poisoned = False

    def recover_after_barrier_failure(self) -> None:
        """Explicitly synchronize both streams, then optionally reverify state."""
        with self._lock:
            if not self._poisoned or self._active_lease is None:
                raise RuntimeError("CUDA NF4 lease pool has no poisoned generation to recover")
            if self._finishing or self._acquiring:
                raise RuntimeError("CUDA NF4 lease pool is already changing state")
            lease = self._active_lease
            consumer_stream = self._recovery_consumer_stream or lease._consumer_stream
            verify = self._verify_during_recovery
            self._finishing = True

        try:
            # Stream synchronizations cover both retained H2D sources and all
            # queued consumer work, including the event-record failure path.
            self._ops.synchronize_stream(self._copy_stream)
            if consumer_stream is not None:
                self._ops.synchronize_stream(consumer_stream)
            if verify:
                if lease._parts is None or tuple(lease._parts) != self._names:
                    raise RuntimeError("cannot verify an incomplete staged NF4 generation during recovery")
                for name in self._names:
                    self._verify_record(name, lease._parts[name])
        except BaseException as exc:
            with self._lock:
                self._finishing = False
                self._append_event(
                    "recovery_failed",
                    phase=lease.phase,
                    generation=lease.generation,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            raise

        with self._lock:
            self._counts["recoveries"] += 1
            self._append_event(
                "recovery",
                phase=lease.phase,
                generation=lease.generation,
                staged_bytes=lease._staged_bytes,
                transfer_digests={} if self._preflight_only and not self._in_preflight else dict(self._verified_digests),
            )
            self._finishing = False
            self._release_lease_locked(lease)


class _TransferIntegrityError(RuntimeError):
    """A staged copy did not match its canonical pinned NF4 transfer digest."""


__all__ = ("CUDANF4Lease", "CUDANF4LeasePool")
