"""Adaptation of qpanda3-runtime ``QTaskManager`` to the backend-task
surface.

:class:`RuntimeBackendTask` wraps a live ``QTaskManager`` so runtime
tasks speak the same surface as local tasks: ``id``, ``status()``,
``try_result()``, ``result(timeout=...)``, and ``checkpoint(path=...)``.
Raw QTaskManager results are decoded into the batch result wrappers,
and only transport failures are converted into the public exception
hierarchy, always retaining the original exception as ``__cause__``.
"""

import json
import time
from pathlib import Path
from typing import Any, NoReturn, Optional

from .backend_task import TaskStatus
from .errors import (
    BackendUnavailableError,
    ResultDecodingError,
    TaskRecoveryError,
    TaskTimeoutError,
)
from .results import EstimateBatchResult, SampleBatchResult

#: Message fragments the qpanda3-runtime timeout error embeds; a query
#: failure mentioning them is reported as a timeout rather than a
#: transport failure.
_TIMEOUT_FRAGMENTS = ("more than", "timeout")

#: qpanda3-runtime rejects blocking queries whose timeout is below this
#: value (its input validation raises "timeout shouldn't be smaller than
#: 120 seconds").  The effective timeout is clamped to it so that call
#: never fails with that misleading validation error.
_RUNTIME_MIN_TIMEOUT = 120.0

#: Submission timeout used when a checkpoint does not carry one (files
#: written before timeout persistence).
_DEFAULT_TIMEOUT = 1800.0


class RuntimeBackendTask:
    """A submitted qpanda3-runtime task adapted to the task protocol.

    ``kind`` is ``"sample"`` or ``"estimate"`` and selects the batch
    result wrapper used to decode raw QTaskManager results.  ``shots``
    and ``timeout`` mirror the values of the submitting
    :class:`~pyqpanda_alg.execution.options.ExecutionOptions`.
    """

    def __init__(
        self,
        qtask: Any,
        *,
        kind: str,
        shots: Optional[int] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if kind not in ("sample", "estimate"):
            raise ValueError(f"kind must be 'sample' or 'estimate', got {kind!r}")
        self.raw_task = qtask
        self.kind = kind
        self.shots = shots
        self.timeout = timeout
        self._decoded: Optional[Any] = None

    @property
    def id(self) -> str:
        """Runtime subtask identifier exposed as a stable string.

        Batch tasks carry a list of subtask IDs; the list is joined so
        the protocol's ``id: str`` contract always holds.
        """
        raw = self.raw_task.id()
        if isinstance(raw, (list, tuple)):
            return ",".join(str(part) for part in raw)
        return str(raw)

    def status(self) -> TaskStatus:
        """Poll without blocking and map to the task lifecycle."""
        _, finished, _ = self._try_get_result()
        return TaskStatus.SUCCEEDED if finished else TaskStatus.RUNNING

    def try_result(self) -> Optional[Any]:
        """Return the decoded batch result once the task is finished."""
        if self._decoded is not None:
            return self._decoded
        raw, finished, _ = self._try_get_result()
        if not finished:
            return None
        self._decoded = self._decode(raw)
        return self._decoded

    def result(self, timeout: Optional[float] = None) -> Any:
        """Block until the raw task finishes, then return the decoded result.

        With ``timeout=None`` the task-level timeout of the submission
        options applies.  Raises :class:`TaskTimeoutError` on timeout
        and the public execution-layer error matching the failure
        otherwise.
        """
        if self._decoded is not None:
            return self._decoded
        effective = self._effective_timeout(timeout)
        deadline = time.monotonic() + effective
        try:
            raw = self.raw_task.get_result_sync(timeout=effective)
        except Exception as exc:
            self._raise_query_error(exc, effective, deadline)
        self._decoded = self._decode(raw)
        return self._decoded

    async def result_async(self, timeout: Optional[float] = None) -> Any:
        """Coroutine variant of :meth:`result` for asyncio callers."""
        if self._decoded is not None:
            return self._decoded
        effective = self._effective_timeout(timeout)
        deadline = time.monotonic() + effective
        try:
            raw = await self.raw_task.get_result_async(timeout=effective)
        except Exception as exc:
            self._raise_query_error(exc, effective, deadline)
        self._decoded = self._decode(raw)
        return self._decoded

    def _effective_timeout(self, timeout: Optional[float]) -> float:
        """Return the requested timeout clamped to the runtime minimum.

        qpanda3-runtime rejects blocking queries below 120 seconds at
        input validation, so a caller-side ``timeout=60`` must never be
        forwarded: it would fail instantly with a misleading
        "timeout shouldn't be smaller than 120 seconds" error instead of
        actually waiting.
        """
        effective = self.timeout if timeout is None else timeout
        return max(effective, _RUNTIME_MIN_TIMEOUT)

    def checkpoint(self, path: Optional[str] = None) -> None:
        """Persist the remote task state through the raw QTaskManager.

        The checkpoint holds the serialized task state only; API keys,
        tokens, live service, and device handles never enter it.  The
        submission timeout is carried in the checkpoint's ``user_data``
        so a recovered task keeps the original deadline.
        """
        self.raw_task.check_point(filepath=path, user_data={"timeout": self.timeout})

    @classmethod
    def recover(cls, service: Any, path: str) -> "RuntimeBackendTask":
        """Rebuild a runtime task from a checkpoint file.

        The task kind (and shots) are read from the checkpointed task
        metadata; a checkpoint without them cannot be adapted and raises
        :class:`~pyqpanda_alg.execution.errors.TaskRecoveryError`.  The
        submission timeout is restored from the checkpoint's
        ``user_data``; a checkpoint without it (written before timeout
        persistence) falls back to the default submission timeout.
        """
        qtask = service.recover_qtask_manager(path)
        metadata = qtask.get_task_state().get("data", {}).get("metadata")
        if not isinstance(metadata, dict):
            raise TaskRecoveryError(
                f"checkpoint {path} does not identify the runtime task kind"
            )
        kind = metadata.get("qtask_type")
        if kind not in ("sample", "estimate"):
            raise TaskRecoveryError(
                f"checkpoint {path} has unsupported runtime task kind {kind!r}"
            )
        return cls(
            qtask,
            kind=kind,
            shots=metadata.get("shots"),
            timeout=_checkpoint_timeout(path),
        )

    def _try_get_result(self) -> tuple[Any, bool, str]:
        """Poll the raw manager, converting transport failures."""
        try:
            return self.raw_task.try_get_result()
        except Exception as exc:
            raise BackendUnavailableError(
                f"runtime task {self.id} could not be queried: {exc}"
            ) from exc

    def _raise_query_error(
        self, exc: Exception, effective: float, deadline: float
    ) -> NoReturn:
        """Map a failed blocking query into the public exception hierarchy."""
        if time.monotonic() >= deadline or any(
            fragment in str(exc).lower() for fragment in _TIMEOUT_FRAGMENTS
        ):
            raise TaskTimeoutError(
                f"runtime task {self.id} did not finish within {effective} seconds"
            ) from exc
        raise BackendUnavailableError(
            f"runtime task {self.id} could not be queried: {exc}"
        ) from exc

    def _decode(self, raw: Any) -> Any:
        """Decode raw QTaskManager results into a batch result wrapper."""
        if self.kind == "sample":
            return self._decode_sample(raw)
        return self._decode_estimate(raw)

    def _decode_sample(self, raw: Any) -> SampleBatchResult:
        try:
            counts = tuple(dict(item) for item in raw)
        except (TypeError, ValueError) as exc:
            raise ResultDecodingError(
                f"could not decode runtime sample result {raw!r}"
            ) from exc
        return SampleBatchResult(counts=counts, shots=self.shots)

    def _decode_estimate(self, raw: Any) -> EstimateBatchResult:
        try:
            values = tuple(float(value) for value in _flatten_estimate(raw))
        except (TypeError, ValueError) as exc:
            raise ResultDecodingError(
                f"could not decode runtime estimate result {raw!r}"
            ) from exc
        return EstimateBatchResult(values=values)


def _flatten_estimate(raw: Any) -> Any:
    """Yield the float leaves of a (possibly nested) estimate result."""
    if isinstance(raw, (int, float)):
        yield raw
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            yield from _flatten_estimate(item)
    else:
        raise TypeError(f"unexpected estimate result element {type(raw).__name__}")


def _checkpoint_timeout(path: str) -> float:
    """Return the timeout persisted in the checkpoint's ``user_data``.

    qpanda3-runtime checkpoint files are JSON objects with a top-level
    ``user_data`` key holding whatever the submitting
    :meth:`RuntimeBackendTask.checkpoint` stored.  An unreadable file, a
    file without ``user_data`` (format version 1 checkpoints written
    before timeout persistence), or a malformed value all fall back to
    the default submission timeout.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _DEFAULT_TIMEOUT
    user_data = payload.get("user_data") if isinstance(payload, dict) else None
    if isinstance(user_data, dict):
        timeout = user_data.get("timeout")
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
            return float(timeout)
    return _DEFAULT_TIMEOUT
