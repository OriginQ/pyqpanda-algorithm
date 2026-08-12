"""Resumable algorithm state machines that coordinate backend tasks.

An :class:`AlgorithmTask` wraps one algorithm run.  Every ``poll()``
call executes exactly one deterministic step through a process-local
``advance`` callback, which submits backend work and returns the task
plus a completion flag; the completed step results accumulate into the
algorithm result.  The callback itself is never serialized: a JSON
checkpoint holds only the state, backend task IDs, backend identity,
and redacted metadata, and ``resume()`` rebuilds the callback from a
factory registered under the algorithm name — arbitrary Python objects
are never deserialized.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Generic, NamedTuple, Optional, TypeVar, Union

from .backend import ExecutionBackend
from .backend_task import BackendTask, TaskStatus
from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    AlgorithmCheckpoint,
    read_checkpoint,
    write_checkpoint,
)
from .errors import AlgorithmInputError, TaskRecoveryError, TaskTimeoutError

T = TypeVar("T")

#: An advance callback drives one step: mutate the state, submit backend
#: work, and report the task plus whether the algorithm has finished.
AlgorithmAdvance = Callable[[Any], tuple[BackendTask[Any], bool]]

#: A registered factory rebuilds a fresh advance callback at resume time.
AlgorithmFactory = Callable[[Optional[ExecutionBackend]], AlgorithmAdvance]


class AlgorithmStep(NamedTuple):
    """One step of an algorithm state machine.

    ``task`` is the backend task submitted for this step and ``done``
    reports whether the algorithm finished with it.
    """

    task: BackendTask[Any]
    done: bool


_ALGORITHM_FACTORIES: dict[str, AlgorithmFactory] = {}


def register_algorithm(algorithm: str, factory: AlgorithmFactory) -> None:
    """Register the process-local factory that rebuilds ``advance`` callbacks.

    ``factory(backend)`` must return a fresh advance callback for the
    algorithm.  :meth:`AlgorithmTask.resume` looks the factory up by the
    checkpoint's algorithm name, which is why the callback itself never
    needs to be serialized.
    """
    if not isinstance(algorithm, str) or not algorithm:
        raise AlgorithmInputError("algorithm name must be a non-empty string")
    _ALGORITHM_FACTORIES[algorithm] = factory


class AlgorithmTask(Generic[T]):
    """Resumable state machine coordinating one or more backend tasks.

    The constructor accepts the algorithm name, the initial (mutable)
    state, and the process-local ``advance`` callback.  ``metadata`` is
    user-supplied context stored in checkpoints after recursive
    credential-key filtering; ``backend`` only contributes its identity
    to checkpoints.  Every remote task ID is collected in
    ``backend_task_ids`` as steps complete.
    """

    def __init__(
        self,
        algorithm: str,
        initial_state: Any,
        advance: AlgorithmAdvance,
        *,
        metadata: Optional[dict[str, Any]] = None,
        backend: Optional[ExecutionBackend] = None,
    ) -> None:
        if not isinstance(algorithm, str) or not algorithm:
            raise AlgorithmInputError("algorithm name must be a non-empty string")
        if not callable(advance):
            raise AlgorithmInputError("advance must be a callable step function")
        self.algorithm = algorithm
        self.id = uuid.uuid4().hex
        self.backend_task_ids: list[str] = []
        self._state = initial_state
        self._metadata = dict(metadata) if metadata else {}
        self._advance = advance
        self._backend = backend
        self._status = TaskStatus.PENDING
        self._result: Any = None
        self._error: Optional[BaseException] = None

    def status(self) -> TaskStatus:
        """Current lifecycle state of the algorithm task."""
        return self._status

    def poll(self) -> TaskStatus:
        """Run exactly one advance step and return the new status.

        Polling is deterministic: each call runs the process-local
        ``advance`` callback once.  Once the task is terminal, polling
        is a no-op.  If a step raises, the task is marked FAILED and the
        exception propagates to the caller.
        """
        if self._status is TaskStatus.SUCCEEDED:
            return self._status
        if self._status is TaskStatus.TIMED_OUT:
            return self._status
        if self._status is TaskStatus.FAILED:
            if self._error is not None:
                raise self._error
            return self._status
        try:
            step = self._advance(self._state)
            task, done = step
            if task.id:
                self.backend_task_ids.append(task.id)
            self._accumulate(task)
        except Exception as exc:
            # The whole step -- including accumulating its result -- is
            # inside the try: a query failure while collecting the step's
            # result must mark the task FAILED instead of leaving it
            # PENDING, which would make the next poll() re-run the step
            # and double-submit backend work.
            self._status = TaskStatus.FAILED
            self._error = exc
            raise
        self._status = TaskStatus.SUCCEEDED if done else TaskStatus.RUNNING
        return self._status

    def try_result(self) -> Optional[T]:
        """Return the algorithm result if succeeded, else None."""
        if self._status is TaskStatus.SUCCEEDED:
            return self._result
        return None

    def result(self, timeout: Optional[float] = None) -> T:
        """Poll until the algorithm succeeds, up to ``timeout`` seconds.

        Raises :class:`~pyqpanda_alg.execution.errors.TaskTimeoutError`
        when the deadline passes first, and re-raises the stored step
        failure when a step raised.  With ``timeout=None`` the task is
        polled until it succeeds or fails.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._status is TaskStatus.SUCCEEDED:
                return self._result
            if self._status is TaskStatus.FAILED:
                if self._error is not None:
                    raise self._error
                raise TaskRecoveryError("algorithm task failed without a recorded error")
            if self._status is TaskStatus.TIMED_OUT:
                raise TaskTimeoutError(
                    f"algorithm task {self.id} previously timed out"
                )
            if deadline is not None and time.monotonic() >= deadline:
                self._status = TaskStatus.TIMED_OUT
                raise TaskTimeoutError(
                    f"algorithm task {self.id} did not finish within {timeout} seconds"
                )
            self.poll()

    async def result_async(self, timeout: Optional[float] = None) -> T:
        """Resolve the algorithm without blocking the caller's event loop."""
        return await asyncio.to_thread(self.result, timeout)

    def checkpoint(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the algorithm state to ``path`` as JSON and return it.

        The checkpoint never contains the advance callback, live task
        handles, or credentials: state and metadata are stored after
        recursive credential-key filtering, and only backend task IDs
        (never the task objects) are recorded.
        """
        if path is None:
            raise AlgorithmInputError("checkpoint() requires an explicit path")
        return write_checkpoint(
            path,
            AlgorithmCheckpoint(
                format_version=CHECKPOINT_FORMAT_VERSION,
                algorithm=self.algorithm,
                status=self._status.value,
                state=self._state,
                result=self._result,
                backend_task_ids=tuple(self.backend_task_ids),
                backend_identity=_backend_identity(self._backend),
                metadata=self._metadata,
            ),
        )

    @classmethod
    def resume(
        cls, path: Union[str, Path], *, backend: Optional[ExecutionBackend] = None
    ) -> "AlgorithmTask[T]":
        """Reconstruct an algorithm task from a checkpoint file.

        The advance callback is rebuilt through the factory registered
        for the checkpoint's algorithm name; arbitrary Python objects
        are never deserialized.  ``backend`` is passed to the factory so
        the caller re-supplies compatible (logged-in) credentials at
        recovery time.  Live backend task handles are not restored —
        only their IDs survive in the checkpoint.  A backend supplied at
        resume time must match the identity recorded in the checkpoint;
        resuming with a different backend raises
        :class:`~pyqpanda_alg.execution.errors.TaskRecoveryError`.
        """
        checkpoint = read_checkpoint(path)
        recorded = checkpoint.backend_identity
        if recorded is not None and _backend_identity(backend) != recorded:
            raise TaskRecoveryError(
                f"checkpoint records backend identity {recorded!r} but resume "
                f"was given "
                f"{type(backend).__name__ if backend is not None else 'no backend'}"
            )
        factory = _ALGORITHM_FACTORIES.get(checkpoint.algorithm)
        if factory is None:
            raise TaskRecoveryError(
                f"no advance factory registered for algorithm "
                f"{checkpoint.algorithm!r}; register one with "
                f"register_algorithm() before resuming"
            )
        return cls._restore(checkpoint, factory(backend), backend=backend)

    @classmethod
    def _restore(
        cls,
        checkpoint: AlgorithmCheckpoint,
        advance: AlgorithmAdvance,
        *,
        backend: Optional[ExecutionBackend] = None,
    ) -> "AlgorithmTask[T]":
        task = cls(
            algorithm=checkpoint.algorithm,
            initial_state=checkpoint.state,
            advance=advance,
            metadata=checkpoint.metadata,
            backend=backend,
        )
        task._status = TaskStatus(checkpoint.status)
        task._result = checkpoint.result
        task.backend_task_ids = list(checkpoint.backend_task_ids)
        return task

    def _accumulate(self, task: BackendTask[Any]) -> None:
        """Add the step's completed result into the algorithm result."""
        value = task.try_result()
        if value is None:
            return
        self._result = value if self._result is None else self._result + value


def _backend_identity(backend: Optional[ExecutionBackend]) -> Any:
    """Describe the backend without ever touching its credentials."""
    if backend is None:
        return None
    return {"type": type(backend).__name__}
