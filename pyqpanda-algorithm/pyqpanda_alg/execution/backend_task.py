"""Backend task abstractions: status enum, task protocol, completed task.

Backends always submit work and return a :class:`BackendTask`; callers
poll ``status()`` and collect results either non-blocking with
``try_result()`` or blocking with ``result(timeout=...)``.  Status
queries are the only operations that may be retried (with bounded
backoff); submission itself is never retried automatically.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


class TaskStatus(Enum):
    """Lifecycle state of a backend task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@runtime_checkable
class BackendTask(Protocol[T]):
    """Handle to work already submitted to a backend.

    A task is the only object a backend ever returns from submission.
    Once a task fails or times out, ``result()`` raises a public
    execution-layer exception carrying the original failure as
    ``__cause__`` where one exists.
    """

    id: str

    def status(self) -> TaskStatus:
        """Return the current lifecycle state without blocking."""
        ...

    def try_result(self) -> Optional[T]:
        """Return the result if the task is finished, else None."""
        ...

    def result(self, timeout: Optional[float] = None) -> T:
        """Block until the result is available, up to ``timeout`` seconds.

        Raises :class:`~pyqpanda_alg.execution.errors.TaskTimeoutError` on
        timeout and the public execution-layer exception matching the
        failure otherwise.
        """
        ...

    def checkpoint(self, path: Optional[str] = None) -> None:
        """Persist enough remote state to resume this task later."""
        ...


@dataclass(frozen=True)
class CompletedBackendTask(Generic[T]):
    """An already-finished backend task with a fixed result.

    Local and synchronous backends return this instead of a live
    handle; the result is available immediately and never changes.
    """

    value: T
    task_id: str = ""

    @property
    def id(self) -> str:
        """Backend task identifier exposed by the task protocol."""
        return self.task_id

    def status(self) -> TaskStatus:
        return TaskStatus.SUCCEEDED

    def try_result(self) -> T:
        return self.value

    def result(self, timeout: Optional[float] = None) -> T:
        return self.value

    def checkpoint(self, path: Optional[str] = None) -> None:
        """Nothing to persist for a task that is already finished."""
