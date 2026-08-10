"""Variational session abstraction shared by local and runtime backends.

A :class:`VariationalSession` binds an ansatz and an observable once and
evaluates them for many parameter sets: ``run(parameters)`` returns a
:class:`~pyqpanda_alg.execution.backend_task.BackendTask` whose result is
the expectation value (a float).  Every variant is a context manager
whose exit releases the underlying session exactly once -- also when the
body raises -- and the release is idempotent.  The state kept for
checkpointing is the parameter history only; a live session handle never
enters it.
"""

import uuid
from typing import Any, Optional

from pyqpanda3.core import QProg, expval_hamiltonian

from .backend_task import BackendTask, CompletedBackendTask, TaskStatus
from .errors import AlgorithmInputError
from .options import ExecutionOptions
from .runtime_task import RuntimeBackendTask


class VariationalSession:
    """Context-managed variational session with idempotent release.

    Subclasses implement :meth:`run` and :meth:`_release_raw`; this base
    owns the context protocol, the at-most-once release, and the
    parameter history kept for checkpointing.
    """

    def __init__(self, *, options: ExecutionOptions) -> None:
        self._options = options
        self._released = False
        #: Every run's parameters -- the only session state that may be
        #: checkpointed; live session handles are never stored here.
        self.history: list = []

    def __enter__(self) -> "VariationalSession":
        # One-way latch: entering never resets the released flag, so a
        # session released by an earlier context (or an explicit
        # release) cannot re-release its raw resource -- the raw session
        # would otherwise be released twice over the network.
        if self._released:
            raise AlgorithmInputError(
                "cannot re-enter a released variational session"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False

    def release(self) -> None:
        """Release the underlying session at most once.

        Calling this repeatedly -- including the release performed by
        the context exit after an error -- is a no-op after the first
        call.
        """
        if self._released:
            return
        self._released = True
        self._release_raw()

    def _release_raw(self) -> None:
        """Backend-specific release of the underlying session."""

    def run(self, parameters) -> BackendTask[float]:
        """Evaluate ``parameters`` and return the expectation task."""
        raise NotImplementedError


class LocalVariationalSession(VariationalSession):
    """Session evaluating concrete parameters synchronously on ``CPUQVM``."""

    def __init__(
        self, ansatz: Any, observable: Any, *, options: ExecutionOptions
    ) -> None:
        super().__init__(options=options)
        self._ansatz = ansatz
        self._observable = observable

    def run(self, parameters) -> CompletedBackendTask[float]:
        """Bind ``parameters`` into the ansatz and compute the expectation.

        Work is synchronous, so the returned task is already finished.
        """
        prog = _bind_parameters(self._ansatz, parameters)
        value = expval_hamiltonian(prog, self._observable, self._options.shots)
        self.history.append(_as_parameter_list(parameters))
        return CompletedBackendTask(
            float(value), task_id=f"local-variational-{uuid.uuid4().hex}"
        )


class RuntimeVariationalSession(VariationalSession):
    """Session submitting parameterized runs through a qpanda3-runtime
    ``VQSession``.

    ``raw_session`` is the live VQSession created by the service; it is
    released once, on context exit or explicit :meth:`release`.
    """

    def __init__(
        self,
        raw_session: Any,
        *,
        options: ExecutionOptions,
        measure_qubits: Optional[list] = None,
    ) -> None:
        super().__init__(options=options)
        self.raw_session = raw_session
        self._measure_qubits = measure_qubits if measure_qubits is not None else []

    def __enter__(self) -> "RuntimeVariationalSession":
        # Guard before touching the raw session: re-entering a released
        # session must never re-activate (or re-release) the live one.
        super().__enter__()
        self.raw_session.__enter__()
        return self

    def _release_raw(self) -> None:
        self.raw_session.release()

    def run(self, parameters) -> "SessionRunTask":
        """Submit the ansatz bound to ``parameters`` and return the task."""
        qtask = self.raw_session.run_vqtask(parameters, measure_list=self._measure_qubits)
        self.history.append(_as_parameter_list(parameters))
        task = RuntimeBackendTask(
            qtask,
            kind="estimate",
            shots=self._options.shots,
            timeout=self._options.timeout,
        )
        return SessionRunTask(task)


class SessionRunTask:
    """A runtime session run adapted to the ``BackendTask[float]`` surface.

    Wraps a :class:`RuntimeBackendTask` so the session's expectation
    value surfaces as a plain float while keeping the shared task
    surface (``id``, ``status()``, ``try_result()``,
    ``result(timeout=...)``, ``checkpoint(path=...)``).
    """

    def __init__(self, task: RuntimeBackendTask) -> None:
        self._task = task

    @property
    def id(self) -> str:
        """Runtime subtask identifier of this run."""
        return self._task.id

    def status(self) -> TaskStatus:
        """Poll the run without blocking."""
        return self._task.status()

    def try_result(self) -> Optional[float]:
        """Return the expectation value once the run is finished."""
        result = self._task.try_result()
        return None if result is None else float(result.single_value())

    def result(self, timeout: Optional[float] = None) -> float:
        """Block until the run finishes and return its expectation value."""
        return float(self._task.result(timeout).single_value())

    def checkpoint(self, path: Optional[str] = None) -> None:
        """Persist the run's remote state through the adapted task."""
        self._task.checkpoint(path)


def _bind_parameters(ansatz: Any, parameters: Any) -> QProg:
    """Return a concrete ``QProg`` with ``parameters`` substituted."""
    bound = ansatz(parameters)
    circuits = bound.circuits()
    if not circuits:
        raise AlgorithmInputError(
            "the ansatz produced no circuit for the given parameters"
        )
    return QProg(circuits[0])


def _as_parameter_list(parameters: Any) -> list:
    """Return ``parameters`` as a plain list, tolerating a scalar."""
    if isinstance(parameters, (int, float)):
        return [parameters]
    return list(parameters)
