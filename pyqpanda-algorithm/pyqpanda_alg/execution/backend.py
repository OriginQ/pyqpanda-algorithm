"""ExecutionBackend protocol: backends submit work and return tasks.

Every backend method submits work immediately and returns a
:class:`~pyqpanda_alg.execution.backend_task.BackendTask`; callers never
block except through ``result(timeout=...)`` on the task.  A runtime
failure never silently falls back to another backend, and submission is
never retried automatically — only idempotent status queries may use
bounded backoff.
"""

from typing import Any, Protocol, runtime_checkable

from .backend_task import BackendTask
from .capabilities import BackendCapabilities
from .options import ExecutionOptions
from .results import EstimateBatchResult, SampleBatchResult, StatevectorBatchResult


@runtime_checkable
class ExecutionBackend(Protocol):
    """Common submission surface implemented by every backend."""

    capabilities: BackendCapabilities

    def submit_sample(
        self, circuit: Any, *, options: ExecutionOptions
    ) -> BackendTask[SampleBatchResult]:
        """Submit sampling of ``circuit`` and return the task immediately."""
        ...

    def submit_estimate(
        self, circuit_and_observable: Any, *, options: ExecutionOptions
    ) -> BackendTask[EstimateBatchResult]:
        """Submit expectation estimation of an observable on a circuit."""
        ...

    def submit_statevector(
        self, circuit: Any, *, options: ExecutionOptions
    ) -> BackendTask[StatevectorBatchResult]:
        """Submit statevector execution of ``circuit``.

        Backends that do not advertise statevector support raise
        :class:`~pyqpanda_alg.execution.errors.DeviceCapabilityError`
        before any submission.
        """
        ...

    def create_variational_session(
        self, ansatz: Any, observable: Any, *, options: ExecutionOptions
    ) -> Any:
        """Create a variational session whose ``run(parameters)`` returns a
        ``BackendTask[float]`` expectation; the session acts as a context
        manager that releases the underlying session on exit."""
        ...
