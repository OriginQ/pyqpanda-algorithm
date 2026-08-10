"""Local CPU execution backend built on pyqpanda3's ``CPUQVM``.

:class:`LocalBackend` is the default execution path for existing
algorithms: work is synchronous and every submission returns an
already-finished :class:`CompletedBackendTask`.  A runtime failure is
raised directly out of the submission call — there is never a silent
fallback to another backend, and submission is never retried.
"""

import uuid
from typing import Any, Optional

from pyqpanda3.core import CPUQVM, QProg, expval_hamiltonian

from .backend import ExecutionBackend
from .backend_task import CompletedBackendTask
from .capabilities import BackendCapabilities
from .errors import AlgorithmInputError, DeviceCapabilityError
from .options import ExecutionOptions
from .result_normalization import probability_to_counts, sort_by_basis_index
from .results import EstimateBatchResult, SampleBatchResult, StatevectorBatchResult


class LocalBackend:
    """CPU-only backend executing circuits synchronously on ``CPUQVM``."""

    capabilities = BackendCapabilities(variational_session=False, tomography=False)

    def submit_sample(
        self, circuit: Any, *, options: ExecutionOptions
    ) -> CompletedBackendTask:
        """Run ``circuit`` with measurements and return the counts task."""
        prog = _as_qprog(circuit)
        qvm = CPUQVM()
        qvm.run(prog, shots=options.shots)
        probabilities = qvm.result().get_prob_dict()
        counts = sort_by_basis_index(
            probability_to_counts(probabilities, shots=options.shots)
        )
        result = SampleBatchResult(counts=(counts,), shots=options.shots)
        return CompletedBackendTask(result, task_id=f"local-sample-{uuid.uuid4().hex}")

    def submit_estimate(
        self, circuit_and_observable: Any, *, options: ExecutionOptions
    ) -> CompletedBackendTask:
        """Run ``(circuit, observable)`` and return the expectation task."""
        circuit, observable = _split_estimate_input(circuit_and_observable)
        prog = _as_qprog(circuit)
        _require_unmeasured(prog, "Expectation estimation")
        qvm = CPUQVM()
        value = expval_hamiltonian(prog, observable, options.shots)
        result = EstimateBatchResult(values=(value,))
        return CompletedBackendTask(result, task_id=f"local-estimate-{uuid.uuid4().hex}")

    def submit_statevector(
        self, circuit: Any, *, options: ExecutionOptions
    ) -> CompletedBackendTask:
        """Run ``circuit`` and return the statevector task."""
        prog = _as_qprog(circuit)
        _require_unmeasured(prog, "State-vector execution")
        qvm = CPUQVM()
        qvm.run(prog, shots=1)
        statevector = qvm.result().get_state_vector()
        result = StatevectorBatchResult(statevectors=(statevector,))
        return CompletedBackendTask(
            result, task_id=f"local-statevector-{uuid.uuid4().hex}"
        )

    def create_variational_session(
        self, ansatz: Any, observable: Any, *, options: ExecutionOptions
    ) -> None:
        """Reject variational sessions: the local backend does not provide them."""
        raise DeviceCapabilityError(
            "LocalBackend does not support variational sessions"
        )


def resolve_backend(backend: Optional[ExecutionBackend] = None) -> ExecutionBackend:
    """Return the supplied backend, or the local CPU default when None.

    A supplied backend is never replaced: runtime failures surface to
    the caller instead of silently falling back to LocalBackend.
    """
    if backend is None:
        return LocalBackend()
    return backend


def _split_estimate_input(pair: Any) -> tuple[Any, Any]:
    """Split ``(circuit, observable)`` with a public input error."""
    try:
        circuit, observable = pair
    except (TypeError, ValueError) as exc:
        raise AlgorithmInputError(
            "submit_estimate expects a (circuit, observable) pair"
        ) from exc
    return circuit, observable


def _as_qprog(circuit: Any) -> QProg:
    """Return ``circuit`` as a :class:`QProg`, composing gate-only inputs.

    pyqpanda3's ``CPUQVM`` only executes ``QProg`` objects, so circuit-
    like inputs (``QCircuit``, ``QGate``) are composed into one.
    """
    if isinstance(circuit, QProg):
        return circuit
    return QProg() << circuit


def _require_unmeasured(prog: QProg, operation: str) -> None:
    """Reject circuits that contain measurement nodes."""
    if prog.get_measure_nodes():
        raise AlgorithmInputError(
            f"{operation} does not support circuits containing measurements"
        )
