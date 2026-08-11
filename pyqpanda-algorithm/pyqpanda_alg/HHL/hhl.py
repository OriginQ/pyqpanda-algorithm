"""HHL solver facade: local execution and post-selection parsing.

:class:`HHL` is the public solver object tying circuit synthesis
(:func:`~pyqpanda_alg.HHL.circuit.build_hhl_circuit`) to an execution
backend.  The constructor validates the linear system immediately —
singular and ill-conditioned systems are rejected before any task
submission — and derives the phase-estimation register size from the
requested numerical ``precision``.  :meth:`HHL.run` submits the
measurement-free program through ``submit_statevector`` and parses the
outcome: the weight of the ancilla success branch is the success
probability, and the post-selected data register is the solution of
the padded system, which is truncated to the original dimension and
rescaled by the original vector norm to recover the solution of the
original unpadded system.  The solver never constructs a ``CPUQVM``
itself; it always goes through the backend's submission surface.

Precision semantics
-------------------
``precision`` is the QPE phase-resolution target: the phase register
must resolve phase fractions down to ``1 / 2**p <= precision``, so the
default phase register size is ``ceil(log2(1 / precision))`` (at least
one qubit).  A smaller precision yields a finer eigenvalue estimate
and a more accurate reciprocal rotation at the cost of more phase
qubits; the smallest eigenvalue must be resolvable by that register
for the reciprocal rotation to invert it faithfully.
"""

import math

import numpy as np

from ..execution import ExecutionOptions, resolve_backend
from .circuit import HHLCircuitBuild, build_hhl_circuit
from .model import HHLConfig, HHLSolution, NormalizedLinearSystem
from .validation import _coerce_matrix, _coerce_vector, normalize_linear_system


class HHL:
    """Local HHL solver for a finite Hermitian linear system.

    Args:
        matrix: Hermitian square matrix of the system, as a 2-D array
            or a flat list whose length is a perfect square (legacy
            flattened form).  Validated at construction: non-Hermitian,
            singular, and over-threshold ill-conditioned matrices raise
            :class:`~pyqpanda_alg.execution.AlgorithmInputError` before
            any task submission.
        vector: Right-hand vector matching the matrix dimension; must
            have nonzero norm.
        precision: QPE phase-resolution target controlling the phase
            register size (see the module docstring).  Must be
            positive.

    Raises:
        ValueError: If ``precision`` is not positive.
        AlgorithmInputError: If the linear system fails validation
            (non-Hermitian, singular, ill-conditioned, or mismatched
            shapes).
    """

    def __init__(self, matrix, vector, precision: float = 1e-3) -> None:
        """Validate the system and derive the phase register size."""
        if precision <= 0:
            raise ValueError(f"precision must be positive, got {precision}")
        self._precision = float(precision)
        phase_qubits = max(1, math.ceil(math.log2(1 / self._precision)))
        self._config = HHLConfig(phase_qubits=phase_qubits)
        self._matrix = _coerce_matrix(matrix)
        self._vector = _coerce_vector(vector, self._matrix.shape[0])
        self._system = normalize_linear_system(self._matrix, self._vector, self._config)

    def build_circuit(self) -> HHLCircuitBuild:
        """Synthesize the HHL circuit for this system.

        Returns:
            The immutable :class:`~pyqpanda_alg.HHL.circuit.HHLCircuitBuild`
            carrying the measurement-free program, the register layout,
            and the evolution-time/reciprocal-scale metadata, so callers
            can inspect the circuit before running it.
        """
        return build_hhl_circuit(self._system.matrix, self._system.vector, self._config)

    def run(
        self,
        *,
        backend=None,
        execution_options: ExecutionOptions | None = None,
        reconstruct: bool = False,
    ) -> HHLSolution:
        """Solve the system on a backend and return the immutable result.

        The measurement-free program is submitted through the backend's
        ``submit_statevector``; the ancilla success branch gives the
        success probability, and the post-selected data register is the
        solution of the padded system.  With ``reconstruct=False`` (the
        default) the run reports the success probability and the
        post-selected data state, matching the runtime contract of not
        reconstructing the full vector; with ``reconstruct=True`` the
        classical solution vector and the residual against the original
        unpadded system are computed as well.

        Args:
            backend: The execution backend to submit the statevector
                execution to.  Keyword-only.  When None, the local CPU
                backend is used.
            execution_options: Submission options for the backend.
                Keyword-only.  When None, defaults apply.
            reconstruct: Whether to reconstruct the classical solution
                vector and compute the residual.  Keyword-only.
                Defaults to False.

        Returns:
            The frozen :class:`HHLSolution`: ``classical_vector`` (unit
            norm, truncated to the original dimension) and ``residual``
            (``||A x - b|| / ||b||`` with ``x`` restored to the original
            scale) when ``reconstruct`` is true, the post-selected data
            state, the success probability, and execution metadata.

        Raises:
            AlgorithmExecutionError: If the backend submission or result
                retrieval fails — the runtime failure is surfaced,
                never silently retried on another backend.
        """
        build = self.build_circuit()
        execution_backend = resolve_backend(backend)
        options = (
            execution_options
            if execution_options is not None
            else ExecutionOptions()
        )
        task = execution_backend.submit_statevector(build.program, options=options)
        statevector = np.asarray(task.result().single_statevector())
        data_state, success_probability = _postselect(build, statevector)
        metadata = {
            "backend": type(execution_backend).__name__,
            "task_ids": [task.id] if task.id else [],
            "precision": self._precision,
            "phase_qubits": len(build.phase_qubits),
            "evolution_time": build.evolution_time,
            "reciprocal_scale": build.reciprocal_scale,
            "eigenvalue_bounds": build.eigenvalue_bounds,
        }
        solution = HHLSolution(
            success_probability=success_probability,
            statevector=_unit_state(data_state),
            metadata=metadata,
        )
        if reconstruct:
            solution = _with_reconstruction(
                solution, build, self._system, data_state, self._matrix, self._vector
            )
        return solution


def _postselect(build: HHLCircuitBuild, statevector: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(data_state, success_probability)`` of the success branch.

    The circuit is measurement-free, so the full statevector is
    available; the success branch is the block where the success qubit
    is ``|1>`` and the (exactly uncomputed) phase register is
    ``|0...0>``.  The returned data state is ordered so that entry
    ``d`` is the amplitude of the data register reading ``d``, which
    matches the amplitude-encoded padded vector ordering.
    """
    data_count = len(build.data_qubits)
    phase_count = len(build.phase_qubits)
    dimension = 1 << (data_count + phase_count + 1)
    indices = np.arange(dimension)
    success = ((indices >> build.success_qubit) & 1) == 1
    zero_phase = ((indices >> data_count) & ((1 << phase_count) - 1)) == 0
    data_state = np.asarray(statevector[success & zero_phase], dtype=np.complex128)
    success_probability = float(np.sum(np.abs(statevector[success]) ** 2))
    return data_state, min(1.0, success_probability)


def _with_reconstruction(
    solution: HHLSolution,
    build: HHLCircuitBuild,
    system: NormalizedLinearSystem,
    data_state: np.ndarray,
    matrix: np.ndarray,
    vector: np.ndarray,
) -> HHLSolution:
    """Fill the classical solution and residual of a run.

    The post-selected data state equals ``reciprocal_scale * A_pad^{-1}
    b_hat`` (the reciprocal rotation applies exactly that scale), so
    scaling by ``original_vector_norm / reciprocal_scale`` and
    truncating to the first ``original_dimension`` entries restores the
    solution ``x`` of the original unpadded system.  ``classical_vector``
    is that solution normalized to unit norm, and ``residual`` is
    ``||A x - b|| / ||b||`` against the original system.
    """
    restored = (system.original_vector_norm / build.reciprocal_scale) * data_state
    truncated = restored[: system.original_dimension]
    norm = float(np.linalg.norm(truncated))
    if norm == 0.0:
        raise RuntimeError(
            "the HHL success branch carries no amplitude; "
            "the phase register cannot resolve the system spectrum"
        )
    residual = float(
        np.linalg.norm(matrix @ truncated - vector) / np.linalg.norm(vector)
    )
    return HHLSolution(
        classical_vector=truncated / norm,
        statevector=solution.statevector,
        success_probability=solution.success_probability,
        residual=residual,
        metadata=solution.metadata,
    )


def _unit_state(data_state: np.ndarray) -> np.ndarray:
    """Return the data state normalized to unit norm."""
    norm = float(np.linalg.norm(data_state))
    if norm == 0.0:
        raise RuntimeError(
            "the HHL success branch carries no amplitude; "
            "the phase register cannot resolve the system spectrum"
        )
    return data_state / norm
