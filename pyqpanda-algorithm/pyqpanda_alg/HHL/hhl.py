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

Runtime measurement plans
-------------------------
The exact state-vector path above is used on the CPU
:class:`~pyqpanda_alg.execution.LocalBackend`.  Any other backend is
executed through sampling with the runtime measurement plans: the
success ancilla and the requested data observables are measured, and
the success probability is the weight of the success outcomes in the
counts.  With ``reconstruct=True`` the solution is recovered by
Pauli-basis tomography — ``3 ** data_qubits`` X/Y/Z measurement
circuits, exactly the count declared by
:func:`~pyqpanda_alg.HHL.resources.estimate_hhl_resources` — submitted
one basis batch per step of a resumable
:class:`~pyqpanda_alg.execution.AlgorithmTask`, with a checkpoint
after every completed batch.  The density matrix is reconstructed from
the post-selected counts, its dominant eigenvector is truncated to the
original dimension, and the reconstruction fidelity and uncertainty
are reported in the result metadata.  The reported uncertainty is a
self-consistency measure (``1 - lambda_max``) of the reconstructed
density matrix, not a statistical error bar; because the
linear-inversion reconstruction carries no positivity constraint, shot
noise can push the raw dominant eigenvalue outside the physical range,
and the reported fidelity and uncertainty are clamped into ``[0, 1]``.
The runtime default is success probability and requested observables,
never full-vector reconstruction.

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
import re

import numpy as np
from pyqpanda3.core import QProg, measure

from ..execution import (
    AlgorithmInputError,
    AlgorithmTask,
    CompletedBackendTask,
    DeviceCapabilityError,
    ExecutionOptions,
    LocalBackend,
    TaskStatus,
    register_algorithm,
    resolve_backend,
)
from .circuit import HHLCircuitBuild, build_hhl_circuit
from .model import HHLConfig, HHLSolution, NormalizedLinearSystem
from .resources import estimate_hhl_resources
from .tomography import (
    apply_basis_rotation,
    dominant_eigenvector,
    pauli_basis_strings,
    pauli_expectation,
    reconstruct_density_matrix,
)
from .validation import _coerce_matrix, _coerce_vector, normalize_linear_system

#: One observable token: a Pauli letter directly followed by the index
#: of the data qubit it acts on (``X0Y1`` is X on qubit 0 and Y on 1).
_OBSERVABLE_TOKEN = re.compile(r"([XYZ])(\d+)")


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
        ValueError: If ``precision`` is not positive.  The phase
            register size is derived from ``1 / precision``, which
            requires a positive target.
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
        observables: list[str] | None = None,
        checkpoint_path=None,
    ) -> HHLSolution:
        """Solve the system on a backend and return the immutable result.

        On the CPU :class:`~pyqpanda_alg.execution.LocalBackend` the
        measurement-free program is submitted through
        ``submit_statevector``; the ancilla success branch gives the
        success probability, and the post-selected data register is the
        solution of the padded system.  With ``reconstruct=False`` (the
        default) the run reports the success probability and the
        post-selected data state, matching the runtime contract of not
        reconstructing the full vector; with ``reconstruct=True`` the
        classical solution vector and the residual against the original
        unpadded system are computed as well.

        On any other (runtime) backend the run is executed through
        sampling: the success ancilla and the requested data observables
        are measured, and the success probability is the weight of the
        success outcomes.  With ``reconstruct=True`` the solution is
        recovered by Pauli-basis tomography over the data register: the
        ``3 ** data_qubits`` basis circuits declared by
        :func:`~pyqpanda_alg.HHL.resources.estimate_hhl_resources` are
        submitted one basis batch per step of a resumable
        :class:`~pyqpanda_alg.execution.AlgorithmTask`, a checkpoint is
        written after every completed batch when ``checkpoint_path`` is
        given, and the density matrix reconstructed from the counts
        yields the classical solution vector and the reconstruction
        fidelity and uncertainty — a self-consistency measure
        (``1 - lambda_max``) of the reconstructed density matrix, not a
        statistical error bar, clamped into ``[0, 1]`` because shot
        noise can make the linear-inversion estimate non-physical.
        The sampling path requires a backend advertising sampling
        capability.

        Args:
            backend: The execution backend to submit the work to.
                Keyword-only.  When None, the local CPU backend is used.
            execution_options: Submission options for the backend.
                Keyword-only.  When None, defaults apply.
            reconstruct: Whether to reconstruct the classical solution
                vector.  Keyword-only.  Defaults to False.  On the local
                path the residual against the original unpadded system
                is computed as well.
            observables: Data-register Pauli strings to measure on the
                sampling path, for example ``["Z0", "X0Y1"]``; each
                expectation value is reported in
                ``metadata["observables"]``.  Keyword-only.  Defaults
                to None (no observable expectations).  Rejected on the
                local state-vector path and cannot be combined with
                ``reconstruct=True``, which measures the full Pauli
                basis instead.
            checkpoint_path: Where to checkpoint the tomography state
                machine after each completed basis batch, so a failed
                run can be resumed with
                :meth:`~pyqpanda_alg.execution.AlgorithmTask.resume`.
                Keyword-only.  Defaults to None (no checkpoints).

        Returns:
            The frozen :class:`HHLSolution`: ``classical_vector`` (unit
            norm, truncated to the original dimension) and ``residual``
            (``||A x - b|| / ||b||`` with ``x`` restored to the original
            scale) when ``reconstruct`` is true, the post-selected data
            state, the success probability, and execution metadata.  On
            the sampling path ``statevector`` is always None and the
            reconstruction fidelity and uncertainty are reported in
            ``metadata["tomography"]``.

        Raises:
            AlgorithmExecutionError: If the backend submission or result
                retrieval fails — the runtime failure is surfaced,
                never silently retried on another backend.
            DeviceCapabilityError: If a sampling backend does not
                advertise sampling capability, before any submission.
            AlgorithmInputError: If an observable is not a valid Pauli
                string over the data register, observables are
                requested on the local state-vector path, or
                observables are combined with ``reconstruct=True`` on
                the sampling path.
        """
        execution_backend = resolve_backend(backend)
        options = (
            execution_options
            if execution_options is not None
            else ExecutionOptions()
        )
        if not isinstance(execution_backend, LocalBackend):
            return self._run_sampled(
                execution_backend,
                options,
                reconstruct=reconstruct,
                observables=observables,
                checkpoint_path=checkpoint_path,
            )
        if observables is not None:
            raise AlgorithmInputError(
                "observables are measured through sampling; "
                "the local state-vector path does not support them"
            )
        build = self.build_circuit()
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


    def _run_sampled(
        self,
        execution_backend,
        options: ExecutionOptions,
        *,
        reconstruct: bool,
        observables: list[str] | None,
        checkpoint_path,
    ) -> HHLSolution:
        """Execute the runtime measurement plans on a sampling backend.

        The default plan measures the success ancilla and, per requested
        observable, the data register in that observable's basis; the
        success probability is the weight of the success outcomes of the
        first measurement circuit, and each observable expectation is
        estimated from the post-selected counts.  The reconstruction
        plan delegates to :meth:`_run_tomography`.
        """
        _require_sampling(execution_backend)
        if reconstruct and observables:
            raise AlgorithmInputError(
                "observables cannot be combined with reconstruct=True; "
                "tomography measures the full Pauli basis instead"
            )
        if reconstruct:
            return self._run_tomography(
                execution_backend, options, checkpoint_path
            )
        build = self.build_circuit()
        plans = _measurement_plans(build, observables)
        success_probability = None
        expectations = {}
        task_ids = []
        # Pair each plan with the observable it was built for so
        # expectations can never be mislabeled; without observables the
        # single plan is paired with None.
        for observable, (program, pauli) in zip(observables or [None], plans):
            task = execution_backend.submit_sample(program, options=options)
            counts = task.result().single_counts()
            if success_probability is None:
                success_probability = _success_probability(counts)
            if pauli is not None:
                expectations[observable] = pauli_expectation(
                    pauli, _postselect_counts(counts)
                )
            task_ids.append(task.id)
        metadata = _runtime_metadata(
            execution_backend, task_ids, self._precision, build
        )
        if expectations:
            metadata["observables"] = expectations
        return HHLSolution(
            success_probability=success_probability,
            statevector=None,
            metadata=metadata,
        )

    def _run_tomography(
        self, execution_backend, options: ExecutionOptions, checkpoint_path
    ) -> HHLSolution:
        """Recover the classical solution by Pauli-basis tomography.

        The ``3 ** data_qubits`` basis circuits declared by
        :func:`estimate_hhl_resources` are submitted one basis batch per
        step of a resumable :class:`AlgorithmTask`; a checkpoint written
        after every completed basis batch carries the accumulated counts
        and reconstruction progress, so a failed run can be resumed with
        :meth:`AlgorithmTask.resume` without re-submitting completed
        batches.  The post-selected counts reconstruct the density
        matrix of the success branch, whose dominant eigenvector is
        truncated to the original dimension and normalized to the
        classical solution vector; its eigenvalue is reported as the
        reconstruction fidelity.  The uncertainty is the
        self-consistency measure ``1 - lambda_max`` of the reconstructed
        density matrix, not a statistical error bar; linear inversion
        has no positivity constraint, so shot noise can push the raw
        eigenvalue above 1, and the reported fidelity and uncertainty
        are clamped into ``[0, 1]`` at the metadata construction site.
        """
        build = self.build_circuit()
        estimate = estimate_hhl_resources(self._matrix, self._vector, self._config)
        basis_strings = pauli_basis_strings(len(build.data_qubits))
        state = {"basis_strings": basis_strings, "basis_index": 0}

        def make_advance(exec_backend):
            def advance(state):
                index = state["basis_index"]
                basis = state["basis_strings"][index]
                program = _basis_measurement_circuit(build, basis)
                sample_task = exec_backend.submit_sample(program, options=options)
                counts = sample_task.result().single_counts()
                state["basis_index"] = index + 1
                done = index + 1 >= len(state["basis_strings"])
                return CompletedBackendTask([counts], task_id=sample_task.id), done

            return advance

        register_algorithm("hhl", make_advance)
        task = AlgorithmTask(
            algorithm="hhl",
            initial_state=state,
            advance=make_advance(execution_backend),
            backend=execution_backend,
        )
        while task.poll() is not TaskStatus.SUCCEEDED:
            if checkpoint_path is not None:
                task.checkpoint(checkpoint_path)
        if checkpoint_path is not None:
            task.checkpoint(checkpoint_path)
        counts_list = task.result()
        basis_counts = {
            basis: _postselect_counts(counts)
            for basis, counts in zip(basis_strings, counts_list)
        }
        density = reconstruct_density_matrix(basis_counts, len(build.data_qubits))
        fidelity, eigenvector = dominant_eigenvector(density)
        truncated = np.asarray(eigenvector)[: self._system.original_dimension]
        norm = float(np.linalg.norm(truncated))
        if norm == 0.0:
            raise RuntimeError(
                "the reconstructed HHL solution has no weight in the "
                "original unpadded registers"
            )
        metadata = _runtime_metadata(
            execution_backend, task.backend_task_ids, self._precision, build
        )
        # The uncertainty is a self-consistency measure (1 - lambda_max)
        # of the reconstructed density matrix, not a statistical error
        # bar.  Linear inversion has no positivity constraint, so shot
        # noise can push the raw eigenvalue outside [0, 1]; the reported
        # fidelity and uncertainty are clamped into the physical range.
        fidelity = min(1.0, max(0.0, fidelity))
        metadata["tomography"] = {
            "circuits": estimate.tomography_circuits,
            "shots": options.shots,
            "fidelity": fidelity,
            "uncertainty": 1.0 - fidelity,
        }
        return HHLSolution(
            classical_vector=truncated / norm,
            statevector=None,
            success_probability=_success_probability(counts_list[0]),
            metadata=metadata,
        )


def _require_sampling(execution_backend) -> None:
    """Reject backends without sampling capability before submission."""
    capabilities = getattr(execution_backend, "capabilities", None)
    if capabilities is None:
        raise DeviceCapabilityError(
            "HHL runtime execution requires a backend advertising "
            "sampling capability"
        )
    if not capabilities.sampling:
        raise DeviceCapabilityError(
            "HHL runtime execution requires a backend with sampling "
            "capability"
        )


def _measurement_plans(
    build: HHLCircuitBuild, observables: list[str] | None
) -> list[tuple[object, str | None]]:
    """Return the ``(program, pauli)`` measurement circuits of the plan.

    Without observables the plan is one circuit measuring only the
    success ancilla.  With observables each gets its own circuit
    measuring the success ancilla and the data register in the
    observable's basis; the paired positional Pauli string drives the
    expectation estimate from the post-selected counts.
    """
    if not observables:
        return [(_success_measurement_circuit(build), None)]
    plans = []
    for observable in observables:
        basis, pauli = _parse_observable(observable, len(build.data_qubits))
        program = QProg()
        program << build.program
        apply_basis_rotation(program, basis, build.data_qubits)
        program << _measure_success_and_data(build)
        plans.append((program, pauli))
    return plans


def _parse_observable(
    observable: str, data_qubits: int
) -> tuple[str, str]:
    """Parse ``"X0Y1"`` into the (basis, positional pauli) strings.

    The measurement basis over the data register measures identity
    positions in Z, and the positional Pauli string keeps the letter on
    every nontrivial data qubit so the expectation can be estimated
    from the counts.  Observables referencing a qubit outside the data
    register or using characters other than a Pauli letter plus an
    index are rejected before any submission.
    """
    if not isinstance(observable, str) or not observable:
        raise AlgorithmInputError(
            "observables must be Pauli strings over the data register, "
            "for example 'Z0' or 'X0Y1'"
        )
    basis = ["Z"] * data_qubits
    pauli = ["I"] * data_qubits
    covered = set()
    position = 0
    for match in _OBSERVABLE_TOKEN.finditer(observable):
        letter, index = match.group(1), int(match.group(2))
        if index >= data_qubits:
            raise AlgorithmInputError(
                f"observable {observable!r} references data qubit {index} "
                f"outside the {data_qubits}-qubit data register"
            )
        if index in covered:
            raise AlgorithmInputError(
                f"observable {observable!r} acts on data qubit {index} twice"
            )
        basis[index] = letter
        pauli[index] = letter
        covered.add(index)
        position = match.end()
    if position != len(observable):
        raise AlgorithmInputError(
            f"observable {observable!r} is not a Pauli string over the "
            "data register"
        )
    return "".join(basis), "".join(pauli)


def _success_measurement_circuit(build: HHLCircuitBuild):
    """The core circuit measuring only the success ancilla."""
    program = QProg()
    program << build.program
    program << measure([build.success_qubit], [0])
    return program


def _basis_measurement_circuit(build: HHLCircuitBuild, basis: str):
    """The core circuit measuring the success ancilla and the data
    register in ``basis``, with the success ancilla first."""
    program = QProg()
    program << build.program
    apply_basis_rotation(program, basis, build.data_qubits)
    program << _measure_success_and_data(build)
    return program


def _measure_success_and_data(build: HHLCircuitBuild):
    """A measure node over the success ancilla followed by the data
    register, so outcome ``key[0]`` is the success bit."""
    return measure(
        [build.success_qubit] + list(build.data_qubits),
        list(range(len(build.data_qubits) + 1)),
    )


def _success_probability(counts: dict[str, int]) -> float:
    """Weight of the success-ancilla outcomes (first bit of each key)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    successes = sum(
        count for key, count in counts.items() if key and key[0] == "1"
    )
    return min(1.0, successes / total)


def _postselect_counts(counts: dict[str, int]) -> dict[str, int]:
    """Drop the outcomes where the success ancilla read 0.

    The returned counts keep only the data bits of the success branch,
    so ``key[j]`` is the outcome of data qubit ``j``.
    """
    return {
        key[1:]: count for key, count in counts.items() if key and key[0] == "1"
    }


def _runtime_metadata(
    execution_backend, task_ids, precision: float, build: HHLCircuitBuild
) -> dict:
    """Base metadata of a sampled run, mirroring the local path."""
    return {
        "backend": type(execution_backend).__name__,
        "task_ids": list(task_ids),
        "precision": precision,
        "phase_qubits": len(build.phase_qubits),
        "evolution_time": build.evolution_time,
        "reciprocal_scale": build.reciprocal_scale,
        "eigenvalue_bounds": build.eigenvalue_bounds,
    }


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
    # Mask of every phase-qubit position, derived from the build's
    # actual register layout instead of assuming the phase register
    # occupies a fixed bit range.
    phase_mask = 0
    for qubit in build.phase_qubits:
        phase_mask |= 1 << qubit
    zero_phase = (indices & phase_mask) == 0
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
