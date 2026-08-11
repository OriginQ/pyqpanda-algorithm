"""Hamiltonian-level variational quantum eigensolver with resumable execution.

:class:`VQE` validates the observable (non-empty, Hermitian, finite
coefficients), infers the qubit count from its Pauli operator, and
builds the built-in hardware-efficient ansatz — or accepts a custom one
following the same callable contract.  :meth:`VQE.submit` submits the
optimization as a resumable :class:`~pyqpanda_alg.execution.AlgorithmTask`:
every poll completes exactly one classical optimization iteration, and
:meth:`VQE.run` is the convenience wrapper ``submit().result()``.  Each
energy evaluation materializes the concrete circuit and prefers a
variational session whenever the backend advertises one, falling back to
estimate batches otherwise; the default CPU path needs no runtime
installation.  The completed run is snapshotted into the immutable
:class:`VQEResult`, and a running task can be checkpointed and resumed
with :meth:`~pyqpanda_alg.execution.AlgorithmTask.resume`.
"""

import dataclasses
from typing import Any, Callable, Optional

import numpy as np
from pyqpanda3.core import QProg, RY, RZ
from pyqpanda3.hamiltonian import Hamiltonian
from pyqpanda3.vqcircuit import VQCircuit

from ..execution import (
    AlgorithmTask,
    CompletedBackendTask,
    ExecutionOptions,
    TaskRecoveryError,
    register_algorithm,
    resolve_backend,
)
from .ansatz import hardware_efficient_ansatz
from .config import VQEConfig
from .optimizer import SciPyOptimizer
from .result import VQEResult


class VQE:
    """Variational quantum eigensolver for a pyqpanda3 :class:`Hamiltonian`."""

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        ansatz: Optional[Any] = None,
        layers: int = 1,
        optimizer: Optional[Any] = None,
    ) -> None:
        """Validate the observable and prepare the ansatz.

        Args:
            hamiltonian: The observable whose ground state is sought.
                Must be non-empty, Hermitian (real coefficients in the
                Pauli basis) and have finite coefficients; the qubit
                count is inferred from the Pauli operator.
            ansatz: Optional variational circuit following the callable
                contract ``ansatz(parameters) -> .circuits()[0] ->
                QProg``.  Defaults to the built-in hardware-efficient
                ansatz, which for a single qubit reduces to the RY/RZ
                layers without the CNOT entanglement chain.
            layers: Number of rotation-and-entanglement repetitions of
                the default ansatz.
            optimizer: Optional custom optimizer adapter replacing the
                built-in SciPy adapter.  The adapter must expose
                ``minimize(evaluate, initial_parameters)``; to be
                resumable it additionally implements ``step(evaluate)``
                (one classical iteration), the built-in state
                attributes (``method``, ``parameters``, ``energy``,
                ``energy_history``, ``iterations``, ``max_iterations``,
                ``converged``), and the ``state_dict()`` /
                ``load_state_dict()`` protocol.  Without the protocol
                the run works but cannot be resumed, which is reported
                explicitly (see :meth:`submit`).

        Raises:
            TypeError: If ``hamiltonian`` is not a pyqpanda3
                :class:`Hamiltonian`, or ``optimizer`` does not expose
                ``minimize`` or ``step``.
            ValueError: If the observable is empty, acts on no qubits,
                is non-Hermitian or has non-finite coefficients, or if
                ``layers`` is smaller than 1.
        """
        num_qubits, hamiltonian = _validate_hamiltonian(hamiltonian)
        if optimizer is not None and not (
            callable(getattr(optimizer, "minimize", None))
            or callable(getattr(optimizer, "step", None))
        ):
            raise TypeError(
                "optimizer must expose minimize(evaluate, initial_parameters) "
                "or step(evaluate), "
                f"got {type(optimizer).__name__}"
            )
        self._hamiltonian = hamiltonian
        self._num_qubits = num_qubits
        self._ansatz = (
            ansatz if ansatz is not None else _default_ansatz(num_qubits, layers)
        )
        self._optimizer = optimizer

    def run(
        self,
        initial_parameters,
        *,
        backend: Optional[Any] = None,
        execution_options: Optional[ExecutionOptions] = None,
        config: Optional[VQEConfig] = None,
    ) -> VQEResult:
        """Minimize the energy expectation and return the immutable result.

        Equivalent to :meth:`submit` followed by ``result()``: the run
        is a resumable algorithm task, and this convenience wrapper
        blocks until it succeeds.

        Args:
            initial_parameters: One-dimensional starting parameters,
                matching the ansatz parameter count.
            backend: The execution backend to submit estimations to.
                Keyword-only.  When None, the local CPU backend is used.
            execution_options: Submission options for the backend.
                Keyword-only.  When None, defaults are used with the
                sampling budget taken from ``config.shots``.
            config: Run configuration.  Keyword-only.  When None,
                :class:`VQEConfig` defaults apply.

        Returns:
            The frozen :class:`VQEResult` snapshot of the run: energy
            and optimal parameters at the found optimum, convergence
            bookkeeping, the per-evaluation energy history, the
            materialized circuit at the optimum, and the task IDs and
            metadata of the execution.

        Raises:
            ValueError: If ``initial_parameters`` is not one-dimensional
                or its length does not match the ansatz parameter count.
            TaskSubmissionError: If a backend submission fails — the
                runtime failure is surfaced, never silently retried on
                the CPU backend.
            TaskTimeoutError: If ``result()`` exceeds the backend
                options' timeout.
        """
        return self.submit(
            initial_parameters,
            backend=backend,
            execution_options=execution_options,
            config=config,
        ).result()

    def submit(
        self,
        initial_parameters,
        *,
        backend: Optional[Any] = None,
        execution_options: Optional[ExecutionOptions] = None,
        config: Optional[VQEConfig] = None,
    ) -> AlgorithmTask:
        """Submit the optimization as a resumable algorithm task.

        The returned task is the optimization state machine: each
        ``poll()`` completes exactly one classical optimization
        iteration — the optimizer probes the energy landscape (each
        probe submitting to the backend and recording its task ID) and
        one recorded evaluation at the returned point closes the
        iteration.  ``result()`` blocks until the iteration budget is
        exhausted or the optimizer reports convergence.

        The backend chooses the execution path: when
        ``backend.capabilities.variational_session`` is true, one
        variational session is created (and entered) lazily on the first
        evaluation and reused for the whole run, then released
        (idempotently) when the run finishes or fails; otherwise every
        evaluation submits an estimate batch.  A runtime submission
        failure surfaces as the execution layer's public error — there
        is never a silent fallback to the CPU backend.

        The task can be checkpointed while running
        (``task.checkpoint(path)``) and reconstructed with
        :meth:`AlgorithmTask.resume`: the checkpoint holds the optimizer
        state (method, parameters, energy history, iteration,
        convergence criteria, task IDs), the resume-support flag, and a
        solver fingerprint — never a live session or credentials.
        Checkpoint a task while it is RUNNING; a completed task's
        immutable result is not JSON-serializable and the checkpoint
        layer rejects it loudly.  A session abandoned by dropping a
        checkpointed task without resuming it expires with the session
        lifetime, like any session whose client disappears.

        Resuming validates the checkpoint against this solver: the
        registered factory is rebuilt from the most recent ``submit``,
        so a checkpoint created by a solver with a different observable
        or ansatz raises
        :class:`~pyqpanda_alg.execution.errors.TaskRecoveryError`
        instead of silently optimizing the wrong problem.

        A custom optimizer (see :meth:`__init__`) is resumable only when
        it implements the ``state_dict()`` / ``load_state_dict()``
        protocol; otherwise ``resume()`` raises
        :class:`~pyqpanda_alg.execution.errors.TaskRecoveryError` with an
        explicit explanation, and the limitation is flagged in
        ``VQEResult.metadata["resume_supported"]``.

        Args:
            initial_parameters: One-dimensional starting parameters,
                matching the ansatz parameter count.
            backend: The execution backend to submit estimations to.
                Keyword-only.  When None, the local CPU backend is used.
            execution_options: Submission options for the backend.
                Keyword-only.  When None, defaults are used with the
                sampling budget taken from ``config.shots``.
            config: Run configuration.  Keyword-only.  When None,
                :class:`VQEConfig` defaults apply.

        Returns:
            The resumable :class:`~pyqpanda_alg.execution.AlgorithmTask`
            whose polls drive the optimization.

        Raises:
            ValueError: If ``initial_parameters`` is not one-dimensional
                or its length does not match the ansatz parameter count.
            TaskSubmissionError: If a backend submission fails; the
                runtime failure is surfaced, never silently retried on
                the CPU backend.
            TaskRecoveryError: If a resumed checkpoint does not match
                this solver, or belongs to a non-resumable optimizer.
            AlgorithmInputError: If a completed task is checkpointed —
                its immutable result is not JSON-serializable.
        """
        cfg = config if config is not None else VQEConfig()
        execution_backend = resolve_backend(backend)
        options = dataclasses.replace(
            execution_options if execution_options is not None else ExecutionOptions(),
            shots=cfg.shots,
        )
        initial = np.asarray(initial_parameters, dtype=float)
        if initial.ndim != 1:
            raise ValueError(
                f"initial_parameters must be one-dimensional, got shape {initial.shape}"
            )
        expected = self._ansatz.mutable_parameter_total()
        if initial.shape[0] != expected:
            raise ValueError(
                f"initial_parameters has {initial.shape[0]} entries, "
                f"the ansatz expects {expected}"
            )

        optimizer = (
            self._optimizer
            if self._optimizer is not None
            else SciPyOptimizer(
                cfg.optimizer,
                max_iterations=cfg.max_iterations,
                tolerance=cfg.tolerance,
            )
        )
        _reset_adapter(optimizer, initial)

        resume_supported = getattr(optimizer, "resume_supported", None)
        if resume_supported is None:
            resume_supported = hasattr(optimizer, "state_dict") and hasattr(
                optimizer, "load_state_dict"
            )
        resume_supported = bool(resume_supported)
        initial_run = initial.copy()
        state = {
            "optimizer": optimizer.state_dict() if resume_supported else None,
            "resume_supported": resume_supported,
            "fingerprint": _solver_fingerprint(self),
        }

        def resume_factory(exec_backend: Any) -> Callable[[dict], tuple]:
            """Rebuild the advance for a resumed run, or refuse explicitly.

            This factory is registered for :meth:`AlgorithmTask.resume`:
            a checkpoint from a non-resumable optimizer is reported
            loudly at recovery time, never silently continued with a
            different optimizer.
            """
            if not resume_supported:
                raise TaskRecoveryError(
                    "the vqe checkpoint was created by an optimizer that does "
                    "not implement the documented state_dict()/load_state_dict() "
                    "resume protocol, so the run cannot be resumed; resubmit "
                    "the optimization with a resumable optimizer"
                )
            return self._make_advance(
                exec_backend, optimizer, options, resume_supported, initial_run
            )

        register_algorithm("vqe", resume_factory)
        return AlgorithmTask(
            "vqe",
            initial_state=state,
            advance=self._make_advance(
                execution_backend, optimizer, options, resume_supported, initial_run
            ),
            backend=execution_backend,
        )

    def _make_advance(
        self,
        execution_backend: Any,
        optimizer: Any,
        options: ExecutionOptions,
        resume_supported: bool,
        initial_run: np.ndarray,
    ) -> Callable[[dict], tuple]:
        """Build the one-iteration-per-poll advance callback for submit.

        The advance restores the optimizer adapter from the task state
        at the start of every poll, so a resumed run continues exactly
        where the checkpoint stopped; the live variational session (if
        any) lives in this closure only and never enters the state, so
        checkpoints never carry it.  The session is entered when it is
        created (the runtime session surface requires an entered
        session) and released idempotently when the run finishes or a
        step raises.  Every poll verifies the state's solver
        fingerprint against this solver, so resuming a checkpoint
        through a solver that optimizes a different problem raises
        :class:`TaskRecoveryError` instead of silently computing the
        wrong energy.
        """
        use_session = bool(
            getattr(
                getattr(execution_backend, "capabilities", None),
                "variational_session",
                False,
            )
        )
        session = None
        last_task_id = ""

        def evaluate(adapter, parameters: np.ndarray) -> float:
            nonlocal session
            if use_session:
                if session is None:
                    session = execution_backend.create_variational_session(
                        self._ansatz, self._hamiltonian, options=options
                    )
                    session.__enter__()
                task = session.run(parameters)
                energy = float(task.result())
            else:
                prog = QProg() << self._ansatz(parameters).circuits()[0]
                task = execution_backend.submit_estimate(
                    (prog, self._hamiltonian), options=options
                )
                energy = float(task.result().single_value())
            task_id = getattr(task, "id", "")
            if task_id:
                adapter.task_ids.append(task_id)
            return energy

        def advance(state: dict) -> tuple:
            nonlocal session, last_task_id
            try:
                if state.get("fingerprint") != _solver_fingerprint(self):
                    raise TaskRecoveryError(
                        "the vqe checkpoint was created by a different solver "
                        "(the observable or the ansatz differ); resuming it "
                        "through this solver would optimize the wrong problem"
                    )
                opt_state = state["optimizer"]
                adapter = (
                    _restore_optimizer(optimizer, opt_state)
                    if opt_state is not None
                    else optimizer
                )
                if callable(getattr(adapter, "step", None)):
                    if (
                        adapter.iterations >= adapter.max_iterations
                        or adapter.converged
                    ):
                        finished = True
                    else:
                        adapter.step(lambda p: evaluate(adapter, p))
                        _sync_optimizer_state(state, adapter)
                        finished = (
                            adapter.iterations >= adapter.max_iterations
                            or adapter.converged
                        )
                else:
                    adapter.minimize(
                        lambda p: evaluate(adapter, p), initial_run
                    )
                    _sync_optimizer_state(state, adapter)
                    finished = True
                if adapter.task_ids:
                    last_task_id = adapter.task_ids[-1]
                if finished:
                    if session is not None:
                        session.release()
                    return CompletedBackendTask(
                        _build_result(adapter, execution_backend, resume_supported, self),
                        task_id=last_task_id,
                    ), True
                return CompletedBackendTask(None, task_id=last_task_id), False
            except Exception:
                if session is not None:
                    session.release()
                raise

        return advance


def _restore_optimizer(adapter: Any, state: dict) -> Any:
    """Restore the adapter's explicit state from the task state.

    The adapter is either the solver's live instance (a resumable custom
    optimizer) or the submit-time built-in; ``load_state_dict`` fully
    overwrites the optimization-relevant attributes, so the restored
    adapter continues exactly where the checkpoint stopped.
    """
    adapter.load_state_dict(state)
    return adapter


def _sync_optimizer_state(state: dict, adapter: Any) -> None:
    """Persist the adapter's explicit state into the task state."""
    if hasattr(adapter, "state_dict"):
        state["optimizer"] = adapter.state_dict()


def _reset_adapter(optimizer: Any, initial: np.ndarray) -> None:
    """Reset the adapter for a fresh run and seed the starting point.

    The built-in adapter is constructed fresh per submit, but a custom
    adapter instance is shared across runs of the same solver, so every
    submit resets the run-relevant attributes before the first poll;
    step-capable adapters then start from ``parameters``.  One adapter
    instance therefore supports one active task at a time.
    """
    for attr, value in (
        ("initial_parameters", initial.copy()),
        ("parameters", initial.copy()),
        ("energy", None),
        ("energy_history", []),
        ("parameter_history", []),
        ("task_ids", []),
        ("iterations", 0),
        ("converged", False),
    ):
        if hasattr(optimizer, attr):
            setattr(optimizer, attr, value)


def _solver_fingerprint(solver: "VQE") -> dict:
    """Stable JSON-safe identity of the problem the solver optimizes.

    The canonical Pauli string covers the observable (terms and
    coefficients) and the qubit count; the ansatz parameter count pins
    the variational circuit shape.  The advance verifies the task
    state's fingerprint on every poll, so resuming a checkpoint through
    a solver that optimizes a different problem is refused instead of
    silently computing the wrong energy.
    """
    return {
        "hamiltonian": str(solver._hamiltonian.pauli_operator()),
        "ansatz_parameters": solver._ansatz.mutable_parameter_total(),
    }


def _build_result(adapter, execution_backend, resume_supported, solver) -> VQEResult:
    """Snapshot the finished run into the immutable :class:`VQEResult`."""
    parameters = adapter.parameters
    return VQEResult(
        energy=adapter.energy,
        optimal_parameters=parameters,
        converged=adapter.converged,
        iterations=adapter.iterations,
        energy_history=tuple(adapter.energy_history),
        optimal_circuit=QProg() << solver._ansatz(parameters).circuits()[0],
        task_ids=tuple(adapter.task_ids),
        metadata={
            "optimizer": adapter.method,
            "backend": type(execution_backend).__name__,
            "resume_supported": resume_supported,
        },
    )


def _validate_hamiltonian(hamiltonian: Hamiltonian) -> tuple[int, Hamiltonian]:
    """Return ``(num_qubits, hamiltonian)``, rejecting invalid observables.

    The qubit count is inferred from ``pauli_operator().qubits()`` as
    the highest qubit index plus one, so the ansatz spans the full
    Hilbert space of the observable.
    """
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError(
            f"hamiltonian must be a pyqpanda3 Hamiltonian, got {type(hamiltonian).__name__}"
        )
    pauli = hamiltonian.pauli_operator()
    terms = pauli.terms()
    if not terms:
        raise ValueError("hamiltonian must contain at least one term")
    identity_qubits, active_qubits = pauli.qubits()
    qubits = identity_qubits + active_qubits
    if not qubits:
        raise ValueError("hamiltonian must act on at least one qubit")
    for term in terms:
        coef = term.coef()
        if not (np.isfinite(coef.real) and np.isfinite(coef.imag)):
            raise ValueError("hamiltonian coefficients must be finite")
        if coef.imag != 0:
            raise ValueError(
                "hamiltonian must be Hermitian: every coefficient in the "
                "Pauli basis must be real"
            )
    return max(qubits) + 1, hamiltonian


def _default_ansatz(num_qubits: int, layers: int) -> VQCircuit:
    """Built-in hardware-efficient ansatz, CNOT-free for a single qubit.

    Two or more qubits delegate to
    :func:`~pyqpanda_alg.VQE.ansatz.hardware_efficient_ansatz`.  A
    single qubit has no entanglement partner, so the default reduces to
    the same layer-major, qubit-major RY-then-RZ parameter ordering
    without the CNOT chain.
    """
    if num_qubits >= 2:
        return hardware_efficient_ansatz(num_qubits, layers)
    if layers < 1:
        raise ValueError(f"layers must be at least 1, got {layers}")
    ansatz = VQCircuit(1)
    ansatz.set_Param([2 * layers])
    index = 0
    for _ in range(layers):
        ansatz << RY(0, ansatz.Param([index]))
        index += 1
        ansatz << RZ(0, ansatz.Param([index]))
        index += 1
    return ansatz
