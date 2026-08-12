"""VQE checkpoint-resume contract: the optimizer state machine round-trips.

A submitted VQE run is a resumable :class:`AlgorithmTask`: every poll
completes exactly one classical optimization iteration, the checkpoint
holds the optimizer state (never a live session), and
:meth:`AlgorithmTask.resume` rebuilds the step driver from the
registered ``vqe`` factory so the next evaluation continues at the next
iteration.  Custom optimizers that do not implement the documented
``state_dict()`` / ``load_state_dict()`` protocol are reported
explicitly at resume time.
"""

import json

import numpy as np
import pytest
from pyqpanda3.core import RY, RZ
from pyqpanda3.hamiltonian import Hamiltonian
from pyqpanda3.vqcircuit import VQCircuit

from pyqpanda_alg.VQE import VQE, VQEConfig
from pyqpanda_alg.execution import (
    AlgorithmTask,
    BackendCapabilities,
    TaskRecoveryError,
    TaskStatus,
)

from test.execution.fakes import FakeQTaskManager, RecordingBackend


def _scripted_energies(count: int) -> list:
    """Strictly decreasing energy script.

    A flat fake landscape makes SLSQP report convergence on the first
    iteration; the strictly decreasing values keep the optimizer
    stepping so the full iteration budget is consumed.
    """
    return [1.0 - 0.02 * i for i in range(count)]


class _SteppingOptimizer:
    """Canned step driver: one deterministic step per poll.

    The adapter exposes the built-in state shape (``parameters``,
    ``energy``, ``energy_history``, ``iterations``, ``max_iterations``,
    ``converged``) and VQE seeds the starting point into ``parameters``
    before the first step.
    """

    def __init__(self, *, step_limit: int) -> None:
        self.method = "canned"
        self.max_iterations = step_limit
        self.tolerance = 1e-12
        self.initial_parameters = None
        self.parameters = None
        self.energy = None
        self.energy_history: list = []
        self.task_ids: list = []
        self.iterations = 0
        self.converged = False

    def step(self, evaluate) -> bool:
        self.parameters = np.asarray(self.parameters, dtype=float).copy() * 0.9
        self.energy = float(evaluate(self.parameters))
        self.energy_history.append(self.energy)
        self.iterations += 1
        self.converged = self.iterations >= self.max_iterations
        return self.converged


class _NoProtocolOptimizer(_SteppingOptimizer):
    """Custom optimizer without the documented resume protocol."""


class _ResumableOptimizer(_SteppingOptimizer):
    """Custom optimizer implementing the documented resume protocol."""

    def state_dict(self) -> dict:
        return {
            "method": self.method,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "initial_parameters": (
                None
                if self.initial_parameters is None
                else self.initial_parameters.tolist()
            ),
            "parameters": (
                None if self.parameters is None else self.parameters.tolist()
            ),
            "energy": self.energy,
            "energy_history": [float(e) for e in self.energy_history],
            "task_ids": list(self.task_ids),
            "iterations": self.iterations,
            "converged": self.converged,
        }

    def load_state_dict(self, state: dict) -> None:
        self.method = state["method"]
        self.max_iterations = state["max_iterations"]
        self.tolerance = state["tolerance"]
        self.initial_parameters = _as_parameters(state["initial_parameters"])
        self.parameters = _as_parameters(state["parameters"])
        self.energy = state["energy"]
        self.energy_history = [float(e) for e in state["energy_history"]]
        self.task_ids = list(state["task_ids"])
        self.iterations = state["iterations"]
        self.converged = bool(state["converged"])


def _as_parameters(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float)


@pytest.mark.runtime_contract
def test_vqe_checkpoint_resume_continues_at_iteration_two(
    runtime_backend, hamiltonian, tmp_path
):
    """A session-path run resumes at iteration two with a fresh session."""
    runtime_backend.capabilities = BackendCapabilities(variational_session=True)
    script = _scripted_energies(60)
    run_index = 0
    sessions = []
    original_vqsession = runtime_backend.service.vqsession

    def scripted_vqsession(vqcircuit, device, shots=1000, life_time=360, observable=None):
        session = original_vqsession(vqcircuit, device, shots, life_time, observable)
        sessions.append(session)

        def run_vqtask(gate_params, measure_list=None):
            nonlocal run_index
            value = script[min(run_index, len(script) - 1)]
            run_index += 1
            return FakeQTaskManager([value], kind="estimate", finished=True)

        session.run_vqtask = run_vqtask
        return session

    runtime_backend.service.vqsession = scripted_vqsession

    solver = VQE(hamiltonian)
    task = solver.submit(
        initial_parameters=[0.1, 0.2],
        backend=runtime_backend,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    assert task.poll() is TaskStatus.RUNNING  # iteration one complete
    path = task.checkpoint(tmp_path / "vqe.json")
    assert task.status() is TaskStatus.RUNNING

    # the checkpoint holds the optimizer state and solver fingerprint
    # only -- never a live session
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["state"]) == {"optimizer", "resume_supported", "fingerprint"}
    # Assert on the fingerprint's own stable fields instead of pinning
    # pyqpanda3's Hamiltonian repr, which is not part of this package's
    # contract.
    fingerprint = payload["state"]["fingerprint"]
    assert set(fingerprint) == {"hamiltonian", "ansatz_parameters", "ansatz"}
    assert "Z0" in fingerprint["hamiltonian"]
    assert fingerprint["ansatz_parameters"] == 2
    assert "RY" in fingerprint["ansatz"] and "RZ" in fingerprint["ansatz"]
    optimizer_state = payload["state"]["optimizer"]
    assert set(optimizer_state) == {
        "method",
        "max_iterations",
        "tolerance",
        "initial_parameters",
        "parameters",
        "energy",
        "energy_history",
        "parameter_history",
        "task_ids",
        "iterations",
        "converged",
    }
    assert optimizer_state["iterations"] == 1
    assert len(optimizer_state["energy_history"]) == 1
    assert optimizer_state["task_ids"]
    assert payload["state"]["resume_supported"] is True
    assert "session" not in path.read_text(encoding="utf-8").lower()

    assert len(sessions) == 1  # exactly one session for the original run
    resumed = AlgorithmTask.resume(path, backend=runtime_backend)
    assert resumed.status() is TaskStatus.RUNNING
    assert resumed.poll() is TaskStatus.RUNNING  # iteration two
    assert resumed.poll() is TaskStatus.SUCCEEDED  # iteration three

    result = resumed.result()
    assert result.iterations == 3
    assert len(result.energy_history) == 3
    assert result.metadata["resume_supported"] is True

    # the resumed run opened a fresh session -- the checkpoint never
    # carried the live one -- and released it when the run completed
    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert sessions[1].release_calls == 1
    assert run_index < len(script)  # the scripted values were never exhausted


@pytest.mark.runtime_contract
def test_vqe_checkpoint_resume_estimate_path_continues(tmp_path):
    """An estimate-path run resumes at iteration two without replaying."""
    backend = RecordingBackend(expectations=_scripted_energies(60))
    backend.capabilities = BackendCapabilities(variational_session=False)
    solver = VQE(Hamiltonian({"Z0": 1.0}))
    task = solver.submit(
        initial_parameters=[0.1, 0.2],
        backend=backend,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    assert task.poll() is TaskStatus.RUNNING  # iteration one
    path = task.checkpoint(tmp_path / "vqe.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"]["optimizer"]["iterations"] == 1
    original_ids = list(payload["backend_task_ids"])

    resumed = AlgorithmTask.resume(path, backend=backend)
    assert resumed.poll() is TaskStatus.RUNNING  # iteration two
    assert resumed.poll() is TaskStatus.SUCCEEDED  # iteration three

    result = resumed.result()
    assert result.iterations == 3
    assert len(result.energy_history) == 3
    # iteration one was never replayed: its task id appears exactly once
    assert resumed.backend_task_ids[0] == original_ids[0]
    assert len(set(resumed.backend_task_ids)) == len(resumed.backend_task_ids)


@pytest.mark.runtime_contract
def test_vqe_custom_optimizer_without_protocol_reports_on_resume(
    estimator_only_backend, hamiltonian, tmp_path
):
    solver = VQE(hamiltonian, optimizer=_NoProtocolOptimizer(step_limit=3))
    task = solver.submit(
        initial_parameters=[0.1, 0.2],
        backend=estimator_only_backend,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    assert task.poll() is TaskStatus.RUNNING  # iteration one
    path = task.checkpoint(tmp_path / "vqe.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"]["optimizer"] is None
    assert payload["state"]["resume_supported"] is False

    with pytest.raises(TaskRecoveryError, match="resume"):
        AlgorithmTask.resume(path, backend=estimator_only_backend)

    # the interrupted run itself is unaffected and finishes normally,
    # with the limitation documented in the result metadata
    assert task.poll() is TaskStatus.RUNNING  # iteration two
    assert task.poll() is TaskStatus.SUCCEEDED  # iteration three
    result = task.result()
    assert result.iterations == 3
    assert result.metadata["resume_supported"] is False


@pytest.mark.runtime_contract
def test_vqe_custom_optimizer_with_protocol_round_trips(
    estimator_only_backend, hamiltonian, tmp_path
):
    solver = VQE(hamiltonian, optimizer=_ResumableOptimizer(step_limit=3))
    task = solver.submit(
        initial_parameters=[0.1, 0.2],
        backend=estimator_only_backend,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    assert task.poll() is TaskStatus.RUNNING  # iteration one
    path = task.checkpoint(tmp_path / "vqe.json")

    resumed = AlgorithmTask.resume(path, backend=estimator_only_backend)
    assert resumed.poll() is TaskStatus.RUNNING  # iteration two
    assert resumed.poll() is TaskStatus.SUCCEEDED  # iteration three

    result = resumed.result()
    assert result.iterations == 3
    assert len(result.energy_history) == 3
    assert result.metadata["resume_supported"] is True


def _ryrz_ansatz() -> VQCircuit:
    """One-qubit RY-then-RZ ansatz with two parameters."""
    ansatz = VQCircuit(1)
    ansatz.set_Param([2])
    ansatz << RY(0, ansatz.Param([0]))
    ansatz << RZ(0, ansatz.Param([1]))
    return ansatz


def _ryry_ansatz() -> VQCircuit:
    """One-qubit RY-then-RY ansatz with the same parameter count."""
    ansatz = VQCircuit(1)
    ansatz.set_Param([2])
    ansatz << RY(0, ansatz.Param([0]))
    ansatz << RY(0, ansatz.Param([1]))
    return ansatz


def _fingerprint_of(solver, backend, tmp_path) -> dict:
    task = solver.submit(
        initial_parameters=[0.1, 0.2],
        backend=backend,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    path = task.checkpoint(tmp_path / "vqe.json")
    return json.loads(path.read_text(encoding="utf-8"))["state"]["fingerprint"]


@pytest.mark.runtime_contract
def test_vqe_fingerprint_distinguishes_same_parameter_count_ansatzes(
    estimator_only_backend, tmp_path
):
    """Two ansatzes with equal parameter counts but different structure
    must not share a solver fingerprint.

    Without the ansatz circuit descriptor in the fingerprint, a
    checkpoint could be resumed through a structurally different solver
    optimizing the same observable with the same parameter count.
    """
    fingerprint_ryrz = _fingerprint_of(
        VQE(Hamiltonian({"Z0": 1.0}), ansatz=_ryrz_ansatz()),
        estimator_only_backend,
        tmp_path,
    )
    fingerprint_ryry = _fingerprint_of(
        VQE(Hamiltonian({"Z0": 1.0}), ansatz=_ryry_ansatz()),
        estimator_only_backend,
        tmp_path,
    )
    assert fingerprint_ryrz["ansatz_parameters"] == 2
    assert fingerprint_ryrz["ansatz_parameters"] == fingerprint_ryry["ansatz_parameters"]
    assert fingerprint_ryrz["ansatz"] != fingerprint_ryry["ansatz"]
    assert fingerprint_ryrz != fingerprint_ryry


@pytest.mark.runtime_contract
def test_vqe_resume_refuses_a_different_solver(estimator_only_backend, tmp_path):
    """A checkpoint resumed through a solver for another observable is refused.

    The resume factory is rebuilt from the most recent submit, so
    without a solver fingerprint check the resumed run would silently
    optimize the wrong observable.
    """
    solver_a = VQE(Hamiltonian({"Z0": 1.0}))
    backend_a = RecordingBackend(expectations=_scripted_energies(60))
    backend_a.capabilities = BackendCapabilities(variational_session=False)
    task_a = solver_a.submit(
        initial_parameters=[0.1, 0.2],
        backend=backend_a,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    assert task_a.poll() is TaskStatus.RUNNING
    path = task_a.checkpoint(tmp_path / "vqe.json")

    # a later submit on another solver overwrites the registered factory
    solver_b = VQE(Hamiltonian({"X0": 0.5}))
    task_b = solver_b.submit(
        initial_parameters=[0.1, 0.2],
        backend=estimator_only_backend,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    task_b.result()

    resumed = AlgorithmTask.resume(path, backend=backend_a)
    with pytest.raises(TaskRecoveryError, match="different solver"):
        resumed.poll()

    resumed = AlgorithmTask.resume(path, backend=estimator_only_backend)
    with pytest.raises(TaskRecoveryError, match="different solver"):
        resumed.poll()


@pytest.mark.runtime_contract
def test_vqe_custom_optimizer_reset_between_runs(estimator_only_backend, hamiltonian):
    """A reused custom adapter instance starts a fresh run, not the last one.

    The adapter is shared across runs of the same solver, so submit must
    reset its run state; otherwise the second run would finish on the
    first poll with the first run's result.
    """
    solver = VQE(hamiltonian, optimizer=_NoProtocolOptimizer(step_limit=3))
    first = solver.run(
        initial_parameters=[0.1, 0.2],
        backend=estimator_only_backend,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    assert first.iterations == 3

    second = solver.submit(
        initial_parameters=[0.5, 0.6],
        backend=estimator_only_backend,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    assert second.poll() is TaskStatus.RUNNING  # a fresh run steps again
    assert second.poll() is TaskStatus.RUNNING
    assert second.poll() is TaskStatus.SUCCEEDED
    assert second.result().iterations == 3


@pytest.mark.runtime_contract
def test_vqe_custom_optimizer_one_shot_minimize_contract(
    estimator_only_backend, hamiltonian
):
    """A minimize-only adapter works through run(); it is not resumable."""

    class _OneShotOptimizer:
        def __init__(self) -> None:
            self.method = "canned-one-shot"
            self.energy = None
            self.parameters = None
            self.energy_history: list = []
            self.task_ids: list = []
            self.iterations = 0
            self.converged = False

        def minimize(self, evaluate, initial_parameters) -> None:
            parameters = np.asarray(initial_parameters, dtype=float).copy()
            for _ in range(2):
                self.parameters = parameters
                self.energy = float(evaluate(parameters))
                self.energy_history.append(self.energy)
                self.iterations += 1
                parameters = parameters * 0.5
            self.converged = True

    solver = VQE(hamiltonian, optimizer=_OneShotOptimizer())
    result = solver.run(
        initial_parameters=[0.1, 0.2],
        backend=estimator_only_backend,
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    assert result.iterations == 2
    assert result.converged is True
    assert result.metadata["resume_supported"] is False
