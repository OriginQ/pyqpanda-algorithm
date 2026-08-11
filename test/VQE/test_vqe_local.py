"""Local VQE solver tests: one- and two-qubit ground states.

The solver runs on the default local CPU backend without any runtime
installation, and the found energies are compared against exact
values (analytical for the one-qubit case, full diagonalization for
the two-qubit case).
"""

import numpy as np
import pytest
from pyqpanda3.hamiltonian import Hamiltonian

from pyqpanda_alg.VQE import VQE, VQEConfig


def test_vqe_finds_z_ground_state_locally():
    solver = VQE(Hamiltonian({"Z0": 1.0}))
    result = solver.run(
        initial_parameters=np.array([0.2, 0.0]),
        config=VQEConfig(max_iterations=80, tolerance=1e-6),
    )
    assert result.converged
    assert abs(result.energy + 1.0) < 1e-5


def test_vqe_rejects_empty_hamiltonian():
    with pytest.raises(ValueError, match="at least one term"):
        VQE(Hamiltonian({}))


def test_vqe_rejects_non_hermitian_hamiltonian():
    with pytest.raises(ValueError, match="Hermitian"):
        VQE(Hamiltonian({"Z0": 1.0j}))


def test_vqe_rejects_non_finite_hamiltonian_coefficients():
    with pytest.raises(ValueError, match="finite"):
        VQE(Hamiltonian({"Z0": float("nan")}))


def test_vqe_rejects_wrong_initial_parameter_count():
    solver = VQE(Hamiltonian({"Z0": 1.0}))
    with pytest.raises(ValueError, match="initial_parameters"):
        solver.run(initial_parameters=np.array([0.2, 0.0, 0.5]))


def test_vqe_infers_qubit_count_from_highest_qubit_index():
    # The ansatz must span the full Hilbert space of the observable, so
    # {"Z1": 1.0} -- which acts only on qubit one -- infers two qubits.
    assert VQE(Hamiltonian({"Z1": 1.0}))._num_qubits == 2


class _ParamlessAnsatz:
    """Custom ansatz following the documented callable contract but
    without ``mutable_parameter_total()``, like a hand-rolled circuit
    object."""

    def __call__(self, parameters):
        raise AssertionError("the run must refuse before evaluating the ansatz")


def test_vqe_custom_ansatz_without_parameter_count_reports_value_error():
    solver = VQE(Hamiltonian({"Z0": 1.0}), ansatz=_ParamlessAnsatz())
    with pytest.raises(ValueError, match="mutable_parameter_total"):
        solver.run(initial_parameters=np.array([0.2, 0.0]))


def test_vqe_two_qubit_zz_matches_exact_diagonalization():
    hamiltonian = Hamiltonian({"Z0 Z1": 1.0, "X0": 0.5})
    solver = VQE(hamiltonian)
    result = solver.run(initial_parameters=np.array([0.1, -0.2, 0.3, 0.4]))
    exact_ground = float(
        np.linalg.eigvalsh(np.asarray(hamiltonian.pauli_operator().matrix()))[0]
    )
    assert result.converged
    assert abs(result.energy - exact_ground) < 1e-4
    assert result.optimal_parameters.shape == (4,)
    assert result.optimal_circuit is not None
    assert len(result.energy_history) == result.iterations
    assert result.task_ids
