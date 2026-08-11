"""Local VQE solver tests: one- and two-qubit ground states.

The solver runs on the default local CPU backend without any runtime
installation, and the found energies are compared against exact
values (analytical for the one-qubit case, full diagonalization for
the two-qubit case).
"""

import numpy as np
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
