import pytest
import sys
from pathlib import Path

# Add project root to path (works from any working directory)
_project_root = Path(__file__).resolve().parent.parent.parent
_pkg_root = _project_root / "pyqpanda-algorithm"
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_pkg_root))

import numpy as np
from pyqpanda3.hamiltonian import PauliOperator
from pyqpanda_alg.VQE import VQE


class TestVQE:
    """Tests for the VQE algorithm."""

    def test_initialization(self):
        """Test that VQE object can be created."""
        H = PauliOperator({"Z0": 1.0})
        v = VQE(H, n_qubits=1, ansatz_depth=1)
        assert v.n_qubits == 1
        assert v.n_params == 2  # 2 * 1 * 1

    def test_initialization_multiqubit(self):
        """Test initialization with multi-qubit Hamiltonian."""
        H = (PauliOperator({"Z0": 1.0})
             + PauliOperator({"Z1": 1.0})
             + PauliOperator({"Z0 Z1": 0.5}))
        v = VQE(H, ansatz_depth=2)
        assert v.n_qubits == 2
        assert v.n_params == 8  # 2 * 2 * 2

    def test_calculate_energy_single_qubit(self):
        """Test calculate_energy for a single-qubit Hamiltonian."""
        # H = Z (eigenvalues: |0⟩→+1, |1⟩→-1)
        H = PauliOperator({"Z0": 1.0})
        v = VQE(H, n_qubits=1, ansatz_depth=1)

        # params = [0, 0] → RY(0)RZ(0) = Identity → |0⟩ → E = +1
        energy = v.calculate_energy([0.0, 0.0], shots=-1)
        assert abs(energy - 1.0) < 1e-10, f"Expected +1.0, got {energy}"

    def test_calculate_energy_two_qubit(self):
        """Test calculate_energy for a two-qubit Hamiltonian."""
        # H = Z0 + Z1: |00⟩→+2, |11⟩→-2 (ground)
        H = PauliOperator({"Z0": 1.0}) + PauliOperator({"Z1": 1.0})
        v = VQE(H, n_qubits=2, ansatz_depth=1)

        # Identity circuit → |00⟩ → E = +2
        energy = v.calculate_energy([0.0, 0.0, 0.0, 0.0], shots=-1)
        assert abs(energy - 2.0) < 1e-10, f"Expected +2.0, got {energy}"

    def test_run_single_qubit(self):
        """Test that run() converges for a simple Hamiltonian."""
        # H = Z, ground = |1⟩, energy = -1
        H = PauliOperator({"Z0": 1.0})
        v = VQE(H, n_qubits=1, ansatz_depth=2)

        energy = v.run(optimizer='SLSQP', maxiter=100)
        assert energy < -0.9, f"Energy {energy} not close to ground state -1"

    def test_run_two_qubit(self):
        """Test that run() converges for a two-qubit Hamiltonian."""
        # H = Z0 + Z1, ground = |11⟩, energy = -2
        H = PauliOperator({"Z0": 1.0}) + PauliOperator({"Z1": 1.0})
        v = VQE(H, n_qubits=2, ansatz_depth=2)

        energy = v.run(optimizer='SLSQP', maxiter=100)
        assert energy < -1.8, f"Energy {energy} not close to ground state -2"

    def test_energy_history(self):
        """Test that energy history is recorded."""
        H = PauliOperator({"Z0": 1.0})
        v = VQE(H, n_qubits=1, ansatz_depth=1)

        v.run(optimizer='SLSQP', maxiter=50)
        assert len(v.energy_history) > 0, "Energy history should not be empty"
        assert len(v.energy_history) > 1, "Should have more than one energy value"

    def test_optimal_params_stored(self):
        """Test that optimal parameters are stored after run."""
        H = PauliOperator({"Z0": 1.0})
        v = VQE(H, n_qubits=1, ansatz_depth=2)

        v.run(optimizer='SLSQP', maxiter=50)
        assert v.optimal_params is not None
        assert len(v.optimal_params) == v.n_params

    def test_n_qubits_inferred(self):
        """Test that n_qubits is correctly inferred from Hamiltonian."""
        H = PauliOperator({"Z3": 1.0})
        v = VQE(H, ansatz_depth=1)
        assert v.n_qubits == 4, f"Expected 4 qubits, got {v.n_qubits}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
