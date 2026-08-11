"""Pure-function tests for HHL data-register tomography.

Covers the 3**data_qubits X/Y/Z basis strings, the single-qubit basis
rotations used to measure them, the parity expectation estimator, and
full density-matrix reconstruction from scripted basis counts — the
dominant eigenvector, one- and two-qubit known states, and the
fidelity/uncertainty report.
"""

import numpy as np
import pytest
from pyqpanda3.core import CPUQVM, H, P, QProg, measure

from pyqpanda_alg.HHL.tomography import (
    apply_basis_rotation,
    dominant_eigenvector,
    pauli_basis_strings,
    pauli_expectation,
    reconstruct_density_matrix,
)


def normalized(vector):
    return vector / np.linalg.norm(vector)


def test_pauli_basis_strings_cover_every_xyz_string():
    assert pauli_basis_strings(1) == ["X", "Y", "Z"]
    strings = pauli_basis_strings(2)
    assert len(strings) == 9
    assert set(strings) == {a + b for a in "XYZ" for b in "XYZ"}
    assert all(
        len(string) == 3 and set(string) <= set("XYZ")
        for string in pauli_basis_strings(3)
    )


def test_basis_rotation_measures_in_the_requested_pauli_basis():
    qvm = CPUQVM()
    # |+x> measured in X returns 0 with certainty.
    program = QProg(1) << H(0)
    apply_basis_rotation(program, "X", (0,))
    program << measure([0], [0])
    qvm.run(program, shots=1000)
    assert qvm.result().get_prob_dict() == {"0": 1.0}
    # |+y> measured in Y returns 0 with certainty.
    program = QProg(1) << H(0) << P(0, np.pi / 2)
    apply_basis_rotation(program, "Y", (0,))
    program << measure([0], [0])
    qvm.run(program, shots=1000)
    assert qvm.result().get_prob_dict() == {"0": 1.0}


def test_basis_rotation_does_not_mutate_the_input_program():
    core = QProg(1) << H(0)
    rotated = QProg() << core
    apply_basis_rotation(rotated, "X", (0,))
    qvm = CPUQVM()
    qvm.run(core, shots=1000)
    distribution = qvm.result().get_prob_dict()
    assert abs(distribution.get("0", 0.0) - 0.5) < 1e-6
    assert abs(distribution.get("1", 0.0) - 0.5) < 1e-6


def test_pauli_expectation_reads_parity_from_counts():
    assert pauli_expectation("X", {"0": 400, "1": 600}) == pytest.approx(-0.2)
    assert pauli_expectation("Z", {"0": 750, "1": 250}) == pytest.approx(0.5)


def test_pauli_expectation_marginalizes_identity_positions():
    counts = {"00": 300, "01": 100, "10": 450, "11": 150}
    assert pauli_expectation("ZI", counts) == pytest.approx(-0.2)


def test_reconstruct_density_matrix_recovers_plus_state():
    basis_counts = {
        "X": {"0": 1000, "1": 0},
        "Y": {"0": 500, "1": 500},
        "Z": {"0": 500, "1": 500},
    }
    rho = reconstruct_density_matrix(basis_counts, 1)
    assert np.allclose(rho, [[0.5, 0.5], [0.5, 0.5]], atol=1e-9)
    eigenvalue, vector = dominant_eigenvector(rho)
    assert eigenvalue == pytest.approx(1.0)
    assert abs(np.vdot(normalized([1, 1]), vector)) > 0.999


def test_reconstruct_density_matrix_recovers_zero_state():
    basis_counts = {
        "X": {"0": 500, "1": 500},
        "Y": {"0": 500, "1": 500},
        "Z": {"0": 1000, "1": 0},
    }
    rho = reconstruct_density_matrix(basis_counts, 1)
    assert np.allclose(rho, [[1.0, 0.0], [0.0, 0.0]], atol=1e-9)
    eigenvalue, vector = dominant_eigenvector(rho)
    assert eigenvalue == pytest.approx(1.0)
    assert abs(vector[0]) == pytest.approx(1.0)
    assert abs(vector[1]) < 1e-9


def test_reconstruct_density_matrix_recovers_two_qubit_product_state():
    # |0> ⊗ |+>: the Z basis returns 0 on qubit 0 with certainty and
    # the X basis returns 0 on qubit 1 with certainty; everything else
    # is uniformly distributed.
    basis_counts = {
        "XX": {"00": 500, "10": 500},
        "XY": {"00": 250, "01": 250, "10": 250, "11": 250},
        "XZ": {"00": 250, "01": 250, "10": 250, "11": 250},
        "YX": {"00": 500, "10": 500},
        "YY": {"00": 250, "01": 250, "10": 250, "11": 250},
        "YZ": {"00": 250, "01": 250, "10": 250, "11": 250},
        "ZX": {"00": 1000},
        "ZY": {"00": 500, "01": 500},
        "ZZ": {"00": 500, "01": 500},
    }
    rho = reconstruct_density_matrix(basis_counts, 2)
    expected = np.outer(normalized([1, 1, 0, 0]), normalized([1, 1, 0, 0]))
    assert np.allclose(rho, expected, atol=1e-9)
    eigenvalue, vector = dominant_eigenvector(rho)
    assert eigenvalue == pytest.approx(1.0)
    assert abs(np.vdot(normalized([1, 1, 0, 0]), vector)) > 0.999


def test_reconstruction_fidelity_is_the_dominant_population():
    basis_counts = {
        "X": {"0": 500, "1": 500},
        "Y": {"0": 500, "1": 500},
        "Z": {"0": 900, "1": 100},
    }
    rho = reconstruct_density_matrix(basis_counts, 1)
    eigenvalue, vector = dominant_eigenvector(rho)
    assert eigenvalue == pytest.approx(0.9)
    assert 1.0 - eigenvalue == pytest.approx(0.1)  # reconstruction uncertainty
    assert abs(vector[1]) < 1e-9


def test_dominant_eigenvector_returns_the_largest_eigenvalue():
    rho = np.array([[0.25, 0.0], [0.0, 0.75]])
    eigenvalue, vector = dominant_eigenvector(rho)
    assert eigenvalue == pytest.approx(0.75)
    assert abs(vector[0]) < 1e-9
    assert abs(vector[1]) == pytest.approx(1.0)
