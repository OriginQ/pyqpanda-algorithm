"""Validation and padding tests for the HHL input pipeline.

Covers the contract of :func:`normalize_linear_system`: rejection of
non-Hermitian, non-finite, singular, and ill-conditioned inputs,
acceptance of real and complex Hermitian fixtures, the reversible
power-of-two padding rule, and flat-legacy-matrix reshaping.
"""

import numpy as np
import pytest

from pyqpanda_alg.HHL import HHLConfig, HHLSolution, normalize_linear_system
from pyqpanda_alg.execution import AlgorithmInputError


def test_non_hermitian_matrix_is_rejected():
    matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(AlgorithmInputError, match="Hermitian"):
        normalize_linear_system(matrix, np.array([1.0, 0.0]), HHLConfig())


def test_three_dimensional_system_pads_to_four():
    matrix = np.diag([1.0, 2.0, 3.0])
    system = normalize_linear_system(matrix, np.ones(3), HHLConfig())
    assert system.padded_dimension == 4
    assert system.original_dimension == 3


def test_two_by_two_real_hermitian_system_is_accepted_and_normalized():
    matrix = np.diag([1.0, 2.0])
    vector = np.array([3.0, 4.0])
    system = normalize_linear_system(matrix, vector, HHLConfig())
    assert system.padded_dimension == 2
    assert system.original_dimension == 2
    assert np.allclose(system.matrix, matrix)
    assert np.isclose(np.linalg.norm(system.vector), 1.0)
    assert np.isclose(system.original_vector_norm, 5.0)
    assert np.allclose(system.vector, vector / 5.0)


def test_complex_hermitian_system_is_accepted():
    matrix = np.array([[2.0, 1.0 + 1.0j], [1.0 - 1.0j, 3.0]])
    system = normalize_linear_system(matrix, np.array([1.0, 1.0j]), HHLConfig())
    assert system.padded_dimension == 2
    assert system.original_dimension == 2
    assert np.allclose(system.matrix, matrix)
    assert np.isclose(np.linalg.norm(system.vector), 1.0)


def test_padding_appends_identity_block_and_zeros():
    matrix = np.diag([1.0, 2.0, 3.0])
    vector = np.array([1.0, 1.0, 1.0])
    system = normalize_linear_system(matrix, vector, HHLConfig())
    assert np.allclose(system.matrix[:3, :3], matrix)
    assert np.allclose(system.matrix[3:, 3:], np.eye(1))
    assert np.allclose(system.matrix[3, :3], 0.0)
    assert np.allclose(system.matrix[:3, 3], 0.0)
    assert np.allclose(system.vector[3], 0.0)
    assert np.isclose(np.linalg.norm(system.vector), 1.0)


def test_singular_matrix_is_rejected():
    matrix = np.diag([1.0, 0.0])
    with pytest.raises(AlgorithmInputError, match="singular"):
        normalize_linear_system(matrix, np.array([1.0, 1.0]), HHLConfig())


def test_ill_conditioned_matrix_is_rejected():
    matrix = np.diag([1.0, 2e-7])
    with pytest.raises(AlgorithmInputError, match="condition"):
        normalize_linear_system(matrix, np.array([1.0, 1.0]), HHLConfig())


def test_non_finite_matrix_entries_are_rejected():
    matrix = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(AlgorithmInputError, match="finite"):
        normalize_linear_system(matrix, np.array([1.0, 1.0]), HHLConfig())


def test_non_finite_vector_entries_are_rejected():
    with pytest.raises(AlgorithmInputError, match="finite"):
        normalize_linear_system(np.eye(2), np.array([1.0, np.inf]), HHLConfig())


def test_zero_norm_vector_is_rejected():
    with pytest.raises(AlgorithmInputError, match="norm"):
        normalize_linear_system(np.eye(2), np.zeros(2), HHLConfig())


def test_flat_matrix_list_with_perfect_square_length_is_reshaped():
    system = normalize_linear_system([1.0, 0.0, 0.0, 2.0], [1.0, 1.0], HHLConfig())
    assert system.original_dimension == 2
    assert np.allclose(system.matrix[:2, :2], np.diag([1.0, 2.0]))


def test_flat_matrix_list_without_perfect_square_length_is_rejected():
    with pytest.raises(AlgorithmInputError):
        normalize_linear_system([1.0, 0.0, 0.0], [1.0, 1.0], HHLConfig())


def test_non_square_2d_matrix_is_rejected():
    with pytest.raises(AlgorithmInputError):
        normalize_linear_system(np.ones((2, 3)), np.ones(2), HHLConfig())


def test_dimension_mismatch_between_matrix_and_vector_is_rejected():
    with pytest.raises(AlgorithmInputError):
        normalize_linear_system(np.eye(3), np.ones(2), HHLConfig())


def test_hhl_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        HHLConfig(max_condition_number=0)
    with pytest.raises(ValueError):
        HHLConfig(phase_qubits=0)


def test_hhl_solution_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        HHLSolution(classical_vector=np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="finite"):
        HHLSolution(success_probability=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        HHLSolution(residual=float("inf"))


def test_hhl_solution_accepts_default_unfilled_fields():
    solution = HHLSolution()
    assert solution.classical_vector is None
    assert solution.statevector is None
    assert solution.success_probability is None
    assert solution.residual is None


def test_normalized_system_arrays_are_immutable_copies():
    matrix = np.diag([1.0, 2.0])
    vector = np.array([1.0, 1.0])
    system = normalize_linear_system(matrix, vector, HHLConfig())
    matrix[0, 0] = 99.0
    assert system.matrix[0, 0] == 1.0
    with pytest.raises(ValueError):
        system.matrix[0, 0] = 5.0
    with pytest.raises(ValueError):
        system.vector[0] = 5.0
