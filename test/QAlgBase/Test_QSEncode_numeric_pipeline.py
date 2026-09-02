import numpy as np
import pytest

from pyqpanda_alg.QSEncode.QSEncode import (
    QSpare_Code,
    _fast_walsh_hadamard_transform,
)


def _hadamard_matrix(size):
    """Build a small independent Sylvester Hadamard matrix."""
    matrix = np.array([[1.0]])
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def _dense_fwht(values):
    """Reference transform using an explicit matrix, not a butterfly."""
    values = np.asarray(values)
    if values.size == 0:
        return np.empty(0, dtype=np.result_type(values.dtype, np.float64))
    padded_size = 1 << (values.size - 1).bit_length()
    padded = np.zeros(padded_size, dtype=np.result_type(values.dtype, np.float64))
    padded[:values.size] = values
    return _hadamard_matrix(padded_size) @ padded


def _dense_sparse_walsh_probabilities(probabilities, cut):
    """Independent end-to-end oracle for the Walsh sparse encoder."""
    probabilities = np.asarray(probabilities, dtype=float)
    amplitudes = np.sqrt(probabilities / probabilities.sum())
    size = amplitudes.size
    hadamard = _hadamard_matrix(size) / np.sqrt(size)
    coefficients = hadamard @ amplitudes

    order = np.argsort(np.abs(coefficients), kind='stable')
    sparse = np.zeros_like(coefficients)
    sparse[order[-cut:]] = coefficients[order[-cut:]]
    sparse /= np.linalg.norm(sparse)
    return np.abs(hadamard @ sparse) ** 2


@pytest.mark.parametrize(
    'values',
    [
        np.array([], dtype=float),
        np.array([3.25]),
        np.array([1.0, -2.0, 4.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.0 + 2.0j, -3.0j, 0.5 - 0.25j]),
    ],
)
def test_fast_walsh_transform_matches_dense_oracle(values):
    actual = _fast_walsh_hadamard_transform(values)
    expected = _dense_fwht(values)

    assert actual.dtype != object
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_fast_walsh_transform_matches_dense_oracle_for_lengths_1_to_64():
    rng = np.random.default_rng(20260901)

    for size in range(1, 65):
        values = rng.normal(size=size)
        if size % 2 == 0:
            values = values + 1j * rng.normal(size=size)
        actual = _fast_walsh_hadamard_transform(values)
        expected = _dense_fwht(values)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize('qubits', range(13))
def test_normalized_walsh_transform_preserves_norm_and_is_self_inverse(qubits):
    size = 1 << qubits
    values = np.random.default_rng(qubits).normal(size=size)
    transformed = _fast_walsh_hadamard_transform(values) / np.sqrt(size)
    restored = _fast_walsh_hadamard_transform(transformed) / np.sqrt(size)

    np.testing.assert_allclose(np.linalg.norm(transformed), np.linalg.norm(values), rtol=1e-13)
    np.testing.assert_allclose(restored, values, rtol=1e-13, atol=1e-13)


def test_fast_walsh_transform_does_not_modify_input_and_promotes_numeric_dtype():
    values = np.array([1, 2, 3], dtype=np.int16)
    original = values.copy()

    result = _fast_walsh_hadamard_transform(values)

    np.testing.assert_array_equal(values, original)
    assert result.dtype == np.float64


def test_fast_walsh_transform_rejects_non_numeric_or_non_vector_input():
    with pytest.raises(ValueError, match='1D'):
        _fast_walsh_hadamard_transform(np.ones((2, 2)))
    with pytest.raises(TypeError, match='numeric'):
        _fast_walsh_hadamard_transform(['a', 'b'])


@pytest.mark.parametrize(
    ('size', 'cut'),
    [(8, 1), (8, 7), (257, 17), (4096, 63)],
)
def test_linear_top_k_matches_full_stable_sort_for_unique_magnitudes(size, cut):
    rng = np.random.default_rng(size + cut)
    phases = np.exp(2j * np.pi * rng.random(size))
    values = np.arange(1, size + 1, dtype=float) * phases
    encoder = QSpare_Code([0.5, 0.5])

    actual = encoder.select_top_n_complex_numbers(values, cut)
    expected = np.zeros_like(values)
    expected_indices = np.argsort(np.abs(values), kind='stable')[-cut:]
    expected[expected_indices] = values[expected_indices]

    np.testing.assert_array_equal(actual, expected)


def test_linear_top_k_has_deterministic_boundary_ties():
    values = np.array([-3.0, 3.0, 2.0, -2.0, 1.0])
    encoder = QSpare_Code([0.5, 0.5])

    actual = encoder.select_top_n_complex_numbers(values, 3)

    np.testing.assert_array_equal(actual, [-3.0, 3.0, 0.0, -2.0, 0.0])


def test_linear_top_k_keeps_input_unchanged_and_fast_paths_full_selection():
    values = np.array([1.0 + 2.0j, -4.0j, 3.0])
    original = values.copy()
    encoder = QSpare_Code([0.5, 0.5])

    result = encoder.select_top_n_complex_numbers(values, values.size)

    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(result, original)
    assert result is not values


def test_linear_top_k_rejects_invalid_shape_and_non_finite_values():
    encoder = QSpare_Code([0.5, 0.5])
    with pytest.raises(ValueError, match='1D'):
        encoder.select_top_n_complex_numbers(np.ones((2, 2)), 1)
    with pytest.raises(ValueError, match='finite'):
        encoder.select_top_n_complex_numbers(np.array([1.0, np.nan]), 1)
    with pytest.raises(ValueError, match='finite'):
        encoder.select_top_n_complex_numbers(np.array([1.0, np.inf]), 2)


@pytest.mark.parametrize(
    ('probabilities', 'cut'),
    [
        ([0.50, 0.30, 0.15, 0.05], 2),
        ([0.31, 0.19, 0.16, 0.12, 0.09, 0.06, 0.04, 0.03], 3),
    ],
)
def test_walsh_quantum_result_matches_independent_dense_pipeline(probabilities, cut):
    encoder = QSpare_Code(list(map(float, probabilities)), mode='walsh', cut_length=cut)

    actual = encoder.Quantum_Res()
    expected = _dense_sparse_walsh_probabilities(probabilities, cut)

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.sum(actual), 1.0, rtol=0.0, atol=1e-12)
