import numpy as np
import pytest

from pyqpanda_alg.QSEncode import InputValidationError
from pyqpanda_alg.QSEncode._error_budget import (
    candidate_neighborhood,
    find_k_star,
    retained_energy_ratio,
    stable_magnitude_order,
    top_k_coefficients,
    top_k_indices,
)


def test_frozen_stable_v1_exact_ties_keep_stable_ascending_order_then_suffix():
    coefficients = np.array([1.0, -1.0, 1.0, -1.0])
    np.testing.assert_array_equal(stable_magnitude_order(coefficients), [0, 1, 2, 3])
    np.testing.assert_array_equal(top_k_indices(coefficients, 2), [2, 3])
    np.testing.assert_array_equal(
        top_k_coefficients(coefficients, 2), [0.0, 0.0, 1.0, -1.0]
    )


def test_normalized_top_k_state_preserves_selected_complex_coefficients():
    coefficients = np.array([1 + 1j, 4j, -2.0, 0.5])
    selected = top_k_coefficients(coefficients, 2, normalize=True)
    expected = np.array([0j, 4j, -2.0 + 0j, 0j]) / np.sqrt(20.0)
    np.testing.assert_allclose(selected, expected, atol=1e-12, rtol=1e-10)


def test_retained_energy_uses_shared_ranking_order():
    coefficients = np.array([1.0, 2.0, 3.0])
    assert retained_energy_ratio(coefficients, 1) == pytest.approx(9 / 14)
    assert retained_energy_ratio(coefficients, 2) == pytest.approx(13 / 14)
    assert retained_energy_ratio(coefficients, 3) == pytest.approx(1.0)


def test_k_star_minimality_middle_boundary():
    coefficients = np.sqrt(np.array([0.6, 0.3, 0.1]))
    result = find_k_star(coefficients, 0.9)
    assert result.k_star == 2
    assert result.previous_retained_energy == pytest.approx(0.6)
    assert result.retained_energy == pytest.approx(0.9)
    assert result.minimality_pass is True
    assert result.candidate_k == (1, 2)
    assert result.ranking_policy == "frozen_stable_v1"


def test_k_star_one_and_target_one_reaching_n():
    dominant = find_k_star(np.sqrt([0.995, 0.005]), 0.99)
    assert dominant.k_star == 1
    assert dominant.previous_retained_energy == 0.0
    assert dominant.candidate_k == (1,)

    full = find_k_star(np.ones(4), 1.0)
    assert full.k_star == 4
    assert full.retained_energy == pytest.approx(1.0)
    assert full.candidate_k == (3,)


@pytest.mark.parametrize(
    ("k_star", "size", "expected"),
    [(1, 2, (1,)), (1, 8, (1, 2)), (4, 8, (3, 4, 5)), (8, 8, (7,))],
)
def test_candidate_neighborhood_exact_boundaries(k_star, size, expected):
    assert candidate_neighborhood(k_star, size) == expected


@pytest.mark.parametrize(
    ("coefficients", "code"),
    [
        ([0.0, 0.0], "zero_coefficient_energy"),
        ([1.0, np.nan], "nonfinite_coefficients"),
        ([1.0, np.inf], "nonfinite_coefficients"),
    ],
)
def test_invalid_coefficient_energy_has_exact_codes(coefficients, code):
    with pytest.raises(InputValidationError) as exc_info:
        find_k_star(coefficients, 0.99)
    assert exc_info.value.code == code


@pytest.mark.parametrize("target", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_fidelity_target_is_rejected(target):
    with pytest.raises(InputValidationError) as exc_info:
        find_k_star([1.0, 1.0], target)
    assert exc_info.value.code == "invalid_fidelity_target"
