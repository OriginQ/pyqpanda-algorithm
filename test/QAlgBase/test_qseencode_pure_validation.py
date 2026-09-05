import numpy as np
import pytest

from pyqpanda_alg.QSEncode import InputPolicy, InputValidationError
from pyqpanda_alg.QSEncode._validation import canonicalize_probabilities


def test_normalize_padding_and_input_summary_contract():
    validated = canonicalize_probabilities([1.0, 3.0, 0.0])

    np.testing.assert_array_equal(validated.probabilities, [0.25, 0.75, 0.0, 0.0])
    assert validated.probabilities.dtype == np.float64
    assert validated.probabilities.flags.writeable is False
    assert validated.summary.original_length == 3
    assert validated.summary.padded_length == 4
    assert validated.summary.original_sum == 4.0
    assert validated.summary.normalized is True
    assert validated.summary.padding_count == 1
    assert validated.summary.original_input_sha256 == (
        "e729a6fb50de27cc3f1868f7a7ccd7c322e735dcb64680ca127852e61e11c797"
    )
    assert validated.summary.effective_probability_sha256 == (
        "2b120f880bd1947febe7ca47d2023eb587a39478584caadae547e43ecbacbf44"
    )


def test_exact_unit_sum_is_divided_but_normalized_flag_describes_change_only():
    validated = canonicalize_probabilities([0.2, 0.3, 0.5])
    assert validated.summary.normalized is False
    np.testing.assert_array_equal(validated.probabilities, [0.2, 0.3, 0.5, 0.0])


def test_strict_policy_accepts_within_tolerance_and_still_divides():
    policy = InputPolicy(normalization="strict")
    validated = canonicalize_probabilities([0.5, 0.5000000000005], policy=policy)
    assert validated.summary.normalized is False
    assert validated.probabilities.sum() == pytest.approx(1.0, abs=1e-15)


def test_strict_policy_rejects_outside_tolerance():
    with pytest.raises(InputValidationError) as exc_info:
        canonicalize_probabilities(
            [0.5, 0.500000000002],
            policy=InputPolicy(normalization="strict"),
        )
    assert exc_info.value.code == "not_normalized"


@pytest.mark.parametrize(
    ("length", "padded_length"),
    [(2, 2), (3, 4), (4, 4), (5, 8), (63, 64), (64, 64)],
)
def test_next_power_of_two_padding_grid(length, padded_length):
    result = canonicalize_probabilities(np.ones(length, dtype=np.float64))
    assert result.summary.original_length == length
    assert result.summary.padded_length == padded_length
    assert result.summary.padding_count == padded_length - length
    assert result.probabilities.sum() == pytest.approx(1.0, abs=1e-15)


def test_reject_padding_policy_rejects_non_power_of_two_only():
    policy = InputPolicy(padding="reject")
    accepted = canonicalize_probabilities([1.0, 1.0, 1.0, 1.0], policy=policy)
    assert accepted.summary.padded_length == 4

    with pytest.raises(InputValidationError) as exc_info:
        canonicalize_probabilities([1.0, 1.0, 1.0], policy=policy)
    assert exc_info.value.code == "non_power_of_two"


@pytest.mark.parametrize(
    ("values", "code"),
    [
        (1.0, "not_one_dimensional"),
        ([[0.5, 0.5]], "not_one_dimensional"),
        ([1.0], "insufficient_dimension"),
        ([np.nan, 1.0], "nonfinite_probability"),
        ([np.inf, 1.0], "nonfinite_probability"),
        ([0.5 + 0.1j, 0.5], "complex_probability"),
        ([-0.1, 1.1], "negative_probability"),
        ([0.0, 0.0], "zero_mass"),
        (["not-a-number", "1"], "not_numeric"),
    ],
)
def test_invalid_inputs_have_exact_reason_codes(values, code):
    with pytest.raises(InputValidationError) as exc_info:
        canonicalize_probabilities(values)
    assert exc_info.value.code == code


def test_caller_array_is_not_modified():
    original = np.array([1.0, 2.0, 3.0])
    before = original.copy()
    canonicalize_probabilities(original)
    np.testing.assert_array_equal(original, before)
