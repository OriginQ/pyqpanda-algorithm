import numpy as np
import pytest

from pyqpanda_alg.QSEncode._preparation import (
    SPARSE_SUPPORT_THRESHOLD,
    adapt_preparation_input,
)
from pyqpanda_alg.QSEncode.exceptions import InputValidationError


def test_dense_adapter_preserves_complex_values_and_caller_input():
    source = np.array([1 / np.sqrt(2), 0, 1j / np.sqrt(2), 0])
    original = source.copy()

    adapted = adapt_preparation_input(source)

    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(adapted.coefficients, original)
    assert np.iscomplexobj(adapted.coefficients)
    assert adapted.coefficients.flags.writeable is False
    assert adapted.logical_dimension == 4
    assert adapted.logical_output_qubits == 2


def test_sparse_adapter_uses_frozen_strict_representation_cleanup_threshold():
    tiny = SPARSE_SUPPORT_THRESHOLD
    source = np.array([np.sqrt(1.0 - (2 * tiny) ** 2), tiny, 2 * tiny, 0.0])
    source /= np.linalg.norm(source)

    adapted = adapt_preparation_input(source)

    assert adapted.support_indices == (0, 2)
    assert tuple(key for key, _ in adapted.sparse_items) == ("00", "10")
    assert adapted.support_threshold == 1e-14


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ([1.0, 0.0, 0.0], "invalid_state_dimension"),
        ([1.0, 1.0, 0.0, 0.0], "state_not_normalized"),
        ([0.0, 0.0, 0.0, 0.0], "state_not_normalized"),
        ([np.nan, 0.0, 0.0, 0.0], "nonfinite_state"),
    ],
)
def test_adapter_rejects_invalid_selected_states(values, code):
    with pytest.raises(InputValidationError) as exc_info:
        adapt_preparation_input(values)
    assert exc_info.value.code == code
