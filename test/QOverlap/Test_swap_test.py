import math

import pytest

from pyqpanda_alg.QOverlap import overlap_squared_from_zero_probability


@pytest.mark.parametrize(
    ("zero_probability", "expected"),
    [(0.5, 0.0), (0.75, 0.5), (1.0, 1.0), (0.4, 0.0)],
)
def test_overlap_squared_from_zero_probability(zero_probability, expected):
    assert math.isclose(
        overlap_squared_from_zero_probability(zero_probability),
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_overlap_decoder_rejects_invalid_probability(value):
    with pytest.raises(ValueError):
        overlap_squared_from_zero_probability(value)
