import math

import pytest

from pyqpanda_alg.QCount import count_from_bitstring, phase_to_count


def test_phase_to_count_zero_and_full_search_space_bounds():
    assert phase_to_count(0.0, 4) == 0
    assert 0 <= phase_to_count(0.5, 4) <= 16


def test_phase_to_count_decodes_known_two_of_eight_fraction():
    phase = math.asin(math.sqrt(2 / 8)) / math.pi
    assert phase_to_count(phase, 3) == 2
    assert phase_to_count(1.0 - phase, 3) == 2


def test_count_from_bitstring_uses_phase_register_precision():
    assert count_from_bitstring("001000", 3) == phase_to_count(8 / 64, 3)


@pytest.mark.parametrize("value", ["", "1021", "hello"])
def test_count_from_bitstring_rejects_invalid_binary(value):
    with pytest.raises(ValueError):
        count_from_bitstring(value, 3)


@pytest.mark.parametrize("phase", [float("nan"), float("inf"), -float("inf")])
def test_phase_to_count_rejects_non_finite_phase(phase):
    with pytest.raises(ValueError):
        phase_to_count(phase, 3)
