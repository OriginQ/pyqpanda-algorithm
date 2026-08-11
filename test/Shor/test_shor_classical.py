"""Classical preprocessing and order-recovery tests for the Shor solver.

Covers the contract of :func:`classical_preprocess` (explicit results
for even, prime, and perfect-power moduli; a needs-quantum marker for
odd composites) and of :func:`recover_order` (continued-fraction
denominator recovery with the multiple-of-denominator fallback), plus
the input rejection classes and the execution provenance guarantees:
no quantum task is ever submitted for a classically resolved modulus.
"""

import pytest

from pyqpanda_alg.Shor import (
    NEEDS_QUANTUM,
    Shor,
    ShorConfig,
    ShorResult,
    classical_preprocess,
    recover_order,
)
from pyqpanda_alg.Shor.classical import _convergent_denominators
from pyqpanda_alg.execution import AlgorithmInputError


def test_even_modulus_returns_without_quantum_execution():
    result = Shor(12).run()
    assert result.factors == (2, 6)
    assert result.used_quantum is False
    assert result.task_ids == ()


def test_prime_modulus_is_reported_without_task():
    result = Shor(13).run()
    assert result.is_prime is True
    assert result.factors is None
    assert result.used_quantum is False


def test_perfect_power_modulus_returns_factor_without_quantum():
    result = Shor(27).run()
    assert result.factors == (3, 9)
    assert result.used_quantum is False
    assert result.task_ids == ()


def test_classical_preprocess_marks_odd_composite_as_needing_quantum():
    outcome = classical_preprocess(15)
    assert outcome.resolved is False
    assert outcome.status == NEEDS_QUANTUM
    assert outcome.factors is None
    assert outcome.is_prime is False


def test_recover_order_recovers_order_four_for_base_two_mod_fifteen():
    # Phase sample 64/256 = 1/4 recovers the order of 2 mod 15 directly.
    assert recover_order(sample=64, phase_bits=8, base=2, modulus=15) == 4


def test_recover_order_recovers_order_six_for_base_two_mod_twenty_one():
    # Phase sample 341/1024 has convergent denominator 3, which is a
    # proper divisor of the true order 6; the multiple scan must find 6.
    assert recover_order(sample=341, phase_bits=10, base=2, modulus=21) == 6


def test_zero_phase_sample_recovers_no_order():
    assert recover_order(sample=0, phase_bits=8, base=2, modulus=15) is None


def test_later_unit_convergent_is_skipped_not_brute_forced():
    """A sample fraction at/above 1/2 produces a later 1/1 convergent.

    Every unit-denominator convergent must be skipped -- not only the
    first -- or the multiple scan of a 1/1 denominator degenerates into
    brute-force order search.  Sample 192/256 = 3/4 has convergents
    [1, 4]: the 0/1 and 1/1 convergents are both skipped, so only the
    1/4 denominator is scanned and order 4 is recovered without ever
    brute-forcing.
    """
    assert list(_convergent_denominators(192, 256)) == [4]
    assert recover_order(sample=192, phase_bits=8, base=2, modulus=15) == 4


@pytest.mark.parametrize("invalid", [True, False])
def test_boolean_modulus_is_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="integer"):
        classical_preprocess(invalid)
    with pytest.raises(AlgorithmInputError, match="integer"):
        Shor(invalid)


@pytest.mark.parametrize("invalid", [12.0, "13", 12 + 0j, None])
def test_non_integer_modulus_is_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="integer"):
        classical_preprocess(invalid)


@pytest.mark.parametrize("invalid", [1, 0, -5])
def test_modulus_below_two_is_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="at least 2"):
        classical_preprocess(invalid)


def test_recover_order_rejects_sample_outside_phase_register():
    with pytest.raises(AlgorithmInputError, match="sample"):
        recover_order(sample=256, phase_bits=8, base=2, modulus=15)


def test_recover_order_rejects_base_not_coprime_to_modulus():
    with pytest.raises(AlgorithmInputError, match="coprime"):
        recover_order(sample=64, phase_bits=8, base=3, modulus=15)


def test_shor_config_defaults_and_validation():
    config = ShorConfig()
    assert config.max_attempts == 6
    assert config.phase_qubits is None
    with pytest.raises(ValueError, match="max_attempts"):
        ShorConfig(max_attempts=0)
    with pytest.raises(ValueError, match="phase_qubits"):
        ShorConfig(phase_qubits=0)


def test_shor_result_normalizes_containers_and_copies_metadata():
    metadata = {"preprocessing": "even"}
    result = ShorResult(factors=[2, 6], task_ids=["t-1"], metadata=metadata)
    assert result.factors == (2, 6)
    assert result.task_ids == ("t-1",)
    assert result.used_quantum is False
    metadata["preprocessing"] = "mutated"
    assert result.metadata["preprocessing"] == "even"
