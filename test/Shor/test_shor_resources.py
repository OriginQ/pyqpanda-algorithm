"""Resource estimation tests for Shor order finding.

Covers the qubit budget of :func:`estimate_shor_resources` — phase
register of twice the modulus bit length by default, value register of
the modulus bit length, ancillas of the modular exponentiation, and the
total consistent with the sum — plus the labeled approximate
gate/depth estimates and the input rejection surface.
"""

import pytest

from pyqpanda_alg.Shor.resources import ShorResourceEstimate, estimate_shor_resources
from pyqpanda_alg.execution import AlgorithmInputError


def test_resource_estimate_grows_with_modulus_bits():
    estimate_15 = estimate_shor_resources(15)
    estimate_21 = estimate_shor_resources(21)
    assert estimate_21.total_qubits >= estimate_15.total_qubits
    assert estimate_15.phase_qubits >= 2 * (15).bit_length()


def test_default_phase_register_is_twice_the_modulus_bit_length():
    estimate = estimate_shor_resources(15)
    assert estimate.phase_qubits == 2 * (15).bit_length()


def test_explicit_phase_qubits_are_respected():
    estimate = estimate_shor_resources(15, phase_qubits=6)
    assert estimate.phase_qubits == 6
    assert estimate.total_qubits == 6 + (15).bit_length() + (15).bit_length() + 1


def test_qubit_budget_matches_circuit_layout():
    estimate = estimate_shor_resources(21)
    assert estimate.value_qubits == (21).bit_length()
    assert estimate.ancilla_qubits == (21).bit_length() + 1
    assert estimate.total_qubits == (
        estimate.phase_qubits + estimate.value_qubits + estimate.ancilla_qubits
    )
    assert estimate.controlled_multiplies == estimate.phase_qubits


def test_approximate_fields_are_labeled():
    estimate = estimate_shor_resources(15)
    assert "synthesized_gates" in estimate.approximate_labels
    assert "estimated_depth" in estimate.approximate_labels
    assert estimate.synthesized_gates >= 1
    assert estimate.estimated_depth >= 1


@pytest.mark.parametrize("invalid", [True, False, 15.0, "15", 15 + 0j, None])
def test_non_integer_modulus_is_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="integer"):
        estimate_shor_resources(invalid)


@pytest.mark.parametrize("invalid", [1, 0, -4])
def test_modulus_below_three_is_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="at least 3"):
        estimate_shor_resources(invalid)


def test_even_modulus_is_rejected():
    with pytest.raises(AlgorithmInputError, match="odd"):
        estimate_shor_resources(14)


@pytest.mark.parametrize("invalid", [0, -1])
def test_phase_qubits_below_one_are_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="phase_qubits"):
        estimate_shor_resources(15, phase_qubits=invalid)


@pytest.mark.parametrize(
    "field",
    ["modulus", "phase_qubits", "value_qubits", "ancilla_qubits", "total_qubits"],
)
def test_direct_construction_rejects_non_positive_fields(field):
    """Direct construction raises the same input error as the estimator."""
    kwargs = {
        "modulus": 15,
        "phase_qubits": 8,
        "value_qubits": 4,
        "ancilla_qubits": 5,
        "total_qubits": 17,
        "controlled_multiplies": 8,
        "synthesized_gates": 1,
        "estimated_depth": 1,
    }
    kwargs[field] = 0
    with pytest.raises(AlgorithmInputError, match="positive integer"):
        ShorResourceEstimate(**kwargs)


def test_direct_construction_rejects_inconsistent_total_qubits():
    with pytest.raises(AlgorithmInputError, match="total_qubits"):
        ShorResourceEstimate(
            modulus=15,
            phase_qubits=8,
            value_qubits=4,
            ancilla_qubits=5,
            total_qubits=99,  # 8 + 4 + 5 = 17
            controlled_multiplies=8,
            synthesized_gates=1,
            estimated_depth=1,
        )
