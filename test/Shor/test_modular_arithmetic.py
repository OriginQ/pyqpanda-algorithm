"""Basis-state permutation tests for reversible modular arithmetic.

Every test prepares a little-endian basis value, applies a
:class:`pyqpanda_alg.Shor.arithmetic` circuit with its control qubit set,
samples once on :class:`LocalBackend`, and asserts that the value
register holds the expected modular image while every declared ancilla
returns to zero.  The contract under test: with the control set the
circuit is the permutation ``x -> base*x mod modulus`` (respectively
``x -> x*base**e mod modulus`` for exponentiation), with the control
clear it is the identity, and non-coprime bases, non-integer inputs, and
even moduli are rejected before any circuit is built.
"""

import pytest

from pyqpanda_alg.Shor.arithmetic import (
    controlled_modular_exponentiation,
    controlled_modular_multiply,
)
from pyqpanda_alg.execution import AlgorithmInputError

from test.Shor.helpers import run_basis_permutation


@pytest.mark.parametrize("value", range(15))
def test_multiply_by_two_mod_fifteen(value):
    circuit = controlled_modular_multiply(base=2, modulus=15, value_qubits=[0, 1, 2, 3], control=4)
    observed = run_basis_permutation(circuit, control=1, value=value)
    assert observed == (2 * value) % 15


@pytest.mark.parametrize("value", range(21))
def test_multiply_by_two_mod_twenty_one(value):
    circuit = controlled_modular_multiply(base=2, modulus=21, value_qubits=[0, 1, 2, 3, 4], control=5)
    observed = run_basis_permutation(circuit, control=1, value=value)
    assert observed == (2 * value) % 21


@pytest.mark.parametrize("value", range(15))
def test_multiply_by_seven_mod_fifteen(value):
    circuit = controlled_modular_multiply(base=7, modulus=15, value_qubits=[0, 1, 2, 3], control=4)
    observed = run_basis_permutation(circuit, control=1, value=value)
    assert observed == (7 * value) % 15


@pytest.mark.parametrize("value", range(21))
def test_multiply_by_four_mod_twenty_one(value):
    circuit = controlled_modular_multiply(base=4, modulus=21, value_qubits=[0, 1, 2, 3, 4], control=5)
    observed = run_basis_permutation(circuit, control=1, value=value)
    assert observed == (4 * value) % 21


@pytest.mark.parametrize("value", range(15))
def test_multiply_with_control_clear_is_identity(value):
    circuit = controlled_modular_multiply(base=2, modulus=15, value_qubits=[0, 1, 2, 3], control=4)
    observed = run_basis_permutation(circuit, control=0, value=value)
    assert observed == value


@pytest.mark.parametrize("value", range(15))
def test_exponentiation_clears_all_exponent_bits(value):
    circuit = controlled_modular_exponentiation(
        base=2,
        modulus=15,
        exponent_qubits=[5, 6, 7],
        value_qubits=[0, 1, 2, 3],
    )
    # All exponent bits set: exponent 2**3 - 1 = 7.
    observed = run_basis_permutation(circuit, control=1, value=value)
    assert observed == (value * pow(2, 7, 15)) % 15


def test_multiply_rejects_base_not_coprime_to_modulus():
    with pytest.raises(AlgorithmInputError, match="coprime"):
        controlled_modular_multiply(base=3, modulus=15, value_qubits=[0, 1, 2, 3], control=4)


def test_exponentiation_rejects_base_not_coprime_to_modulus():
    with pytest.raises(AlgorithmInputError, match="coprime"):
        controlled_modular_exponentiation(
            base=6, modulus=15, exponent_qubits=[5, 6, 7], value_qubits=[0, 1, 2, 3]
        )


@pytest.mark.parametrize("invalid", [True, False])
def test_boolean_modulus_is_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="integer"):
        controlled_modular_multiply(base=2, modulus=invalid, value_qubits=[0, 1, 2, 3], control=4)


@pytest.mark.parametrize("invalid", [12.0, "13", 12 + 0j, None])
def test_non_integer_modulus_is_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="integer"):
        controlled_modular_multiply(base=2, modulus=invalid, value_qubits=[0, 1, 2, 3], control=4)


def test_even_modulus_is_rejected():
    with pytest.raises(AlgorithmInputError, match="odd"):
        controlled_modular_multiply(base=3, modulus=14, value_qubits=[0, 1, 2, 3], control=4)


@pytest.mark.parametrize("invalid", [1, 0, -5])
def test_modulus_below_three_is_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="at least 3"):
        controlled_modular_multiply(base=2, modulus=invalid, value_qubits=[0, 1, 2, 3], control=4)


def test_value_register_too_small_for_modulus_is_rejected():
    with pytest.raises(AlgorithmInputError, match="value_qubits"):
        controlled_modular_multiply(base=2, modulus=21, value_qubits=[0, 1, 2, 3], control=4)
