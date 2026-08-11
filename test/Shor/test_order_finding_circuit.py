"""Order-finding circuit construction tests for the Shor solver.

Construction is deliberately sampling-free: the no-answer-injection
tests monkeypatch the factorization and order-recovery helpers to raise
and assert circuit construction still succeeds using only modular
arithmetic, and the structural tests pin the register layout — phase
register of twice the modulus bit length by default (explicit sizes
respected), value register wide enough for the modulus and prepared to
|1>, ancilla budget of the modular exponentiation — together with the
pre-transpilation depth and gate-count metadata.
"""

import pytest
import sympy

from pyqpanda_alg.Shor import classical
from pyqpanda_alg.Shor.circuit import build_order_finding_circuit
from pyqpanda_alg.execution import AlgorithmInputError


def _forbid(name: str):
    """Return a helper that raises when a forbidden answer helper runs."""

    def _raise(*args, **kwargs):
        raise AssertionError(
            f"{name} must not be called during order-finding circuit construction"
        )

    return _raise


@pytest.mark.parametrize("base, modulus", [(2, 15), (2, 21)])
def test_construction_uses_only_modular_arithmetic(monkeypatch, base, modulus):
    # Factorization and order recovery would inject the answer into the
    # circuit; construction must never touch them.
    monkeypatch.setattr(sympy, "factorint", _forbid("factorint"))
    monkeypatch.setattr(classical, "classical_preprocess", _forbid("classical_preprocess"))
    monkeypatch.setattr(classical, "recover_order", _forbid("recover_order"))
    build = build_order_finding_circuit(base, modulus)
    assert build.program is not None


def test_default_phase_register_is_twice_the_modulus_bit_length():
    for base, modulus in [(2, 15), (2, 21)]:
        build = build_order_finding_circuit(base, modulus)
        assert len(build.phase_qubits) == 2 * modulus.bit_length()


def test_explicit_phase_qubits_are_respected():
    build = build_order_finding_circuit(2, 15, phase_qubits=6)
    assert len(build.phase_qubits) == 6


def test_value_register_holds_the_modulus_and_prepares_one():
    build = build_order_finding_circuit(2, 15)
    assert len(build.value_qubits) == (15).bit_length()
    # The |1> preparation is the only value-register setup: modular
    # exponentiation of the base's zeroth power leaves |1> unchanged.
    assert build.gate_counts.get("X", 0) >= 1


def test_ancilla_register_matches_exponentiation_budget():
    build = build_order_finding_circuit(2, 15)
    assert len(build.ancilla_qubits) == (15).bit_length() + 1
    assert build.total_qubits == (
        len(build.phase_qubits)
        + len(build.value_qubits)
        + len(build.ancilla_qubits)
    )


def test_build_reports_pre_transpilation_depth_and_gate_counts():
    build = build_order_finding_circuit(2, 15)
    assert build.depth >= 1
    assert sum(build.gate_counts.values()) >= 1


@pytest.mark.parametrize("invalid", [0, -1])
def test_phase_qubits_below_one_are_rejected(invalid):
    with pytest.raises(AlgorithmInputError, match="phase_qubits"):
        build_order_finding_circuit(2, 15, phase_qubits=invalid)


def test_even_modulus_is_rejected():
    with pytest.raises(AlgorithmInputError, match="odd"):
        build_order_finding_circuit(2, 14)


def test_non_coprime_base_is_rejected():
    with pytest.raises(AlgorithmInputError, match="coprime"):
        build_order_finding_circuit(3, 15)
