import numpy as np
import pytest

from pyqpanda_alg.QDrift import (
    PauliTerm,
    apply_qdrift_plan,
    dense_hamiltonian,
    qdrift_error_bound,
    qdrift_segment_count,
    sample_qdrift,
    state_fidelity,
)


def test_segment_bound_meets_requested_epsilon():
    terms = [PauliTerm(0.7, "X"), PauliTerm(-0.3, "Z")]
    count = qdrift_segment_count(terms, time=2.0, epsilon=0.04)

    assert count == 200
    assert qdrift_error_bound(terms, 2.0, count) <= 0.04


def test_seeded_schedule_is_reproducible_and_signed():
    terms = [PauliTerm(3.0, "X"), PauliTerm(-1.0, "Z")]
    first = sample_qdrift(terms, time=0.5, segments=12, seed=1234)
    second = sample_qdrift(terms, time=0.5, segments=12, seed=1234)

    assert first.rotations == second.rotations
    assert first.probabilities == pytest.approx((0.75, 0.25))
    assert all(
        rotation.exponent > 0.0
        if rotation.pauli == "X"
        else rotation.exponent < 0.0
        for rotation in first.rotations
    )


def test_single_term_qdrift_matches_exact_pauli_evolution():
    term = PauliTerm(-0.8, "X")
    initial = np.asarray([1.0, 0.0], dtype=np.complex128)
    plan = sample_qdrift([term], time=1.3, segments=7, seed=9)

    approximate = apply_qdrift_plan(initial, plan)

    alpha = term.coefficient * 1.3
    expected = (
        np.cos(alpha) * np.eye(2)
        - 1.0j * np.sin(alpha) * dense_hamiltonian(
            [PauliTerm(1.0, "X")]
        )
    ).dot(initial)

    assert state_fidelity(approximate, expected) == pytest.approx(
        1.0, abs=1e-12
    )


def test_rejects_mixed_pauli_widths():
    with pytest.raises(ValueError, match="same qubit width"):
        sample_qdrift(
            [PauliTerm(1.0, "X"), PauliTerm(1.0, "ZZ")],
            time=1.0,
            segments=2,
        )
