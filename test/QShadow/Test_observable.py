import numpy as np
import pytest

from pyqpanda_alg.QShadow import PauliObservable, PauliTerm, pauli_expectation


def test_pauli_term_normalises_and_reports_support():
    term = PauliTerm(" x i z ", 2)
    assert term.pauli == "XIZ"
    assert term.support == (0, 2)
    assert term.weight == 2
    assert term.n_qubits == 3


def test_observable_combines_duplicate_terms():
    observable = PauliObservable.from_terms(
        [("ZI", 0.5), PauliTerm("ZI", 1.25), ("IZ", -0.25)], name="energy"
    )
    assert observable.name == "energy"
    assert observable.terms == (PauliTerm("ZI", 1.75), PauliTerm("IZ", -0.25))


def test_pauli_term_rejects_invalid_or_non_hermitian_input():
    with pytest.raises(ValueError, match="invalid Pauli"):
        PauliTerm("XA")
    with pytest.raises(ValueError, match="must be real"):
        PauliTerm("X", 1 + 2j)
    with pytest.raises(ValueError, match="finite"):
        PauliTerm("X", np.inf)


def test_observable_rejects_mixed_width_and_duplicate_names_are_handled_elsewhere():
    with pytest.raises(ValueError, match="equal length"):
        PauliObservable((PauliTerm("X"), PauliTerm("ZZ")))


def test_exact_single_qubit_pauli_expectations():
    plus = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    plus_i = np.asarray([1.0, 1.0j]) / np.sqrt(2.0)
    assert pauli_expectation([1, 0], "Z") == pytest.approx(1.0)
    assert pauli_expectation(plus, "X") == pytest.approx(1.0)
    assert pauli_expectation(plus_i, "Y") == pytest.approx(1.0)


def test_exact_expectation_uses_q0_first_pauli_strings():
    # Statevector index 1 is |q1 q0> = |01>: q0=1 and q1=0.
    state = np.asarray([0, 1, 0, 0], dtype=complex)
    observable = PauliObservable.from_terms(
        {"ZI": 2.0, "IZ": 0.5, "II": 0.25}, name="q0-first"
    )
    assert observable.exact_expectation(state) == pytest.approx(-1.25)


def test_exact_expectation_rejects_wrong_or_unnormalised_statevector():
    with pytest.raises(ValueError, match="expected 4"):
        pauli_expectation([1, 0], "ZZ")
    with pytest.raises(ValueError, match="normalised"):
        pauli_expectation([2, 0], "Z")
