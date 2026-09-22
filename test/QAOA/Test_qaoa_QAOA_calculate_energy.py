"""Sparse operator addresses must match QAOA energy and CPU register indices."""

import numpy as np
import pytest
from pyqpanda3.core import QCircuit, X
from pyqpanda3.hamiltonian import Hamiltonian, PauliOperator
from pyqpanda_alg.QAOA.qaoa import QAOA

CASES = [
    ({"Z2": 1.25, "": -0.5}, 3),
    ({"Z0 Z3": 0.75, "Z3": -0.25, "": 0.3}, 4),
    ({"Z1 Z4": 0.7, "Z4": -0.2, "": 0.1}, 5),
    ({"Z10": -0.9, "": 0.2}, 11),
    ({"Z0": 0.1, "Z1": 0.2}, 2),
]


def _energies(terms: dict[str, float], width: int) -> np.ndarray:
    """Evaluate Z parities directly on integer basis states."""
    indices = np.arange(1 << width)
    energies = np.zeros(1 << width)
    for word, coefficient in terms.items():
        signs = np.ones(1 << width)
        for pauli in word.split():
            signs *= 1 - 2 * ((indices >> int(pauli[1:])) & 1)
        energies += coefficient * signs
    return energies


def _reference_probabilities(
    energies: np.ndarray, width: int, gammas: list[float], betas: list[float]
) -> np.ndarray:
    """Apply diagonal phases and analytic RX(-2 beta), without SDK circuits."""
    state = np.ones(1 << width, dtype=complex) / np.sqrt(1 << width)
    indices = np.arange(1 << width)
    for gamma, beta in zip(gammas, betas, strict=True):
        state *= np.exp(-1j * gamma * energies)
        for qubit in range(width):
            state = (
                np.cos(beta) * state + 1j * np.sin(beta) * state[indices ^ (1 << qubit)]
            )
    return np.abs(state) ** 2


@pytest.mark.parametrize("wrapper", [PauliOperator, Hamiltonian])
@pytest.mark.parametrize(("terms", "width"), CASES)
@pytest.mark.parametrize(
    ("gammas", "betas"), [([0.31], [0.17]), ([0.31, -0.13], [0.17, 0.29])]
)
def test_sparse_operator_energy_and_cpu_probabilities(
    wrapper: type,
    terms: dict[str, float],
    width: int,
    gammas: list[float],
    betas: list[float],
) -> None:
    """Retain leading/internal holes, high addresses and dense compatibility."""
    problem = QAOA(wrapper(terms))
    assert problem.problem_dimension == width
    energies = _energies(terms, width)
    for index, expected in enumerate(energies):
        bits = [(index >> qubit) & 1 for qubit in range(width)]
        assert problem.calculate_energy(bits) == pytest.approx(expected)
    actual = problem.run_qaoa_circuit(gammas, betas, shots=-1)
    assert all(len(key) == width for key in actual)
    probabilities = np.array(
        [actual.get(format(i, f"0{width}b"), 0.0) for i in range(1 << width)]
    )
    np.testing.assert_allclose(
        probabilities,
        _reference_probabilities(energies, width, gammas, betas),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("wrapper", [PauliOperator, Hamiltonian])
def test_custom_circuits_receive_original_qubit_addresses(wrapper: type) -> None:
    """Do not compact addresses when passing the register to user circuits."""
    seen = []

    def initial(qubits: list[int]) -> QCircuit:
        seen.append(list(qubits))
        return QCircuit() << X(qubits[3])

    problem = QAOA(
        wrapper({"Z3": 1.0}),
        init_circuit=initial,
        mixer_circuit=lambda qubits, angle: QCircuit(),
    )
    actual = problem.run_qaoa_circuit([0.23], [0.17], shots=-1)
    assert seen == [[0, 1, 2, 3]]
    assert actual.get("1000", 0.0) == pytest.approx(1.0)
    assert problem.calculate_energy([0, 0, 0, 1]) == -1.0


@pytest.mark.parametrize("wrapper", [PauliOperator, Hamiltonian])
def test_identity_only_keeps_zero_register_width(wrapper: type) -> None:
    """Retain the existing constructor behavior for an empty support."""
    problem = QAOA(wrapper({"": 2.0}))
    assert problem.problem_dimension == 0
    assert problem.calculate_energy([]) == 2.0
