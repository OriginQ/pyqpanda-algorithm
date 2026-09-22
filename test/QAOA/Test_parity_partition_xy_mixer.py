"""Verify parity-ring mixers against independent dense XY Hamiltonians."""

import numpy as np
import pytest
from pyqpanda3.core import CPUQVM, QCircuit, QProg, X
from pyqpanda_alg.QAOA.default_circuits import (
    parity_partition_xy_mixer,
    xy_mixer,
)
from scipy.linalg import expm


def _unitary(circuit: QCircuit, width: int) -> np.ndarray:
    """Simulate every computational-basis column, including complex phases."""
    columns = []
    for basis in range(1 << width):
        program = QProg(width)
        for qubit in range(width):
            if basis & (1 << qubit):
                program << X(qubit)
        program << circuit
        machine = CPUQVM()
        machine.run(program, shots=1)
        columns.append(machine.result().get_state_vector())
    return np.asarray(columns).T


def _reference(domain: list[int], width: int, beta: float) -> np.ndarray:
    """Exponentiate disjoint XY partitions; retain the existing angle sign."""
    size = len(domain)
    identity = np.eye(1 << width, dtype=complex)
    if size < 2:
        return identity
    if size == 2:
        partitions = [[(domain[0], domain[1])]]
    else:
        edges = [(domain[i], domain[(i + 1) % size]) for i in range(size)]
        # For odd rings, the wrap edge is a separate third partition.
        interior = edges[:-1] if size % 2 else edges
        partitions = [interior[::2], interior[1::2]]
        if size % 2:
            partitions.append([edges[-1]])
    result = identity
    for partition in partitions:
        hamiltonian = np.zeros_like(identity)
        for left, right in partition:
            for basis in range(1 << width):
                if ((basis >> left) & 1) != ((basis >> right) & 1):
                    swapped = basis ^ (1 << left) ^ (1 << right)
                    # XX + YY exchanges |01> and |10> with amplitude 2.
                    hamiltonian[swapped, basis] += 2
        result = expm(1j * beta * hamiltonian) @ result
    return result


@pytest.mark.parametrize("width", range(1, 7))
@pytest.mark.parametrize("beta", [0.0, 0.37, -0.23])
def test_full_parity_ring_unitary(width: int, beta: float) -> None:
    """Cover singleton identity, one two-qubit edge, and odd/even rings."""
    domain = list(range(width))
    circuit = parity_partition_xy_mixer(domain, beta)
    actual = _unitary(circuit, width)
    np.testing.assert_allclose(actual, _reference(domain, width, beta), atol=1e-12)
    for basis in range(1 << width):
        outside = [i for i in range(1 << width) if i.bit_count() != basis.bit_count()]
        assert np.sum(np.abs(actual[outside, basis]) ** 2) < 1e-24
    edges = 0 if width == 1 else 1 if width == 2 else width
    assert dict(circuit.count_ops()).get("CNOT", 0) == 2 * edges


@pytest.mark.parametrize("domain", [[4, 1, 3], [5, 0, 3, 1, 4]])
def test_permuted_qubits_and_spectators(domain: list[int]) -> None:
    """Use caller qubit order, leaving spectator qubits unchanged."""
    actual = _unitary(parity_partition_xy_mixer(domain, 0.19), 6)
    np.testing.assert_allclose(actual, _reference(domain, 6, 0.19), atol=1e-12)


@pytest.mark.parametrize(
    ("argument", "domains", "width"),
    [
        ([[0], [1, 4], [2, 5, 3]], [[0], [1, 4], [2, 5, 3]], 6),
        (3, [[0], [1], [2]], 3),
        (2, [[0, 1, 2], [3, 4, 5]], 6),
    ],
)
def test_factory_preserves_each_domain(
    argument: int | list[list[int]], domains: list[list[int]], width: int
) -> None:
    """Exercise public integer/list domain paths over all Hamming sectors."""
    actual = _unitary(xy_mixer(argument, "PXY")(list(range(width)), 0.37), width)
    expected = np.eye(1 << width, dtype=complex)
    for domain in domains:
        expected = _reference(domain, width, 0.37) @ expected
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    for basis in range(1 << width):
        outside = [
            other
            for other in range(1 << width)
            if any(
                sum((basis >> q) & 1 for q in domain)
                != sum((other >> q) & 1 for q in domain)
                for domain in domains
            )
        ]
        assert np.sum(np.abs(actual[outside, basis]) ** 2) < 1e-24


def test_empty_domain_remains_a_no_op() -> None:
    """Preserve the previously accepted empty-circuit behavior."""
    circuit = parity_partition_xy_mixer([], 0.37)
    assert not dict(circuit.count_ops())
    np.testing.assert_allclose(_unitary(circuit, 1), np.eye(2), atol=1e-12)
