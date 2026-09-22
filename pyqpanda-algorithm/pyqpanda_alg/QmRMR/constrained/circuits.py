"""pyQPanda3 circuits for constrained QAOA QmRMR feature selection."""

from __future__ import annotations

from itertools import combinations
from math import comb, pi, sqrt
from typing import Sequence

import numpy as np
from pyqpanda3.core import CNOT, Encode, QCircuit, QProg, RX, RY, RZ, X

from .problem import QmRMRProblem


def _as_qubit_list(qubits: Sequence[int]) -> list[int]:
    values = [int(qubit) for qubit in qubits]
    if values != list(range(len(values))):
        raise ValueError("qubits must be the consecutive indices [0, ..., n-1]")
    if len(values) > 20:
        raise ValueError("the CPU state-vector backend is limited to 20 qubits")
    return values


def build_dicke_state(
    qubits: Sequence[int],
    select_num: int,
    *,
    mode: str = "auto",
    weights: Sequence[float] | None = None,
    strength: float = 1.0,
) -> QCircuit:
    """Prepare the uniform fixed-weight state.

    ``mode='amplitude'`` prepares a uniform fixed-weight state with the
    pyQPanda3 amplitude encoder. ``mode='relevance'`` (or ``'biased'``)
    applies a soft relevance bias while remaining in the same fixed-weight
    subspace. ``'auto'`` is an alias for the uniform amplitude construction.
    """

    qbs = _as_qubit_list(qubits)
    n = len(qbs)
    if not 0 <= int(select_num) <= n:
        raise ValueError("select_num must satisfy 0 <= select_num <= n")
    mode = str(mode).lower()
    if mode not in {"auto", "amplitude", "relevance", "biased"}:
        raise ValueError(
            "mode must be 'auto', 'amplitude', 'relevance', or 'biased'"
        )

    if mode in {"relevance", "biased"}:
        if weights is None:
            raise ValueError("weights are required for a biased Dicke state")
        weight_array = np.asarray(weights, dtype=float).reshape(-1)
        if weight_array.size != n or not np.all(np.isfinite(weight_array)):
            raise ValueError("weights must be a finite vector matching qubit count")
        if not np.isfinite(strength) or float(strength) < 0.0:
            raise ValueError("strength must be a non-negative finite number")

        amplitudes = np.zeros(1 << n, dtype=float)
        scores: list[tuple[int, tuple[int, ...]]] = []
        for selected in combinations(range(n), int(select_num)):
            index = sum(1 << position for position in selected)
            scores.append((index, selected))
        logits = np.asarray(
            [float(strength) * float(weight_array[list(selected)].sum()) for _, selected in scores],
            dtype=float,
        )
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        for (index, _), probability in zip(scores, probabilities):
            amplitudes[index] = sqrt(float(probability))
        encoder = Encode()
        encoder.amplitude_encode(qbs, amplitudes.tolist())
        return encoder.get_circuit()

    amplitudes = np.zeros(1 << n, dtype=float)
    weight = comb(n, int(select_num))
    if weight <= 0:
        raise ValueError("invalid Dicke-state dimension")
    amplitude = 1.0 / sqrt(weight)
    for selected in combinations(range(n), int(select_num)):
        index = sum(1 << position for position in selected)
        amplitudes[index] = amplitude

    encoder = Encode()
    encoder.amplitude_encode(qbs, amplitudes.tolist())
    return encoder.get_circuit()


def xy_exchange(q1: int, q2: int, angle: float) -> QCircuit:
    """Implement exp(-i * angle * (XX + YY) / 2) with pyQPanda3 gates."""

    circuit = QCircuit()
    circuit << RX(q1, pi / 2)
    circuit << RX(q2, pi / 2)
    circuit << CNOT(q1, q2)
    circuit << RX(q1, angle)
    circuit << RZ(q2, angle)
    circuit << CNOT(q1, q2)
    circuit << RX(q1, -pi / 2)
    circuit << RX(q2, -pi / 2)
    return circuit


def parity_xy_mixer(
    qubits: Sequence[int],
    beta: float,
    *,
    ring: bool = False,
) -> QCircuit:
    """Apply even and odd nearest-neighbor XY matchings."""

    qbs = _as_qubit_list(qubits)
    circuit = QCircuit()
    n = len(qbs)
    for parity in (0, 1):
        for start in range(parity, n - 1, 2):
            circuit << xy_exchange(qbs[start], qbs[start + 1], float(beta))
    if ring and n >= 3 and n % 2 == 0:
        # The edge (n-1, 0) is a third matching for the open-chain layout.
        circuit << xy_exchange(qbs[-1], qbs[0], float(beta))
    return circuit


def cost_phase(problem: QmRMRProblem, gamma: float) -> QCircuit:
    """Build exp(-i * gamma * H_C) for the diagonal Ising polynomial."""

    _, z_terms, zz_terms = problem.ising_coefficients()
    circuit = QCircuit()
    for index, coefficient in enumerate(z_terms):
        if coefficient != 0.0:
            circuit << RZ(index, 2.0 * float(gamma) * coefficient)
    for (left, right), coefficient in zz_terms.items():
        if coefficient == 0.0:
            continue
        circuit << CNOT(left, right)
        circuit << RZ(right, 2.0 * float(gamma) * coefficient)
        circuit << CNOT(left, right)
    return circuit


def constrained_qaoa_circuit(
    problem: QmRMRProblem,
    parameters: Sequence[float],
    *,
    initializer: str = "auto",
    initializer_strength: float = 1.0,
    ring: bool = False,
) -> QCircuit:
    """Build a p-layer constrained QAOA circuit.

    Parameters are ordered as [gamma_0, beta_0, ..., gamma_{p-1}, beta_{p-1}].
    """

    params = np.asarray(parameters, dtype=float).reshape(-1)
    if params.size == 0 or params.size % 2:
        raise ValueError("parameters must contain gamma/beta pairs")
    qubits = list(range(problem.n_features))
    circuit = QCircuit()
    circuit << build_dicke_state(
        qubits,
        problem.select_num,
        mode=initializer,
        weights=problem.relevance_vector
        if str(initializer).lower() in {"relevance", "biased"}
        else None,
        strength=initializer_strength,
    )
    for gamma, beta in params.reshape(-1, 2):
        circuit << cost_phase(problem, float(gamma))
        circuit << parity_xy_mixer(qubits, float(beta), ring=ring)
    return circuit


def exchange_ansatz_circuit(
    problem: QmRMRProblem,
    parameters: Sequence[float],
) -> tuple[QCircuit, int]:
    """Build the exchange-ansatz baseline circuit.

    The public baseline accepts the conventional ``floor(n/2)*n`` parameter
    layout while the circuit consumes ``floor(n/2)*(n-1)`` active values. The
    returned integer is the active parameter count; extra supplied values are
    ignored so callers can use either layout.
    """

    params = np.asarray(parameters, dtype=float).reshape(-1)
    qbs = list(range(problem.n_features))
    layers = problem.n_features // 2
    even_edges = [(i, i + 1) for i in range(0, problem.n_features - 1, 2)]
    odd_edges = [(i, i + 1) for i in range(1, problem.n_features - 1, 2)]
    active = layers * (len(even_edges) + len(odd_edges))
    if params.size < active:
        raise ValueError(
            f"exchange ansatz needs at least {active} parameters, got {params.size}"
        )

    circuit = QCircuit()
    for index in range(problem.select_num):
        circuit << X(qbs[index])

    parameter_index = 0
    for _ in range(layers):
        for left, right in even_edges + odd_edges:
            # The controlled rotation implements one nearest-neighbour exchange.
            circuit << CNOT(qbs[right], qbs[left])
            circuit << RY(qbs[right], float(params[parameter_index])).control(qbs[left])
            circuit << CNOT(qbs[right], qbs[left])
            parameter_index += 1
    return circuit, active


def append_circuit_to_program(circuit: QCircuit, n_qubits: int) -> QProg:
    """Create a program without measurement for exact state-vector execution."""

    program = QProg(int(n_qubits))
    program.append(circuit)
    return program

