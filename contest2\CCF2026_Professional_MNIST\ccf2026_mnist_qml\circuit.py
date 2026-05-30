import random
from typing import List, Optional, Sequence

from .statevector import apply_cnot, apply_h, apply_ry, apply_rz, expectation_z, zero_state


def circuit_features(
    x: Sequence[float],
    theta: Sequence[float],
    n_qubits: int,
    ansatz_layers: int,
    shots: Optional[int],
    noise_prob: float,
    angle_jitter: float,
    rng: random.Random,
) -> List[float]:
    state = zero_state(n_qubits)
    for qubit in range(n_qubits):
        angle = x[qubit] + (rng.gauss(0.0, angle_jitter) if angle_jitter else 0.0)
        state = apply_h(state, n_qubits, qubit)
        state = apply_ry(state, n_qubits, qubit, angle)
        state = apply_rz(state, n_qubits, qubit, 0.5 * angle)
    for qubit in range(n_qubits - 1):
        state = apply_cnot(state, n_qubits, qubit, qubit + 1)

    cursor = 0
    for _ in range(ansatz_layers):
        for qubit in range(n_qubits):
            state = apply_ry(state, n_qubits, qubit, theta[cursor])
            cursor += 1
            state = apply_rz(state, n_qubits, qubit, theta[cursor])
            cursor += 1
        for qubit in range(n_qubits - 1):
            state = apply_cnot(state, n_qubits, qubit, qubit + 1)

    values = [expectation_z(state, qubit) for qubit in range(n_qubits)]
    if noise_prob > 0:
        attenuation = max(0.0, (1.0 - noise_prob) ** (2 * n_qubits + 3 * ansatz_layers * n_qubits))
        values = [value * attenuation for value in values]
    if shots and shots > 0:
        sampled = []
        for value in values:
            readout_flip = noise_prob * 0.5
            p_one = min(1.0, max(0.0, (1.0 - value) / 2.0))
            p_one = p_one * (1.0 - readout_flip) + (1.0 - p_one) * readout_flip
            ones = sum(1 for _ in range(shots) if rng.random() < p_one)
            sampled.append(1.0 - 2.0 * ones / shots)
        values = sampled
    return values


def resource_profile(n_qubits: int, ansatz_layers: int, shots: int, model: str, backend: str):
    one_qubit_gates = 3 * n_qubits + ansatz_layers * 2 * n_qubits
    two_qubit_gates = (ansatz_layers + 1) * (n_qubits - 1)
    circuit_params = ansatz_layers * 2 * n_qubits
    head_params = n_qubits + 1
    return {
        "model": model,
        "qubits": n_qubits,
        "depth": 3 + (ansatz_layers + 1) * (n_qubits - 1) + ansatz_layers * 2,
        "circuit_params": circuit_params,
        "head_params": head_params,
        "total_params": circuit_params + head_params,
        "one_qubit_gates": one_qubit_gates,
        "two_qubit_gates": two_qubit_gates,
        "shots": shots,
        "backend": backend,
    }

