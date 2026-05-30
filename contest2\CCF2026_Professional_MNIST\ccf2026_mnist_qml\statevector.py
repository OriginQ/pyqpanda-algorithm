import math
from typing import List


State = List[complex]


def zero_state(n_qubits: int) -> State:
    state = [0j] * (1 << n_qubits)
    state[0] = 1.0 + 0j
    return state


def apply_ry(state: State, n_qubits: int, qubit: int, angle: float) -> State:
    c = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    return apply_single_qubit(state, n_qubits, qubit, ((c, -s), (s, c)))


def apply_rz(state: State, n_qubits: int, qubit: int, angle: float) -> State:
    p0 = complex(math.cos(-angle / 2.0), math.sin(-angle / 2.0))
    p1 = complex(math.cos(angle / 2.0), math.sin(angle / 2.0))
    return apply_single_qubit(state, n_qubits, qubit, ((p0, 0j), (0j, p1)))


def apply_h(state: State, n_qubits: int, qubit: int) -> State:
    inv = 1.0 / math.sqrt(2.0)
    return apply_single_qubit(state, n_qubits, qubit, ((inv, inv), (inv, -inv)))


def apply_single_qubit(state: State, n_qubits: int, qubit: int, matrix) -> State:
    output = state[:]
    step = 1 << qubit
    block = step << 1
    for base in range(0, 1 << n_qubits, block):
        for offset in range(step):
            idx0 = base + offset
            idx1 = idx0 + step
            amp0 = state[idx0]
            amp1 = state[idx1]
            output[idx0] = matrix[0][0] * amp0 + matrix[0][1] * amp1
            output[idx1] = matrix[1][0] * amp0 + matrix[1][1] * amp1
    return output


def apply_cnot(state: State, n_qubits: int, control: int, target: int) -> State:
    output = [0j] * (1 << n_qubits)
    control_mask = 1 << control
    target_mask = 1 << target
    for idx, amp in enumerate(state):
        output[idx ^ target_mask if idx & control_mask else idx] += amp
    return output


def expectation_z(state: State, qubit: int) -> float:
    mask = 1 << qubit
    total = 0.0
    for idx, amp in enumerate(state):
        prob = amp.real * amp.real + amp.imag * amp.imag
        total += -prob if idx & mask else prob
    return total

