# Licensed under the Apache License, Version 2.0.
"""Small NumPy statevector backend used as an independent reference model."""
from __future__ import annotations

import numpy as np

from ..core.pauli import PauliString


def apply_pauli(state: np.ndarray, pauli_string: PauliString) -> np.ndarray:
    """Apply a Pauli string to a little-endian statevector."""
    n = len(pauli_string)
    out = np.zeros_like(state, dtype=complex)
    for index, amplitude in enumerate(state):
        if amplitude == 0:
            continue
        target = index
        phase = 1.0 + 0.0j
        for qubit, operator in enumerate(pauli_string):
            bit = (index >> qubit) & 1
            if operator == "I":
                continue
            if operator == "X":
                target ^= 1 << qubit
            elif operator == "Y":
                target ^= 1 << qubit
                phase *= 1j if bit == 0 else -1j
            elif operator == "Z":
                phase *= 1.0 if bit == 0 else -1.0
            else:
                raise ValueError(f"unsupported Pauli operator {operator}")
        out[target] += phase * amplitude
    return out


def apply_pauli_rotation(
    state: np.ndarray,
    pauli_string: PauliString,
    theta: float,
) -> np.ndarray:
    """Apply ``exp(-i theta P)`` using the identity ``P**2 = I``."""
    if abs(theta) < 1e-16:
        return state
    return (
        np.cos(theta) * state
        - 1j * np.sin(theta) * apply_pauli(state, pauli_string)
    )


def product_state_from_ry(angles: np.ndarray) -> np.ndarray:
    """Prepare a product state obtained by applying ``RY(angle[q])`` to |0...0>."""
    angles = np.asarray(angles, dtype=float)
    n = len(angles)
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0
    for qubit, angle in enumerate(angles):
        cosine = np.cos(angle / 2.0)
        sine = np.sin(angle / 2.0)
        updated = np.zeros_like(state)
        for index in range(1 << n):
            if (index >> qubit) & 1:
                continue
            amplitude = state[index]
            if amplitude == 0:
                continue
            updated[index] += cosine * amplitude
            updated[index | (1 << qubit)] += sine * amplitude
        state = updated
    return state


def plus_state(n: int) -> np.ndarray:
    """Return the uniform ``|+>**n`` statevector."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=complex)


def probabilities(state: np.ndarray) -> np.ndarray:
    """Return a normalized computational-basis probability vector."""
    prob = np.abs(np.asarray(state, dtype=complex)) ** 2
    norm = float(prob.sum())
    if norm <= 0:
        raise ValueError("state has zero norm")
    return prob / norm
