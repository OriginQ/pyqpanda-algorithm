from __future__ import annotations

from functools import reduce
from typing import Iterable

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_all(mats: Iterable[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, mats)


def site_operator(num_qubits: int, site: int, op: np.ndarray) -> np.ndarray:
    mats = [I2] * num_qubits
    mats[site] = op
    return kron_all(mats)


def pair_operator(num_qubits: int, left: int, left_op: np.ndarray, right: int, right_op: np.ndarray) -> np.ndarray:
    mats = [I2] * num_qubits
    mats[left] = left_op
    mats[right] = right_op
    return kron_all(mats)


def expectation(state: np.ndarray, operator: np.ndarray) -> complex:
    return np.vdot(state, operator @ state)


def normalize(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("不能归一化零态")
    return state / norm


def pauli_rotation_state(state: np.ndarray, pauli_matrix: np.ndarray, theta: float) -> np.ndarray:
    dim = pauli_matrix.shape[0]
    return np.cos(theta) * state - 1j * np.sin(theta) * (pauli_matrix @ state)
