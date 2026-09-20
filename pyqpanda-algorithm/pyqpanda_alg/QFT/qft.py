# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable Quantum Fourier Transform circuits and reference semantics."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from pyqpanda3.core import CR, H, QCircuit, SWAP


def _validated_qubits(qubits: Iterable[int]) -> list[int]:
    """Return a validated, materialized low-to-high qubit list."""
    values = list(qubits)
    if not values:
        raise ValueError("QFT requires at least one qubit")
    for position, qubit in enumerate(values):
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise TypeError(
                f"qubit at position {position} must be an integer index, "
                f"got {type(qubit).__name__}"
            )
        if qubit < 0:
            raise ValueError(
                f"qubit at position {position} must be non-negative, got {qubit}"
            )
    if len(set(values)) != len(values):
        raise ValueError("QFT qubit indices must be unique")
    return values


def qft_circuit(
    qubits: Sequence[int],
    *,
    inverse: bool = False,
    do_swaps: bool = True,
) -> QCircuit:
    """Build a forward or inverse Quantum Fourier Transform circuit.

    Qubits are ordered least-significant to most-significant, matching the
    existing pyqpanda-algorithm QFT helper.  With do_swaps=True the output is
    in ordinary computational-basis index order; disabling swaps intentionally
    leaves bit-reversed output.

    The inverse is the dagger of the exact forward circuit, so the controlled
    phases and swap convention remain paired.
    """
    ordered = _validated_qubits(qubits)
    circuit = QCircuit()
    width = len(ordered)

    for layer in range(width):
        target = ordered[width - 1 - layer]
        circuit << H(target)
        for control_layer in range(layer + 1, width):
            control = ordered[width - 1 - control_layer]
            angle = 2.0 * math.pi / (1 << (control_layer - layer + 1))
            circuit << CR(control, target, angle)

    if do_swaps:
        for index in range(width // 2):
            circuit << SWAP(ordered[index], ordered[width - 1 - index])

    return circuit.dagger() if inverse else circuit


def reference_qft(
    state: Sequence[complex],
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Apply the mathematical QFT to a state-vector-like sequence.

    The forward convention is y[k] = sum_j x[j] exp(+2*pi*i*j*k/N)/sqrt(N);
    the inverse uses the negative phase. Input length must be a non-zero power
    of two. Values are transformed linearly as supplied and are not silently
    normalized.
    """
    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError("QFT reference input must be one-dimensional")
    size = int(vector.size)
    if size == 0 or size & (size - 1):
        raise ValueError("QFT reference input length must be a non-zero power of two")
    if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
        raise ValueError("QFT reference input must contain only finite amplitudes")

    indices = np.arange(size, dtype=np.float64)
    sign = -1.0 if inverse else 1.0
    kernel = np.exp(
        sign * 2j * np.pi * np.outer(indices, indices) / float(size)
    )
    return (kernel @ vector) / math.sqrt(size)


def bit_reversed_indices(num_qubits: int) -> np.ndarray:
    """Return the index permutation produced when final QFT swaps are omitted."""
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least one")

    size = 1 << num_qubits
    result = np.empty(size, dtype=np.int64)
    for value in range(size):
        reversed_value = 0
        work = value
        for _ in range(num_qubits):
            reversed_value = (reversed_value << 1) | (work & 1)
            work >>= 1
        result[value] = reversed_value
    return result


class QuantumFourierTransform:
    """Configured QFT builder with a stable qubit-order contract."""

    def __init__(self, qubits: Sequence[int], *, do_swaps: bool = True):
        self.qubits = tuple(_validated_qubits(qubits))
        self.do_swaps = bool(do_swaps)

    def circuit(self, *, inverse: bool = False) -> QCircuit:
        """Return a fresh transform circuit for this configuration."""
        return qft_circuit(
            self.qubits,
            inverse=inverse,
            do_swaps=self.do_swaps,
        )

    @staticmethod
    def reference(
        state: Sequence[complex],
        *,
        inverse: bool = False,
    ) -> np.ndarray:
        """Apply the exact mathematical reference transform."""
        return reference_qft(state, inverse=inverse)


__all__ = [
    "QuantumFourierTransform",
    "bit_reversed_indices",
    "qft_circuit",
    "reference_qft",
]
