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

"""Superdense coding with an explicit two-bit message convention.

The ordered qubit pair is [alice, bob]. The first message bit controls
the phase flip (Z) and the second message bit controls the bit flip (X) on
Alice's half of the Bell pair. After decoding, measuring the same ordered
qubit pair yields the original two-bit message.
"""

from __future__ import annotations

from collections.abc import Sequence

from pyqpanda3.core import CNOT, H, QCircuit, X, Z


def normalize_message(message: str | int) -> str:
    """Return a canonical two-bit message."""
    if isinstance(message, bool):
        raise TypeError("message must be a two-bit string or integer 0..3")

    if isinstance(message, int):
        if not 0 <= message <= 3:
            raise ValueError("integer message must be in the range 0..3")
        return f"{message:02b}"

    if isinstance(message, str):
        message = message.strip()
        if len(message) == 2 and set(message) <= {"0", "1"}:
            return message
        raise ValueError("string message must contain exactly two binary digits")

    raise TypeError("message must be a two-bit string or integer 0..3")


def _ordered_qubits(qubits: Sequence) -> tuple:
    ordered = tuple(qubits)
    if len(ordered) != 2:
        raise ValueError("superdense coding requires exactly two ordered qubits")
    if ordered[0] == ordered[1]:
        raise ValueError("superdense coding requires two distinct qubits")
    return ordered


def prepare_bell_pair(qubits: Sequence) -> QCircuit:
    """Create the shared Bell pair used by superdense coding."""
    alice, bob = _ordered_qubits(qubits)
    circuit = QCircuit()
    circuit << H(alice)
    circuit << CNOT(alice, bob)
    return circuit


def encode_message(qubits: Sequence, message: str | int) -> QCircuit:
    """Encode two classical bits on Alice's half of the Bell pair.

    The first bit controls Z and the second bit controls X. For "11" both
    gates are applied; their relative global phase does not affect decoding.
    """
    alice, _ = _ordered_qubits(qubits)
    bits = normalize_message(message)

    circuit = QCircuit()
    if bits[0] == "1":
        circuit << Z(alice)
    if bits[1] == "1":
        circuit << X(alice)
    return circuit


def decode_message(qubits: Sequence) -> QCircuit:
    """Decode Alice's transmitted qubit back into the two classical bits."""
    alice, bob = _ordered_qubits(qubits)
    circuit = QCircuit()
    circuit << CNOT(alice, bob)
    circuit << H(alice)
    return circuit


def superdense_coding_circuit(qubits: Sequence, message: str | int) -> QCircuit:
    """Build the complete prepare -> encode -> decode protocol circuit."""
    ordered = _ordered_qubits(qubits)
    circuit = QCircuit()
    circuit << prepare_bell_pair(ordered)
    circuit << encode_message(ordered, message)
    circuit << decode_message(ordered)
    return circuit


class SuperdenseCoding:
    """Reusable superdense-coding protocol builder."""

    def __init__(self, message: str | int):
        self._message = normalize_message(message)

    @property
    def expected_bits(self) -> str:
        """Canonical two-bit message recovered after decoding."""
        return self._message

    def cir(self, qubits: Sequence) -> QCircuit:
        """Return the complete superdense-coding circuit."""
        return superdense_coding_circuit(qubits, self._message)
