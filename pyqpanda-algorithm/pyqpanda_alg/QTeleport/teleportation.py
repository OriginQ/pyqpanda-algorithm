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

"""Three-qubit coherent quantum teleportation.

The public circuit builder uses the deferred-measurement form of the standard
teleportation protocol. The message qubit may contain an arbitrary state before
the returned circuit is appended; the two resource qubits must begin in |0>.

Qubit convention
----------------
q[0] is the message qubit, q[1] is Alice's Bell-pair qubit, and q[2] is Bob's
output qubit.

Instead of measuring the first two qubits and applying classically controlled
X/Z corrections, the builder applies those corrections coherently. This is
equivalent by the deferred-measurement principle and avoids requiring a
backend-specific classical-control API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)


def _validate_qubits(qubits: Sequence[Any]) -> list[Any]:
    resolved = list(qubits)
    if len(resolved) != 3:
        raise ValueError("qubits must contain exactly three qubits")
    for index, qubit in enumerate(resolved):
        if qubit in resolved[:index]:
            raise ValueError("qubits must be distinct")
    return resolved


def _load_qpanda_gates():
    try:
        from pyqpanda3.core import CNOT, H, QCircuit, Z
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ImportError(
            "PyQPanda3 is required for QTeleport circuit construction; "
            "reference_teleportation() remains available without it"
        ) from exc
    return QCircuit, CNOT, H, Z


def coherent_teleportation_circuit(qubits: Sequence[Any]):
    """Build a coherent three-qubit teleportation circuit.

    Before the circuit, qubits[0] holds the message state and qubits[1:3] must
    be |00>. After the circuit, qubits[2] carries the original message state.
    The first two qubits coherently retain the two Bell-basis correction bits.

    Gate sequence::

        H(q1)
        CNOT(q1, q2)
        CNOT(q0, q1)
        H(q0)
        CNOT(q1, q2)
        CZ(q0, q2)

    The controlled-Z is expressed as a controlled Z gate to stay within the
    PyQPanda3 gate interface used elsewhere in this repository.
    """

    q_message, q_alice, q_bob = _validate_qubits(qubits)
    QCircuit, CNOT, H, Z = _load_qpanda_gates()

    circuit = QCircuit()
    circuit << H(q_alice)
    circuit << CNOT(q_alice, q_bob)
    circuit << CNOT(q_message, q_alice)
    circuit << H(q_message)
    circuit << CNOT(q_alice, q_bob)
    circuit << Z(q_bob).control([q_message])
    return circuit


def _normalise_logical_state(state: Sequence[complex]) -> np.ndarray:
    vector = np.asarray(state, dtype=complex)
    if vector.shape != (2,):
        raise ValueError("logical_state must contain exactly two amplitudes")
    if not np.all(np.isfinite(vector)):
        raise ValueError("logical_state amplitudes must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("logical_state must have non-zero norm")
    return vector / norm


def _apply_single_qubit(
    state: np.ndarray,
    gate: np.ndarray,
    target: int,
) -> np.ndarray:
    out = np.zeros_like(state)
    target_mask = 1 << target
    for basis in range(state.size):
        if basis & target_mask:
            continue
        paired = basis | target_mask
        a0 = state[basis]
        a1 = state[paired]
        out[basis] = gate[0, 0] * a0 + gate[0, 1] * a1
        out[paired] = gate[1, 0] * a0 + gate[1, 1] * a1
    return out


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    out = np.zeros_like(state)
    target_mask = 1 << target
    for basis, amplitude in enumerate(state):
        destination = basis
        if (basis >> control) & 1:
            destination ^= target_mask
        out[destination] += amplitude
    return out


def _apply_cz(state: np.ndarray, control: int, target: int) -> np.ndarray:
    out = state.copy()
    for basis in range(state.size):
        if ((basis >> control) & 1) and ((basis >> target) & 1):
            out[basis] *= -1.0
    return out


def bob_density_matrix(statevector: Sequence[complex]) -> np.ndarray:
    """Trace out message/Alice qubits from a three-qubit q0-first state."""

    state = np.asarray(statevector, dtype=complex)
    if state.shape != (8,):
        raise ValueError("statevector must contain exactly eight amplitudes")
    rho = np.zeros((2, 2), dtype=complex)
    for lower_bits in range(4):
        amplitudes = np.array(
            [state[lower_bits], state[lower_bits | (1 << 2)]],
            dtype=complex,
        )
        rho += np.outer(amplitudes, np.conjugate(amplitudes))
    return rho


def reference_teleportation(
    logical_state: Sequence[complex],
    *,
    return_statevector: bool = False,
) -> dict[str, Any]:
    """Execute the protocol with an independent NumPy state-vector reference.

    logical_state may be any non-zero pair of complex amplitudes; it is
    normalized before use. Qubit 0 is the least-significant state-vector bit,
    matching the q0-first convention used by the contribution modules.

    Returns Bob's reduced density matrix, fidelity with the normalized input,
    and optionally the final three-qubit state vector.
    """

    logical = _normalise_logical_state(logical_state)
    state = np.zeros(8, dtype=complex)
    state[0] = logical[0]
    state[1] = logical[1]

    state = _apply_single_qubit(state, _H, 1)
    state = _apply_cnot(state, 1, 2)
    state = _apply_cnot(state, 0, 1)
    state = _apply_single_qubit(state, _H, 0)
    state = _apply_cnot(state, 1, 2)
    state = _apply_cz(state, 0, 2)

    rho_bob = bob_density_matrix(state)
    fidelity = float(np.real(np.vdot(logical, rho_bob @ logical)))
    result: dict[str, Any] = {
        "logical_state": logical,
        "bob_density_matrix": rho_bob,
        "fidelity": fidelity,
    }
    if return_statevector:
        result["statevector"] = state
    return result


def teleportation_gate_counts() -> dict[str, int]:
    """Return the logical gate counts of the coherent protocol."""

    return {
        "H": 2,
        "CNOT": 3,
        "controlled_Z": 1,
    }
