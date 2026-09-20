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

"""Three-qubit repetition quantum error-correction codes.

The module implements the canonical three-qubit bit-flip code and its
Hadamard-basis phase-flip counterpart. Circuit construction uses PyQPanda3
lazily, while an independent NumPy state-vector reference remains usable
without PyQPanda3.

Qubit convention
----------------
``q_code[0]`` carries the logical input before encoding. ``q_code[1]`` and
``q_code[2]`` must begin in ``|0>``. Recovery decodes the logical state back to
``q_code[0]`` and leaves the two syndrome bits in ``q_code[1:3]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np


_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _validate_code_qubits(q_code: Sequence[Any]) -> list[Any]:
    q_code = list(q_code)
    if len(q_code) != 3:
        raise ValueError("q_code must contain exactly three qubits")
    for index, qubit in enumerate(q_code):
        if qubit in q_code[:index]:
            raise ValueError("q_code must not contain duplicate qubits")
    return q_code


def _load_qpanda_gates():
    """Import the PyQPanda3 gates used by the circuit builders."""

    try:
        from pyqpanda3.core import QCircuit, CNOT, H, TOFFOLI, X, Z
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ImportError(
            "PyQPanda3 is required for QEC circuit construction; the NumPy "
            "reference helpers remain available without it"
        ) from exc
    return QCircuit, CNOT, H, TOFFOLI, X, Z


def bit_flip_encode(q_code: Sequence[Any]):
    """Encode one logical qubit into the three-qubit repetition code.

    The input is ``alpha|0> + beta|1>`` on ``q_code[0]`` with the other two
    qubits initialized to ``|0>``. The output is
    ``alpha|000> + beta|111>``.
    """

    q0, q1, q2 = _validate_code_qubits(q_code)
    QCircuit, CNOT, _, _, _, _ = _load_qpanda_gates()
    circuit = QCircuit()
    circuit << CNOT(q0, q1)
    circuit << CNOT(q0, q2)
    return circuit


def bit_flip_recover(q_code: Sequence[Any]):
    """Decode and coherently recover from any one Pauli-X error.

    Recovery returns the logical state to ``q_code[0]`` and retains syndrome
    information in ``(q_code[1], q_code[2])``. The syndrome convention is::

        no error -> 00
        X on q0  -> 11
        X on q1  -> 10
        X on q2  -> 01

    where the printed order is ``q1 q2``.
    """

    q0, q1, q2 = _validate_code_qubits(q_code)
    QCircuit, CNOT, _, TOFFOLI, _, _ = _load_qpanda_gates()
    circuit = QCircuit()
    circuit << CNOT(q0, q1)
    circuit << CNOT(q0, q2)
    circuit << TOFFOLI(q1, q2, q0)
    return circuit


def phase_flip_encode(q_code: Sequence[Any]):
    """Encode the logical qubit into the Hadamard-basis repetition code.

    This applies :func:`bit_flip_encode` and then Hadamard gates to all three
    code qubits. The code corrects one Pauli-Z error.
    """

    q_code = _validate_code_qubits(q_code)
    QCircuit, _, H, _, _, _ = _load_qpanda_gates()
    circuit = QCircuit()
    circuit << bit_flip_encode(q_code)
    for qubit in q_code:
        circuit << H(qubit)
    return circuit


def phase_flip_recover(q_code: Sequence[Any]):
    """Decode and coherently recover from any one Pauli-Z error.

    Hadamards convert phase flips to bit flips, after which the ordinary
    repetition decoder/recovery is applied. The decoded logical state is left
    on ``q_code[0]`` with the same syndrome convention as bit-flip recovery.
    """

    q_code = _validate_code_qubits(q_code)
    QCircuit, _, H, _, _, _ = _load_qpanda_gates()
    circuit = QCircuit()
    for qubit in q_code:
        circuit << H(qubit)
    circuit << bit_flip_recover(q_code)
    return circuit


def inject_single_pauli_error(
    q_code: Sequence[Any], error_qubit: int, pauli: str
):
    """Return a circuit applying one explicit ``X`` or ``Z`` error.

    This helper is intended for examples, tests, and controlled experiments;
    it is not a noise model.
    """

    q_code = _validate_code_qubits(q_code)
    if isinstance(error_qubit, bool) or not isinstance(error_qubit, Integral):
        raise TypeError("error_qubit must be an integer in [0, 2]")
    error_qubit = int(error_qubit)
    if error_qubit not in (0, 1, 2):
        raise ValueError("error_qubit must be in [0, 2]")
    if not isinstance(pauli, str):
        raise TypeError("pauli must be 'X' or 'Z'")
    pauli = pauli.upper()
    if pauli not in ("X", "Z"):
        raise ValueError("pauli must be 'X' or 'Z'")

    QCircuit, _, _, _, X, Z = _load_qpanda_gates()
    circuit = QCircuit()
    circuit << (X(q_code[error_qubit]) if pauli == "X" else Z(q_code[error_qubit]))
    return circuit


def expected_syndrome(error_qubit: int | None) -> tuple[int, int]:
    """Return the deterministic ``(q1, q2)`` syndrome for one correctable error."""

    if error_qubit is None:
        return (0, 0)
    if isinstance(error_qubit, bool) or not isinstance(error_qubit, Integral):
        raise TypeError("error_qubit must be None or an integer in [0, 2]")
    error_qubit = int(error_qubit)
    mapping = {0: (1, 1), 1: (1, 0), 2: (0, 1)}
    if error_qubit not in mapping:
        raise ValueError("error_qubit must be None or in [0, 2]")
    return mapping[error_qubit]


def _validated_logical_state(logical_state: Sequence[complex]) -> np.ndarray:
    state = np.asarray(logical_state, dtype=complex)
    if state.shape != (2,):
        raise ValueError("logical_state must contain exactly two amplitudes")
    if not np.all(np.isfinite(state)):
        raise ValueError("logical_state amplitudes must be finite")
    norm = float(np.vdot(state, state).real)
    if not np.isclose(norm, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("logical_state must be normalized")
    return state.copy()


def _apply_single_qubit(
    state: np.ndarray, gate: np.ndarray, qubit: int, num_qubits: int = 3
) -> np.ndarray:
    out = np.asarray(state, dtype=complex).copy()
    step = 1 << qubit
    block = step << 1
    for base in range(0, 1 << num_qubits, block):
        for offset in range(step):
            i0 = base + offset
            i1 = i0 + step
            a0, a1 = out[i0], out[i1]
            out[i0] = gate[0, 0] * a0 + gate[0, 1] * a1
            out[i1] = gate[1, 0] * a0 + gate[1, 1] * a1
    return out


def _apply_controlled_x(
    state: np.ndarray,
    controls: Sequence[int],
    target: int,
    num_qubits: int = 3,
) -> np.ndarray:
    out = np.asarray(state, dtype=complex).copy()
    target_mask = 1 << target
    control_mask = sum(1 << control for control in controls)
    for basis in range(1 << num_qubits):
        if basis & target_mask:
            continue
        if basis & control_mask == control_mask:
            partner = basis | target_mask
            out[basis], out[partner] = out[partner], out[basis]
    return out


def _encode_reference(state: np.ndarray, code: str) -> np.ndarray:
    state = _apply_controlled_x(state, [0], 1)
    state = _apply_controlled_x(state, [0], 2)
    if code == "phase_flip":
        for qubit in range(3):
            state = _apply_single_qubit(state, _H, qubit)
    return state


def _recover_reference(state: np.ndarray, code: str) -> np.ndarray:
    if code == "phase_flip":
        for qubit in range(3):
            state = _apply_single_qubit(state, _H, qubit)
    state = _apply_controlled_x(state, [0], 1)
    state = _apply_controlled_x(state, [0], 2)
    state = _apply_controlled_x(state, [1, 2], 0)
    return state


def logical_density_matrix(state: Sequence[complex]) -> np.ndarray:
    """Trace out syndrome qubits and return the decoded q0 density matrix."""

    state = np.asarray(state, dtype=complex)
    if state.shape != (8,):
        raise ValueError("state must be a three-qubit state vector of length 8")
    norm = float(np.vdot(state, state).real)
    if not np.isclose(norm, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("state vector must be normalized")

    density = np.zeros((2, 2), dtype=complex)
    for syndrome in range(4):
        base = syndrome << 1  # q0 is the least-significant qubit.
        logical = state[[base, base | 1]]
        density += np.outer(logical, logical.conj())
    return density


def syndrome_probabilities(state: Sequence[complex]) -> np.ndarray:
    """Return probabilities for syndrome integer ``q1 + 2*q2``."""

    state = np.asarray(state, dtype=complex)
    if state.shape != (8,):
        raise ValueError("state must be a three-qubit state vector of length 8")
    norm = float(np.vdot(state, state).real)
    if not np.isclose(norm, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("state vector must be normalized")
    probabilities = np.empty(4, dtype=float)
    for syndrome in range(4):
        base = syndrome << 1
        probabilities[syndrome] = float(
            np.abs(state[base]) ** 2 + np.abs(state[base | 1]) ** 2
        )
    probabilities[np.abs(probabilities) < 1e-15] = 0.0
    return probabilities


def logical_fidelity(
    density_matrix: Sequence[Sequence[complex]], logical_state: Sequence[complex]
) -> float:
    """Return fidelity against a pure target logical qubit state."""

    density = np.asarray(density_matrix, dtype=complex)
    if density.shape != (2, 2):
        raise ValueError("density_matrix must have shape (2, 2)")
    logical_state = _validated_logical_state(logical_state)
    fidelity = float(np.vdot(logical_state, density @ logical_state).real)
    if fidelity < 0 and fidelity > -1e-12:
        fidelity = 0.0
    if fidelity > 1 and fidelity < 1 + 1e-12:
        fidelity = 1.0
    return fidelity


def reference_repetition_recovery(
    code: str,
    logical_state: Sequence[complex] = (1.0, 0.0),
    error_qubit: int | None = None,
    error_pauli: str | None = None,
    return_state: bool = False,
) -> dict[str, Any]:
    """Exact NumPy reference for one encode/error/recover experiment.

    Parameters
    ----------
    code:
        ``"bit_flip"`` or ``"phase_flip"``.
    logical_state:
        Normalized amplitudes ``(alpha, beta)`` of the logical input qubit.
    error_qubit:
        ``None`` for no error or one of ``0, 1, 2``.
    error_pauli:
        Explicit ``"X"`` or ``"Z"``. When omitted, uses ``X`` for the bit-flip
        code and ``Z`` for the phase-flip code. Supplying the opposite Pauli is
        useful for demonstrating the code's documented error-model boundary.
    return_state:
        Include the final three-qubit state vector in the returned dictionary.

    Returns
    -------
    dict
        Contains ``logical_density``, ``logical_fidelity``,
        ``syndrome_probabilities``, and metadata. For a correctable single error,
        fidelity is one (within floating-point precision) for every logical
        input state.
    """

    if code not in ("bit_flip", "phase_flip"):
        raise ValueError("code must be 'bit_flip' or 'phase_flip'")
    logical_state = _validated_logical_state(logical_state)
    if error_qubit is not None:
        if isinstance(error_qubit, bool) or not isinstance(error_qubit, Integral):
            raise TypeError("error_qubit must be None or an integer in [0, 2]")
        error_qubit = int(error_qubit)
        if error_qubit not in (0, 1, 2):
            raise ValueError("error_qubit must be None or in [0, 2]")

    correctable_pauli = "X" if code == "bit_flip" else "Z"
    if error_pauli is None:
        error_pauli = correctable_pauli
    if not isinstance(error_pauli, str):
        raise TypeError("error_pauli must be 'X' or 'Z'")
    error_pauli = error_pauli.upper()
    if error_pauli not in ("X", "Z"):
        raise ValueError("error_pauli must be 'X' or 'Z'")

    state = np.zeros(8, dtype=complex)
    state[0] = logical_state[0]
    state[1] = logical_state[1]
    state = _encode_reference(state, code)

    if error_qubit is not None:
        gate = _X if error_pauli == "X" else _Z
        state = _apply_single_qubit(state, gate, error_qubit)

    state = _recover_reference(state, code)
    density = logical_density_matrix(state)
    syndromes = syndrome_probabilities(state)
    result: dict[str, Any] = {
        "code": code,
        "error_qubit": error_qubit,
        "error_pauli": error_pauli if error_qubit is not None else None,
        "correctable_pauli": correctable_pauli,
        "logical_density": density,
        "logical_fidelity": logical_fidelity(density, logical_state),
        "syndrome_probabilities": syndromes,
    }
    if return_state:
        result["state"] = state
    return result


class ThreeQubitRepetitionCode:
    """Convenience wrapper around the bit-flip or phase-flip repetition code."""

    def __init__(self, code: str = "bit_flip"):
        if code not in ("bit_flip", "phase_flip"):
            raise ValueError("code must be 'bit_flip' or 'phase_flip'")
        self.code = code

    @property
    def correctable_pauli(self) -> str:
        return "X" if self.code == "bit_flip" else "Z"

    def encode(self, q_code: Sequence[Any]):
        return (
            bit_flip_encode(q_code)
            if self.code == "bit_flip"
            else phase_flip_encode(q_code)
        )

    def recover(self, q_code: Sequence[Any]):
        return (
            bit_flip_recover(q_code)
            if self.code == "bit_flip"
            else phase_flip_recover(q_code)
        )

    def reference(
        self,
        logical_state: Sequence[complex] = (1.0, 0.0),
        error_qubit: int | None = None,
        error_pauli: str | None = None,
        return_state: bool = False,
    ) -> dict[str, Any]:
        return reference_repetition_recovery(
            self.code,
            logical_state=logical_state,
            error_qubit=error_qubit,
            error_pauli=error_pauli,
            return_state=return_state,
        )
