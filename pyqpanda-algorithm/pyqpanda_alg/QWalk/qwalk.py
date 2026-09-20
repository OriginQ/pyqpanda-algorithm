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

"""Discrete-time coined quantum walks on cyclic position registers.

The quantum circuit path uses PyQPanda3 lazily so the independent NumPy
reference implementation remains usable in environments where PyQPanda3 is not
installed. Position qubits are interpreted little-endian: ``q_position[0]`` is
the least-significant bit of the integer position.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Any

import numpy as np


_HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)


def _require_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_register(q_position: Sequence[Any], q_coin: Any) -> list[Any]:
    q_position = list(q_position)
    if not q_position:
        raise ValueError("q_position must contain at least one qubit")
    for index, qubit in enumerate(q_position):
        if qubit in q_position[:index]:
            raise ValueError("q_position must not contain duplicate qubits")
    if q_coin in q_position:
        raise ValueError("q_coin must be distinct from all position qubits")
    return q_position


def _load_qpanda_gates():
    """Import the small PyQPanda3 surface needed by the circuit builders."""

    try:
        from pyqpanda3.core import QCircuit, H, X
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ImportError(
            "PyQPanda3 is required for QWalk circuit construction; "
            "reference_walk_cycle() remains available without it"
        ) from exc
    return QCircuit, H, X


def controlled_increment_cycle(q_position: Sequence[Any], q_control: Any):
    """Return a controlled ``+1 mod 2**n`` circuit on ``q_position``.

    ``q_position[0]`` is the least-significant bit. The increment is active
    only when ``q_control`` is ``|1>``. No ancilla qubits are required.
    """

    q_position = list(q_position)
    if not q_position:
        raise ValueError("q_position must contain at least one qubit")
    if q_control in q_position:
        raise ValueError("q_control must be distinct from all position qubits")

    QCircuit, _, X = _load_qpanda_gates()
    circuit = QCircuit()

    # Toggle a high bit iff every lower bit was 1 before the increment. Walking
    # from high to low preserves those lower control values until they are used.
    for target in range(len(q_position) - 1, 0, -1):
        controls = [q_control] + q_position[:target]
        circuit << X(q_position[target]).control(controls)
    circuit << X(q_position[0]).control(q_control)
    return circuit


def controlled_decrement_cycle(q_position: Sequence[Any], q_control: Any):
    """Return a controlled ``-1 mod 2**n`` circuit on ``q_position``.

    This is the exact inverse of :func:`controlled_increment_cycle`; all gates
    are self-inverse, so reversing their order implements modular decrement.
    """

    q_position = list(q_position)
    if not q_position:
        raise ValueError("q_position must contain at least one qubit")
    if q_control in q_position:
        raise ValueError("q_control must be distinct from all position qubits")

    QCircuit, _, X = _load_qpanda_gates()
    circuit = QCircuit()
    circuit << X(q_position[0]).control(q_control)
    for target in range(1, len(q_position)):
        controls = [q_control] + q_position[:target]
        circuit << X(q_position[target]).control(controls)
    return circuit


def conditional_cycle_shift(q_position: Sequence[Any], q_coin: Any):
    """Build the coined-walk shift on a cycle.

    Coin ``|0>`` moves the walker one site left and coin ``|1>`` moves it one
    site right, with wraparound modulo ``2**len(q_position)``.
    """

    q_position = _validate_register(q_position, q_coin)
    QCircuit, _, X = _load_qpanda_gates()
    circuit = QCircuit()

    # Activate the decrement on the original coin-|0> subspace by temporarily
    # flipping the coin. Restore it before acting on the coin-|1> subspace.
    circuit << X(q_coin)
    circuit << controlled_decrement_cycle(q_position, q_coin)
    circuit << X(q_coin)
    circuit << controlled_increment_cycle(q_position, q_coin)
    return circuit


def walk_step(
    q_position: Sequence[Any],
    q_coin: Any,
    coin_operator: Callable[[Any], Any] | None = None,
):
    """Build one coined-walk step: coin operation followed by cyclic shift.

    Parameters
    ----------
    q_position:
        Position register in little-endian order.
    q_coin:
        Coin qubit, distinct from every position qubit.
    coin_operator:
        Optional callable receiving ``q_coin`` and returning a PyQPanda gate or
        circuit. The default is a Hadamard coin.
    """

    q_position = _validate_register(q_position, q_coin)
    QCircuit, H, _ = _load_qpanda_gates()
    circuit = QCircuit()
    circuit << (H(q_coin) if coin_operator is None else coin_operator(q_coin))
    circuit << conditional_cycle_shift(q_position, q_coin)
    return circuit


def coined_walk_cycle(
    q_position: Sequence[Any],
    q_coin: Any,
    steps: int,
    initial_position: int = 0,
    initial_coin: int = 0,
    coin_operator: Callable[[Any], Any] | None = None,
):
    """Build a complete discrete-time coined quantum walk on a cycle.

    The circuit prepares a computational-basis initial position and coin, then
    applies ``steps`` repetitions of ``coin -> conditional shift``.

    Parameters
    ----------
    q_position:
        Position register. ``q_position[0]`` is the least-significant bit.
    q_coin:
        Coin qubit, distinct from every position qubit.
    steps:
        Number of walk steps. Zero returns state preparation only.
    initial_position:
        Integer in ``[0, 2**len(q_position))``.
    initial_coin:
        Either 0 or 1.
    coin_operator:
        Optional custom coin gate/circuit callable. Default: Hadamard.
    """

    q_position = _validate_register(q_position, q_coin)
    steps = _require_nonnegative_int("steps", steps)
    initial_position = _require_nonnegative_int("initial_position", initial_position)
    if initial_position >= 2 ** len(q_position):
        raise ValueError("initial_position does not fit in q_position")
    if isinstance(initial_coin, bool) or not isinstance(initial_coin, Integral):
        raise TypeError("initial_coin must be 0 or 1")
    initial_coin = int(initial_coin)
    if initial_coin not in (0, 1):
        raise ValueError("initial_coin must be 0 or 1")

    QCircuit, _, X = _load_qpanda_gates()
    circuit = QCircuit()

    for bit, qubit in enumerate(q_position):
        if (initial_position >> bit) & 1:
            circuit << X(qubit)
    if initial_coin:
        circuit << X(q_coin)

    for _ in range(steps):
        circuit << walk_step(q_position, q_coin, coin_operator=coin_operator)
    return circuit


def _validated_coin_matrix(coin_matrix: np.ndarray | None) -> np.ndarray:
    if coin_matrix is None:
        return _HADAMARD
    matrix = np.asarray(coin_matrix, dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("coin_matrix must have shape (2, 2)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("coin_matrix must contain finite values")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=1e-12, rtol=1e-12):
        raise ValueError("coin_matrix must be unitary")
    return matrix


def reference_walk_cycle(
    num_position_qubits: int,
    steps: int,
    initial_position: int = 0,
    initial_coin: int = 0,
    coin_matrix: np.ndarray | None = None,
    return_state: bool = False,
):
    """Exact NumPy reference for the same coined walk.

    This function intentionally has no PyQPanda dependency. It is both a small
    problem reference oracle and a convenient way to reason about expected
    distributions before constructing a quantum circuit.

    The returned state, when requested, has shape ``(2, 2**n)`` and is indexed
    as ``state[coin, position]``.
    """

    num_position_qubits = _require_nonnegative_int(
        "num_position_qubits", num_position_qubits
    )
    if num_position_qubits == 0:
        raise ValueError("num_position_qubits must be at least 1")
    steps = _require_nonnegative_int("steps", steps)
    initial_position = _require_nonnegative_int("initial_position", initial_position)
    size = 2**num_position_qubits
    if initial_position >= size:
        raise ValueError("initial_position does not fit in the position register")
    if isinstance(initial_coin, bool) or not isinstance(initial_coin, Integral):
        raise TypeError("initial_coin must be 0 or 1")
    initial_coin = int(initial_coin)
    if initial_coin not in (0, 1):
        raise ValueError("initial_coin must be 0 or 1")

    coin_matrix = _validated_coin_matrix(coin_matrix)
    state = np.zeros((2, size), dtype=complex)
    state[initial_coin, initial_position] = 1.0

    for _ in range(steps):
        state = coin_matrix @ state
        shifted = np.zeros_like(state)
        shifted[0] = np.roll(state[0], -1)  # coin 0 -> left
        shifted[1] = np.roll(state[1], 1)   # coin 1 -> right
        state = shifted

    if return_state:
        return state
    probabilities = np.sum(np.abs(state) ** 2, axis=0)
    # Numerical cleanup keeps the public contract exact enough for downstream
    # invariants without hiding a genuine normalization defect.
    probabilities[np.abs(probabilities) < 1e-15] = 0.0
    return probabilities


def principal_displacement_moments(
    probabilities: Sequence[float], origin: int = 0
) -> tuple[float, float]:
    """Return mean and variance of principal signed displacement on the cycle.

    For an even cycle, the antipodal site is assigned displacement ``-N/2``.
    This branch convention is explicit because a circular walk does not have a
    globally unique linear mean.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(probabilities))
    if total <= 0:
        raise ValueError("probabilities must have positive total weight")

    size = int(probabilities.size)
    if isinstance(origin, bool) or not isinstance(origin, Integral):
        raise TypeError("origin must be an integer")
    origin = int(origin) % size
    p = probabilities / total
    displacement = ((np.arange(size) - origin + size // 2) % size) - size // 2
    mean = float(np.dot(p, displacement))
    variance = float(np.dot(p, (displacement - mean) ** 2))
    return mean, variance


class CoinedQuantumWalkCycle:
    """Convenience object for a Hadamard coined walk on a ``2**n`` cycle.

    ``circuit()`` builds the PyQPanda3 circuit. ``reference()`` returns the
    exact independent NumPy distribution and therefore remains usable when
    PyQPanda3 is unavailable.
    """

    def __init__(
        self,
        num_position_qubits: int,
        steps: int,
        initial_position: int = 0,
        initial_coin: int = 0,
    ):
        num_position_qubits = _require_nonnegative_int(
            "num_position_qubits", num_position_qubits
        )
        if num_position_qubits == 0:
            raise ValueError("num_position_qubits must be at least 1")
        self.num_position_qubits = num_position_qubits
        self.steps = _require_nonnegative_int("steps", steps)
        self.initial_position = _require_nonnegative_int(
            "initial_position", initial_position
        )
        if self.initial_position >= 2**self.num_position_qubits:
            raise ValueError("initial_position does not fit in the position register")
        if isinstance(initial_coin, bool) or not isinstance(initial_coin, Integral):
            raise TypeError("initial_coin must be 0 or 1")
        self.initial_coin = int(initial_coin)
        if self.initial_coin not in (0, 1):
            raise ValueError("initial_coin must be 0 or 1")

    @property
    def size(self) -> int:
        return 2**self.num_position_qubits

    def circuit(self):
        """Build the walk on contiguous qubits ``0..n`` (coin is qubit ``n``)."""

        q_position = list(range(self.num_position_qubits))
        q_coin = self.num_position_qubits
        return coined_walk_cycle(
            q_position,
            q_coin,
            self.steps,
            initial_position=self.initial_position,
            initial_coin=self.initial_coin,
        )

    def reference(self) -> np.ndarray:
        """Return the exact independent position distribution."""

        return reference_walk_cycle(
            self.num_position_qubits,
            self.steps,
            initial_position=self.initial_position,
            initial_coin=self.initial_coin,
        )

    def run_exact(self) -> np.ndarray:
        """Simulate the PyQPanda circuit exactly and return position probability.

        The coin is the most-significant qubit in this convenience layout, so
        the little-endian state vector reshapes to ``(coin, position)``.
        """

        try:
            from pyqpanda3.quantum_info import StateVector
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise ImportError("PyQPanda3 is required for run_exact()") from exc

        state = np.asarray(
            StateVector(self.num_position_qubits + 1).evolve(self.circuit()).ndarray(),
            dtype=complex,
        ).reshape(2, self.size)
        probabilities = np.sum(np.abs(state) ** 2, axis=0)
        probabilities[np.abs(probabilities) < 1e-15] = 0.0
        return probabilities

    def moments(self) -> tuple[float, float]:
        """Return reference mean/variance of principal displacement."""

        return principal_displacement_moments(
            self.reference(), origin=self.initial_position
        )
