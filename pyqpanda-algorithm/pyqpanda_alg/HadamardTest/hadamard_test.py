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

"""Hadamard-test expectation estimation.

The Hadamard test estimates the complex overlap <psi|U|psi> with one ancilla.
The real component is measured in the ancilla X basis and the imaginary
component in the ancilla Y basis. This implementation keeps state preparation
and the target unitary as callables so it composes with the existing
PyQPanda3 circuit-building style used by pyqpanda_alg.

Qubit convention
----------------
Data qubits are supplied to user callbacks in q0-first order. One ancilla is
allocated after the data register and is never passed to those callbacks.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np


Component = Literal["real", "imag"]


def reference_expectation(state, unitary) -> complex:
    """Return the exact state-vector value <state|unitary|state>.

    This small NumPy reference is useful for examples and for checking the
    sampling estimator independently of the quantum runtime.

    Args:
        state: One-dimensional normalized complex state vector.
        unitary: Square unitary matrix acting on state.

    Raises:
        ValueError: If shapes, normalization, or unitarity are invalid.
    """
    vector = np.asarray(state, dtype=np.complex128)
    matrix = np.asarray(unitary, dtype=np.complex128)

    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("state must be a non-empty one-dimensional vector")
    if vector.size & (vector.size - 1):
        raise ValueError("state dimension must be a power of two")
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("unitary shape must match the state dimension")

    norm = float(np.vdot(vector, vector).real)
    if not np.isclose(norm, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("state must be normalized")

    identity = np.eye(vector.size, dtype=np.complex128)
    if not np.allclose(matrix.conj().T @ matrix, identity, rtol=1e-10, atol=1e-12):
        raise ValueError("unitary must satisfy U†U = I")

    return complex(np.vdot(vector, matrix @ vector))


class HadamardTest:
    """Estimate <psi|U|psi> with a one-ancilla Hadamard test.

    Args:
        state_preparation: Callable receiving the q0-first data-qubit list and
            returning a PyQPanda3 circuit/program that prepares |psi>.
        unitary: Callable receiving the same data-qubit list and returning a
            controllable PyQPanda3 circuit/program implementing U.
        qnumber: Number of data qubits. The ancilla is allocated separately.

    The estimator intentionally does not assume that U is Hermitian.
    estimate_complex therefore measures both real and imaginary parts.
    """

    def __init__(
        self,
        state_preparation: Callable,
        unitary: Callable,
        qnumber: int,
    ):
        if not callable(state_preparation):
            raise TypeError("state_preparation must be callable")
        if not callable(unitary):
            raise TypeError("unitary must be callable")
        if isinstance(qnumber, bool) or not isinstance(qnumber, int) or qnumber <= 0:
            raise ValueError("qnumber must be a positive integer")

        self.state_preparation = state_preparation
        self.unitary = unitary
        self.qnumber = qnumber

    @staticmethod
    def _normalize_component(component: str) -> Component:
        value = str(component).lower()
        if value not in ("real", "imag"):
            raise ValueError("component must be 'real' or 'imag'")
        return value

    def build_program(self, component: Component = "real"):
        """Build one Hadamard-test program and return (program, ancilla).

        For the real part, the final H maps ancilla X to computational Z.
        For the imaginary part, RX(pi/2) maps ancilla Y to computational Z.
        Consequently P(0)-P(1) is respectively Re or Im of the overlap.
        """
        from pyqpanda3.core import H, RX, QProg

        component = self._normalize_component(component)
        register = QProg(self.qnumber + 1)
        qubits = register.qubits()
        data_qubits = qubits[: self.qnumber]
        ancilla = qubits[self.qnumber]

        preparation = self.state_preparation(data_qubits)
        controlled_unitary = self.unitary(data_qubits)

        if preparation is None:
            raise ValueError("state_preparation must return a circuit or program")
        if controlled_unitary is None:
            raise ValueError("unitary must return a controllable circuit or program")

        program = QProg()
        program << preparation
        program << H(ancilla)
        program << controlled_unitary.control([ancilla])

        if component == "real":
            program << H(ancilla)
        else:
            program << RX(ancilla, np.pi / 2)

        return program, ancilla

    def estimate(self, component: Component = "real", shots: int = 4096) -> float:
        """Estimate one overlap component from ancilla probabilities."""
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")

        from pyqpanda3.core import CPUQVM

        component = self._normalize_component(component)
        program, ancilla = self.build_program(component)

        machine = CPUQVM()
        machine.run(program, shots)
        probabilities = machine.result().get_prob_dict([ancilla])

        p0 = float(probabilities.get("0", 0.0))
        p1 = float(probabilities.get("1", 0.0))
        return p0 - p1

    def estimate_complex(self, shots: int = 4096) -> complex:
        """Estimate both components and return a complex expectation value."""
        return complex(
            self.estimate("real", shots=shots),
            self.estimate("imag", shots=shots),
        )
