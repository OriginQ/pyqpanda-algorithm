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

"""Reusable multi-qubit SWAP Test for quantum-state overlap estimation.

For pure states |psi> and |phi>, the ancilla-zero probability is

    p(0) = (1 + |<psi|phi>|**2) / 2,

so the squared overlap is 2*p(0)-1. The same circuit estimates Tr(rho*sigma)
for mixed-state inputs prepared by compatible caller circuits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Optional, Sequence

from pyqpanda3.core import CPUQVM, H, QProg, SWAP


CircuitFactory = Callable[[Sequence[Any]], Any]


@dataclass(frozen=True)
class SwapTestResult:
    """Measurement summary returned by SwapTest.run()."""

    overlap_squared: float
    overlap: float
    zero_probability: float
    bitstring: str
    shots: int


def overlap_squared_from_zero_probability(zero_probability: float) -> float:
    """Convert ancilla P(0) to the ideal squared state overlap.

    Shot noise or hardware noise can push the raw estimator slightly outside
    the physical interval. The returned value is therefore clamped to [0, 1].
    """

    if isinstance(zero_probability, bool) or not isinstance(
        zero_probability, (int, float)
    ):
        raise ValueError("zero_probability must be a finite real number")

    zero_probability = float(zero_probability)
    if not math.isfinite(zero_probability):
        raise ValueError("zero_probability must be a finite real number")
    if zero_probability < 0.0 or zero_probability > 1.0:
        raise ValueError("zero_probability must be in [0, 1]")

    return min(1.0, max(0.0, 2.0 * zero_probability - 1.0))


class SwapTest:
    """Estimate overlap between two equally sized prepared quantum states.

    Parameters
    ----------
    qubits_per_state:
        Number of qubits in each state register.
    prepare_left:
        Callable taking the left register and returning its preparation circuit.
    prepare_right:
        Callable taking the right register and returning its preparation circuit.
    shots:
        Default number of CPU-simulator shots for run().

    The implementation uses one ancilla plus two equal-width state registers.
    Each corresponding register pair is joined by a SWAP gate controlled by
    the ancilla, giving the standard multi-qubit SWAP Test without imposing a
    specific state-preparation method on callers.
    """

    def __init__(
        self,
        qubits_per_state: int,
        prepare_left: CircuitFactory,
        prepare_right: CircuitFactory,
        shots: int = 1024,
    ):
        if isinstance(qubits_per_state, bool) or not isinstance(qubits_per_state, int):
            raise ValueError("qubits_per_state must be an integer")
        if qubits_per_state <= 0:
            raise ValueError("qubits_per_state must be positive")
        if not callable(prepare_left) or not callable(prepare_right):
            raise ValueError("prepare_left and prepare_right must be callable")
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")

        self.qubits_per_state = qubits_per_state
        self.prepare_left = prepare_left
        self.prepare_right = prepare_right
        self.shots = shots

    def build_program(self):
        """Build the SWAP-test program and return its logical registers."""

        qubits = QProg(1 + 2 * self.qubits_per_state).qubits()
        ancilla = qubits[0]
        left = qubits[1 : 1 + self.qubits_per_state]
        right = qubits[1 + self.qubits_per_state :]

        program = QProg()
        program << self.prepare_left(left)
        program << self.prepare_right(right)
        program << H(ancilla)

        for left_qubit, right_qubit in zip(left, right):
            program << SWAP(left_qubit, right_qubit).control([ancilla])

        program << H(ancilla)
        return program, ancilla, left, right

    def run(self, shots: Optional[int] = None) -> SwapTestResult:
        """Run the SWAP Test on the CPU simulator."""

        if shots is None:
            shots = self.shots
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")

        program, ancilla, _, _ = self.build_program()
        machine = CPUQVM()
        machine.run(program, shots)
        probabilities = machine.result().get_prob_dict([ancilla])

        if not probabilities:
            raise RuntimeError("SWAP Test returned no ancilla probabilities")

        zero_probability = float(probabilities.get("0", 0.0))
        overlap_squared = overlap_squared_from_zero_probability(zero_probability)
        bitstring = max(probabilities, key=probabilities.get)

        return SwapTestResult(
            overlap_squared=overlap_squared,
            overlap=math.sqrt(overlap_squared),
            zero_probability=zero_probability,
            bitstring=bitstring,
            shots=shots,
        )
