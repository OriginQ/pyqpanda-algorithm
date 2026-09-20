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

"""Quantum Counting built on PyQPanda3 amplitude amplification.

Quantum Counting estimates the number of marked states without requiring the
caller to know that number in advance. It applies phase estimation to the
Grover amplitude-amplification operator and converts the measured eigenphase
back to a marked-state count.

For a search space of size N=2**n with M marked states, Grover's eigenphases
are +/-2*theta, where sin(theta)**2 = M/N. Phase estimation therefore yields
phi=theta/pi (or its mirror 1-phi), and M = N * sin(pi * phi)**2.

The public decoding helpers are independent of PyQPanda runtime state so
measured phase/count results can also be post-processed separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Optional, Sequence

from pyqpanda3.core import CPUQVM, H, QProg

from pyqpanda_alg.Grover.Grover_core import amp_operator
from ..plugin import QFT, hadamard_circuit


CircuitFactory = Callable[[Sequence[Any]], Any]


@dataclass(frozen=True)
class QuantumCountResult:
    """A Quantum Counting estimate and its measured phase evidence."""

    marked_count: int
    marked_fraction: float
    phase: float
    bitstring: str
    probability: float
    shots: int


def _validate_search_qubits(search_qubits: int) -> int:
    if isinstance(search_qubits, bool) or not isinstance(search_qubits, int):
        raise ValueError("search_qubits must be an integer")
    if search_qubits <= 0:
        raise ValueError("search_qubits must be positive")
    return search_qubits


def _validate_precision_bits(precision_bits: int) -> int:
    if isinstance(precision_bits, bool) or not isinstance(precision_bits, int):
        raise ValueError("precision_bits must be an integer")
    if precision_bits <= 0:
        raise ValueError("precision_bits must be positive")
    if precision_bits > 20:
        raise ValueError("precision_bits above 20 would build an impractically large circuit")
    return precision_bits


def phase_to_count(phase: float, search_qubits: int) -> int:
    """Convert a Grover eigenphase into an estimated marked-state count.

    The conversion is symmetric for the two Grover eigenphases phi and 1-phi.
    The final estimate is rounded and clamped to the physical interval
    [0, 2**search_qubits].
    """

    search_qubits = _validate_search_qubits(search_qubits)
    if isinstance(phase, bool) or not isinstance(phase, (int, float)):
        raise ValueError("phase must be a finite real number")

    phase = float(phase)
    if not math.isfinite(phase):
        raise ValueError("phase must be a finite real number")

    phase %= 1.0
    marked_fraction = math.sin(math.pi * phase) ** 2
    search_size = 1 << search_qubits
    estimate = int(round(search_size * marked_fraction))
    return min(search_size, max(0, estimate))


def count_from_bitstring(bitstring: str, search_qubits: int) -> int:
    """Decode a phase-register bitstring directly to a marked-state count."""

    if not isinstance(bitstring, str) or not bitstring:
        raise ValueError("bitstring must be a non-empty binary string")
    if any(bit not in "01" for bit in bitstring):
        raise ValueError("bitstring must contain only '0' and '1'")

    phase = int(bitstring, 2) / float(1 << len(bitstring))
    return phase_to_count(phase, search_qubits)


class QuantumCount:
    """Estimate how many states are marked by a Grover phase oracle.

    Parameters
    ----------
    search_qubits:
        Number of qubits in the search register.
    flip_operator:
        Callable f(qubits) returning the phase-marking circuit, with the same
        contract used by Grover.
    precision_bits:
        Number of phase-estimation qubits. More bits improve count resolution
        while increasing controlled-Grover depth exponentially.
    in_operator:
        Search-state preparation circuit. Defaults to Hadamards.
    zero_flip:
        Optional custom reflection about zero, forwarded to amp_operator.
    shots:
        Default number of simulator shots used by run().
    """

    def __init__(
        self,
        search_qubits: int,
        flip_operator: CircuitFactory,
        precision_bits: int = 6,
        in_operator: Optional[CircuitFactory] = None,
        zero_flip: Optional[CircuitFactory] = None,
        shots: int = 1024,
    ):
        self.search_qubits = _validate_search_qubits(search_qubits)
        self.precision_bits = _validate_precision_bits(precision_bits)

        if not callable(flip_operator):
            raise ValueError("flip_operator must be callable")
        if in_operator is not None and not callable(in_operator):
            raise ValueError("in_operator must be callable")
        if zero_flip is not None and not callable(zero_flip):
            raise ValueError("zero_flip must be callable")
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")

        self.flip_operator = flip_operator
        self.in_operator = hadamard_circuit if in_operator is None else in_operator
        self.zero_flip = zero_flip
        self.shots = shots

    def build_program(self):
        """Build the counting program and return it with both qubit registers."""

        qubits = QProg(self.search_qubits + self.precision_bits).qubits()
        search = qubits[: self.search_qubits]
        counting = qubits[self.search_qubits :]

        program = QProg()
        program << self.in_operator(search)
        for qubit in counting:
            program << H(qubit)

        grover_step = amp_operator(
            q_input=search,
            q_flip=search,
            q_zero=search,
            in_operator=self.in_operator,
            flip_operator=self.flip_operator,
            zero_flip=self.zero_flip,
        )

        for power, control in enumerate(counting):
            for _ in range(1 << power):
                program << grover_step.control([control])

        program << QFT(counting).dagger()
        return program, search, counting

    def run(self, shots: Optional[int] = None) -> QuantumCountResult:
        """Run Quantum Counting on the CPU simulator and return the estimate."""

        if shots is None:
            shots = self.shots
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")

        program, _, counting = self.build_program()
        machine = CPUQVM()
        machine.run(program, shots)

        probabilities = machine.result().get_prob_dict(counting)
        if not probabilities:
            raise RuntimeError("quantum counting returned no phase-register probabilities")

        bitstring = max(probabilities, key=probabilities.get)
        phase = int(bitstring, 2) / float(1 << self.precision_bits)
        marked_count = phase_to_count(phase, self.search_qubits)
        search_size = 1 << self.search_qubits

        return QuantumCountResult(
            marked_count=marked_count,
            marked_fraction=marked_count / float(search_size),
            phase=phase,
            bitstring=bitstring,
            probability=float(probabilities[bitstring]),
            shots=shots,
        )
