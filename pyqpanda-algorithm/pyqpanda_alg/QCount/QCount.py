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

from ..plugin import hadamard_circuit
from ..QuantumCounting import QuantumCounting
from ..QuantumCounting.decoding import phase_to_probability


CircuitFactory = Callable[[Sequence[Any]], Any]


@dataclass(frozen=True)
class QuantumCountResult:
    """Compatibility result: marked_fraction is probability before rounding.

    Custom preparation has no cardinality estimate: marked_count and
    estimated_count are None. shots records the native simulator run argument,
    not a sample count: run() reads exact statevector probabilities.
    """

    marked_count: Optional[int]
    marked_fraction: float
    phase: float
    bitstring: str
    probability: float
    shots: int
    estimated_count: Optional[float] = None
    pair_probability: Optional[float] = None
    count_available: bool = True


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
    marked_fraction = phase_to_probability(phase)
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
    """Compatibility API sharing the QuantumCounting circuit and decoder.

    Uniform preparation produces a count. Custom preparation produces marked
    probability only, with count fields None. Factories must implement the
    exact reflection and preparation contracts documented by QuantumCounting.

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
        Optional I - 2|0><0| reflection, including the default global phase.
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
        self._uniform_preparation = in_operator is None
        self.in_operator = hadamard_circuit if in_operator is None else in_operator
        self.zero_flip = zero_flip
        self.shots = shots

    def _counter(self, shots):
        return QuantumCounting(
            qnumber=self.search_qubits, phase_bits=self.precision_bits,
            flip_operator=self.flip_operator,
            in_operator=(None if self._uniform_preparation
                         and self.in_operator is hadamard_circuit
                         else self.in_operator),
            zero_flip=self.zero_flip, shots=shots,
        )

    def build_program(self):
        """Build through the shared engine; retain the original register return."""
        program, counting = self._counter(self.shots).circuit()
        return program, list(range(self.search_qubits)), counting

    def run(self, shots: Optional[int] = None) -> QuantumCountResult:
        """Read exact CPU probabilities through the shared counting engine."""
        shots = self.shots if shots is None else shots
        result = self._counter(shots).run()
        return QuantumCountResult(
            marked_count=result.rounded_count,
            marked_fraction=result.estimated_probability,
            phase=result.phase, bitstring=result.phase_state,
            probability=result.phase_probability, shots=shots,
            estimated_count=result.estimated_count,
            pair_probability=result.pair_probability,
            count_available=result.count_available,
        )
