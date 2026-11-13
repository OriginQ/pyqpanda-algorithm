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

"""Quantum counting by phase estimation of the Grover operator."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
from pyqpanda3.core import CPUQVM, H, QCircuit, QProg, RX

from ..Grover import amp_operator, mark_data_reflection
from ..plugin import QFT, apply_QGate
from .decoding import decode_phase_distribution, phase_to_probability, positive_integer


@dataclass(frozen=True)
class QuantumCountingResult:
    """Structured result returned by QuantumCounting."""

    estimated_count: Optional[float]
    rounded_count: Optional[int]
    phase: float
    phase_state: str
    phase_probability: float
    search_space_size: int
    estimated_probability: Optional[float] = None
    pair_probability: Optional[float] = None
    count_available: bool = True


class QuantumCounting:
    """Estimate marked probability, and cardinality for uniform preparation.

    Quantum counting combines Grover amplitude amplification with quantum phase
    estimation. If the search space has size N and contains M marked states,
    the normalized Grover eigenphase phi obeys M / N = sin(pi * phi) ** 2.

    The implementation follows the QFT and phase-register convention already
    used by pyqpanda_alg.QAE so callers can compose it with this package.

    Args:
        qnumber: Number of search qubits. Search-space size is 2 ** qnumber.
        phase_bits: Number of phase-estimation qubits.
        flip_operator: Callable producing the Grover phase-oracle circuit.
        mark_data: Convenience alternative to flip_operator. A bit string or
            sequence of bit strings marked through Grover.mark_data_reflection.
        in_operator: Optional state-preparation callable. When omitted, a
            uniform Hadamard superposition is used.
        shots: Native simulator run argument. Readout uses exact probabilities,
            not sampled counts; it does not provide shot uncertainty.
        zero_flip: Optional circuit implementing I - 2|0><0| exactly, with
            the same global phase as the default reflection.

    With custom preparation, estimated_probability is the marked probability
    of that state and both count fields are None. It is not generally M/N.
    The oracle must implement I - 2 Pi_good, including its global phase.
    Circuit factories must be deterministic. The Grover operator is constructed
    once per build; its preparation factory is used for A and A-dagger.
    Peak estimates have finite phase resolution; rounded counts are not guaranteed.

    Exactly one of flip_operator and mark_data must be supplied.
    """

    def __init__(
        self,
        qnumber: int,
        phase_bits: int = 6,
        flip_operator: Optional[Callable[[Sequence[int]], QCircuit]] = None,
        mark_data: Optional[Union[str, Sequence[str]]] = None,
        in_operator: Optional[Callable[[Sequence[int]], QCircuit]] = None,
        shots: int = 2048,
        zero_flip: Optional[Callable[[Sequence[int]], QCircuit]] = None,
    ) -> None:
        positive_integer(qnumber, "qnumber")
        positive_integer(phase_bits, "phase_bits")
        positive_integer(shots, "shots")
        if phase_bits > 20:
            raise ValueError("phase_bits above 20 would build an impractically large circuit")
        for name, factory in (("flip_operator", flip_operator),
                              ("in_operator", in_operator), ("zero_flip", zero_flip)):
            if factory is not None and not callable(factory):
                raise ValueError(f"{name} must be callable")
        if (flip_operator is None) == (mark_data is None):
            raise ValueError("provide exactly one of flip_operator or mark_data")

        self.qnumber = qnumber
        self.phase_bits = phase_bits
        self.in_operator = in_operator
        self.zero_flip = zero_flip
        self.count_available = in_operator is None
        self.shots = shots
        self.search_space_size = 1 << qnumber

        if mark_data is not None:
            states = [mark_data] if isinstance(mark_data, str) else list(mark_data)
            if len(states) > self.search_space_size:
                raise ValueError("mark_data cannot exceed the search-space size")
            for state in states:
                if not isinstance(state, str):
                    raise TypeError("each marked state must be a bit string")
                if len(state) != qnumber or set(state) - {"0", "1"}:
                    raise ValueError(
                        "each marked state must be a binary string of length qnumber"
                    )

            if len(set(states)) != len(states):
                raise ValueError("mark_data contains duplicate states")

            def marked_state_oracle(qubits):
                return mark_data_reflection(qubits=qubits, mark_data=states)

            self.flip_operator = marked_state_oracle
        else:
            self.flip_operator = flip_operator

    def _initial_state(self, search_qubits: Sequence[int]) -> QCircuit:
        if self.in_operator is None:
            return apply_QGate(list(search_qubits), H)
        return self.in_operator(search_qubits)

    def _grover_operator(self, search_qubits: Sequence[int]) -> QCircuit:
        operator = amp_operator(
            in_operator=self.in_operator,
            flip_operator=self.flip_operator,
            q_input=list(search_qubits),
            q_flip=list(search_qubits),
            q_zero=list(search_qubits),
            zero_flip=self.zero_flip,
        )
        # Credit: mnn31 PR #47 identifies and repairs this controlled-phase
        # requirement. amp_operator is -Q; phase estimation requires Q.
        operator << RX(search_qubits[0], 2 * np.pi)
        return operator

    @staticmethod
    def count_from_phase(qnumber: int, phase: float) -> float:
        """Convert a normalized Grover eigenphase to a marked-state count."""
        positive_integer(qnumber, "qnumber")
        return float((1 << qnumber) * phase_to_probability(phase))

    def circuit(self):
        """Build the quantum-counting program and return its phase qubits."""
        qubits = QProg(self.qnumber + self.phase_bits).qubits()
        search_qubits = list(qubits[: self.qnumber])
        phase_qubits = list(qubits[self.qnumber :])

        program = QProg()
        program << self._initial_state(search_qubits)
        program << apply_QGate(phase_qubits, H)

        grover = self._grover_operator(search_qubits)
        for index, control_qubit in enumerate(phase_qubits):
            controlled_grover = grover.control([control_qubit])
            for _ in range(1 << index):
                program << controlled_grover

        program << QFT(phase_qubits).dagger()
        return program, phase_qubits

    def run(self) -> QuantumCountingResult:
        """Execute quantum counting and return the dominant count estimate."""
        machine = CPUQVM()
        program, phase_qubits = self.circuit()
        machine.run(program, self.shots)
        probabilities = machine.result().get_prob_dict(phase_qubits)
        if not probabilities:
            raise RuntimeError("quantum counting returned no phase probabilities")

        estimate = decode_phase_distribution(probabilities)
        count = (self.search_space_size * estimate.probability
                 if self.count_available else None)
        return QuantumCountingResult(
            estimated_count=count,
            rounded_count=(max(0, min(self.search_space_size, int(round(count))))
                           if count is not None else None),
            phase=estimate.phase, phase_state=estimate.phase_state,
            phase_probability=estimate.phase_probability,
            search_space_size=self.search_space_size,
            estimated_probability=estimate.probability,
            pair_probability=estimate.pair_probability,
            count_available=self.count_available,
        )
