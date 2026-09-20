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

"""Bernstein-Vazirani hidden-bitstring algorithm for PyQPanda3.

The algorithm learns an n-bit secret string s from the Boolean oracle

    f(x) = s · x XOR b

with one quantum oracle query. A deterministic classical reference oracle is
included so callers can validate oracle contracts independently of a quantum
backend.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from pyqpanda3.core import CNOT, CPUQVM, H, QCircuit, QProg, X

from ..plugin import parse_quantum_result_dict


BitTuple = Tuple[int, ...]


def _normalise_bits(bits: Sequence[int] | str, *, name: str) -> BitTuple:
    """Return a validated tuple containing only 0/1 values."""
    if isinstance(bits, str):
        if not bits:
            raise ValueError(f"{name} must contain at least one bit")
        if any(ch not in "01" for ch in bits):
            raise ValueError(f"{name} must contain only '0' and '1'")
        return tuple(int(ch) for ch in bits)

    try:
        values = tuple(int(bit) for bit in bits)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a bit sequence") from exc

    if not values:
        raise ValueError(f"{name} must contain at least one bit")
    if any(bit not in (0, 1) for bit in values):
        raise ValueError(f"{name} must contain only 0 and 1")
    return values


def bitstring(bits: Sequence[int] | str) -> str:
    """Return a validated bit sequence in canonical string form."""
    return "".join(str(bit) for bit in _normalise_bits(bits, name="bits"))


def evaluate_oracle(
    secret: Sequence[int] | str,
    input_bits: Sequence[int] | str,
    bias: int = 0,
) -> int:
    """Evaluate f(x) = s·x XOR b for the Bernstein-Vazirani oracle."""
    secret_bits = _normalise_bits(secret, name="secret")
    x_bits = _normalise_bits(input_bits, name="input_bits")
    if len(secret_bits) != len(x_bits):
        raise ValueError("input_bits must have the same width as secret")
    if bias not in (0, 1):
        raise ValueError("bias must be 0 or 1")
    return (sum(s * x for s, x in zip(secret_bits, x_bits)) + bias) & 1


def build_phase_oracle(
    secret: Sequence[int] | str,
    input_qubits: Sequence[int],
    ancilla: int,
    bias: int = 0,
) -> QCircuit:
    """Build the reversible y -> y XOR f(x) oracle used by the algorithm."""
    secret_bits = _normalise_bits(secret, name="secret")
    q_input = tuple(input_qubits)
    if len(q_input) != len(secret_bits):
        raise ValueError("input_qubits must have the same width as secret")
    if len(set(q_input)) != len(q_input):
        raise ValueError("input_qubits must be distinct")
    if ancilla in q_input:
        raise ValueError("ancilla must be distinct from input_qubits")
    if bias not in (0, 1):
        raise ValueError("bias must be 0 or 1")

    circuit = QCircuit()
    for qubit, enabled in zip(q_input, secret_bits):
        if enabled:
            circuit << CNOT(qubit, ancilla)
    if bias:
        circuit << X(ancilla)
    return circuit


@dataclass(frozen=True)
class BernsteinVaziraniResult:
    """Execution result returned by BernsteinVazirani.run."""

    secret: str
    measured_secret: str
    success_probability: float
    probabilities: Mapping[str, float]
    shots: int

    @property
    def success(self) -> bool:
        """Whether the most likely measured state is the hidden string."""
        return self.measured_secret == self.secret


class BernsteinVazirani:
    """Recover a hidden linear Boolean function with one oracle query.

    Bit ordering follows the explicit input_qubits sequence. The first
    character of secret is the coefficient of input_qubits[0]. The repository
    parse_quantum_result_dict helper normalises PyQPanda result ordering back
    to that same logical sequence.
    """

    def __init__(self, secret: Sequence[int] | str, bias: int = 0):
        self._secret_bits = _normalise_bits(secret, name="secret")
        if bias not in (0, 1):
            raise ValueError("bias must be 0 or 1")
        self.bias = int(bias)

    @property
    def secret(self) -> str:
        return "".join(str(bit) for bit in self._secret_bits)

    @property
    def width(self) -> int:
        return len(self._secret_bits)

    @property
    def qubits_required(self) -> int:
        """n data qubits plus one phase-kickback ancilla."""
        return self.width + 1

    @property
    def quantum_oracle_queries(self) -> int:
        return 1

    @property
    def deterministic_classical_queries(self) -> int:
        """Queries needed by the standard deterministic classical strategy."""
        return self.width

    def oracle_value(self, input_bits: Sequence[int] | str) -> int:
        """Evaluate this instance's classical reference oracle."""
        return evaluate_oracle(self._secret_bits, input_bits, self.bias)

    def build_circuit(
        self,
        input_qubits: Optional[Sequence[int]] = None,
        ancilla: Optional[int] = None,
    ) -> QCircuit:
        """Construct the Bernstein-Vazirani circuit without measurements."""
        if input_qubits is None:
            input_qubits = tuple(range(self.width))
        else:
            input_qubits = tuple(input_qubits)
        if ancilla is None:
            ancilla = self.width

        if len(input_qubits) != self.width:
            raise ValueError("input_qubits must have the same width as secret")
        if len(set(input_qubits)) != len(input_qubits):
            raise ValueError("input_qubits must be distinct")
        if ancilla in input_qubits:
            raise ValueError("ancilla must be distinct from input_qubits")

        circuit = QCircuit()
        circuit << X(ancilla)
        circuit << H(ancilla)
        for qubit in input_qubits:
            circuit << H(qubit)

        circuit << build_phase_oracle(
            self._secret_bits,
            input_qubits,
            ancilla,
            self.bias,
        )

        for qubit in input_qubits:
            circuit << H(qubit)
        return circuit

    def build_program(
        self,
        input_qubits: Optional[Sequence[int]] = None,
        ancilla: Optional[int] = None,
    ) -> QProg:
        """Wrap build_circuit in a PyQPanda quantum program."""
        program = QProg()
        program << self.build_circuit(input_qubits=input_qubits, ancilla=ancilla)
        return program

    def run(
        self,
        shots: int = 1,
        machine: Optional[CPUQVM] = None,
    ) -> BernsteinVaziraniResult:
        """Execute the circuit and decode the hidden string."""
        if shots < 1:
            raise ValueError("shots must be a positive integer")

        q_input = list(range(self.width))
        ancilla = self.width
        qvm = machine if machine is not None else CPUQVM()
        program = self.build_program(q_input, ancilla)

        qvm.run(program, shots=shots)
        raw = qvm.result().get_prob_dict(q_input)
        probabilities: Dict[str, float] = dict(
            parse_quantum_result_dict(raw, q_input, select_max=-1)
        )
        if not probabilities:
            raise RuntimeError("quantum backend returned an empty result")

        measured = max(probabilities, key=probabilities.get)
        success_probability = float(probabilities.get(self.secret, 0.0))

        return BernsteinVaziraniResult(
            secret=self.secret,
            measured_secret=measured,
            success_probability=success_probability,
            probabilities=probabilities,
            shots=shots,
        )


__all__ = [
    "BernsteinVazirani",
    "BernsteinVaziraniResult",
    "bitstring",
    "build_phase_oracle",
    "evaluate_oracle",
]
