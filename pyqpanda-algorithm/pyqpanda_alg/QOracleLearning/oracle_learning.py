"""Deutsch-Jozsa and Bernstein-Vazirani oracle-learning algorithms.

Bit-order convention
--------------------
Secret strings are q0-first: ``"101"`` means q0=1, q1=0, q2=1.
Truth-table index ``x`` uses the same convention through ordinary integer bits,
i.e. q0 is the least-significant bit of ``x``.

The pure helpers in this module have no PyQPanda dependency. Circuit-building
and execution helpers import ``pyqpanda3`` lazily.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def _normalise_bits(bits: str | Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(bits, str):
        if not bits:
            raise ValueError(f"{name} must not be empty")
        if any(ch not in "01" for ch in bits):
            raise ValueError(f"{name} must contain only '0' and '1'")
        return tuple(int(ch) for ch in bits)
    try:
        values = tuple(int(v) for v in bits)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a bit sequence") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(v not in (0, 1) for v in values):
        raise ValueError(f"{name} must contain only 0 and 1")
    return values


def validate_truth_table(table: Sequence[int]) -> tuple[int, ...]:
    """Validate a Boolean truth table and return it as an immutable tuple."""
    try:
        values = tuple(int(v) for v in table)
    except (TypeError, ValueError) as exc:
        raise ValueError("truth table must be a sequence of bits") from exc
    if len(values) < 2 or len(values) & (len(values) - 1):
        raise ValueError("truth table length must be a power of two >= 2")
    if any(v not in (0, 1) for v in values):
        raise ValueError("truth table entries must be 0 or 1")
    return values


def input_qubit_count(table: Sequence[int]) -> int:
    values = validate_truth_table(table)
    return (len(values) - 1).bit_length()


def index_to_q0_bits(index: int, width: int) -> tuple[int, ...]:
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(index, int) or index < 0 or index >= 2**width:
        raise ValueError("index does not fit the requested width")
    return tuple((index >> qubit) & 1 for qubit in range(width))


def q0_bits_to_index(bits: str | Sequence[int]) -> int:
    values = _normalise_bits(bits, name="bits")
    return sum(bit << qubit for qubit, bit in enumerate(values))


def affine_truth_table(secret: str | Sequence[int], bias: int = 0) -> tuple[int, ...]:
    """Return ``f(x) = secret·x XOR bias`` in q0-first convention."""
    secret_bits = _normalise_bits(secret, name="secret")
    if bias not in (0, 1, False, True):
        raise ValueError("bias must be 0 or 1")
    b = int(bias)
    table = []
    for x in range(2 ** len(secret_bits)):
        parity = b
        for qubit, coefficient in enumerate(secret_bits):
            if coefficient:
                parity ^= (x >> qubit) & 1
        table.append(parity)
    return tuple(table)


def classify_deutsch_jozsa_promise(table: Sequence[int]) -> str:
    """Classify a promised oracle as ``constant`` or ``balanced``."""
    values = validate_truth_table(table)
    ones = sum(values)
    if ones in (0, len(values)):
        return "constant"
    if ones * 2 == len(values):
        return "balanced"
    raise ValueError("Deutsch-Jozsa requires a constant or balanced truth table")


def phase_oracle_diagonal(table: Sequence[int]) -> tuple[int, ...]:
    """Return the exact diagonal of the phase oracle ``(-1)**f(x)``."""
    return tuple(1 if bit == 0 else -1 for bit in validate_truth_table(table))


def walsh_probabilities(table: Sequence[int]) -> tuple[float, ...]:
    """Return exact output probabilities by independent Walsh transform."""
    values = validate_truth_table(table)
    size = len(values)
    probs: list[float] = []
    for y in range(size):
        total = 0
        for x, fx in enumerate(values):
            parity = (x & y).bit_count() & 1
            total += -1 if (fx ^ parity) else 1
        amplitude = total / size
        probs.append(float(amplitude * amplitude))
    return tuple(probs)


def recover_affine_secret(table: Sequence[int]) -> str:
    """Recover a q0-first BV secret, rejecting non-affine tables."""
    values = validate_truth_table(table)
    n = input_qubit_count(values)
    bias = values[0]
    secret = tuple(values[1 << qubit] ^ bias for qubit in range(n))
    if values != affine_truth_table(secret, bias):
        raise ValueError("truth table is not affine and is invalid for Bernstein-Vazirani")
    return "".join(str(bit) for bit in secret)


def dominant_q0_bitstring(probabilities: Sequence[float], width: int) -> str:
    """Decode a unique dominant probability-list entry as q0-first bits."""
    if len(probabilities) != 2**width:
        raise ValueError("probability list length does not match width")
    if any(p < -1e-12 for p in probabilities):
        raise ValueError("probabilities must be non-negative")
    maximum = max(probabilities)
    winners = [i for i, p in enumerate(probabilities) if abs(p - maximum) <= 1e-12]
    if len(winners) != 1:
        raise ValueError("probability distribution has no unique dominant state")
    return "".join(str(bit) for bit in index_to_q0_bits(winners[0], width))


def _resolve_qubits(width: int, qubits: Iterable[Any] | None) -> list[Any]:
    resolved = list(range(width)) if qubits is None else list(qubits)
    if len(resolved) != width:
        raise ValueError(f"expected exactly {width} qubits")
    if len({str(q) for q in resolved}) != width:
        raise ValueError("qubits must be distinct")
    return resolved


def truth_table_phase_oracle(table: Sequence[int], qubits: Iterable[Any] | None = None):
    """Build a direct PyQPanda3 phase oracle for an explicit truth table.

    Each marked basis state is surrounded by X gates for zero controls and gets
    one n-qubit controlled-Z phase flip. This generic teaching synthesis is
    worst-case O(2^n), not a scalability claim.
    """
    values = validate_truth_table(table)
    n = input_qubit_count(values)
    q = _resolve_qubits(n, qubits)
    from pyqpanda3.core import QCircuit, X, Z

    circuit = QCircuit()
    for x, fx in enumerate(values):
        if not fx:
            continue
        zero_qubits = [q[i] for i in range(n) if ((x >> i) & 1) == 0]
        for qb in zero_qubits:
            circuit << X(qb)
        if n == 1:
            circuit << Z(q[0])
        else:
            circuit << Z(q[-1]).control(q[:-1])
        for qb in reversed(zero_qubits):
            circuit << X(qb)
    return circuit


def affine_phase_oracle(secret: str | Sequence[int], qubits: Iterable[Any] | None = None):
    """Build the O(n) phase oracle for ``f(x)=secret·x XOR bias``.

    The affine bias is a global phase and therefore intentionally absent.
    """
    secret_bits = _normalise_bits(secret, name="secret")
    q = _resolve_qubits(len(secret_bits), qubits)
    from pyqpanda3.core import QCircuit, Z

    circuit = QCircuit()
    for coefficient, qb in zip(secret_bits, q):
        if coefficient:
            circuit << Z(qb)
    return circuit


def deutsch_jozsa_circuit(table: Sequence[int], qubits: Iterable[Any] | None = None):
    """Construct the ancilla-free phase-oracle Deutsch-Jozsa circuit."""
    values = validate_truth_table(table)
    classify_deutsch_jozsa_promise(values)
    n = input_qubit_count(values)
    q = _resolve_qubits(n, qubits)
    from pyqpanda3.core import H, QCircuit

    circuit = QCircuit()
    for qb in q:
        circuit << H(qb)
    circuit << truth_table_phase_oracle(values, q)
    for qb in q:
        circuit << H(qb)
    return circuit


def bernstein_vazirani_circuit(secret: str | Sequence[int], qubits: Iterable[Any] | None = None):
    """Construct the ancilla-free Bernstein-Vazirani circuit in O(n) gates."""
    secret_bits = _normalise_bits(secret, name="secret")
    q = _resolve_qubits(len(secret_bits), qubits)
    from pyqpanda3.core import H, QCircuit

    circuit = QCircuit()
    for qb in q:
        circuit << H(qb)
    circuit << affine_phase_oracle(secret_bits, q)
    for qb in q:
        circuit << H(qb)
    return circuit


def _run_probabilities(circuit: Any, qubits: Sequence[Any], shots: int = 1024) -> tuple[float, ...]:
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    from pyqpanda3.core import CPUQVM, QProg

    machine = CPUQVM()
    program = QProg()
    program << circuit
    machine.run(program, shots)
    return tuple(float(p) for p in machine.result().get_prob_list(list(qubits)))


def run_deutsch_jozsa(table: Sequence[int], shots: int = 1024) -> tuple[str, tuple[float, ...]]:
    """Execute Deutsch-Jozsa and return ``(classification, probabilities)``."""
    values = validate_truth_table(table)
    classify_deutsch_jozsa_promise(values)
    n = input_qubit_count(values)
    qubits = list(range(n))
    probs = _run_probabilities(deutsch_jozsa_circuit(values, qubits), qubits, shots)
    classification = "constant" if probs[0] > 0.5 else "balanced"
    return classification, probs


def run_bernstein_vazirani(secret: str | Sequence[int], shots: int = 1024) -> tuple[str, tuple[float, ...]]:
    """Execute Bernstein-Vazirani and return q0-first secret + probabilities."""
    secret_bits = _normalise_bits(secret, name="secret")
    qubits = list(range(len(secret_bits)))
    probs = _run_probabilities(bernstein_vazirani_circuit(secret_bits, qubits), qubits, shots)
    return dominant_q0_bitstring(probs, len(secret_bits)), probs
