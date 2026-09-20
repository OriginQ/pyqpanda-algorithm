"""Simon's hidden-XOR quantum algorithm and exact GF(2) recovery helpers.

Bit-order convention
--------------------
Public bit strings are q0-first: ``"101"`` means q0=1, q1=0, q2=1.
Integer basis-state indices use q0 as the least-significant bit.

The pure helpers have no PyQPanda dependency. Circuit-building and execution
helpers import ``pyqpanda3`` lazily.

This module uses a canonical linear 2-to-1 oracle. For a non-zero secret ``s``
and a pivot p with s[p]=1, the output bits are

    f(x)[p] = 0
    f(x)[j] = x[j] XOR s[j] * x[p]     (j != p)

so the kernel is exactly ``{0, s}`` and therefore ``f(x) = f(x XOR s)``.
The reversible XOR-oracle implementation uses n-1 + wt(s)-1 CNOT gates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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


def _normalise_secret(secret: str | Sequence[int]) -> tuple[int, ...]:
    values = _normalise_bits(secret, name="secret")
    if not any(values):
        raise ValueError("Simon's promise requires a non-zero secret")
    return values


def index_to_q0_bits(index: int, width: int) -> tuple[int, ...]:
    """Convert a basis-state index to q0-first bits."""
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(index, int) or index < 0 or index >= 2**width:
        raise ValueError("index does not fit the requested width")
    return tuple((index >> qubit) & 1 for qubit in range(width))


def q0_bits_to_index(bits: str | Sequence[int]) -> int:
    """Convert q0-first bits to a basis-state index."""
    values = _normalise_bits(bits, name="bits")
    return sum(bit << qubit for qubit, bit in enumerate(values))


def dot_mod2(left: str | Sequence[int], right: str | Sequence[int]) -> int:
    """Return the GF(2) dot product of equally sized bit vectors."""
    a = _normalise_bits(left, name="left")
    b = _normalise_bits(right, name="right")
    if len(a) != len(b):
        raise ValueError("bit vectors must have equal width")
    return sum(x * y for x, y in zip(a, b)) & 1


def linear_simon_function(
    bits: str | Sequence[int], secret: str | Sequence[int]
) -> tuple[int, ...]:
    """Evaluate the canonical classical function with hidden period ``secret``.

    Every output has exactly the two preimages ``x`` and ``x XOR secret``.
    The returned tuple is q0-first and has the same width as the input.
    """
    x = _normalise_bits(bits, name="bits")
    s = _normalise_secret(secret)
    if len(x) != len(s):
        raise ValueError("bits and secret must have equal width")
    pivot = next(i for i, bit in enumerate(s) if bit)
    out = [0] * len(s)
    for j in range(len(s)):
        if j == pivot:
            continue
        out[j] = x[j] ^ (s[j] & x[pivot])
    return tuple(out)


def simon_reference_probabilities(secret: str | Sequence[int]) -> tuple[float, ...]:
    """Return Simon's ideal input-register measurement distribution.

    Exactly the q0-first strings ``y`` satisfying ``y·secret = 0 (mod 2)``
    have non-zero probability, each with probability ``1 / 2**(n-1)``.
    """
    s = _normalise_secret(secret)
    support_probability = 1.0 / (2 ** (len(s) - 1))
    return tuple(
        support_probability if dot_mod2(index_to_q0_bits(i, len(s)), s) == 0 else 0.0
        for i in range(2 ** len(s))
    )


def _normalise_equations(
    equations: Iterable[str | Sequence[int]], width: int | None = None
) -> tuple[list[list[int]], int]:
    rows_raw = list(equations)
    if width is not None and (not isinstance(width, int) or width <= 0):
        raise ValueError("width must be a positive integer")
    if not rows_raw:
        if width is None:
            raise ValueError("width is required when no equations are supplied")
        return [], width

    first = _normalise_bits(rows_raw[0], name="equation")
    n = len(first) if width is None else width
    if len(first) != n:
        raise ValueError("equation width does not match requested width")
    rows = [list(first)]
    for raw in rows_raw[1:]:
        row = _normalise_bits(raw, name="equation")
        if len(row) != n:
            raise ValueError("all equations must have the same width")
        rows.append(list(row))
    return rows, n


def _rref(
    equations: Iterable[str | Sequence[int]], width: int | None = None
) -> tuple[list[list[int]], tuple[int, ...], int]:
    rows, n = _normalise_equations(equations, width)
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(n):
        source = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column]),
            None,
        )
        if source is None:
            continue
        rows[pivot_row], rows[source] = rows[source], rows[pivot_row]
        for row in range(len(rows)):
            if row != pivot_row and rows[row][column]:
                rows[row] = [a ^ b for a, b in zip(rows[row], rows[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return rows, tuple(pivot_columns), n


def gf2_rank(
    equations: Iterable[str | Sequence[int]], width: int | None = None
) -> int:
    """Return the rank of homogeneous equations over GF(2)."""
    _, pivots, _ = _rref(equations, width)
    return len(pivots)


def independent_equations(
    equations: Iterable[str | Sequence[int]], width: int | None = None
) -> tuple[str, ...]:
    """Return a deterministic independent basis spanning the supplied rows."""
    rows, pivots, n = _rref(equations, width)
    basis = rows[: len(pivots)]
    return tuple("".join(str(bit) for bit in row[:n]) for row in basis)


def recover_secret_from_equations(
    equations: Iterable[str | Sequence[int]], width: int | None = None
) -> str:
    """Recover the unique non-zero Simon secret from ``y·s = 0`` equations.

    A unique Simon secret requires rank exactly ``n-1``. Undersampled or
    contradictory-width inputs fail closed instead of guessing a secret.
    """
    rows, pivots, n = _rref(equations, width)
    if len(pivots) != n - 1:
        raise ValueError(
            f"Simon recovery requires rank {n - 1}, got {len(pivots)}"
        )
    free_columns = [column for column in range(n) if column not in pivots]
    if len(free_columns) != 1:
        raise ValueError("equations do not define a unique non-zero secret")

    solution = [0] * n
    solution[free_columns[0]] = 1
    for row_index in range(len(pivots) - 1, -1, -1):
        pivot = pivots[row_index]
        solution[pivot] = sum(
            rows[row_index][column] * solution[column]
            for column in range(pivot + 1, n)
        ) & 1

    if not any(solution):
        raise ValueError("equations yielded only the zero vector")
    return "".join(str(bit) for bit in solution)


def equations_from_probability_list(
    probabilities: Sequence[float],
    width: int,
    *,
    tolerance: float = 1e-12,
) -> tuple[str, ...]:
    """Extract non-zero-probability q0-first equations from a probability list."""
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if len(probabilities) != 2**width:
        raise ValueError("probability list length does not match width")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    values = tuple(float(p) for p in probabilities)
    if any(p < -tolerance for p in values):
        raise ValueError("probabilities must be non-negative")
    equations = []
    for index, probability in enumerate(values):
        if probability <= tolerance:
            continue
        bits = index_to_q0_bits(index, width)
        if any(bits):
            equations.append("".join(str(bit) for bit in bits))
    return tuple(equations)


def recover_secret_from_probabilities(
    probabilities: Sequence[float],
    width: int,
    *,
    tolerance: float = 1e-12,
) -> str:
    """Recover a Simon secret from an ideal/marginal probability list."""
    equations = equations_from_probability_list(
        probabilities, width, tolerance=tolerance
    )
    return recover_secret_from_equations(equations, width)


def recover_secret_from_counts(
    counts: Mapping[str, int | float],
    width: int,
    *,
    key_order: str = "q0-first",
) -> str:
    """Recover a secret from sampled bit-string counts.

    ``key_order`` must state whether mapping keys are already q0-first or are
    conventional qN-first display strings. Only positive-count outcomes are
    used; the all-zero outcome contributes no equation.
    """
    if key_order not in ("q0-first", "qN-first"):
        raise ValueError("key_order must be 'q0-first' or 'qN-first'")
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    equations: list[str] = []
    for key, count in sorted(counts.items(), key=lambda item: str(item[0])):
        try:
            positive = float(count) > 0
        except (TypeError, ValueError) as exc:
            raise ValueError("counts must be numeric") from exc
        if not positive:
            continue
        bits = _normalise_bits(key, name="count key")
        if len(bits) != width:
            raise ValueError("count key width does not match requested width")
        if key_order == "qN-first":
            bits = tuple(reversed(bits))
        if any(bits):
            equations.append("".join(str(bit) for bit in bits))
    return recover_secret_from_equations(equations, width)


def _resolve_registers(
    width: int,
    input_qubits: Iterable[Any] | None,
    output_qubits: Iterable[Any] | None,
) -> tuple[list[Any], list[Any]]:
    inputs = list(range(width)) if input_qubits is None else list(input_qubits)
    outputs = (
        list(range(width, 2 * width))
        if output_qubits is None
        else list(output_qubits)
    )
    if len(inputs) != width or len(outputs) != width:
        raise ValueError(f"expected exactly {width} input and {width} output qubits")
    labels = [str(q) for q in inputs + outputs]
    if len(set(labels)) != 2 * width:
        raise ValueError("input and output qubits must be distinct")
    return inputs, outputs


def linear_simon_oracle(
    secret: str | Sequence[int],
    input_qubits: Iterable[Any] | None = None,
    output_qubits: Iterable[Any] | None = None,
):
    """Build the reversible XOR oracle ``|x,y> -> |x,y XOR f(x)>``.

    The canonical linear function has kernel exactly ``{0, secret}`` and uses
    ``n - 1 + wt(secret) - 1`` CNOT gates.
    """
    s = _normalise_secret(secret)
    inputs, outputs = _resolve_registers(len(s), input_qubits, output_qubits)
    pivot = next(i for i, bit in enumerate(s) if bit)
    from pyqpanda3.core import CNOT, QCircuit

    circuit = QCircuit()
    for j in range(len(s)):
        if j == pivot:
            continue
        circuit << CNOT(inputs[j], outputs[j])
        if s[j]:
            circuit << CNOT(inputs[pivot], outputs[j])
    return circuit


def simon_circuit(
    secret: str | Sequence[int],
    input_qubits: Iterable[Any] | None = None,
    output_qubits: Iterable[Any] | None = None,
):
    """Construct Simon's circuit, leaving measurement to the caller."""
    s = _normalise_secret(secret)
    inputs, outputs = _resolve_registers(len(s), input_qubits, output_qubits)
    from pyqpanda3.core import H, QCircuit

    circuit = QCircuit()
    for qubit in inputs:
        circuit << H(qubit)
    circuit << linear_simon_oracle(s, inputs, outputs)
    for qubit in inputs:
        circuit << H(qubit)
    return circuit


def run_simon(
    secret: str | Sequence[int],
    shots: int = 1024,
    *,
    tolerance: float = 1e-12,
) -> tuple[str, tuple[float, ...]]:
    """Execute Simon's circuit and recover ``(secret, input probabilities)``."""
    s = _normalise_secret(secret)
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    inputs = list(range(len(s)))
    outputs = list(range(len(s), 2 * len(s)))
    from pyqpanda3.core import CPUQVM, QProg

    program = QProg()
    program << simon_circuit(s, inputs, outputs)
    machine = CPUQVM()
    machine.run(program, shots)
    probabilities = tuple(
        float(p) for p in machine.result().get_prob_list(inputs)
    )
    recovered = recover_secret_from_probabilities(
        probabilities, len(s), tolerance=tolerance
    )
    return recovered, probabilities
