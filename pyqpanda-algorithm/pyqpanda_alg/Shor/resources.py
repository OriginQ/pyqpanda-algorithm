"""Backend-independent resource estimation for Shor order finding.

:func:`estimate_shor_resources` validates the modulus exactly as
:func:`build_order_finding_circuit` does and then reports the qubit
counts, controlled multiplications, and approximate gate and depth
budgets of the order-finding circuit — before any circuit is
synthesized.  The estimates are independent of a backend: they describe
the circuit and submission plan, not any transpilation a particular
machine would perform.

Counting rules
--------------
``phase_qubits``
    The phase-estimation register size: twice the modulus bit length by
    default, or the explicitly configured value.
``value_qubits``
    ``modulus.bit_length()`` — the register holding the modular image,
    wide enough to represent the modulus and prepared to ``|1>``.
``ancilla_qubits``
    The modular exponentiation's scratch register (``value_qubits``
    wide) plus one comparison qubit: exactly ``value_qubits + 1``.
``total_qubits``
    ``phase_qubits + value_qubits + ancilla_qubits``.
``controlled_multiplies``
    One controlled modular multiplication per phase qubit (powers
    ``base**(2**0) .. base**(2**(phase_qubits - 1))``), so exactly
    ``phase_qubits``.  Exact.
``synthesized_gates`` and ``estimated_depth``
    Approximate pre-transpilation estimates computed with the
    order-of-magnitude heuristic under "Counting heuristic" below.
    These are the only approximate fields; see
    ``approximate_labels``.

Counting heuristic
------------------
Every component is QFT-based.  The plugin QFT on ``w`` qubits costs
``w(w+1)/2 + w//2`` gates at depth ``2w``.  A translation adds one QFT,
``w`` phase gates, and one inverse QFT; a comparison adds two QFTs on
``w+1`` qubits, two QFTs on ``w`` qubits, phase gates, and one X; a
modular add/sub uses two comparisons and two translations.  One
controlled multiply then sweeps the value bits with ``w`` modular adds,
``w`` cyclic rotations, ``w`` exchange SWAPs, and ``w`` modular
subtracts::

    qft_gates(w)    = w(w+1)/2 + w/2
    qft_add(w)      = 2 qft_gates(w) + w
    comparator(w)   = 2 qft_gates(w+1) + (w+1) + 2 qft_gates(w) + w + 1
    mod_add(w)      = 2 comparator(w) + 2 qft_add(w)
    multiply(w)     = w (2 mod_add(w) + 2(w-1) + 1)
    synthesized_gates = phase_qubits * multiply(w) + phase_qubits
                        + qft_gates(phase_qubits) + 1

with the depth heuristic the same shape using ``qft_depth(w) = 2w``.
"""

import numbers
from dataclasses import dataclass
from typing import Any

from pyqpanda_alg.execution import AlgorithmInputError

#: Field names of :class:`ShorResourceEstimate` that are estimates.
_APPROXIMATE_FIELDS = frozenset({"synthesized_gates", "estimated_depth"})


@dataclass(frozen=True)
class ShorResourceEstimate:
    """Immutable backend-independent estimate of order-finding resources.

    Qubit counts and the controlled-multiplication count are exact by
    construction; ``synthesized_gates`` and ``estimated_depth`` are
    order-of-magnitude pre-transpilation estimates, as recorded by
    ``approximate_labels``.  See the module docstring for the counting
    rules and the counting heuristic.
    """

    modulus: int
    phase_qubits: int
    value_qubits: int
    ancilla_qubits: int
    total_qubits: int
    controlled_multiplies: int
    synthesized_gates: int
    estimated_depth: int
    approximate_labels: frozenset[str] = _APPROXIMATE_FIELDS

    def __post_init__(self) -> None:
        _require_positive(self.modulus, "modulus")
        _require_positive(self.phase_qubits, "phase_qubits")
        _require_positive(self.value_qubits, "value_qubits")
        _require_positive(self.ancilla_qubits, "ancilla_qubits")
        _require_positive(self.controlled_multiplies, "controlled_multiplies")
        _require_positive(self.synthesized_gates, "synthesized_gates")
        _require_positive(self.estimated_depth, "estimated_depth")
        _require_positive(self.total_qubits, "total_qubits")
        expected = self.phase_qubits + self.value_qubits + self.ancilla_qubits
        if self.total_qubits != expected:
            raise AlgorithmInputError(
                f"total_qubits {self.total_qubits} does not match "
                f"phase + value + ancilla = {expected}"
            )
        object.__setattr__(
            self, "approximate_labels", frozenset(self.approximate_labels)
        )


def estimate_shor_resources(modulus, phase_qubits=None) -> ShorResourceEstimate:
    """Estimate the order-finding circuit resources for ``modulus``.

    ``modulus`` is validated exactly as in
    :func:`build_order_finding_circuit`: integral, at least three, and
    odd.  ``phase_qubits`` is the phase register size; None selects the
    derived default of twice the modulus bit length.  The returned
    estimate is backend-independent and reports the qubit counts, the
    number of controlled multiplications, and the approximate
    pre-transpilation gate count and depth.
    """
    modulus = _validate_modulus(modulus)
    bits = _require_int(
        phase_qubits if phase_qubits is not None else 2 * modulus.bit_length(),
        "phase_qubits",
        1,
    )
    value_bits = modulus.bit_length()
    ancilla_bits = value_bits + 1
    return ShorResourceEstimate(
        modulus=modulus,
        phase_qubits=bits,
        value_qubits=value_bits,
        ancilla_qubits=ancilla_bits,
        total_qubits=bits + value_bits + ancilla_bits,
        controlled_multiplies=bits,
        synthesized_gates=_synthesized_gates(value_bits, bits),
        estimated_depth=_estimated_depth(value_bits, bits),
        approximate_labels=_APPROXIMATE_FIELDS,
    )


def _qft_gates(width: int) -> int:
    """Gate count of the plugin QFT on ``width`` qubits."""
    return width * (width + 1) // 2 + width // 2


def _qft_depth(width: int) -> int:
    """Depth of the plugin QFT on ``width`` qubits (approximate)."""
    return 2 * width


def _synthesized_gates(value_bits: int, phase_bits: int) -> int:
    """Approximate pre-transpilation gate count of the order-finding circuit."""
    qft_add = 2 * _qft_gates(value_bits) + value_bits
    comparator = (
        2 * _qft_gates(value_bits + 1)
        + (value_bits + 1)
        + 2 * _qft_gates(value_bits)
        + value_bits
        + 1
    )
    mod_add = 2 * comparator + 2 * qft_add
    multiply = value_bits * (2 * mod_add + 2 * (value_bits - 1) + 1)
    return phase_bits * multiply + phase_bits + _qft_gates(phase_bits) + 1


def _estimated_depth(value_bits: int, phase_bits: int) -> int:
    """Approximate pre-transpilation depth of the order-finding circuit."""
    qft_add = 2 * _qft_depth(value_bits) + 1
    comparator = 2 * _qft_depth(value_bits + 1) + 2 * _qft_depth(value_bits) + 1
    mod_add = 2 * comparator + 2 * qft_add
    multiply = value_bits * (2 * mod_add + 2 * (value_bits - 1) + 1) + value_bits
    return phase_bits * multiply + _qft_depth(phase_bits) + 1


def _require_positive(value: Any, name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise AlgorithmInputError(f"{name} must be a positive integer, got {value!r}")


def _require_int(value, name: str, minimum: int) -> int:
    """Coerce ``value`` to an int of at least ``minimum``, or reject it."""
    if isinstance(value, bool):
        raise AlgorithmInputError(f"{name} must be an integer, got bool")
    if not isinstance(value, numbers.Integral):
        raise AlgorithmInputError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    value = int(value)
    if value < minimum:
        raise AlgorithmInputError(f"{name} must be at least {minimum}, got {value}")
    return value


def _validate_modulus(modulus) -> int:
    """Validate the modulus: integral, at least three, and odd."""
    value = _require_int(modulus, "modulus", 3)
    if value % 2 == 0:
        raise AlgorithmInputError(f"modulus {value} must be odd")
    return value
