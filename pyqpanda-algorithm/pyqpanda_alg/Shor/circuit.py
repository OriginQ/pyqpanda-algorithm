"""Order-finding circuit construction for the Shor solver.

:func:`build_order_finding_circuit` synthesizes the textbook order-
finding circuit for a base ``base`` modulo an odd ``modulus``: a phase
register of at least twice the modulus bit length (unless explicitly
configured), a value register wide enough to represent the modulus and
prepared to ``|1>``, Hadamards on the phase register, one controlled
modular multiplication per phase qubit, an inverse QFT on the phase
register, and a measurement of the phase register.  The returned
:class:`ShorCircuitBuild` carries the program together with the
register layout and pre-transpilation depth and gate counts so a solver
can check the qubit budget against a backend and submit the phase
samples without re-deriving the layout.

Circuit layout
--------------
* ``phase_qubits``: ``2 * modulus.bit_length()`` qubits by default
  (``phase_qubits`` overrides), holding the phase estimate.
* ``value_qubits``: ``modulus.bit_length()`` qubits, prepared to
  ``|1>`` so the exponentiation of the zeroth power leaves the value
  register holding one.
* ``ancilla_qubits``: the scratch register and comparison qubit of the
  modular exponentiation (``value_qubits``-wide scratch plus one), all
  returned to ``|0...0>``.

Synthesis
---------
The phase register is placed on the lowest qubit indices and Hadamards
prepare its uniform superposition.  The controlled modular
exponentiation of :mod:`pyqpanda_alg.Shor.arithmetic` uses the phase
register as its exponent register — phase qubit ``k`` gates the
multiplication by ``base**(2**k) mod modulus`` — and maps
``|e>|x> -> |e>|x*base**e mod modulus>`` with clean ancillas.  The
inverse QFT is the dagger of the plugin QFT (the standard convention:
order finding for base 2 modulo 15 with an 8-qubit phase register
samples 0, 64, 128, and 192), and the phase register is then measured.

No answer injection: construction derives every classical constant
from ``pow(base, 2**k, modulus)`` alone; factorization and
order-recovery helpers are never called.  Every public entry point
rejects booleans, non-integers, even moduli, and bases not coprime to
the modulus with an :class:`AlgorithmInputError` naming the failing
argument.
"""

import math
import numbers
from dataclasses import dataclass

from pyqpanda3.core import H, QProg, X, measure

from pyqpanda_alg.execution import AlgorithmInputError
from pyqpanda_alg.plugin import QFT

from .arithmetic import controlled_modular_exponentiation


@dataclass(frozen=True)
class ShorCircuitBuild:
    """Immutable order-finding circuit build with its register metadata.

    ``program`` is the :class:`QProg` implementing the full order-
    finding circuit, ending with a measurement of the phase register.
    ``phase_qubits``, ``value_qubits``, and ``ancilla_qubits`` name the
    register layout inside the program, and ``total_qubits`` its qubit
    count.  ``gate_counts`` (per gate type) and ``depth`` are the
    pre-transpilation counts of the program as constructed; they
    describe the built circuit, not any transpilation a backend would
    perform.
    """

    program: QProg
    phase_qubits: tuple[int, ...]
    value_qubits: tuple[int, ...]
    ancilla_qubits: tuple[int, ...]
    total_qubits: int
    gate_counts: dict[str, int]
    depth: int


def build_order_finding_circuit(base, modulus, phase_qubits=None) -> ShorCircuitBuild:
    """Synthesize the order-finding circuit for ``base`` modulo ``modulus``.

    ``base`` must be an integer of at least two coprime to the odd
    modulus ``modulus`` of at least three.  ``phase_qubits`` is the
    size of the phase register; None selects the derived default of
    twice the modulus bit length.  The returned build carries the
    program — Hadamards, controlled modular exponentiation, inverse
    QFT, phase measurement — its register layout, and pre-transpilation
    depth and gate counts.
    """
    modulus = _validate_modulus(modulus)
    base = _validate_base(base, modulus)
    bits = _require_int(
        phase_qubits if phase_qubits is not None else 2 * modulus.bit_length(),
        "phase_qubits",
        1,
    )
    value_bits = modulus.bit_length()

    phase = list(range(bits))
    value = list(range(bits, bits + value_bits))
    exponentiation = controlled_modular_exponentiation(base, modulus, phase, value)

    total_qubits = bits + 2 * value_bits + 1
    program = QProg(total_qubits)
    program << X(value[0])
    for qubit in phase:
        program << H(qubit)
    program << exponentiation
    program << QFT(phase).dagger()
    program << measure(phase, phase)

    return ShorCircuitBuild(
        program=program,
        phase_qubits=tuple(phase),
        value_qubits=tuple(value),
        ancilla_qubits=tuple(exponentiation.ancilla_qubits),
        total_qubits=total_qubits,
        gate_counts=program.count_ops(),
        depth=program.depth(),
    )


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


def _validate_base(base, modulus: int) -> int:
    """Validate the multiplier: integral, at least two, coprime to modulus."""
    value = _require_int(base, "base", 2)
    if math.gcd(value, modulus) != 1:
        raise AlgorithmInputError(f"base {value} must be coprime to modulus {modulus}")
    return value
