"""Reversible modular arithmetic circuits for Shor order finding.

Validation
----------
Every public builder rejects booleans, non-integers, even moduli, and
bases not coprime to the modulus with an :class:`AlgorithmInputError`
naming the failing argument, so callers fail fast before any circuit is
constructed.  Qubit lists must be non-empty, non-repeating, and (for
the value register) wide enough to represent the modulus.

Construction
------------
:func:`controlled_modular_multiply` returns the controlled permutation
``|x> -> |base*x mod modulus>``: the value register is rotated one bit
at a time, each set bit conditionally adds the classically precomputed
power ``base*2**k mod modulus`` into a scratch register, the registers
are exchanged, and the scratch register is uncomputed with the inverse
powers ``base**-1*2**k mod modulus``.  Every gate is controlled on the
control qubit, so a clear control leaves the state untouched and every
ancilla qubit returns to |0>.

:func:`controlled_modular_exponentiation` composes one controlled
multiply per exponent bit, using the classically precomputed powers
``base**(2**k) mod modulus`` derived from the base and modulus alone,
and maps ``|e>|x> -> |e>|x*base**e mod modulus>``.

The returned circuits are :class:`ModularArithmeticCircuit` instances
whose register layout (value, ancilla, control, exponent) is carried as
metadata so later stages and test harnesses can wire them up and assert
ancilla cleanup without re-deriving the layout.
"""

import math
import numbers

from pyqpanda3.core import SWAP, U1, QCircuit

from pyqpanda_alg.QCmp import qft_comparator
from pyqpanda_alg.execution import AlgorithmInputError
from pyqpanda_alg.plugin import QFT


class ModularArithmeticCircuit(QCircuit):
    """A :class:`QCircuit` carrying the register layout it was built for.

    ``value_qubits`` and ``ancilla_qubits`` name the value and work
    qubits; a controlled multiply additionally declares its single
    ``control_qubit`` and an exponentiation its ``exponent_qubits``
    register.  Every circuit returns all ``ancilla_qubits`` to |0...0>.
    """

    def __init__(
        self,
        *,
        value_qubits: list[int],
        ancilla_qubits: list[int],
        control_qubit: int | None = None,
        exponent_qubits: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.value_qubits = list(value_qubits)
        self.ancilla_qubits = list(ancilla_qubits)
        self.control_qubit = control_qubit
        self.exponent_qubits = list(exponent_qubits) if exponent_qubits is not None else None


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


def _validate_qubit_list(qubits, name: str, minimum: int = 1) -> list[int]:
    """Coerce ``qubits`` to a non-empty list of distinct qubit indices."""
    if not isinstance(qubits, list):
        raise AlgorithmInputError(f"{name} must be a list of qubit indices")
    result = [_require_int(qubit, name, 0) for qubit in qubits]
    if len(result) < minimum:
        raise AlgorithmInputError(f"{name} must contain at least {minimum} qubit")
    if len(set(result)) != len(result):
        raise AlgorithmInputError(f"{name} must not repeat qubit indices")
    return result


def _validate_value_register(value_qubits: list[int], modulus: int) -> list[int]:
    """Reject a value register too small to represent ``modulus``."""
    if 1 << len(value_qubits) <= modulus:
        raise AlgorithmInputError(
            f"value_qubits must be able to represent modulus {modulus}"
        )
    return value_qubits


def _qft_add(reg: list[int], addend: int, controls: tuple) -> QCircuit:
    """Translate ``reg`` by ``addend`` modulo ``2**len(reg)``.

    The translation gates are each controlled on ``controls``, so a
    clear control makes the whole translation the identity.
    """
    width = len(reg)
    cir = QCircuit()
    cir << QFT(reg)
    for j in range(width):
        angle = 2 * math.pi * ((addend << j) % (1 << width)) / (1 << width)
        cir << U1(reg[j], angle).control(list(controls))
    cir << QFT(reg).dagger()
    return cir


def _mod_add_constant(reg: list[int], addend: int, modulus: int, controls: tuple) -> QCircuit:
    """Map ``reg -> (reg + addend) mod modulus`` controlled on ``controls``.

    A comparison qubit above the register holds ``[reg >= modulus - addend]``
    before the translation and is uncomputed afterwards via the identity
    ``[reg' < addend] == [reg >= modulus - addend]`` for the reduced value.
    """
    width = len(reg)
    cmp_qubit = [max(reg) + 1]
    cir = QCircuit()
    cir << qft_comparator(modulus - addend, reg, cmp_qubit, function="geq").control(
        list(controls)
    )
    cir << _qft_add(reg, addend, controls)
    cir << _qft_add(reg, (1 << width) - modulus, list(controls) + cmp_qubit)
    cir << qft_comparator(addend, reg, cmp_qubit, function="s").control(list(controls))
    return cir


def _mod_sub_constant(reg: list[int], addend: int, modulus: int, controls: tuple) -> QCircuit:
    """Map ``reg -> (reg - addend) mod modulus`` controlled on ``controls``.

    The mirror image of :func:`_mod_add_constant`: the comparison qubit
    holds ``[reg < addend]`` before the translation and is uncomputed via
    ``[reg' >= modulus - addend] == [reg < addend]``.
    """
    width = len(reg)
    cmp_qubit = [max(reg) + 1]
    cir = QCircuit()
    cir << qft_comparator(addend, reg, cmp_qubit, function="s").control(list(controls))
    cir << _qft_add(reg, (1 << width) - addend, controls)
    cir << _qft_add(reg, modulus, list(controls) + cmp_qubit)
    cir << qft_comparator(modulus - addend, reg, cmp_qubit, function="geq").control(
        list(controls)
    )
    return cir


def _rotate_right(reg: list[int], controls: tuple) -> QCircuit:
    """Cyclically rotate ``reg`` one step towards lower significance.

    A right rotation brings bit ``k`` of the original register into
    position ``0`` on the ``k``-th application, so the precomputed
    power ``base*2**k mod modulus`` matches the bit under test.
    """
    cir = QCircuit()
    for i in range(len(reg) - 1):
        cir << SWAP(reg[i], reg[i + 1]).control(list(controls))
    return cir


def _controlled_multiply_circuit(
    base: int,
    modulus: int,
    value_qubits: list[int],
    control: int,
    ancilla: tuple[list[int], int],
) -> QCircuit:
    """Map ``|x>|0> -> |base*x mod modulus>|0>`` gated on ``control``.

    ``ancilla`` is a pair ``(scratch, cmp_qubit)``: an ``n``-qubit
    scratch register and one comparison qubit, all returned to |0>.
    The right rotation sweeps the value bits past ``value_qubits[0]``
    least-significant bit first, so the same scratch register can
    accumulate the precomputed powers ``base*2**k mod modulus`` and
    then be uncomputed with the inverse powers
    ``base**-1*2**k mod modulus`` after the exchange.
    """
    width = len(value_qubits)
    scratch, cmp_qubit = ancilla
    cir = QCircuit()
    for k in range(width):
        addend = (base << k) % modulus
        cir << _mod_add_constant(scratch, addend, modulus, (control, value_qubits[0]))
        cir << _rotate_right(value_qubits, (control,))
    for i in range(width):
        cir << SWAP(value_qubits[i], scratch[i]).control(control)
    inverse = pow(base, -1, modulus)
    for k in range(width):
        addend = (inverse << k) % modulus
        cir << _mod_sub_constant(scratch, addend, modulus, (control, value_qubits[0]))
        cir << _rotate_right(value_qubits, (control,))
    return cir


def _allocate_ancilla(first: int, width: int) -> tuple[list[int], int]:
    """Allocate a scratch register and comparison qubit above ``first``."""
    scratch = list(range(first, first + width))
    return scratch, first + width


def controlled_modular_multiply(base, modulus, value_qubits, control) -> ModularArithmeticCircuit:
    """Return the controlled circuit ``|x> -> |base*x mod modulus>``.

    ``base`` must be an integer of at least two coprime to ``modulus``,
    ``modulus`` an odd integer of at least three, and ``value_qubits`` a
    list of distinct qubit indices wide enough to represent the modulus.
    ``control`` names the qubit gating the whole circuit: a clear
    control leaves every register untouched, a set control applies the
    modular multiplication, and all ancilla qubits return to |0>.
    """
    modulus = _validate_modulus(modulus)
    base = _validate_base(base, modulus)
    value_qubits = _validate_qubit_list(value_qubits, "value_qubits")
    _validate_value_register(value_qubits, modulus)
    control = _require_int(control, "control", 0)
    if control in value_qubits:
        raise AlgorithmInputError("control must be distinct from value_qubits")

    first = max(max(value_qubits), control) + 1
    scratch, cmp_qubit = _allocate_ancilla(first, len(value_qubits))
    circuit = _controlled_multiply_circuit(
        base, modulus, value_qubits, control, (scratch, cmp_qubit)
    )
    result = ModularArithmeticCircuit(
        value_qubits=value_qubits,
        ancilla_qubits=scratch + [cmp_qubit],
        control_qubit=control,
    )
    result << circuit
    return result


def controlled_modular_exponentiation(
    base, modulus, exponent_qubits, value_qubits
) -> ModularArithmeticCircuit:
    """Return the circuit ``|e>|x> -> |e>|x*base**e mod modulus>``.

    ``base`` must be an integer of at least two coprime to ``modulus``;
    ``exponent_qubits`` and ``value_qubits`` are disjoint lists of
    distinct qubit indices, the value register wide enough to represent
    the odd modulus.  The exponent register gates one controlled
    multiply per bit, applying the classically precomputed powers
    ``base**(2**k) mod modulus``, and the shared ancilla qubits return
    to |0> between blocks.
    """
    modulus = _validate_modulus(modulus)
    base = _validate_base(base, modulus)
    exponent_qubits = _validate_qubit_list(exponent_qubits, "exponent_qubits")
    value_qubits = _validate_qubit_list(value_qubits, "value_qubits")
    _validate_value_register(value_qubits, modulus)
    if set(exponent_qubits) & set(value_qubits):
        raise AlgorithmInputError("exponent_qubits and value_qubits must be disjoint")

    first = max(max(value_qubits), max(exponent_qubits)) + 1
    scratch, cmp_qubit = _allocate_ancilla(first, len(value_qubits))
    result = ModularArithmeticCircuit(
        value_qubits=value_qubits,
        ancilla_qubits=scratch + [cmp_qubit],
        exponent_qubits=exponent_qubits,
    )
    for k, control in enumerate(exponent_qubits):
        power = pow(base, 1 << k, modulus)
        result << _controlled_multiply_circuit(
            power, modulus, value_qubits, control, (scratch, cmp_qubit)
        )
    return result
