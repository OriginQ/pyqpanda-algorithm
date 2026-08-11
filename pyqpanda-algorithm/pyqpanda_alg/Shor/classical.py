"""Classical preprocessing and order recovery for the Shor solver.

Validation
----------
Every public entry point rejects booleans, non-integers, and values
below two with an :class:`AlgorithmInputError` naming the failing
class, so callers fail fast before any task submission.

Preprocessing
-------------
:func:`classical_preprocess` resolves a modulus explicitly when it is
prime (via SymPy primality), even (factor pair ``(2, n // 2)``), or a
perfect power (smallest nontrivial integer root pair); anything else
is marked as needing quantum order finding.

Order recovery
--------------
:func:`recover_order` turns a measured phase-register sample
``s / 2**phase_bits`` into a candidate multiplicative order: it walks
the continued-fraction convergent denominators of the sample and, for
each denominator, checks the denominator and its multiples against
``pow(base, order, modulus) == 1``.  The multiple scan is what finds
the true order when a convergent denominator is only a proper divisor
of it.  Every unit-denominator convergent is skipped — the trivial
0/1 convergent and any later 1/1 convergent — so the scan never
degenerates into brute force.

The :class:`Shor` orchestration facade lives in
:mod:`pyqpanda_alg.Shor.shor`, which consumes these helpers.
"""

import math
import numbers

import sympy

from pyqpanda_alg.execution import AlgorithmInputError

from .model import NEEDS_QUANTUM, RESOLVED, PreprocessOutcome


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
    """Validate the public modulus: integral and at least two."""
    return _require_int(modulus, "modulus", 2)


def classical_preprocess(modulus) -> PreprocessOutcome:
    """Classify a modulus into an explicit result or a needs-quantum marker.

    ``modulus`` must be an integer of at least two.  Prime moduli
    resolve with ``is_prime=True``, even moduli with the factor pair
    ``(2, modulus // 2)``, and perfect powers with their smallest
    nontrivial root pair; odd composites resolve to ``NEEDS_QUANTUM``.
    """
    value = _validate_modulus(modulus)
    if sympy.isprime(value):
        return PreprocessOutcome(status=RESOLVED, is_prime=True)
    if value % 2 == 0:
        return PreprocessOutcome(status=RESOLVED, factors=(2, value // 2))
    factor = _perfect_power_factor(value)
    if factor is not None:
        return PreprocessOutcome(status=RESOLVED, factors=(factor, value // factor))
    return PreprocessOutcome(status=NEEDS_QUANTUM)


def _perfect_power_factor(value: int) -> int | None:
    """Return the smallest nontrivial factor of an odd perfect power.

    A perfect power ``value = root**exponent`` with ``exponent >= 2``
    is found by integer nth-root search.  Exponents are tried from
    largest to smallest so the returned root — and therefore the
    factor pair ``(root, value // root)`` — is the smallest nontrivial
    factor of ``value``.
    """
    for exponent in range(value.bit_length() - 1, 1, -1):
        root = _integer_nth_root(value, exponent)
        if root is not None:
            return root
    return None


def _integer_nth_root(value: int, exponent: int) -> int | None:
    """Return the integer ``exponent``-th root of ``value``, or None.

    Binary search for ``root`` with ``root**exponent == value``;
    ``value`` itself never qualifies because ``root`` is bounded above
    by ``value`` and ``exponent >= 2``.
    """
    low, high = 2, value
    while low <= high:
        mid = (low + high) // 2
        power = mid**exponent
        if power == value:
            return mid
        if power < value:
            low = mid + 1
        else:
            high = mid - 1
    return None


def recover_order(sample, phase_bits, base, modulus) -> int | None:
    """Recover a candidate multiplicative order from a phase sample.

    ``sample`` is a measured value of a ``phase_bits``-wide phase
    register, so it must lie in ``[0, 2**phase_bits)``; ``base`` must
    be an integer of at least two coprime to ``modulus``.  The sample
    fraction ``sample / 2**phase_bits`` is expanded as a continued
    fraction and every convergent denominator, together with its
    multiples below ``modulus``, is validated against
    ``pow(base, order, modulus) == 1``.  Returns the first validated
    candidate — the true order whenever the sample is a good phase
    estimate, otherwise a multiple of it — and None when no convergent
    yields one (including a zero sample, which carries no phase
    information).
    """
    phase_bits = _require_int(phase_bits, "phase_bits", 1)
    base = _require_int(base, "base", 2)
    modulus = _validate_modulus(modulus)
    sample = _require_int(sample, "sample", 0)
    if sample >= 1 << phase_bits:
        raise AlgorithmInputError(
            f"sample must be below 2**phase_bits, got {sample} "
            f"with phase_bits {phase_bits}"
        )
    if math.gcd(base, modulus) != 1:
        raise AlgorithmInputError(
            f"base {base} must be coprime to modulus {modulus}"
        )
    if sample == 0:
        return None
    for denominator in _convergent_denominators(sample, 1 << phase_bits):
        order = _first_valid_multiple(denominator, base, modulus)
        if order is not None:
            return order
    return None


def _convergent_denominators(numerator: int, denominator: int):
    """Yield the continued-fraction convergent denominators of a fraction.

    Every convergent with denominator one is skipped — the trivial
    0/1 convergent and any later 1/1 convergent (a sample fraction at
    or above 1/2 always produces one).  A unit denominator divides
    every order, so scanning its multiples would degenerate into brute
    force.  Denominators of the remaining convergents strictly
    increase.
    """
    previous, current = 1, 0
    a, b = numerator, denominator
    while b:
        quotient, remainder = divmod(a, b)
        a, b = b, remainder
        current, previous = quotient * current + previous, current
        if current == 1:
            continue
        yield current


def _first_valid_multiple(denominator: int, base: int, modulus: int) -> int | None:
    """Return the smallest multiple of ``denominator`` with unit base power.

    Checks ``denominator`` and then its multiples while the candidate
    stays below ``modulus`` — the true order never reaches ``modulus``
    — and returns the first one satisfying
    ``pow(base, candidate, modulus) == 1``, or None.
    """
    if denominator >= modulus:
        return None
    multiple = denominator
    while multiple < modulus:
        if pow(base, multiple, modulus) == 1:
            return multiple
        multiple += denominator
    return None
