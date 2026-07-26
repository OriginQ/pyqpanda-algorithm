"""Pauli observable data structures used by :mod:`pyqpanda_alg.QShadow`.

The public convention in this module is deliberately explicit: character ``q``
in a Pauli string acts on qubit ``q``.  Therefore, ``"XI"`` means X on qubit 0
and identity on qubit 1.  This avoids leaking backend-specific bit-string
endianness into the statistical code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

_ALLOWED_PAULIS = frozenset("IXYZ")


def _normalise_pauli(pauli: str) -> str:
    if not isinstance(pauli, str):
        raise TypeError("pauli must be a string")
    normalised = "".join(pauli.split()).upper()
    if not normalised:
        raise ValueError("a Pauli string cannot be empty")
    invalid = sorted(set(normalised) - _ALLOWED_PAULIS)
    if invalid:
        raise ValueError(
            f"invalid Pauli character(s) {invalid}; expected only I, X, Y, Z"
        )
    return normalised


@dataclass(frozen=True, slots=True)
class PauliTerm:
    """A real-weighted Pauli string.

    Parameters
    ----------
    pauli:
        Pauli string in q0-first order.  Character ``pauli[q]`` acts on qubit
        ``q``.
    coefficient:
        Finite real coefficient.  QShadow estimates Hermitian observables, so a
        complex coefficient is intentionally rejected.
    """

    pauli: str
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pauli", _normalise_pauli(self.pauli))
        if isinstance(self.coefficient, complex):
            if self.coefficient.imag != 0:
                raise ValueError("PauliTerm coefficient must be real")
            coefficient = float(self.coefficient.real)
        else:
            try:
                coefficient = float(self.coefficient)
            except (TypeError, ValueError) as exc:
                raise TypeError("PauliTerm coefficient must be a real number") from exc
        if not np.isfinite(coefficient):
            raise ValueError("PauliTerm coefficient must be finite")
        object.__setattr__(self, "coefficient", coefficient)

    @property
    def n_qubits(self) -> int:
        """Number of qubits addressed by the Pauli string."""

        return len(self.pauli)

    @property
    def support(self) -> tuple[int, ...]:
        """Qubit indices on which the term is non-identity."""

        return tuple(q for q, axis in enumerate(self.pauli) if axis != "I")

    @property
    def weight(self) -> int:
        """Pauli weight (number of non-identity factors)."""

        return len(self.support)

    @property
    def is_identity(self) -> bool:
        """Whether this term is the all-identity operator."""

        return self.weight == 0


@dataclass(frozen=True, slots=True)
class PauliObservable:
    """A named real linear combination of Pauli strings."""

    terms: tuple[PauliTerm, ...]
    name: str = "observable"

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not terms:
            raise ValueError("an observable must contain at least one Pauli term")
        if not all(isinstance(term, PauliTerm) for term in terms):
            raise TypeError("terms must contain only PauliTerm instances")
        n_qubits = terms[0].n_qubits
        if any(term.n_qubits != n_qubits for term in terms):
            raise ValueError("all Pauli terms in an observable must have equal length")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("observable name must be a non-empty string")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "name", self.name.strip())

    @classmethod
    def from_terms(
        cls,
        terms: (
            Mapping[str, float]
            | Iterable[PauliTerm | tuple[str, float]]
        ),
        *,
        name: str = "observable",
    ) -> "PauliObservable":
        """Build an observable and combine duplicate Pauli strings.

        ``terms`` may be a mapping such as ``{"ZI": 0.5, "IZ": 0.5}``, an
        iterable of ``(pauli, coefficient)`` pairs, or existing
        :class:`PauliTerm` objects.  Duplicate strings are summed in insertion
        order.  Zero coefficients are retained so the number of qubits remains
        unambiguous even for a zero observable.
        """

        source: Iterable[PauliTerm | tuple[str, float]]
        source = terms.items() if isinstance(terms, Mapping) else terms
        combined: dict[str, float] = {}
        for item in source:
            term = item if isinstance(item, PauliTerm) else PauliTerm(*item)
            combined[term.pauli] = combined.get(term.pauli, 0.0) + term.coefficient
        if not combined:
            raise ValueError("terms cannot be empty")
        return cls(
            tuple(PauliTerm(pauli, coefficient) for pauli, coefficient in combined.items()),
            name=name,
        )

    @property
    def n_qubits(self) -> int:
        """Number of qubits addressed by this observable."""

        return self.terms[0].n_qubits

    def exact_expectation(self, statevector: Sequence[complex]) -> float:
        """Return the exact statevector expectation for validation.

        The statevector follows the usual little-endian simulator layout: bit
        ``q`` of an amplitude index is qubit ``q``.  This method is a reference
        calculation for examples and tests, not a replacement for measurements.
        """

        value = sum(
            term.coefficient * pauli_expectation(statevector, term.pauli)
            for term in self.terms
        )
        return float(value)


def pauli_expectation(statevector: Sequence[complex], pauli: str) -> float:
    """Compute ``<psi|P|psi>`` without constructing a dense Pauli matrix."""

    pauli = _normalise_pauli(pauli)
    state = np.asarray(statevector, dtype=np.complex128)
    if state.ndim != 1:
        raise ValueError("statevector must be one-dimensional")
    expected_size = 1 << len(pauli)
    if state.size != expected_size:
        raise ValueError(
            f"statevector has length {state.size}, expected {expected_size} "
            f"for {len(pauli)} qubits"
        )
    norm = float(np.vdot(state, state).real)
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError("statevector must be finite and normalised")

    transformed = np.zeros_like(state)
    for source, amplitude in enumerate(state):
        target = source
        phase = 1.0 + 0.0j
        for qubit, axis in enumerate(pauli):
            if axis == "I":
                continue
            bit = (source >> qubit) & 1
            if axis == "X":
                target ^= 1 << qubit
            elif axis == "Y":
                target ^= 1 << qubit
                phase *= 1j if bit == 0 else -1j
            else:  # Z
                phase *= 1.0 if bit == 0 else -1.0
        transformed[target] += phase * amplitude

    expectation = np.vdot(state, transformed)
    if abs(expectation.imag) > 1e-9:
        raise ArithmeticError(
            "a Hermitian Pauli expectation acquired a non-negligible imaginary part"
        )
    return float(expectation.real)


def coerce_observables(
    observables: PauliObservable | Sequence[PauliObservable],
) -> tuple[PauliObservable, ...]:
    """Validate and normalise one or more same-size observables."""

    result: tuple[PauliObservable, ...]
    if isinstance(observables, PauliObservable):
        result = (observables,)
    else:
        result = tuple(observables)
    if not result:
        raise ValueError("at least one observable is required")
    if not all(isinstance(observable, PauliObservable) for observable in result):
        raise TypeError("observables must contain only PauliObservable instances")
    n_qubits = result[0].n_qubits
    if any(observable.n_qubits != n_qubits for observable in result):
        raise ValueError("all observables must act on the same number of qubits")
    names = [observable.name for observable in result]
    if len(set(names)) != len(names):
        raise ValueError("observable names must be unique")
    return result
