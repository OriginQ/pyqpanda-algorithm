# Licensed under the Apache License, Version 2.0.
"""Native higher-order polynomial Ising/HUBO representation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Tuple
import numpy as np


def _canonical_term(term: Iterable[int], n: int) -> tuple[int, ...]:
    t = tuple(int(i) for i in term)
    if len(set(t)) != len(t):
        raise ValueError(f"repeated variable/qubit in term {t}; reduce powers before constructing the problem")
    if any(i < 0 or i >= n for i in t):
        raise ValueError(f"term {t} contains an index outside [0,{n})")
    return tuple(sorted(t))


@dataclass(frozen=True)
class PolynomialIsingProblem:
    """Native higher-order Ising/HUBO minimization problem.

    The Hamiltonian is

        H = offset + sum_S J_S prod_{i in S} Z_i,

    where ``S`` may have arbitrary size.  Computational-basis bit ``b_i`` is
    mapped to the spin eigenvalue ``z_i = 1 - 2 b_i``.

    ``terms`` uses tuples of qubit indices as keys, e.g. ``(0, 2, 4)`` for
    ``Z0 Z2 Z4``.  The empty tuple is not stored; constants belong in
    ``offset``.
    """

    n: int
    terms: Mapping[tuple[int, ...], float]
    offset: float = 0.0
    name: str = "polynomial_ising"

    def __post_init__(self):
        n = int(self.n)
        if n < 1:
            raise ValueError("n must be >= 1")
        canonical: dict[tuple[int, ...], float] = {}
        offset = float(self.offset)
        for raw_term, raw_coeff in dict(self.terms).items():
            coeff = float(raw_coeff)
            term = _canonical_term(raw_term, n)
            if len(term) == 0:
                offset += coeff
                continue
            canonical[term] = canonical.get(term, 0.0) + coeff
        canonical = {t: float(c) for t, c in canonical.items() if abs(c) > 1e-15}
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "terms", canonical)
        object.__setattr__(self, "offset", offset)

    @property
    def degree(self) -> int:
        return max((len(t) for t in self.terms), default=0)

    @property
    def num_terms(self) -> int:
        return len(self.terms)

    @property
    def term_weights(self) -> tuple[int, ...]:
        return tuple(sorted({len(t) for t in self.terms}))

    def energy_spins(self, spins: Iterable[float]) -> float:
        z = np.asarray(list(spins), dtype=float)
        if z.shape != (self.n,):
            raise ValueError("spins length must equal problem size")
        e = float(self.offset)
        for term, coeff in self.terms.items():
            e += coeff * float(np.prod(z[list(term)]))
        return float(e)

    def energy_bits(self, bits: Iterable[int]) -> float:
        b = np.asarray(list(bits), dtype=int)
        if b.shape != (self.n,):
            raise ValueError("bits length must equal problem size")
        if np.any((b != 0) & (b != 1)):
            raise ValueError("bits must contain only 0/1")
        return self.energy_spins(1.0 - 2.0 * b)

    def energy_index(self, index: int) -> float:
        return self.energy_bits([(int(index) >> q) & 1 for q in range(self.n)])

    def energy_bitstring(self, bitstring: str) -> float:
        if len(bitstring) != self.n:
            raise ValueError("bitstring length must equal problem size")
        # display convention is q_(n-1)...q_0
        return self.energy_bits([int(c) for c in bitstring[::-1]])

    def energies(self) -> np.ndarray:
        return np.array([self.energy_index(i) for i in range(1 << self.n)], dtype=float)

    def exact_ground_states(self) -> Tuple[float, list[str]]:
        e = self.energies()
        emin = float(e.min())
        inds = np.flatnonzero(np.isclose(e, emin, atol=1e-12))
        return emin, [format(int(i), f"0{self.n}b") for i in inds]

    @classmethod
    def from_quadratic_ising(cls, problem) -> "PolynomialIsingProblem":
        terms: dict[tuple[int, ...], float] = {}
        for i, hi in enumerate(np.asarray(problem.h, dtype=float)):
            if abs(hi) > 1e-15:
                terms[(i,)] = float(hi)
        J = np.asarray(problem.J, dtype=float)
        for i in range(problem.n):
            for j in range(i + 1, problem.n):
                if abs(J[i, j]) > 1e-15:
                    terms[(i, j)] = float(J[i, j])
        return cls(problem.n, terms, offset=float(problem.offset), name=problem.name)


def binary_hubo_to_ising(
    n: int,
    terms: Mapping[tuple[int, ...], float],
    *,
    offset: float = 0.0,
    name: str = "binary_hubo",
) -> PolynomialIsingProblem:
    """Convert a binary HUBO into a native polynomial Ising Hamiltonian.

    Input objective:

        f(x) = offset + sum_S c_S prod_{i in S} x_i,  x_i in {0,1}.

    Using ``x_i=(1-z_i)/2`` each k-body binary monomial is expanded exactly
    into all Ising-Z subsets of that monomial.  No auxiliary variables and no
    quadratization are introduced.
    """
    n = int(n)
    out_terms: dict[tuple[int, ...], float] = {}
    out_offset = float(offset)
    for raw_term, raw_coeff in dict(terms).items():
        term = _canonical_term(raw_term, n)
        coeff = float(raw_coeff)
        if not term:
            out_offset += coeff
            continue
        scale = coeff / (2.0 ** len(term))
        # prod_i (1-z_i) = sum_{T subset S} (-1)^|T| prod_{i in T} z_i
        for r in range(len(term) + 1):
            sign = -1.0 if (r % 2) else 1.0
            for subset in combinations(term, r):
                contribution = scale * sign
                if len(subset) == 0:
                    out_offset += contribution
                else:
                    out_terms[subset] = out_terms.get(subset, 0.0) + contribution
    return PolynomialIsingProblem(n=n, terms=out_terms, offset=out_offset, name=name)


def binary_hubo_energy(
    n: int,
    terms: Mapping[tuple[int, ...], float],
    bits: Iterable[int],
    *,
    offset: float = 0.0,
) -> float:
    """Direct reference evaluator for a binary HUBO before Ising conversion."""
    x = np.asarray(list(bits), dtype=int)
    if x.shape != (int(n),):
        raise ValueError("bits length must equal n")
    e = float(offset)
    for raw_term, coeff in dict(terms).items():
        term = _canonical_term(raw_term, int(n))
        if len(term) == 0:
            e += float(coeff)
        else:
            e += float(coeff) * float(np.prod(x[list(term)]))
    return float(e)
