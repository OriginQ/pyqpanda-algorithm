# Licensed under the Apache License, Version 2.0.
"""Weighted MAX-3-SAT encoding, native HUBO conversion, and cubic quadratization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence
import numpy as np

from ..problems.polynomial import PolynomialIsingProblem, binary_hubo_to_ising, binary_hubo_energy
from ..problems.ising import IsingProblem


def _add_poly(a: dict[tuple[int, ...], float], b: Mapping[tuple[int, ...], float], scale: float = 1.0):
    out = dict(a)
    for t, c in b.items():
        key = tuple(sorted(set(int(i) for i in t)))
        out[key] = out.get(key, 0.0) + scale * float(c)
        if abs(out[key]) < 1e-15:
            del out[key]
    return out


def _mul_poly(a: Mapping[tuple[int, ...], float], b: Mapping[tuple[int, ...], float]):
    """Multiply binary polynomials using x_i^2=x_i."""
    out: dict[tuple[int, ...], float] = {}
    for ta, ca in a.items():
        for tb, cb in b.items():
            term = tuple(sorted(set(ta).union(tb)))
            out[term] = out.get(term, 0.0) + float(ca) * float(cb)
    return {t: c for t, c in out.items() if abs(c) > 1e-15}


@dataclass(frozen=True)
class WeightedClause:
    """One weighted 3-SAT clause in DIMACS-style literal notation.

    Literal ``+k`` means x_(k-1); literal ``-k`` means NOT x_(k-1).
    The clause is satisfied when at least one literal is true.
    """

    literals: tuple[int, int, int]
    weight: float = 1.0

    def __post_init__(self):
        lits = tuple(int(x) for x in self.literals)
        if len(lits) != 3 or any(x == 0 for x in lits):
            raise ValueError("MAX-3-SAT clauses must contain exactly three nonzero DIMACS literals")
        if len({abs(x) for x in lits}) != 3:
            raise ValueError("MAX-3-SAT clauses require three distinct variables")
        if float(self.weight) <= 0:
            raise ValueError("clause weight must be positive")
        object.__setattr__(self, "literals", lits)
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True)
class Max3SATProblem:
    """Weighted MAX-3-SAT instance with native cubic-HUBO encoding."""

    n: int
    clauses: tuple[WeightedClause, ...]
    name: str = "max3sat"

    @classmethod
    def from_clauses(
        cls,
        n: int,
        clauses: Iterable[Sequence[int] | tuple[Sequence[int], float]],
        *,
        name: str = "max3sat",
    ) -> "Max3SATProblem":
        parsed: list[WeightedClause] = []
        for item in clauses:
            # Weighted input: ((1,-2,3), 1.5). Plain input: (1,-2,3).
            if len(item) == 2 and isinstance(item[0], (tuple, list, np.ndarray)):
                lits, w = item
                parsed.append(WeightedClause(tuple(int(x) for x in lits), float(w)))
            else:
                parsed.append(WeightedClause(tuple(int(x) for x in item), 1.0))
        n = int(n)
        if n < 3:
            raise ValueError("MAX-3-SAT needs n>=3")
        for c in parsed:
            if any(abs(l) > n for l in c.literals):
                raise ValueError(f"literal index exceeds n={n}: {c}")
        return cls(n=n, clauses=tuple(parsed), name=name)

    @property
    def total_weight(self) -> float:
        return float(sum(c.weight for c in self.clauses))

    def literal_truth(self, literal: int, bits: Sequence[int]) -> int:
        x = int(bits[abs(literal) - 1])
        return x if literal > 0 else 1 - x

    def unsatisfied_weight_bits(self, bits: Sequence[int]) -> float:
        b = list(int(x) for x in bits)
        if len(b) != self.n or any(x not in (0, 1) for x in b):
            raise ValueError("bits must be a 0/1 vector of length n")
        cost = 0.0
        for c in self.clauses:
            if not any(self.literal_truth(l, b) for l in c.literals):
                cost += c.weight
        return float(cost)

    def satisfied_weight_bits(self, bits: Sequence[int]) -> float:
        return float(self.total_weight - self.unsatisfied_weight_bits(bits))

    def satisfied_weight_bitstring(self, bitstring: str) -> float:
        if len(bitstring) != self.n:
            raise ValueError("bitstring length mismatch")
        # Display convention q_(n-1)...q_0; logical variables are q0-first.
        return self.satisfied_weight_bits([int(c) for c in bitstring[::-1]])

    def binary_hubo(self) -> tuple[float, dict[tuple[int, ...], float]]:
        """Return exact binary HUBO for weighted UNSAT cost.

        For a positive literal x_i the unsatisfied factor is (1-x_i).
        For a negative literal NOT x_i the unsatisfied factor is x_i.
        The product of the three factors is 1 exactly when the clause is false.
        """
        poly: dict[tuple[int, ...], float] = {}
        for clause in self.clauses:
            cp: dict[tuple[int, ...], float] = {(): 1.0}
            for lit in clause.literals:
                q = abs(lit) - 1
                # factor = 1-x_q for positive literal; x_q for negated literal
                factor = {(): 1.0, (q,): -1.0} if lit > 0 else {(q,): 1.0}
                cp = _mul_poly(cp, factor)
            poly = _add_poly(poly, cp, clause.weight)
        offset = float(poly.pop((), 0.0))
        return offset, poly

    def to_native_ising(self) -> PolynomialIsingProblem:
        offset, terms = self.binary_hubo()
        return binary_hubo_to_ising(self.n, terms, offset=offset, name=self.name + "_native_hubo")

    def exact_solution(self) -> tuple[float, list[str]]:
        best = -np.inf
        states: list[str] = []
        for idx in range(1 << self.n):
            bits = [(idx >> q) & 1 for q in range(self.n)]
            val = self.satisfied_weight_bits(bits)
            s = format(idx, f"0{self.n}b")
            if val > best + 1e-12:
                best, states = val, [s]
            elif abs(val - best) <= 1e-12:
                states.append(s)
        return float(best), states


@dataclass(frozen=True)
class QuadratizedQUBO:
    """Quadratic reduction of a cubic HUBO with explicit ancilla metadata."""

    logical_n: int
    total_n: int
    offset: float
    terms: dict[tuple[int, ...], float]
    ancilla_for_cubic: dict[tuple[int, int, int], int]
    penalties: dict[int, float]
    name: str

    @property
    def ancilla_count(self) -> int:
        return self.total_n - self.logical_n

    def energy_bits(self, bits: Sequence[int]) -> float:
        return binary_hubo_energy(self.total_n, self.terms, bits, offset=self.offset)

    def to_ising(self) -> IsingProblem:
        poly = binary_hubo_to_ising(self.total_n, self.terms, offset=self.offset, name=self.name)
        if poly.degree > 2:
            raise RuntimeError("quadratization unexpectedly produced degree>2")
        h = np.zeros(poly.n, dtype=float)
        J = np.zeros((poly.n, poly.n), dtype=float)
        for term, coeff in poly.terms.items():
            if len(term) == 1:
                h[term[0]] += coeff
            elif len(term) == 2:
                i, j = term
                J[i, j] += coeff
                J[j, i] += coeff
            else:
                raise RuntimeError("degree>2 term after quadratization")
        return IsingProblem(h=h, J=J, offset=poly.offset, name=self.name)

    def min_energy_for_logical_bits(self, logical_bits: Sequence[int]) -> tuple[float, tuple[int, ...]]:
        b = tuple(int(x) for x in logical_bits)
        if len(b) != self.logical_n:
            raise ValueError("logical_bits length mismatch")
        best = float("inf")
        best_a: tuple[int, ...] | None = None
        for anc in product((0, 1), repeat=self.ancilla_count):
            e = self.energy_bits(b + anc)
            if e < best:
                best, best_a = e, tuple(anc)
        assert best_a is not None
        return float(best), best_a


def quadratize_cubic_hubo(
    n: int,
    terms: Mapping[tuple[int, ...], float],
    *,
    offset: float = 0.0,
    name: str = "quadratized_hubo",
    penalty_margin: float = 1.0,
) -> QuadratizedQUBO:
    """Exact Rosenberg quadratization for binary degree<=3 HUBOs.

    Each cubic monomial c*x_i*x_j*x_k gets a dedicated ancilla y=x_i*x_j:

        c*x_i*x_j*x_k -> c*y*x_k

    plus the exact binary-product penalty

        M (x_i*x_j - 2 x_i*y - 2 x_j*y + 3y),

    with M=|c|+penalty_margin > |c|.  A unique ancilla per aggregated cubic
    monomial keeps the correctness argument local and makes validation simple.
    """
    if penalty_margin <= 0:
        raise ValueError("penalty_margin must be positive")
    out: dict[tuple[int, ...], float] = {}
    cubic: list[tuple[tuple[int, ...], float]] = []
    for raw, c0 in dict(terms).items():
        t = tuple(sorted(set(int(i) for i in raw)))
        c = float(c0)
        if len(t) <= 2:
            out[t] = out.get(t, 0.0) + c
        elif len(t) == 3:
            cubic.append((t, c))
        else:
            raise ValueError("Rosenberg quadratization in this module supports degree <= 3 only")
    ancilla_for: dict[tuple[int, int, int], int] = {}
    penalties: dict[int, float] = {}
    next_var = int(n)
    for t, c in sorted(cubic):
        i, j, k = t
        y = next_var
        next_var += 1
        ancilla_for[t] = y
        M = abs(c) + float(penalty_margin)
        penalties[y] = M
        # c*y*x_k
        out[tuple(sorted((y, k)))] = out.get(tuple(sorted((y, k))), 0.0) + c
        # Rosenberg penalty M[x_i x_j -2 x_i y -2 x_j y +3y]
        out[(i, j)] = out.get((i, j), 0.0) + M
        out[tuple(sorted((i, y)))] = out.get(tuple(sorted((i, y))), 0.0) - 2.0 * M
        out[tuple(sorted((j, y)))] = out.get(tuple(sorted((j, y))), 0.0) - 2.0 * M
        out[(y,)] = out.get((y,), 0.0) + 3.0 * M
    out = {t: float(c) for t, c in out.items() if abs(c) > 1e-15}
    return QuadratizedQUBO(
        logical_n=int(n), total_n=next_var, offset=float(offset), terms=out,
        ancilla_for_cubic=ancilla_for, penalties=penalties, name=name,
    )


def max3sat_to_qubo(problem: Max3SATProblem, *, penalty_margin: float = 1.0) -> QuadratizedQUBO:
    """Encode a MAX-3-SAT instance as a Rosenberg-quadratized QUBO."""
    offset, terms = problem.binary_hubo()
    return quadratize_cubic_hubo(problem.n, terms, offset=offset, name=problem.name + "_qubo", penalty_margin=penalty_margin)
