# Licensed under the Apache License, Version 2.0.
"""Maximum-weight independent-set application model and QUBO conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

from ..problems.ising import IsingProblem, qubo_to_ising


@dataclass(frozen=True)
class MWISProblem:
    """Maximum-Weight Independent Set encoded as a QUBO/Ising problem.

    Binary variable ``x_i=1`` means vertex ``i`` is selected.  The minimization
    objective is

        E(x) = -sum_i w_i x_i + penalty * sum_(i,j in E) x_i x_j.

    For strictly positive vertex weights, ``penalty > max_i w_i`` is sufficient
    to make every global minimum feasible: from any violating assignment one
    can remove an endpoint of a violated edge and strictly lower the energy.
    """

    weights: np.ndarray
    edges: tuple[tuple[int, int], ...]
    penalty: float | None = None
    name: str = "mwis"

    def __post_init__(self):
        w = np.asarray(self.weights, dtype=float)
        if w.ndim != 1 or w.size < 1:
            raise ValueError("weights must be a non-empty 1D array")
        if np.any(w <= 0):
            raise ValueError("MWIS weights must be strictly positive")
        n = int(w.size)
        norm = set()
        for raw in self.edges:
            if len(raw) != 2:
                raise ValueError("MWIS edge must be (u,v)")
            u, v = int(raw[0]), int(raw[1])
            if u == v or not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"invalid MWIS edge {(u, v)}")
            if u > v:
                u, v = v, u
            norm.add((u, v))
        p = 1.25 * float(np.max(w)) if self.penalty is None else float(self.penalty)
        if p <= float(np.max(w)):
            raise ValueError("penalty must be > max(weights) for the guaranteed-feasible encoding")
        object.__setattr__(self, "weights", w.copy())
        object.__setattr__(self, "edges", tuple(sorted(norm)))
        object.__setattr__(self, "penalty", p)

    @property
    def n(self) -> int:
        return int(self.weights.size)

    def _bits(self, bits: Iterable[int]) -> np.ndarray:
        b = np.asarray(list(bits), dtype=int)
        if b.shape != (self.n,):
            raise ValueError("bits length must equal problem size")
        if np.any((b != 0) & (b != 1)):
            raise ValueError("bits must contain only 0/1")
        return b

    def is_independent_bits(self, bits: Iterable[int]) -> bool:
        b = self._bits(bits)
        return all(not (b[u] and b[v]) for u, v in self.edges)

    def weight_bits(self, bits: Iterable[int]) -> float:
        b = self._bits(bits)
        return float(self.weights @ b)

    def violations_bits(self, bits: Iterable[int]) -> int:
        b = self._bits(bits)
        return int(sum(int(b[u] and b[v]) for u, v in self.edges))

    def qubo_energy_bits(self, bits: Iterable[int]) -> float:
        b = self._bits(bits)
        return float(-self.weights @ b + self.penalty * sum(b[u] * b[v] for u, v in self.edges))

    def bits_from_bitstring(self, bitstring: str) -> np.ndarray:
        if len(bitstring) != self.n:
            raise ValueError("bitstring length must equal problem size")
        return np.asarray([int(c) for c in bitstring[::-1]], dtype=int)

    def bitstring_from_bits(self, bits: Iterable[int]) -> str:
        b = self._bits(bits)
        return "".join(str(int(x)) for x in b[::-1])

    def is_independent_bitstring(self, bitstring: str) -> bool:
        return self.is_independent_bits(self.bits_from_bitstring(bitstring))

    def weight_bitstring(self, bitstring: str) -> float:
        return self.weight_bits(self.bits_from_bitstring(bitstring))

    def violations_bitstring(self, bitstring: str) -> int:
        return self.violations_bits(self.bits_from_bitstring(bitstring))

    def to_qubo_matrix(self) -> np.ndarray:
        """Return upper-triangular-convention Q with E=sum Qii xi+sum_i<j Qij xi xj."""
        Q = np.zeros((self.n, self.n), dtype=float)
        for i, wi in enumerate(self.weights):
            Q[i, i] = -float(wi)
        for u, v in self.edges:
            Q[u, v] += float(self.penalty)
        return Q

    def to_ising(self) -> IsingProblem:
        return qubo_to_ising(self.to_qubo_matrix(), convention="upper", name=self.name)

    def exact_mwis(self) -> tuple[float, list[str]]:
        best = -float("inf")
        states: list[str] = []
        for idx in range(1 << self.n):
            bits = [(idx >> q) & 1 for q in range(self.n)]
            if not self.is_independent_bits(bits):
                continue
            value = self.weight_bits(bits)
            s = format(idx, f"0{self.n}b")
            if value > best + 1e-12:
                best, states = value, [s]
            elif np.isclose(value, best, atol=1e-12):
                states.append(s)
        return float(best), states
