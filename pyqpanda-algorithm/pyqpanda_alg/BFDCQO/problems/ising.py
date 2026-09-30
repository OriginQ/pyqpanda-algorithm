# Licensed under the Apache License, Version 2.0.
"""Quadratic Ising/QUBO problem representation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np

@dataclass(frozen=True)
class IsingProblem:
    """Quadratic Ising minimization H=offset+sum h_i Z_i+sum_{i<j}J_ij Z_i Z_j."""
    h: np.ndarray
    J: np.ndarray
    offset: float = 0.0
    name: str = "ising"

    def __post_init__(self):
        h = np.asarray(self.h, dtype=float)
        J = np.asarray(self.J, dtype=float)
        if h.ndim != 1: raise ValueError("h must be 1D")
        if J.shape != (h.size, h.size): raise ValueError("J has wrong shape")
        if not np.allclose(J, J.T, atol=1e-12): raise ValueError("J must be symmetric")
        if not np.allclose(np.diag(J), 0.0, atol=1e-12): raise ValueError("J diagonal must be zero")
        object.__setattr__(self, "h", h.copy())
        object.__setattr__(self, "J", J.copy())

    @property
    def n(self) -> int: return int(self.h.size)

    def energy_bits(self, bits: Iterable[int]) -> float:
        bits = np.asarray(list(bits), dtype=int)
        if bits.shape != (self.n,): raise ValueError("bits length must equal problem size")
        z = 1.0 - 2.0 * bits
        return float(self.offset + self.h @ z + 0.5 * z @ self.J @ z)

    def energy_index(self, index: int) -> float:
        return self.energy_bits([(index >> q) & 1 for q in range(self.n)])

    def energy_bitstring(self, bitstring: str) -> float:
        if len(bitstring) != self.n: raise ValueError("bitstring length must equal problem size")
        return self.energy_bits([int(c) for c in bitstring[::-1]])

    def energies(self) -> np.ndarray:
        return np.array([self.energy_index(i) for i in range(1 << self.n)], dtype=float)

    def exact_ground_states(self) -> Tuple[float, list[str]]:
        e = self.energies(); emin=float(e.min())
        inds=np.flatnonzero(np.isclose(e,emin,atol=1e-12))
        return emin,[format(int(i),f"0{self.n}b") for i in inds]


def qubo_to_ising(Q: np.ndarray, *, convention: str = "upper", name: str = "qubo") -> IsingProblem:
    """Convert a binary QUBO objective exactly to a quadratic Ising model.

    ``convention="upper"`` interprets the upper triangle as unique pair
    coefficients, while ``"xtqx"`` interprets the input as ``x.T @ Q @ x``.
    """
    Q=np.asarray(Q,dtype=float)
    if Q.ndim!=2 or Q.shape[0]!=Q.shape[1]: raise ValueError("Q must be square")
    n=Q.shape[0]
    if convention=="upper":
        linear=np.diag(Q).copy(); pair=np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n): pair[i,j]=pair[j,i]=Q[i,j]
    elif convention=="xtqx":
        linear=np.diag(Q).copy(); pair=np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n): pair[i,j]=pair[j,i]=Q[i,j]+Q[j,i]
    else: raise ValueError("convention must be 'upper' or 'xtqx'")
    offset=float(0.5*linear.sum()); h=-0.5*linear; J=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            qij=pair[i,j]; offset+=0.25*qij; h[i]-=0.25*qij; h[j]-=0.25*qij; J[i,j]=J[j,i]=0.25*qij
    return IsingProblem(h=h,J=J,offset=offset,name=name)
