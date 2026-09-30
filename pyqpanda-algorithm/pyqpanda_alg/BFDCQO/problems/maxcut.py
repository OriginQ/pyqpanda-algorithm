# Licensed under the Apache License, Version 2.0.
"""Weighted MaxCut application model."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
from .ising import IsingProblem

@dataclass(frozen=True)
class MaxCutProblem:
    """Weighted MaxCut instance with exact conversion to an Ising objective."""

    n: int
    edges: tuple[tuple[int,int,float], ...]
    name: str = "maxcut"

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Sequence[float]], name: str="maxcut"):
        norm=[]
        for e in edges:
            if len(e)==2: u,v=e; w=1.0
            elif len(e)==3: u,v,w=e
            else: raise ValueError("edge must be (u,v) or (u,v,w)")
            u=int(u); v=int(v); w=float(w)
            if u==v or not (0<=u<n and 0<=v<n): raise ValueError(f"invalid edge {e}")
            if u>v: u,v=v,u
            norm.append((u,v,w))
        return cls(n=int(n), edges=tuple(norm), name=name)

    def cut_value_bits(self,bits) -> float:
        b=np.asarray(list(bits),dtype=int)
        if b.shape!=(self.n,): raise ValueError("bits length mismatch")
        return float(sum(w for u,v,w in self.edges if b[u]!=b[v]))

    def cut_value_bitstring(self,s: str) -> float:
        return self.cut_value_bits([int(c) for c in s[::-1]])

    def to_ising(self) -> IsingProblem:
        # Minimize -Cut = -sum w/2 + sum (w/2) z_u z_v.
        h=np.zeros(self.n); J=np.zeros((self.n,self.n)); offset=0.0
        for u,v,w in self.edges:
            offset -= 0.5*w; J[u,v]+=0.5*w; J[v,u]+=0.5*w
        return IsingProblem(h=h,J=J,offset=offset,name=self.name)

    def exact_maxcut(self):
        p=self.to_ising(); e,states=p.exact_ground_states()
        return -e, states
