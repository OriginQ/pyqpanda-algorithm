# Licensed under the Apache License, Version 2.0.
"""Deterministic MaxCut instances used by regression benchmarks."""

from __future__ import annotations
from ..problems.maxcut import MaxCutProblem

def default_maxcut_instances():
    """Return deterministic small MaxCut instances used by regression benchmarks."""
    cycle6=[(i,(i+1)%6,1.0) for i in range(6)]
    complete5=[(i,j,1.0) for i in range(5) for j in range(i+1,5)]
    weighted6=[
        (0,1,1.0),(1,2,1.4),(2,3,0.8),(3,4,1.3),(4,5,0.9),(5,0,1.2),
        (0,3,0.7),(1,4,1.1),(2,5,0.6),(0,2,0.5),(3,5,0.4)
    ]
    return [
        MaxCutProblem.from_edges(6,cycle6,"cycle6"),
        MaxCutProblem.from_edges(5,complete5,"complete5"),
        MaxCutProblem.from_edges(6,weighted6,"weighted6"),
    ]
