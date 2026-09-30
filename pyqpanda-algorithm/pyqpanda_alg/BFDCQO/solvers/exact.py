# Licensed under the Apache License, Version 2.0.
"""Exact enumeration solver used as a small-instance ground-truth baseline."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ExactResult:
    """Ground-truth result returned by exhaustive enumeration."""

    best_energy: float
    best_solutions: list[str]
    states_evaluated: int

class ExactSolver:
    """Brute-force reference solver for small QUBO/Ising/HUBO instances."""
    def solve(self,problem)->ExactResult:
        if not hasattr(problem,"exact_ground_states") or not hasattr(problem,"n"):
            raise TypeError("problem must provide exact_ground_states() and n")
        e,states=problem.exact_ground_states()
        return ExactResult(float(e),states,1<<int(problem.n))
