"""Small classical references for QmRMR feature selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import QmRMRProblem


@dataclass(frozen=True)
class GreedySolution:
    """Forward-selection result for a fixed-cardinality objective."""

    state: tuple[int, ...]
    value: float
    evaluations: int


def greedy_forward_selection(problem: QmRMRProblem) -> GreedySolution:
    """Select one feature at a time using the signed objective.

    The method is intentionally simple: it is a transparent heuristic
    reference, not an additional quantum algorithm.  At each step the
    candidate giving the lowest objective for the currently selected subset is
    retained.  The exact enumerator in :class:`QmRMRProblem` remains the oracle.
    """

    selected: list[int] = []
    evaluations = 0
    for _ in range(problem.select_num):
        candidates: list[tuple[float, int]] = []
        for index in range(problem.n_features):
            if index in selected:
                continue
            bits = np.zeros(problem.n_features, dtype=int)
            bits[selected] = 1
            bits[index] = 1
            candidates.append((problem.objective_unconstrained(bits), index))
            evaluations += 1
        if not candidates:
            raise RuntimeError("no candidate remains for greedy selection")
        _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        selected.append(chosen)

    state = tuple(int(index in selected) for index in range(problem.n_features))
    return GreedySolution(state, problem.objective(state), evaluations)

