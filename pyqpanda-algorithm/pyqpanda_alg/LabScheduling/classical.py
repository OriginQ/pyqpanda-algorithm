"""Independent exact certification by Cartesian enumeration of task choices."""

from itertools import product
from math import prod
from time import perf_counter
from typing import Any

from .model import Model, evaluate


def solve_exact(model: Model, max_combinations: int = 1_000_000) -> dict[str, Any]:
    """Certify the optimum using schedule constraints, never QUBO minimization.

    An empty domain proves infeasibility. A search budget overflow raises before
    enumeration; it is never misreported as an infeasible instance.
    """
    if type(max_combinations) is not int or max_combinations < 1:
        raise ValueError("max_combinations must be a positive integer")
    started = perf_counter()
    combinations = prod(map(len, model.domains))
    if combinations > max_combinations:
        raise ValueError(
            f"exact search needs {combinations} combinations; limit={max_combinations}"
        )
    best = None
    feasible_count = 0
    for choice in product(*model.domains):
        bits = [0] * len(model.variables)
        for index in choice:
            bits[index] = 1
        result = evaluate(model, bits)
        if result["feasible"]:
            feasible_count += 1
            if best is None or result["objective"] < best["objective"]:
                best = result
    return {
        "status": "optimal" if best is not None else "infeasible",
        "best": best,
        "combinations": combinations,
        "feasible_count": feasible_count,
        "runtime_seconds": perf_counter() - started,
    }


def solve_greedy(model: Model) -> dict[str, Any]:
    """Choose the cheapest currently compatible option in input task order.

    No backtracking or repair is performed. A dead end is not an infeasibility
    certificate; this is a transparent operational heuristic for comparisons.
    """
    started = perf_counter()
    bits = [0] * len(model.variables)
    selected: set[int] = set()
    conflicts = {frozenset((i, j)) for i, j, _ in model.conflicts}
    for domain in model.domains:
        options = sorted(domain, key=lambda i: (model.variables[i].cost, i))
        chosen = next(
            (
                i
                for i in options
                if all(frozenset((i, j)) not in conflicts for j in selected)
            ),
            None,
        )
        if chosen is None:
            return {
                "status": "dead_end",
                "best": None,
                "runtime_seconds": perf_counter() - started,
            }
        bits[chosen] = 1
        selected.add(chosen)
    return {
        "status": "feasible",
        "best": evaluate(model, bits),
        "runtime_seconds": perf_counter() - started,
    }
