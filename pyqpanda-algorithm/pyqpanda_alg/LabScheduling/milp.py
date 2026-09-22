"""Independent raw-input MILP oracle; does not import the QUBO compiler."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from .problem import Option, Problem, Task

MAX_RAW_CANDIDATES = 2048


def validate_assignment(problem: Problem, choices: Sequence[int]) -> dict[str, Any]:
    """Check raw option indices, unary calendars, pairs and original-booking cost.

    This checker deliberately uses neither compiled variables nor conflict edges.
    One raw option index per task is mandatory; invalid indices raise ValueError.
    """
    if len(choices) != len(problem.tasks) or any(
        type(i) is not int or not 0 <= i < len(t.options)
        for t, i in zip(problem.tasks, choices, strict=True)
    ):
        raise ValueError("expected one valid raw option index per task")
    selected = [(t, t.options[i]) for t, i in zip(problem.tasks, choices, strict=True)]
    violations: list[dict[str, Any]] = []
    for task, option in selected:
        if not _available(problem, task, option):
            violations.append({"kind": "calendar", "task": task.id})
    for (a, oa), (b, ob) in combinations(selected, 2):
        if _incompatible(problem, a, oa, b, ob):
            violations.append({"kind": "pair", "tasks": [a.id, b.id]})
    preference = sum(o.cost for _, o in selected)
    changed = [
        (t, o)
        for t, o in selected
        if t.original is not None and t.original != (o.resource, o.start)
    ]
    change = sum(t.change_cost for t, _ in changed)
    return {
        "feasible": not violations,
        "violations": violations,
        "violation_count": len(violations),
        "objective": preference + change if not violations else None,
        "preference_cost": preference,
        "change_cost": change,
        "changed_tasks": len(changed),
        "max_start_shift": max(
            (abs(o.start - t.original[1]) for t, o in changed if t.original), default=0
        ),
        "choices": list(choices),
        "schedule": [
            {
                "task": t.id,
                "resource": o.resource,
                "start": o.start,
                "end": o.start + t.duration,
            }
            for t, o in selected
        ],
    }


def solve_milp(problem: Problem, time_limit: float = 30.0) -> dict[str, Any]:
    """Minimize raw-option cost with exact-one, calendar bounds and pair rows.

    HiGHS status, incumbent, dual bound and relative gap are reported separately.
    A time limit (solver time only) is never presented as an optimality certificate.
    At most 2048 raw candidates are accepted to bound quadratic model construction.
    A task with no calendar-compatible raw option proves infeasibility before
    constructing pair rows; this independent check does not use QUBO pruning.
    """
    if isinstance(time_limit, bool) or not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    started = perf_counter()
    options = [
        (ti, oi, t, o)
        for ti, t in enumerate(problem.tasks)
        for oi, o in enumerate(t.options)
    ]
    if len(options) > MAX_RAW_CANDIDATES:
        raise ValueError(f"MILP supports at most {MAX_RAW_CANDIDATES} raw candidates")
    base: dict[str, Any] = {
        "variables": len(options),
        "best": None,
        "dual_bound": None,
        "gap": None,
    }
    if not problem.tasks or any(not t.options for t in problem.tasks):
        feasible = not problem.tasks
        base.update(
            status="optimal" if feasible else "infeasible",
            best=validate_assignment(problem, []) if feasible else None,
            runtime_seconds=perf_counter() - started,
        )
        return base
    costs = [
        o.cost
        + (
            t.change_cost
            if t.original is not None and t.original != (o.resource, o.start)
            else 0
        )
        for _, _, t, o in options
    ]
    upper = [int(_available(problem, t, o)) for _, _, t, o in options]
    available_tasks = {
        ti for allowed, (ti, _, _, _) in zip(upper, options, strict=True) if allowed
    }
    for ti, task in enumerate(problem.tasks):
        if ti not in available_tasks:
            base.update(
                status="infeasible",
                infeasible_task=task.id,
                reason="no raw option satisfies the resource calendar",
                constraints=0,
                runtime_seconds=perf_counter() - started,
            )
            return base
    rows: list[list[int]] = [
        [i for i, (ti, _, _, _) in enumerate(options) if ti == task]
        for task in range(len(problem.tasks))
    ]
    lower = [1.0] * len(rows)
    bounds = [1.0] * len(rows)
    # If upper[i] == 0, y_i + y_j <= 1 follows from y_j <= 1 already.
    # Keep every raw variable and its index, but omit these redundant pair rows.
    available_indices = [i for i, allowed in enumerate(upper) if allowed]
    for i, j in combinations(available_indices, 2):
        ti, _, a, oa = options[i]
        tj, _, b, ob = options[j]
        if ti != tj and _incompatible(problem, a, oa, b, ob):
            rows.append([i, j])
            lower.append(-np.inf)
            bounds.append(1.0)
    row_indices = [r for r, columns in enumerate(rows) for _ in columns]
    column_indices = [i for columns in rows for i in columns]
    matrix = csc_matrix(
        (np.ones(len(row_indices)), (row_indices, column_indices)),
        shape=(len(rows), len(options)),
    )
    result = milp(
        np.asarray(costs, dtype=float),
        integrality=np.ones(len(options)),
        bounds=Bounds(np.zeros(len(options)), upper),
        constraints=LinearConstraint(matrix, lower, bounds),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    status = {0: "optimal", 1: "limit", 2: "infeasible"}.get(result.status, "error")
    if result.x is not None:
        if not np.allclose(result.x, np.rint(result.x), atol=1e-6, rtol=0):
            raise RuntimeError("MILP incumbent is not integral")
        for ti in range(len(problem.tasks)):
            if (
                sum(
                    int(x > 0.5)
                    for x, (task, _, _, _) in zip(result.x, options, strict=True)
                    if task == ti
                )
                != 1
            ):
                raise RuntimeError("MILP incumbent violates exact-one assignment")
        chosen = [
            oi for x, (_, oi, _, _) in zip(result.x, options, strict=True) if x > 0.5
        ]
        best = validate_assignment(problem, chosen)
        if not best["feasible"] or abs(best["objective"] - result.fun) > 1e-5:
            raise RuntimeError("MILP incumbent failed independent semantic check")
        base["best"] = best
    if status == "optimal" and base["best"] is None:
        raise RuntimeError("MILP reported optimal without an incumbent")
    for output, attribute in (("dual_bound", "mip_dual_bound"), ("gap", "mip_gap")):
        value = getattr(result, attribute, None)
        base[output] = (
            float(value) if value is not None and np.isfinite(value) else None
        )
    base.update(
        status=status,
        solver_status=int(result.status),
        message=str(result.message),
        constraints=len(rows),
        runtime_seconds=perf_counter() - started,
    )
    return base


def _available(problem: Problem, task: Task, option: Option) -> bool:
    resource = next(r for r in problem.resources if r.id == option.resource)
    finish = option.start + task.duration + resource.cleanup
    return finish <= problem.horizon and all(
        finish <= begin or option.start >= end for begin, end in resource.downtime
    )


def _incompatible(problem: Problem, a: Task, oa: Option, b: Task, ob: Option) -> bool:
    end_a, end_b = oa.start + a.duration, ob.start + b.duration
    if oa.resource == ob.resource:
        cleanup = next(r.cleanup for r in problem.resources if r.id == oa.resource)
        if not (end_a + cleanup <= ob.start or end_b + cleanup <= oa.start):
            return True
    if a.group == b.group and not (end_a <= ob.start or end_b <= oa.start):
        return True
    return any(
        (before == a.id and after == b.id and end_a > ob.start)
        or (before == b.id and after == a.id and end_b > oa.start)
        for before, after in problem.precedence
    )
