"""Auditable candidate pruning, QUBO compilation and independent validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .problem import Option, Problem, Task


@dataclass(frozen=True)
class Variable:
    """One surviving candidate; its tuple position is its qubit index."""

    task: int
    option: Option
    cost: int


@dataclass(frozen=True)
class Model:
    """Upper-triangular QUBO: offset + linear*x + sum(i<j) quadratic*x_i*x_j."""

    problem: Problem
    variables: tuple[Variable, ...]
    domains: tuple[tuple[int, ...], ...]
    linear: tuple[int, ...]
    quadratic: tuple[tuple[int, int, int], ...]
    offset: int
    penalty: int
    objective_bound: int
    conflicts: tuple[tuple[int, int, tuple[str, ...]], ...]
    pruned: tuple[dict[str, Any], ...]

    def energy(self, bits: Sequence[int]) -> int:
        """Evaluate the full QUBO, including its constant offset."""
        _validate_bits(bits, len(self.variables))
        return (
            self.offset
            + sum(c * b for c, b in zip(self.linear, bits, strict=True))
            + sum(c * bits[i] * bits[j] for i, j, c in self.quadratic)
        )

    def audit(self) -> dict[str, Any]:
        """Return JSON-ready coefficients and a stable variable-to-qubit map."""
        return {
            "qubits": len(self.variables),
            "penalty": self.penalty,
            "objective_bound": self.objective_bound,
            "offset": self.offset,
            "linear": list(self.linear),
            "quadratic": list(self.quadratic),
            "domains": self.domains,
            "pruned": self.pruned,
            "variables": [
                {
                    "index": i,
                    "task": self.problem.tasks[v.task].id,
                    "resource": v.option.resource,
                    "start": v.option.start,
                    "cost": v.cost,
                }
                for i, v in enumerate(self.variables)
            ],
            "conflicts": [
                {"i": i, "j": j, "reasons": reasons} for i, j, reasons in self.conflicts
            ],
        }


def compile_qubo(problem: Problem, penalty: int | None = None) -> Model:
    """Prune unary-invalid options and compile exact-one and pair penalties.

    Let U be the sum of the largest surviving cost per task. For nonnegative
    costs, A=U+1 makes every infeasible bit vector worse than every feasible
    schedule. A custom penalty must be an integer strictly greater than U.
    """
    resources = {r.id: r for r in problem.resources}
    variables: list[Variable] = []
    domains: list[tuple[int, ...]] = []
    pruned: list[dict[str, Any]] = []
    for ti, task in enumerate(problem.tasks):
        domain = []
        for option in task.options:
            resource = resources[option.resource]
            end = option.start + task.duration + resource.cleanup
            reasons = []
            if end > problem.horizon:
                reasons.append("horizon")
            if any(_overlap(option.start, end, a, b) for a, b in resource.downtime):
                reasons.append("downtime")
            if reasons:
                pruned.append(
                    {
                        "task": task.id,
                        "resource": option.resource,
                        "start": option.start,
                        "reasons": reasons,
                    }
                )
            else:
                domain.append(len(variables))
                variables.append(Variable(ti, option, _cost(task, option)))
                if len(variables) > 256:
                    raise ValueError(
                        "model compilation supports at most 256 candidates"
                    )
        domains.append(tuple(domain))
    bound = sum(max((variables[i].cost for i in d), default=0) for d in domains)
    if penalty is None:
        penalty = bound + 1
    if type(penalty) is not int or penalty <= bound:
        raise ValueError(f"penalty must be an integer greater than {bound}")
    linear = [v.cost - penalty for v in variables]
    quadratic: dict[tuple[int, int], int] = {}
    for indices in domains:
        for i, j in combinations(indices, 2):
            quadratic[i, j] = 2 * penalty
    conflicts = []
    precedence = set(problem.precedence)
    for i, j in combinations(range(len(variables)), 2):
        left, right = variables[i], variables[j]
        if left.task == right.task:
            continue
        a, b = problem.tasks[left.task], problem.tasks[right.task]
        oa, ob = left.option, right.option
        ea, eb = oa.start + a.duration, ob.start + b.duration
        reasons = []
        if oa.resource == ob.resource:
            cleanup = resources[oa.resource].cleanup
            if _overlap(oa.start, ea + cleanup, ob.start, eb + cleanup):
                reasons.append("resource")
        if a.group == b.group and _overlap(oa.start, ea, ob.start, eb):
            reasons.append("group")
        if (
            (a.id, b.id) in precedence
            and ea > ob.start
            or (b.id, a.id) in precedence
            and eb > oa.start
        ):
            reasons.append("precedence")
        if reasons:
            # Count a forbidden candidate pair once, even with several reasons.
            conflicts.append((i, j, tuple(reasons)))
            quadratic[i, j] = penalty
    return Model(
        problem,
        tuple(variables),
        tuple(domains),
        tuple(linear),
        tuple((i, j, c) for (i, j), c in sorted(quadratic.items())),
        penalty * len(problem.tasks),
        penalty,
        bound,
        tuple(conflicts),
        tuple(pruned),
    )


def evaluate(model: Model, bits: Sequence[int]) -> dict[str, Any]:
    """Decode and check constraints from input semantics, independently of QUBO.

    Violation count counts violated exact-one task constraints and distinct
    forbidden selected pairs. Detailed pair reasons may contain multiple causes.
    An infeasible vector has objective=None; raw_cost remains available to audit.
    """
    _validate_bits(bits, len(model.variables))
    selected = [i for i, bit in enumerate(bits) if bit]
    rows: list[dict[str, Any]] = []
    violations = []
    raw_cost = 0
    resources = {r.id: r for r in model.problem.resources}
    for ti, task in enumerate(model.problem.tasks):
        chosen = [i for i in selected if model.variables[i].task == ti]
        if len(chosen) != 1:
            violations.append(
                {"kind": "exact_one", "task": task.id, "count": len(chosen)}
            )
        for i in chosen:
            option = model.variables[i].option
            raw_cost += _cost(task, option)
            rows.append(
                {
                    "variable": i,
                    "task": task.id,
                    "group": task.group,
                    "resource": option.resource,
                    "start": option.start,
                    "end": option.start + task.duration,
                    "reserved_end": option.start
                    + task.duration
                    + resources[option.resource].cleanup,
                    "changed": task.original is not None
                    and task.original != (option.resource, option.start),
                }
            )
    for a, b in combinations(rows, 2):
        if a["task"] == b["task"]:
            continue
        reasons = []
        if a["resource"] == b["resource"] and _overlap(
            a["start"], a["reserved_end"], b["start"], b["reserved_end"]
        ):
            reasons.append("resource")
        if a["group"] == b["group"] and _overlap(
            a["start"], a["end"], b["start"], b["end"]
        ):
            reasons.append("group")
        for before, after in model.problem.precedence:
            if (
                (a["task"], b["task"]) == (before, after)
                and a["end"] > b["start"]
                or (b["task"], a["task"]) == (before, after)
                and b["end"] > a["start"]
            ):
                reasons.append("precedence")
        if reasons:
            violations.append(
                {
                    "kind": "conflict",
                    "tasks": [a["task"], b["task"]],
                    "variables": [a["variable"], b["variable"]],
                    "reasons": reasons,
                }
            )
    return {
        "feasible": not violations,
        "objective": raw_cost if not violations else None,
        "raw_cost": raw_cost,
        "violation_count": len(violations),
        "violations": violations,
        "schedule": sorted(rows, key=lambda row: (row["start"], row["task"])),
        "bits": list(bits),
        "qubo_energy": model.energy(bits),
    }


def _cost(task: Task, option: Option) -> int:
    changed = task.original is not None and task.original != (
        option.resource,
        option.start,
    )
    return option.cost + task.change_cost * changed


def _overlap(a: int, b: int, c: int, d: int) -> bool:
    return a < d and c < b


def _validate_bits(bits: Sequence[int], size: int) -> None:
    if len(bits) != size or any(b not in (0, 1) for b in bits):
        raise ValueError(f"expected {size} binary values in qubit-index order")
