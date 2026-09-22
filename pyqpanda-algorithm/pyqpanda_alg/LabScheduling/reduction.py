"""Lossless constraint propagation, components and auditable restoration."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from .model import Model, compile_qubo, evaluate
from .quantum import MAX_QUBITS, solve_qaoa


@dataclass(frozen=True)
class Reduction:
    """Lossless feasible-schedule map; indices always refer to the original model."""

    source: Model
    fixed: tuple[int, ...]
    components: tuple[Model, ...]
    mappings: tuple[tuple[int, ...], ...]
    removals: tuple[dict[str, Any], ...]
    infeasible_task: str | None
    pruning: str = "singleton"

    def audit(self) -> dict[str, Any]:
        """Explain every deletion, fixed cost and component-to-original mapping."""
        return {
            "method": (
                "arc consistency + singleton propagation"
                " + disconnected conflict components"
                if self.pruning == "arc"
                else "singleton propagation + disconnected conflict components"
            ),
            "original_qubits": len(self.source.variables),
            "fixed_variables": self.fixed,
            "fixed_cost": sum(self.source.variables[i].cost for i in self.fixed),
            "removals": self.removals,
            "infeasible_task": self.infeasible_task,
            "component_qubits": [len(m.variables) for m in self.components],
            "max_component_qubits": max(map(len, self.mappings), default=0),
            "component_to_original": self.mappings,
            "components": [m.audit() for m in self.components],
        }

    def restore(self, assignments: Sequence[Sequence[int]]) -> dict[str, Any]:
        """Reinsert fixed choices and validate all original task constraints."""
        if self.infeasible_task is not None:
            raise ValueError("cannot restore a proven infeasible reduction")
        if len(assignments) != len(self.components):
            raise ValueError("one bit vector per component is required")
        bits = [0] * len(self.source.variables)
        for i in self.fixed:
            bits[i] = 1
        for model, mapping, local in zip(
            self.components, self.mappings, assignments, strict=True
        ):
            evaluate(model, local)  # Validate dimensions and binary values.
            for i, bit in zip(mapping, local, strict=True):
                bits[i] = bit
        return evaluate(self.source, bits)


def _enforce_arc_consistency(
    model: Model,
    active: list[set[int]],
    neighbours: dict[int, dict[int, tuple[str, ...]]],
    removals: list[dict[str, Any]],
) -> None:
    """Run deterministic AC-3 and retain the full opposing domain as a witness."""
    adjacent: dict[int, set[int]] = {t: set() for t in range(len(active))}
    for i, j, _ in model.conflicts:
        a, b = model.variables[i].task, model.variables[j].task
        adjacent[a].add(b)
        adjacent[b].add(a)
    queue = deque((a, b) for a in adjacent for b in sorted(adjacent[a]))
    queued = set(queue)
    while queue:
        a, b = queue.popleft()
        queued.remove((a, b))
        revised = False
        for i in sorted(active[a]):
            if any(j not in neighbours[i] for j in active[b]):
                continue
            removals.append(
                {
                    "variable": i,
                    "rule": "no_support",
                    "against_task": model.problem.tasks[b].id,
                    "blockers": [
                        {"variable": j, "reasons": neighbours[i][j]}
                        for j in sorted(active[b])
                    ],
                }
            )
            active[a].remove(i)
            revised = True
            if not active[a]:
                return
        if revised:
            for other in sorted(adjacent[a] - {b}):
                arc = (other, a)
                if arc not in queued:
                    queue.append(arc)
                    queued.add(arc)


def reduce_model(model: Model, *, pruning: str = "singleton") -> Reduction:
    """Preserve every feasible schedule by propagating logically forced choices.

    A singleton must be selected, hence all its conflicting candidates are
    impossible. Optional pruning="arc" first removes choices incompatible with
    every surviving choice of another task, using AC-3. Neither mode proves
    global feasibility from nonempty domains. Then partition by surviving
    forbidden pairs. No dominance rules or heuristic freezing occur.
    """
    if pruning not in ("singleton", "arc"):
        raise ValueError("pruning must be singleton or arc")
    active = [set(domain) for domain in model.domains]
    neighbours: dict[int, dict[int, tuple[str, ...]]] = {
        i: {} for i in range(len(model.variables))
    }
    for i, j, reasons in model.conflicts:
        neighbours[i][j] = reasons
        neighbours[j][i] = reasons
    fixed: list[int] = []
    removals: list[dict[str, Any]] = []
    if pruning == "arc" and all(active):
        _enforce_arc_consistency(model, active, neighbours, removals)
    processed: set[int] = set()
    while True:
        empty = next((t for t, domain in enumerate(active) if not domain), None)
        if empty is not None:
            return Reduction(
                model,
                tuple(fixed),
                (),
                (),
                tuple(removals),
                model.problem.tasks[empty].id,
                pruning,
            )
        task = next(
            (t for t, d in enumerate(active) if len(d) == 1 and t not in processed),
            None,
        )
        if task is None:
            break
        processed.add(task)
        forced = min(active[task])
        fixed.append(forced)
        for other, reasons in sorted(neighbours[forced].items()):
            ti = model.variables[other].task
            if other in active[ti]:
                active[ti].remove(other)
                removals.append(
                    {
                        "variable": other,
                        "forced_by": forced,
                        "reasons": reasons,
                    }
                )
    remaining = set(range(len(active))) - processed
    adjacency: dict[int, set[int]] = {t: set() for t in remaining}
    for i, j, _ in model.conflicts:
        a, b = model.variables[i].task, model.variables[j].task
        if a in remaining and b in remaining and i in active[a] and j in active[b]:
            adjacency[a].add(b)
            adjacency[b].add(a)
    components = []
    mappings = []
    while remaining:
        pending = [min(remaining)]
        tasks: set[int] = set()
        while pending:
            task = pending.pop()
            if task in tasks:
                continue
            tasks.add(task)
            pending.extend(sorted(adjacency[task] - tasks, reverse=True))
        remaining -= tasks
        ordered = sorted(tasks)
        ids = {model.problem.tasks[t].id for t in ordered}
        problem = replace(
            model.problem,
            tasks=tuple(
                replace(
                    model.problem.tasks[t],
                    options=tuple(model.variables[i].option for i in sorted(active[t])),
                )
                for t in ordered
            ),
            precedence=tuple(
                (a, b) for a, b in model.problem.precedence if a in ids and b in ids
            ),
        )
        components.append(compile_qubo(problem))
        mappings.append(tuple(i for t in ordered for i in sorted(active[t])))
    return Reduction(
        model,
        tuple(fixed),
        tuple(components),
        tuple(mappings),
        tuple(removals),
        None,
        pruning,
    )


def solve_reduced(
    model: Model,
    seed: int = 7,
    layers: int = 1,
    shots: int = 512,
    maxiter: int = 60,
    restarts: int = 2,
    mixer: str = "xy",
    *,
    pruning: str = "singleton",
) -> dict[str, Any]:
    """Solve each exact component on CPU and restore; never fall back silently.

    Each component receives its own deterministic training/sampling seed and
    the stated shot budget. Independent component draws give product feasibility.
    A component exceeding MAX_QUBITS is reported before any quantum execution.
    Use pruning="arc" for lossless unsupported-candidate deletion before fixing.
    """
    started = perf_counter()
    # Validate solver arguments even when propagation solves the entire problem.
    empty = compile_qubo(replace(model.problem, tasks=(), precedence=()))
    solve_qaoa(empty, seed, layers, shots, maxiter, restarts, mixer)
    reduction = reduce_model(model, pruning=pruning)
    result: dict[str, Any] = {
        "seed": seed,
        "mixer": mixer,
        "reduction": reduction.audit(),
        "best": None,
        "components": [],
        "circuit_evaluations": 0,
    }
    if reduction.infeasible_task is not None:
        result["status"] = "infeasible_propagation"
    elif any(len(m.variables) > MAX_QUBITS for m in reduction.components):
        result["status"] = "resource_limit"
    else:
        seeds = np.random.SeedSequence(seed).spawn(len(reduction.components))
        runs = [
            solve_qaoa(
                m, int(s.generate_state(1)[0]), layers, shots, maxiter, restarts, mixer
            )
            for m, s in zip(reduction.components, seeds, strict=True)
        ]
        best = reduction.restore([run["best"]["bits"] for run in runs])
        result.update(
            {
                "status": ("feasible" if runs else "classical_propagation")
                if best["feasible"]
                else "no_feasible_sample",
                "best": best,
                "components": runs,
                "circuit_evaluations": sum(r["circuit_evaluations"] for r in runs),
                "feasible_probability": float(
                    np.prod([r["feasible_probability"] for r in runs])
                ),
                "total_shots": shots * len(runs),
            }
        )
    result["runtime_seconds"] = perf_counter() - started
    return result
