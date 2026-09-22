"""Verify lossless arc pruning and reproduce a constructed resource boundary."""

import argparse
import hashlib
import json
from dataclasses import replace
from itertools import product
from math import prod
from pathlib import Path
from typing import Any

from ablation import feasible_vectors
from pyqpanda_alg.LabScheduling import (
    Model,
    Problem,
    Reduction,
    compile_qubo,
    load_problem,
    reduce_model,
    validate_assignment,
)
from pyqpanda_alg.LabScheduling.report import run_experiment
from stress_benchmark import write_json
from summarize_matched import without_timing

HERE = Path(__file__).resolve().parent


def verify_removals(model: Model, reduction: Reduction) -> None:
    """Replay each deletion witness, including its complete then-current domain."""
    active = [set(d) for d in model.domains]
    conflicts = {(i, j): reasons for i, j, reasons in model.conflicts}
    tasks = {t.id: i for i, t in enumerate(model.problem.tasks)}
    for row in reduction.removals:
        i = row["variable"]
        if i not in active[model.variables[i].task]:
            raise ValueError("deletion witness refers to an inactive candidate")
        if row.get("rule") == "no_support":
            opposing = active[tasks[row["against_task"]]]
            blockers = row["blockers"]
            if not opposing or [b["variable"] for b in blockers] != sorted(opposing):
                raise ValueError("deletion witness must contain the entire live domain")
        else:
            forced = row["forced_by"]
            if active[model.variables[forced].task] != {forced}:
                raise ValueError("forced deletion witness is not a singleton")
            blockers = [{"variable": forced, "reasons": row["reasons"]}]
        for blocker in blockers:
            pair = tuple(sorted((i, blocker["variable"])))
            if tuple(blocker["reasons"]) != conflicts.get(pair):
                raise ValueError("deletion witness lacks a matching forbidden pair")
        active[model.variables[i].task].remove(i)
    if reduction.infeasible_task is not None:
        if active[tasks[reduction.infeasible_task]]:
            raise ValueError("infeasibility witness has no empty domain")
    else:
        surviving = set().union(*active) if active else set()
        mapped = set(reduction.fixed).union(*map(set, reduction.mappings))
        if surviving != mapped:
            raise ValueError("restoration map differs from the surviving domains")


def audit_problem(problem: Problem) -> dict[str, Any]:
    """Compare both reductions to the full raw-input feasible set, with costs."""
    if prod(len(t.options) for t in problem.tasks) > 1_000_000:
        raise ValueError("support audit is limited to one million raw combinations")
    model = compile_qubo(problem)
    raw = {}
    for choices in product(*(range(len(t.options)) for t in problem.tasks)):
        result = validate_assignment(problem, choices)
        if result["feasible"]:
            raw[choices] = result["objective"]
    modes = {}
    for mode in ("singleton", "arc"):
        reduction = reduce_model(model, pruning=mode)
        verify_removals(model, reduction)
        restored = {}
        if reduction.infeasible_task is None:
            local = [feasible_vectors(m) for m in reduction.components]
            for assignment in product(*(list(vectors) for vectors in local)):
                result = reduction.restore(assignment)
                expected = reduction.audit()["fixed_cost"] + sum(
                    vectors[bits]
                    for vectors, bits in zip(local, assignment, strict=True)
                )
                if not result["feasible"] or result["objective"] != expected:
                    raise RuntimeError("restoration changed feasibility or cost")
                choices = tuple(
                    task.options.index(
                        model.variables[
                            next(i for i in domain if result["bits"][i])
                        ].option
                    )
                    for task, domain in zip(problem.tasks, model.domains, strict=True)
                )
                if choices in restored:
                    raise RuntimeError("restoration is not injective")
                restored[choices] = result["objective"]
        if restored != raw:
            raise RuntimeError("reduction changed the complete raw feasible set")
        modes[mode] = reduction.audit()
    return {
        "problem": problem.name,
        "feasible_count": len(raw),
        "exact_objective": min(raw.values(), default=None),
        "proof": "all raw feasible schedules and costs equal all restored schedules",
        "modes": modes,
    }


def run_evaluation() -> dict[str, Any]:
    """Audit all 36 existing V2 inputs and run three real CPU boundary-case seeds."""
    sources = sorted((HERE / "data/corpus-v2").glob("n*.json"))
    if len(sources) != 36:
        raise ValueError("expected the complete 36-instance V2 corpus")
    boundary = HERE / "data/support-chain.json"
    problem = load_problem(boundary)
    original = [
        next(i for i, o in enumerate(t.options) if (o.resource, o.start) == t.original)
        for t in problem.tasks
    ]
    control = replace(
        problem, resources=tuple(replace(r, downtime=()) for r in problem.resources)
    )
    rows = []
    for path in [*sources, boundary]:
        row = audit_problem(load_problem(path))
        row["input"] = path.relative_to(HERE).as_posix()
        row["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    # Solver results are kept separate from enumeration, never repaired with it.
    default = run_experiment(boundary, reduce=True)
    runs = [
        run_experiment(boundary, reduce=True, pruning="arc", seed=seed)
        for seed in (7, 19, 42)
    ]
    for report in runs:
        for component in report["quantum"]["components"]:
            key = "".join(map(str, component["best"]["bits"][::-1]))
            if component["counts"].get(key, 0) <= 0:
                raise RuntimeError("selected component result was not sampled")
    package = HERE.parents[1] / "pyqpanda_alg/LabScheduling"
    root = HERE.parents[2]
    files = [
        *sorted(package.glob("*.py")),
        Path(__file__),
        HERE / "ablation.py",
        root / "Tutorials/LabScheduling/ARC_PROTOCOL.md",
    ]
    return {
        "protocol": "ARC_PROTOCOL.md; known V2 corpus and constructed boundary case",
        "source_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "instances": rows,
        "boundary_original": {
            "before_outage": validate_assignment(control, original),
            "after_outage": validate_assignment(problem, original),
        },
        "boundary_default": default,
        "boundary_arc": runs,
    }


def main() -> None:
    """Write complete evidence, optionally requiring an exact non-timing replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/support-pruning"))
    parser.add_argument("--verify-against", type=Path)
    args = parser.parse_args()
    # Compare the JSON representation on both sides, including tuple domains.
    result = json.loads(json.dumps(run_evaluation(), allow_nan=False))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_against:
        original = json.loads(args.verify_against.read_text(encoding="utf-8"))
        if without_timing(original) != without_timing(result):
            raise ValueError("support-pruning replay differs beyond timing")
        write_json(args.output / "replay.json", {"identical_except_timing": True})
    write_json(args.output / "results.json", result)
    corpus = result["instances"][:-1]
    print(
        json.dumps(
            {
                "audited_instances": len(result["instances"]),
                "v2_strictly_smaller_max_component": sum(
                    r["modes"]["arc"]["max_component_qubits"]
                    < r["modes"]["singleton"]["max_component_qubits"]
                    for r in corpus
                ),
                "boundary_component_qubits": {
                    mode: result["instances"][-1]["modes"][mode]["component_qubits"]
                    for mode in ("singleton", "arc")
                },
                "boundary_default_status": result["boundary_default"]["status"],
                "boundary_arc": [
                    {
                        "seed": r["quantum"]["seed"],
                        "status": r["status"],
                        "objective": r["quantum"]["best"]["objective"]
                        if r["quantum"]["best"]
                        else None,
                        "gap": r["absolute_gap"],
                    }
                    for r in result["boundary_arc"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
