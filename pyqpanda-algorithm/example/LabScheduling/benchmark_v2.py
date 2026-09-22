"""Run every frozen heterogeneous instance and both quantum execution paths."""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from importlib.metadata import version
from math import prod
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from ablation import audit_stages
from pyqpanda_alg.LabScheduling import (
    circuit_probabilities,
    compile_qubo,
    evaluate,
    load_problem,
    reduce_model,
    solve_milp,
    solve_qaoa,
    solve_reduced,
    validate_assignment,
)
from pyqpanda_alg.LabScheduling.metrics import distribution_metrics
from stress_benchmark import write_json


def check_sample(model: Any, run: dict[str, Any]) -> None:
    """Reject a selected result absent from counts or invalid in raw input."""
    if "counts" in run:
        key = "".join(map(str, run["best"]["bits"][::-1]))
        if run["counts"].get(key, 0) <= 0:
            raise RuntimeError("best result is not an observed quantum sample")
    if run["best"] and run["best"]["feasible"]:
        choices = [
            task.options.index(
                next(
                    v.option
                    for v, bit in zip(model.variables, run["best"]["bits"], strict=True)
                    if bit and v.task == ti
                )
            )
            for ti, task in enumerate(model.problem.tasks)
        ]
        raw = validate_assignment(model.problem, choices)
        if not raw["feasible"] or raw["objective"] != run["best"]["objective"]:
            raise RuntimeError("sample failed raw semantic check")


def diagnostics(model: Any, run: dict[str, Any]) -> dict[str, Any] | None:
    """Certify a trained distribution without feeding the optimum into training."""
    if "parameters" not in run:
        return None
    probabilities = circuit_probabilities(model, run["parameters"])
    full = circuit_probabilities(model, run["parameters"], phase_mode="full")
    error = float(np.max(np.abs(probabilities - full)))
    if error > 1e-10:
        raise RuntimeError("full and simplified phases disagree")
    metrics = distribution_metrics(model, probabilities)
    samples = []
    for shots in (8, 16, 32, 64, 128, 512):
        rng = np.random.default_rng(
            np.random.SeedSequence([run["seed"], shots, 20260929])
        )
        objectives = []
        for value in set(rng.choice(len(probabilities), size=shots, p=probabilities)):
            row = evaluate(
                model, [(int(value) >> j) & 1 for j in range(len(model.variables))]
            )
            if row["feasible"]:
                objectives.append(row["objective"])
        samples.append(
            {"shots": shots, "best_objective": min(objectives, default=None)}
        )
    return {**metrics, "phase_max_error": error, "independent_samples": samples}


def run(corpus: Path, output: Path) -> None:
    """Save all instances and seed runs; failed/fully classical cases remain visible."""
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    here = Path(__file__).parent
    if (
        hashlib.sha256((here / "make_corpus_v2.py").read_bytes()).hexdigest()
        != manifest["generator_sha256"]
    ):
        raise ValueError("frozen generator changed")
    for spec in manifest["instances"]:
        if (
            hashlib.sha256((corpus / spec["file"]).read_bytes()).hexdigest()
            != spec["sha256"]
        ):
            raise ValueError(f"frozen input changed: {spec['file']}")
    output.mkdir(parents=True, exist_ok=True)
    import pyqpanda_alg.LabScheduling as package

    write_json(
        output / "provenance.json",
        {
            "manifest": manifest,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                n: version(n) for n in ("numpy", "scipy", "pyqpanda3", "pyqpanda_alg")
            },
            "sources": {
                str(p.relative_to(here))
                if p.is_relative_to(here)
                else f"core/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    here / "ablation.py",
                    here / "stress_benchmark.py",
                    *Path(package.__file__).parent.glob("*.py"),
                ]
            },
        },
    )
    started = perf_counter()
    instances = []
    count = 0
    with (output / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for spec in manifest["instances"]:
            problem = load_problem(corpus / spec["file"])
            control = replace(
                problem,
                resources=tuple(replace(r, downtime=()) for r in problem.resources),
            )
            original = validate_assignment(control, [0] * len(control.tasks))
            if not original["feasible"] or original["objective"] != 0:
                raise RuntimeError(
                    "original bookings are not a valid zero-cost control"
                )
            model = compile_qubo(problem)
            reduction = reduce_model(model)
            stages = audit_stages(problem)
            milp = solve_milp(problem)
            if (milp["status"] == "optimal") != bool(stages["feasible_count"]) or milp[
                "status"
            ] not in ("optimal", "infeasible"):
                raise RuntimeError("MILP and exhaustive proof disagree")
            if milp["best"] and milp["best"]["objective"] != stages["exact_objective"]:
                raise RuntimeError("MILP optimum disagrees")
            instances.append(
                {
                    "input": spec,
                    "model": model.audit(),
                    "ablation": stages,
                    "milp": milp,
                    "control": original,
                }
            )
            for seed in manifest["optimization_seeds"]:
                # Both paths execute before any distribution diagnostics.
                direct = solve_qaoa(model, seed=seed, **manifest["training"])
                reduced = solve_reduced(model, seed=seed, **manifest["training"])
                check_sample(model, direct)
                check_sample(model, reduced)
                diagnostic_started = perf_counter()
                profile = diagnostics(model, direct)
                local = []
                for component, result in zip(
                    reduction.components, reduced["components"], strict=True
                ):
                    check_sample(component, result)
                    local.append(diagnostics(component, result))
                record = {
                    "instance": spec["file"],
                    "seed": seed,
                    "direct": direct,
                    "reduced": reduced,
                    "direct_metrics": profile,
                    "component_metrics": local,
                    "reduced_single_joint_draw_p_opt": prod(
                        m["optimal_probability"] for m in local
                    )
                    if reduced["status"]
                    in ("feasible", "no_feasible_sample", "classical_propagation")
                    else None,
                    "diagnostic_seconds": perf_counter() - diagnostic_started,
                }
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                stream.flush()
                count += 1
            print(
                spec["file"],
                milp["status"],
                "components",
                stages["reduction"]["component_qubits"],
                flush=True,
            )
    with (output / "instances.jsonl").open("w", encoding="utf-8") as stream:
        for record in instances:
            stream.write(
                json.dumps(
                    record, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
                + "\n"
            )
    write_json(
        output / "execution.json",
        {
            "instances": len(instances),
            "paired_seed_records": count,
            "runtime_seconds": perf_counter() - started,
        },
    )


def main() -> None:
    """Run the frozen second corpus independently of the first archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path(__file__).parent / "data/corpus-v2"
    )
    parser.add_argument("--output", type=Path, default=Path("reports/lab-v2"))
    args = parser.parse_args()
    run(args.corpus, args.output)


if __name__ == "__main__":
    main()
