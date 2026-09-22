"""Compare four scheduling paths under equal total sampling/training caps."""

import argparse
import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import version
from math import prod
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from benchmark_v2 import check_sample
from pyqpanda_alg.LabScheduling import (
    Model,
    Reduction,
    circuit_probabilities,
    compile_qubo,
    evaluate,
    load_problem,
    reduce_model,
    solve_exact,
    solve_qaoa,
)
from pyqpanda_alg.LabScheduling.metrics import distribution_metrics, hit_probability
from pyqpanda_alg.LabScheduling.quantum import MAX_QUBITS
from stress_benchmark import write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def allocate(total: int, count: int, minimum: int = 1) -> list[int]:
    """Split a positive budget deterministically; never exceed the total cap."""
    if any(type(v) is not int for v in (total, count, minimum)):
        raise ValueError("budgets and counts must be integers")
    if total < 1 or count < 0 or minimum < 1 or total < count * minimum:
        raise ValueError("insufficient or invalid budget")
    if not count:
        return []
    quotient, remainder = divmod(total, count)
    return [quotient + int(i < remainder) for i in range(count)]


def uniform(model: Model, seed: int, shots: int) -> dict[str, Any]:
    """Sample one-hot schedules without repair or consulting an exact optimum."""
    started = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20260930]))
    counts: Counter[str] = Counter()
    best = None
    if all(model.domains):
        for _ in range(shots):
            bits = [0] * len(model.variables)
            for domain in model.domains:
                bits[int(rng.choice(domain))] = 1
            counts["".join(map(str, bits[::-1]))] += 1
            row = evaluate(model, bits)
            if best is None or (not row["feasible"], row["qubo_energy"], bits) < (
                not best["feasible"],
                best["qubo_energy"],
                best["bits"],
            ):
                best = row
    return {
        "status": "infeasible_domain"
        if best is None
        else "feasible"
        if best["feasible"]
        else "no_feasible_sample",
        "best": best,
        **({"counts": dict(counts)} if best is not None else {}),
        "shots": sum(counts.values()),
        "seed": seed,
        "circuit_evaluations": 0,
        "runtime_seconds": perf_counter() - started,
    }


def components(
    reduction: Reduction,
    seed: int,
    *,
    quantum: bool,
    shots: int = 512,
    per_restart: int = 60,
) -> dict[str, Any]:
    """Divide a fixed total budget over exact components, then restore samples."""
    started = perf_counter()
    models = reduction.components
    runs = []
    best = None
    allocations = []
    if reduction.infeasible_task is not None:
        status = "infeasible_propagation"
    elif quantum and any(len(m.variables) > MAX_QUBITS for m in models):
        status = "resource_limit"
    else:
        try:
            sampled = allocate(shots, len(models))
            iterations = (
                allocate(per_restart, len(models), 6) if quantum else [0] * len(models)
            )
        except ValueError:
            status = "budget_insufficient"
        else:
            seeds = (
                [seed]
                if len(models) == 1
                else [
                    int(s.generate_state(1)[0])
                    for s in np.random.SeedSequence(seed).spawn(len(models))
                ]
            )
            for model, child_seed, budget, maxiter in zip(
                models, seeds, sampled, iterations, strict=True
            ):
                run = (
                    solve_qaoa(model, child_seed, shots=budget, maxiter=maxiter)
                    if quantum
                    else uniform(model, child_seed, budget)
                )
                check_sample(model, run)
                runs.append(run)
                allocations.append(
                    {
                        "shots": budget,
                        "maxiter_per_restart": maxiter,
                        "seed": child_seed,
                    }
                )
            best = reduction.restore([r["best"]["bits"] for r in runs])
            status = (
                "classical_propagation"
                if not models
                else "feasible"
                if best["feasible"]
                else "no_feasible_sample"
            )
    return {
        "status": status,
        "best": best,
        "components": runs,
        "allocations": allocations,
        "shots": sum(r["shots"] for r in runs),
        "circuit_evaluations": sum(r["circuit_evaluations"] for r in runs),
        "runtime_seconds": perf_counter() - started,
    }


def joint_hit(probabilities: list[float], shots: int) -> float:
    """Chance every component yields an optimum with a shared total shot cap."""
    return prod(
        hit_probability(p, s)
        for p, s in zip(probabilities, allocate(shots, len(probabilities)), strict=True)
    )


def profile(model: Model, run: dict[str, Any]) -> dict[str, Any] | None:
    """Certify all tied optima only after every solver has finished."""
    if "parameters" not in run:
        return None
    result = distribution_metrics(
        model, circuit_probabilities(model, run["parameters"])
    )
    return {
        k: result[k]
        for k in (
            "exact_objective",
            "optimal_count",
            "combinations",
            "optimal_probability",
            "uniform_optimal_probability",
        )
    }


def compare(model: Model, reduction: Reduction, seed: int) -> dict[str, Any]:
    """Run four uncorrected samplers, then add certificates and budget audits."""
    direct = solve_qaoa(model, seed=seed)
    reduced = components(reduction, seed, quantum=True)
    random_direct = uniform(model, seed, 512)
    random_reduced = components(reduction, seed, quantum=False)
    paths = {
        "direct": direct,
        "reduced": reduced,
        "uniform_direct": random_direct,
        "uniform_reduced": random_reduced,
    }
    for run in paths.values():
        check_sample(model, run)
        if run["circuit_evaluations"] > 120:
            raise RuntimeError("training cap exceeded")
        used = run.get("shots", 0)
        if used > 512 or (run.get("counts") and sum(run["counts"].values()) != used):
            raise RuntimeError("sampling cap violated")
    started = perf_counter()
    exact = solve_exact(model)
    optimum = exact["best"]["objective"] if exact["best"] else None
    direct_profile = profile(model, direct)
    local = [
        profile(m, r)
        for m, r in zip(reduction.components, reduced["components"], strict=True)
    ]
    curves = []
    if optimum is not None:
        for shots in (8, 16, 32, 64, 128, 512):
            curves.append(
                {
                    "total_shots": shots,
                    "direct": hit_probability(
                        direct_profile["optimal_probability"], shots
                    ),
                    "uniform_direct": hit_probability(
                        direct_profile["uniform_optimal_probability"], shots
                    ),
                    "reduced": joint_hit(
                        [m["optimal_probability"] for m in local], shots
                    ),
                    "uniform_reduced": joint_hit(
                        [m["uniform_optimal_probability"] for m in local], shots
                    ),
                }
            )
    return {
        "seed": seed,
        "paths": paths,
        "exact_objective": optimum,
        "absolute_gaps": {
            key: run["best"]["objective"] - optimum
            if optimum is not None and run["best"] and run["best"]["feasible"]
            else None
            for key, run in paths.items()
        },
        "direct_profile": direct_profile,
        "component_profiles": local,
        "conditional_hit_curve": curves,
        "diagnostic_seconds": perf_counter() - started,
    }


def run(corpus: Path, output: Path) -> None:
    """Execute every frozen input/seed; save all failures and source fingerprints."""
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    generator = HERE / "make_corpus_v2.py"
    if (
        hashlib.sha256(generator.read_bytes()).hexdigest()
        != manifest["generator_sha256"]
    ):
        raise ValueError("frozen generator changed")
    for item in manifest["instances"]:
        if (
            hashlib.sha256((corpus / item["file"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError(f"frozen input changed: {item['file']}")
    import pyqpanda_alg.LabScheduling as package

    output.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        HERE / "benchmark_v2.py",
        HERE / "ablation.py",
        HERE / "stress_benchmark.py",
        generator,
        ROOT / "Tutorials/LabScheduling/MATCHED_BUDGET_PROTOCOL.md",
        *Path(package.__file__).parent.glob("*.py"),
    ]
    write_json(
        output / "provenance.json",
        {
            "manifest": manifest,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "sources": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                n: version(n) for n in ("numpy", "scipy", "pyqpanda3", "pyqpanda_alg")
            },
            "training": {
                "layers": 1,
                "restarts": 2,
                "total_evaluation_cap": 120,
                "total_shots": 512,
            },
        },
    )
    started = perf_counter()
    count = 0
    with (output / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for item in manifest["instances"]:
            model = compile_qubo(load_problem(corpus / item["file"]))
            reduction = reduce_model(model)
            for seed in manifest["optimization_seeds"]:
                row = {"instance": item["file"], **compare(model, reduction, seed)}
                stream.write(
                    json.dumps(
                        row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                    )
                    + "\n"
                )
                count += 1
            stream.flush()
            print(item["file"], "complete", flush=True)
    write_json(
        output / "execution.json",
        {"paired_records": count, "runtime_seconds": perf_counter() - started},
    )


def main() -> None:
    """Run the fixed supplemental budget audit without modifying older archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "data/corpus-v2")
    parser.add_argument("--output", type=Path, default=Path("reports/lab-matched"))
    args = parser.parse_args()
    run(args.corpus, args.output)


if __name__ == "__main__":
    main()
