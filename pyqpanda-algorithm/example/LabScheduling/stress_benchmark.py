"""Run the frozen corpus without tuning/filtering; save every optimizer outcome."""

import argparse
import csv
import hashlib
import json
import platform
import resource
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pyqpanda_alg.LabScheduling import (
    circuit_probabilities,
    compile_qubo,
    evaluate,
    load_problem,
    reduce_model,
    solve_exact,
    solve_milp,
    solve_qaoa,
)
from pyqpanda_alg.LabScheduling.classical import solve_greedy
from pyqpanda_alg.LabScheduling.metrics import circuit_resources, distribution_metrics


def write_json(path: Path, value: Any) -> None:
    """Persist finite JSON with a final newline."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(corpus: Path, output: Path, split: str) -> None:
    """Train 10 seeds per registered instance and audit fixed-parameter equivalence."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((corpus / "manifest.json").read_text())
    source = Path(__file__).with_name("make_corpus.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["generator_sha256"]:
        raise ValueError("generator changed after corpus freeze")
    for row in manifest["instances"]:
        if (
            hashlib.sha256((corpus / row["file"]).read_bytes()).hexdigest()
            != row["sha256"]
        ):
            raise ValueError(f"input hash mismatch: {row['file']}")
    started = perf_counter()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(
        output / "provenance.json",
        {
            "manifest": manifest,
            "split": split,
            "benchmark_sha256": script_hash,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                n: version(n) for n in ("numpy", "scipy", "pyqpanda3", "pyqpanda_alg")
            },
            "source_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(
                    __import__(
                        "pyqpanda_alg.LabScheduling", fromlist=["__file__"]
                    ).__file__
                ).parent.glob("*.py")
            },
        },
    )
    rows = []
    for spec in manifest["instances"]:
        if spec["split"] != split:
            continue
        problem = load_problem(corpus / spec["file"])
        model = compile_qubo(problem)
        exact, oracle = solve_exact(model), solve_milp(problem)
        if exact["status"] != oracle["status"] or (
            exact["best"] and exact["best"]["objective"] != oracle["best"]["objective"]
        ):
            raise RuntimeError("independent oracle disagreement")
        greedy = solve_greedy(model)
        for seed in manifest["optimization_seeds"]:
            quantum = solve_qaoa(model, seed=seed, **manifest["training"])
            diagnostics_started = perf_counter()
            probabilities = circuit_probabilities(model, quantum["parameters"])
            full = circuit_probabilities(
                model, quantum["parameters"], phase_mode="full"
            )
            error = float(np.max(np.abs(probabilities - full)))
            if error > 1e-10:
                raise RuntimeError("phase equivalence failed")
            metrics = distribution_metrics(model, probabilities)
            samples = []
            for shots in manifest["shots"]:
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, shots, 20260928])
                )
                # Fresh samples; initialization and training RNG streams are not reused.
                draws = rng.choice(len(probabilities), size=shots, p=probabilities)
                candidates = [
                    evaluate(
                        model, [(int(i) >> j) & 1 for j in range(len(model.variables))]
                    )
                    for i in set(draws)
                ]
                feasible = [r["objective"] for r in candidates if r["feasible"]]
                uniform = []
                for _ in range(shots):
                    bits = [0] * len(model.variables)
                    for domain in model.domains:
                        bits[int(rng.choice(domain))] = 1
                    candidate = evaluate(model, bits)
                    if candidate["feasible"]:
                        uniform.append(candidate["objective"])
                samples.append(
                    {
                        "shots": shots,
                        "quantum_objective": min(feasible, default=None),
                        "uniform_objective": min(uniform, default=None),
                    }
                )
            gates = circuit_resources(model, quantum["parameters"])
            record = {
                "input": spec,
                "model": model.audit(),
                "quantum": quantum,
                "metrics": metrics,
                "raw_milp": oracle,
                "exact": exact,
                "greedy": greedy,
                "resources": gates,
                "phase_max_probability_error": error,
                "samples": samples,
                "reduction": reduce_model(model).audit(),
                "diagnostic_seconds": perf_counter() - diagnostics_started,
            }
            name = f"{Path(spec['file']).stem}-seed{seed}"
            write_json(output / f"{name}.json", record)
            rows.append(
                {
                    "instance": spec["file"],
                    "seed": seed,
                    "qubits": len(model.variables),
                    "conflicts": len(model.conflicts),
                    "exact_status": exact["status"],
                    "exact_objective": metrics["exact_objective"],
                    "quantum_objective": quantum["best"]["objective"],
                    "greedy_objective": greedy["best"]["objective"]
                    if greedy["best"]
                    else None,
                    "p_opt": metrics["optimal_probability"],
                    "p_feasible": metrics["feasible_probability"],
                    "uniform_p_opt": metrics["uniform_optimal_probability"],
                    "hit16": metrics["hit_curve"][1]["quantum"],
                    "uniform_hit16": metrics["hit_curve"][1]["uniform"],
                    "cnot_full": gates["full"]["cnot"],
                    "cnot_auto": gates["auto"]["cnot"],
                    "depth_full": gates["full"]["depth"],
                    "depth_auto": gates["auto"]["depth"],
                    "phase_max_error": error,
                    "evaluations": quantum["circuit_evaluations"],
                    "quantum_seconds": quantum["runtime_seconds"],
                    "milp_seconds": oracle["runtime_seconds"],
                    "exact_seconds": exact["runtime_seconds"],
                }
            )
            print(
                name,
                quantum["status"],
                f"p_opt={metrics['optimal_probability']:.5f}",
                flush=True,
            )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "execution.json",
        {
            "records": len(rows),
            "runtime_seconds": perf_counter() - started,
            "peak_process_rss_kib_linux": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "rss_scope": "whole process high-water mark on Linux; not per circuit",
        },
    )


def main() -> None:
    """Run development separately; evaluation uses the same frozen training budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path(__file__).parent / "data/corpus"
    )
    parser.add_argument("--output", type=Path, default=Path("reports/lab-enhanced"))
    parser.add_argument(
        "--split", choices=("development", "evaluation"), default="evaluation"
    )
    args = parser.parse_args()
    run(args.corpus, args.output, args.split)


if __name__ == "__main__":
    main()
