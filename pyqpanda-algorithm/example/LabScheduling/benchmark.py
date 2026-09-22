"""Reproduce seed/mixer ablations and write raw reports plus CSV metrics."""

import argparse
import csv
import json
from pathlib import Path

from pyqpanda_alg.LabScheduling.report import run_experiment


def main() -> None:
    """Benchmark both feasible datasets with three seeds and two mixers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/lab-scheduling"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 42])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in ("simple", "medium"):
        for seed in args.seeds:
            for mixer in ("xy", "x"):
                report = run_experiment(
                    Path(__file__).parent / "data" / f"{dataset}.json",
                    seed=seed,
                    mixer=mixer,
                )
                name = f"{dataset}-{mixer}-{seed}"
                (args.output / f"{name}.json").write_text(
                    json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                q = report["quantum"]
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "mixer": mixer,
                        "qubits": report["model"]["qubits"],
                        "objective": q["best"]["objective"],
                        "violations": q["best"]["violation_count"],
                        "exact_objective": report["exact"]["best"]["objective"],
                        "gap": report["absolute_gap"],
                        "feasible_probability": q["feasible_probability"],
                        "one_hot_probability": q["one_hot_probability"],
                        "uniform_feasible_probability": report["uniform_one_hot"][
                            "feasible_probability"
                        ],
                        "quantum_seconds": q["runtime_seconds"],
                        "exact_seconds": report["exact"]["runtime_seconds"],
                        "circuit_evaluations": q["circuit_evaluations"],
                    }
                )
                print(name, rows[-1], flush=True)
    with (args.output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
