"""Show the cost of shifting three bookings versus renting a backup instrument."""

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ablation import audit_stages, feasible_vectors
from pyqpanda_alg.LabScheduling import compile_qubo, evaluate, load_problem
from pyqpanda_alg.LabScheduling.report import run_experiment
from stress_benchmark import write_json


def run_demo(output: Path) -> None:
    """Run actual QAOA, then show every feasible alternative and an outage control."""
    source = Path(__file__).parent / "data/tradeoff.json"
    problem = load_problem(source)
    output.mkdir(parents=True, exist_ok=True)
    report = run_experiment(source, reduce=True)
    write_json(output / "outage-report.json", report)
    raw = json.loads(source.read_text(encoding="utf-8"))
    control = copy.deepcopy(raw)
    for resource in control["resources"]:
        resource["downtime"] = []
    write_json(output / "outage-input.json", raw)
    write_json(output / "control-input.json", control)
    write_json(
        output / "control-report.json",
        run_experiment(output / "control-input.json", reduce=True),
    )
    write_json(output / "ablation.json", audit_stages(problem))
    model = compile_qubo(problem)
    best = report["quantum"]["best"]
    alternatives = []
    for bits, cost in feasible_vectors(model).items():
        candidate = evaluate(model, bits)
        alternatives.append(
            {
                "objective": cost,
                "changed_tasks": sum(r["changed"] for r in candidate["schedule"]),
                "schedule": "; ".join(
                    f"{r['task']}:{r['resource']}@{r['start']}"
                    for r in candidate["schedule"]
                ),
                "sampled_choice": bool(best and list(bits) == best["bits"]),
            }
        )
    alternatives.sort(key=lambda r: (r["objective"], r["schedule"]))
    with (output / "alternatives.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(alternatives[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(alternatives)
    if best and best["feasible"]:
        matplotlib.rcParams["svg.hashsalt"] = "lab-tradeoff"
        matplotlib.rcParams["svg.fonttype"] = "none"
        fig, axes = plt.subplots(
            2, 1, figsize=(9, 4.6), layout="constrained", sharex=True
        )
        for ax, after in zip(axes, (False, True), strict=True):
            for ti, task in enumerate(problem.tasks):
                row = next(r for r in best["schedule"] if r["task"] == task.id)
                rid, start = (row["resource"], row["start"]) if after else task.original
                y = {"oven": 0, "backup": 1}[rid]
                ax.barh(
                    y,
                    task.duration,
                    left=start,
                    height=0.5,
                    color=plt.get_cmap("tab10")(ti),
                )
                ax.barh(
                    y,
                    1,
                    left=start + task.duration,
                    height=0.5,
                    color="#dddddd",
                    hatch="//",
                )
                ax.text(start + task.duration / 2, y, task.id, ha="center", va="center")
            ax.axvspan(0, 1, ymin=0.08, ymax=0.46, color="red", alpha=0.15)
            ax.set(
                yticks=[0, 1],
                yticklabels=["Oven", "Backup"],
                ylim=(-0.7, 1.5),
                xlim=(0, 14),
                title="Actual sampled reschedule"
                if after
                else "Original reservations; outage on oven",
            )
            ax.grid(axis="x", alpha=0.2)
        axes[-1].set_xlabel("Time slot; gray hatch = cleanup; synthetic data")
        fig.savefig(output / "tradeoff.svg", metadata={"Date": None})
        plt.close(fig)
    print(
        json.dumps(
            {
                "status": report["status"],
                "sampled_objective": best["objective"] if best else None,
                "certified_optimum": report["milp"]["best"]["objective"],
                "gap": report["absolute_gap"],
                "component_qubits": report["quantum"]["reduction"]["component_qubits"],
                "fixed_variables": report["quantum"]["reduction"]["fixed_variables"],
                "feasible_alternatives": len(alternatives),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """Write the complete offline demonstration and all alternative schedules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/tradeoff-demo"))
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="also run 21 frozen change-cost experiments",
    )
    parser.add_argument(
        "--verify-sensitivity",
        type=Path,
        help="previous sensitivity/results.json to replay",
    )
    args = parser.parse_args()
    if args.verify_sensitivity and not args.sensitivity:
        parser.error("--verify-sensitivity requires --sensitivity")
    run_demo(args.output)
    if args.sensitivity:
        from sensitivity import run_sensitivity

        run_sensitivity(
            args.output / "sensitivity", verify_against=args.verify_sensitivity
        )


if __name__ == "__main__":
    main()
