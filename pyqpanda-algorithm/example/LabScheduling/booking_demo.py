"""One-command booking CSV -> safe reduction -> CPU QAOA -> verified change ledger."""

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyqpanda_alg.LabScheduling.bookings import import_bookings
from pyqpanda_alg.LabScheduling.report import run_experiment


def run_demo(output: Path) -> None:
    """Run outage and unchanged control, save raw reports, CSV changes and SVG."""
    source = Path(__file__).parent / "data/bookings"
    calendar = json.loads((source / "outage.json").read_text(encoding="utf-8"))
    problem = import_bookings(source / "chain.csv", calendar)
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in ("outage", "control"):
        raw = copy.deepcopy(problem)
        if name == "control":
            for resource in raw["resources"]:
                resource["downtime"] = []
            raw["name"] = "合成无停机对照：原预约全部保留"
        path = output / f"{name}-input.json"
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        report = run_experiment(path, reduce=True)
        (output / f"{name}-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if report["status"] != "feasible":
            raise RuntimeError(f"demo failed: {name}: {report['status']}")
        reports[name] = report
    best = reports["outage"]["quantum"]["best"]
    reduction = reports["outage"]["quantum"]["reduction"]
    changes = []
    for row in best["schedule"]:
        task = next(t for t in problem["tasks"] if t["id"] == row["task"])
        original = task["original"]
        option = next(
            o
            for o in task["options"]
            if (o["resource"], o["start"]) == (row["resource"], row["start"])
        )
        pruned = any(
            p["task"] == task["id"]
            and p["resource"] == original["resource"]
            and p["start"] == original["start"]
            for p in reports["outage"]["model"]["pruned"]
        )
        changes.append(
            {
                "task": row["task"],
                "original_resource": original["resource"],
                "original_start": original["start"],
                "new_resource": row["resource"],
                "new_start": row["start"],
                "changed": row["changed"],
                "preference_cost": option["cost"],
                "change_cost": task["change_cost"] if row["changed"] else 0,
                "explanation": "原预约被停机排除"
                if pruned
                else "约束传播要求连锁改约，见 reduction.removals"
                if row["changed"]
                else "保留原预约",
            }
        )
    with (output / "changes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(changes[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(changes)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), layout="constrained", sharex=True)
    for ax, title, after in zip(
        axes,
        ("Original bookings + outage", "Verified reschedule"),
        (False, True),
        strict=True,
    ):
        for i, task in enumerate(problem["tasks"]):
            original = task["original"]
            row = next(r for r in best["schedule"] if r["task"] == task["id"])
            start = row["start"] if after else original["start"]
            rid = row["resource"] if after else original["resource"]
            y = {"oven": 0, "scope": 1}[rid]
            ax.barh(
                y,
                task["duration"],
                left=start,
                height=0.55,
                color=plt.get_cmap("tab10")(i),
            )
            ax.text(
                start + task["duration"] / 2, y, task["id"], ha="center", va="center"
            )
        ax.barh(
            0, 1, left=0, height=0.75, facecolor="none", edgecolor="red", hatch="xx"
        )
        ax.set(
            title=title, yticks=[0, 1], yticklabels=["Oven", "Microscope"], xlim=(0, 6)
        )
        ax.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("Time slot; red hatch = outage; synthetic booking data")
    matplotlib.rcParams["svg.hashsalt"] = "lab-booking-demo"
    matplotlib.rcParams["svg.fonttype"] = "none"
    fig.savefig(output / "booking-changes.svg", metadata={"Date": None})
    plt.close(fig)
    print(
        json.dumps(
            {
                "outage_cost": best["objective"],
                "control_cost": reports["control"]["quantum"]["best"]["objective"],
                "changed_tasks": sum(row["changed"] for row in changes),
                "original_qubits": reduction["original_qubits"],
                "component_qubits": reduction["component_qubits"],
                "outputs": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """Generate the complete offline demonstration in one command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/booking-demo"))
    run_demo(parser.parse_args().output)


if __name__ == "__main__":
    main()
