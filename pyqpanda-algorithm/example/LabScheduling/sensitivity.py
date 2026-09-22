"""Frozen change-cost analysis for the existing three-booking tradeoff demo."""

import copy
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyqpanda_alg.LabScheduling import Problem, load_problem, validate_assignment
from pyqpanda_alg.LabScheduling import report as report_module
from stress_benchmark import write_json
from summarize_matched import without_timing

HERE = Path(__file__).resolve().parent


def weighted_input(base: dict[str, Any], weight: int) -> dict[str, Any]:
    """Copy an input and change only the uniform nonnegative integer change cost."""
    if type(weight) is not int or not 0 <= weight <= 1_000_000:
        raise ValueError("weight must be an integer in [0, 1000000]")
    result = copy.deepcopy(base)
    for task in result["tasks"]:
        task["change_cost"] = weight
    Problem.from_dict(result)
    return result


def feasible_schedules(problem: Problem) -> list[dict[str, Any]]:
    """Enumerate raw choices with the independent semantic checker for this demo."""
    combinations = 1
    for task in problem.tasks:
        combinations *= len(task.options)
    if combinations > 1_000_000:
        raise ValueError("sensitivity enumeration is limited to one million choices")
    rows = []
    for choices in product(*(range(len(t.options)) for t in problem.tasks)):
        row = validate_assignment(problem, choices)
        if row["feasible"]:
            rows.append(row)
    return rows


def sample_metrics(
    report: dict[str, Any], alternatives: list[dict[str, Any]]
) -> dict[str, Any]:
    """Accept any tied optimum; keep infeasible and nonoptimal samples unmodified."""
    best = report["quantum"]["best"]
    feasible = best is not None and best["feasible"]
    optimum = min(row["objective"] for row in alternatives)
    business = report["quantum"].get("business_metrics") if feasible else None
    if feasible and business is None:
        raise ValueError("feasible sample lacks a raw-input business audit")
    gap = best["objective"] - optimum if feasible else None
    if gap != report["absolute_gap"] or gap is not None and gap < 0:
        raise ValueError("sample gap contradicts independent enumeration")
    if feasible:
        matching = [r for r in alternatives if r["choices"] == business["choices"]]
        if len(matching) != 1 or matching[0] != business:
            raise ValueError("sample business audit differs from raw enumeration")
    return {
        "seed": report["quantum"]["seed"],
        "status": report["status"],
        "objective": best["objective"] if feasible else None,
        "violation_count": best["violation_count"] if best is not None else None,
        "preference_cost": business["preference_cost"] if business else None,
        "change_cost": business["change_cost"] if business else None,
        "changed_tasks": business["changed_tasks"] if business else None,
        "choices": business["choices"] if business else None,
        "absolute_gap": gap,
        "optimal": feasible and gap == 0,
        "runtime_seconds": report["runtime_seconds"],
    }


def verify_replay(original: Path, current: dict[str, Any]) -> None:
    """Reject provenance or sample changes, normalizing both sides through JSON."""
    recorded = json.loads(original.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(current, allow_nan=False))
    if without_timing(recorded) != without_timing(normalized):
        raise ValueError("sensitivity replay differs beyond timing")


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    """Plot every certified cost line and actual gaps without hiding failures."""
    matplotlib.rcParams["svg.hashsalt"] = "lab-sensitivity"
    matplotlib.rcParams["svg.fonttype"] = "none"
    fig, (costs, gaps) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    weights = [row["weight"] for row in rows]
    for index in range(len(rows[0]["alternatives"])):
        costs.plot(
            weights,
            [r["alternatives"][index]["objective"] for r in rows],
            color="#bbbbbb",
            linewidth=1,
            alpha=0.6,
        )
    costs.plot(weights, [6 + 3 * w for w in weights], label="Shift all: 6 + 3w")
    costs.plot(weights, [12 + w for w in weights], label="Backup: 12 + w")
    costs.scatter(
        weights,
        [r["optimum"] for r in rows],
        marker="*",
        s=80,
        color="black",
        label="Certified optimum",
        zorder=4,
    )
    costs.axvline(3, linestyle=":", color="gray")
    costs.set(
        xlabel="Change cost w per changed booking",
        ylabel="Total cost (points)",
        title="All nine feasible schedules; synthetic data",
    )
    costs.legend(fontsize=8)
    for index, marker in enumerate(("o", "x", "+")):
        samples = [r["samples"][index] for r in rows]
        gaps.scatter(
            [
                w
                for w, s in zip(weights, samples, strict=True)
                if s["absolute_gap"] is not None
            ],
            [s["absolute_gap"] for s in samples if s["absolute_gap"] is not None],
            marker=marker,
            s=70,
            label=f"Seed {samples[0]['seed']}",
        )
    failures = [
        (r["weight"], s["seed"])
        for r in rows
        for s in r["samples"]
        if s["absolute_gap"] is None
    ]
    gaps.set(
        xlabel="Change cost w per changed booking",
        ylabel="Sampled optimality gap",
        title="Actual QAOA samples (overlapping points retained)",
    )
    gaps.text(
        0.02,
        0.96,
        f"No feasible sample: {failures or 'none'}",
        transform=gaps.transAxes,
        va="top",
        fontsize=8,
    )
    gaps.legend(loc="lower right", fontsize=8)
    for ax in (costs, gaps):
        ax.set_xticks(weights)
        ax.grid(alpha=0.2)
    fig.savefig(output / "sensitivity.svg", metadata={"Date": None})
    plt.close(fig)


def run_sensitivity(
    output: Path, *, verify_against: Path | None = None
) -> dict[str, Any]:
    """Run the frozen 21 experiments, certify all choices, and save full evidence."""
    config_path = HERE / "data/sensitivity.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = HERE / "data" / config["base_input"]
    snapshot = source.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != config["base_sha256"]:
        raise ValueError("sensitivity base input changed after protocol freeze")
    base = json.loads(snapshot)
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rows, records, input_hashes = [], [], []
    choices_reference = None
    for weight in config["weights"]:
        path = inputs / f"w{weight}.json"
        write_json(path, weighted_input(base, weight))
        reports = [
            report_module.run_experiment(path, seed=seed, **config["solver"])
            for seed in config["seeds"]
        ]
        # Enumeration is downstream of all actual quantum runs for this input.
        alternatives = feasible_schedules(load_problem(path))
        choices = [row["choices"] for row in alternatives]
        if (
            not alternatives
            or choices_reference is not None
            and choices != choices_reference
        ):
            raise RuntimeError("changing costs changed the feasible schedule set")
        choices_reference = choices
        optimum = min(row["objective"] for row in alternatives)
        optimal = [row for row in alternatives if row["objective"] == optimum]
        samples = []
        for report in reports:
            if (
                report["exact"]["best"]["objective"] != optimum
                or report["milp"]["status"] != "optimal"
                or report["milp"]["best"]["objective"] != optimum
                or report["exact"]["feasible_count"] != len(alternatives)
            ):
                raise RuntimeError("independent certificates disagree")
            for component in report["quantum"]["components"]:
                key = "".join(map(str, component["best"]["bits"][::-1]))
                if component["counts"].get(key, 0) <= 0:
                    raise RuntimeError("chosen quantum result was not sampled")
            samples.append(sample_metrics(report, alternatives))
            records.append(
                {"weight": weight, "seed": report["quantum"]["seed"], "report": report}
            )
        rows.append(
            {
                "weight": weight,
                "optimum": optimum,
                "optimal_choices": [row["choices"] for row in optimal],
                "optimal_changed_tasks": sorted(
                    {row["changed_tasks"] for row in optimal}
                ),
                "alternatives": alternatives,
                "samples": samples,
            }
        )
        input_hashes.append(
            {
                "weight": weight,
                "file": f"inputs/{path.name}",
                "sha256": reports[0]["input_sha256"],
            }
        )
    package = Path(report_module.__file__).parent
    files = {f"LabScheduling/{p.name}": p for p in sorted(package.glob("*.py"))}
    files.update(
        {
            f"example/{name}": HERE / name
            for name in (
                "sensitivity.py",
                "tradeoff_demo.py",
                "stress_benchmark.py",
                "summarize_matched.py",
            )
        }
    )
    files["protocol"] = (
        HERE.parents[2] / "Tutorials/LabScheduling/SENSITIVITY_PROTOCOL.md"
    )
    files["configuration"] = config_path
    result = {
        "config": config,
        "input_sha256": input_hashes,
        "source_sha256": {
            name: hashlib.sha256(p.read_bytes()).hexdigest()
            for name, p in files.items()
        },
        "environment": records[0]["report"]["environment"],
        "weights": rows,
        "runs": records,
    }
    if verify_against:
        verify_replay(verify_against, result)
        write_json(
            output / "replay.json",
            {
                "identical_except_timing": True,
                "weights": len(rows),
                "runs": len(records),
            },
        )
    # Compact full JSON avoids duplicating thousands of formatted optimizer lines.
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    summaries = [
        {
            "weight": row["weight"],
            "certified_optimum": row["optimum"],
            "optimal_choices": json.dumps(row["optimal_choices"]),
            "optimal_changed_tasks": json.dumps(row["optimal_changed_tasks"]),
            "feasible_runs": sum(s["absolute_gap"] is not None for s in row["samples"]),
            "optimal_runs": sum(s["optimal"] for s in row["samples"]),
            "sampled_objectives": json.dumps([s["objective"] for s in row["samples"]]),
            "sampled_gaps": json.dumps([s["absolute_gap"] for s in row["samples"]]),
        }
        for row in rows
    ]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)
    _plot(rows, output)
    print(
        json.dumps(
            {
                "sensitivity_runs": len(records),
                "decisions": summaries,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result
