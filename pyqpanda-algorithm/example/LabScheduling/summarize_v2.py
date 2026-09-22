"""Summarize every heterogeneous instance without discarding failed runs."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stress_benchmark import write_json


def summarize(source: Path) -> None:
    """Add compact instance/cell tables and stage-resource plots to a run archive."""
    instances = [
        json.loads(line)
        for line in (source / "instances.jsonl").read_text().splitlines()
    ]
    runs = [
        json.loads(line) for line in (source / "runs.jsonl").read_text().splitlines()
    ]
    groups = defaultdict(list)
    for run in runs:
        groups[run["instance"]].append(run)
    rows = []
    for instance in instances:
        spec = instance["input"]
        group = groups[spec["file"]]
        stages = instance["ablation"]["stages"]
        opt = instance["ablation"]["exact_objective"]
        profiles = [
            r["direct_metrics"] for r in group if r["direct_metrics"] is not None
        ]
        final = stages.get("components_auto_phase")
        rows.append(
            {
                "instance": spec["file"],
                "tasks": spec["tasks"],
                "scenario": spec["scenario"],
                "replicate": spec["replicate"],
                "feasible": opt is not None,
                "optimum": opt,
                "raw_candidates": stages["raw_input"]["candidate_count"],
                "calendar_qubits": stages["calendar_full_phase"]["total_qubits"],
                "propagation_qubits": stages.get("propagation_full_phase", {}).get(
                    "total_qubits"
                ),
                "max_component_qubits": final["max_component_qubits"]
                if final
                else None,
                "cnot_calendar_full": stages["calendar_full_phase"]["cnot"],
                "cnot_components_full": stages.get("components_full_phase", {}).get(
                    "cnot"
                ),
                "cnot_components_auto": final["cnot"] if final else None,
                "mean_p_opt": float(
                    np.mean([p["optimal_probability"] for p in profiles])
                )
                if profiles
                else 0,
                "min_p_opt": min(
                    (p["optimal_probability"] for p in profiles), default=0
                ),
                "uniform_p_opt": profiles[0]["uniform_optimal_probability"]
                if profiles
                else 0,
                "direct_optimal_samples": sum(
                    r["direct"]["best"] is not None
                    and opt is not None
                    and r["direct"]["best"]["objective"] == opt
                    for r in group
                ),
                "reduced_optimal_samples": sum(
                    r["reduced"]["best"] is not None
                    and opt is not None
                    and r["reduced"]["best"]["objective"] == opt
                    for r in group
                ),
                "direct_mean_evaluations": float(
                    np.mean([r["direct"]["circuit_evaluations"] for r in group])
                ),
                "reduced_mean_evaluations": float(
                    np.mean([r["reduced"]["circuit_evaluations"] for r in group])
                ),
                "direct_mean_seconds": float(
                    np.mean([r["direct"]["runtime_seconds"] for r in group])
                ),
                "reduced_mean_seconds": float(
                    np.mean([r["reduced"]["runtime_seconds"] for r in group])
                ),
                "direct_shots": group[0]["direct"].get("shots", 0),
                "reduced_total_shots": group[0]["reduced"].get("total_shots", 0),
                "milp_seconds": instance["milp"]["runtime_seconds"],
                "feasible_assignments": instance["ablation"]["feasible_count"],
            }
        )
    with (source / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    cells = []
    for tasks in (3, 4, 5):
        for scenario in range(3):
            cell = [
                r for r in rows if r["tasks"] == tasks and r["scenario"] == scenario
            ]
            feasible = [r for r in cell if r["feasible"]]
            cells.append(
                {
                    "tasks": tasks,
                    "scenario": scenario,
                    "instances": len(cell),
                    "feasible": len(feasible),
                    "instance_mean_p_opt": float(
                        np.mean([r["mean_p_opt"] for r in feasible])
                    )
                    if feasible
                    else None,
                    "instance_mean_uniform_p_opt": float(
                        np.mean([r["uniform_p_opt"] for r in feasible])
                    )
                    if feasible
                    else None,
                    "instance_mean_delta_range": [
                        min(r["mean_p_opt"] - r["uniform_p_opt"] for r in feasible),
                        max(r["mean_p_opt"] - r["uniform_p_opt"] for r in feasible),
                    ]
                    if feasible
                    else None,
                }
            )
    feasible_profiles = [
        r["direct_metrics"]
        for r in runs
        if r["direct_metrics"] and r["direct_metrics"]["exact_objective"] is not None
    ]
    aggregate = {
        "instances": len(instances),
        "feasible_instances": sum(r["feasible"] for r in rows),
        "paired_seed_records": len(runs),
        "direct_statuses": dict(Counter(r["direct"]["status"] for r in runs)),
        "reduced_statuses": dict(Counter(r["reduced"]["status"] for r in runs)),
        "direct_quantum_executions": sum(
            r["direct"]["circuit_evaluations"] > 0 for r in runs
        ),
        "reduced_quantum_executions": sum(
            r["reduced"]["circuit_evaluations"] > 0 for r in runs
        ),
        "reduced_component_trainings": sum(
            len(r["reduced"]["components"]) for r in runs
        ),
        "feasible_training_seeds": len(feasible_profiles),
        "seeds_better_than_uniform": sum(
            p["optimal_probability"] > p["uniform_optimal_probability"] + 1e-12
            for p in feasible_profiles
        ),
        "seeds_worse_than_uniform": sum(
            p["optimal_probability"] < p["uniform_optimal_probability"] - 1e-12
            for p in feasible_profiles
        ),
        "phase_max_error": max(
            (
                r["direct_metrics"]["phase_max_error"]
                for r in runs
                if r["direct_metrics"]
            ),
            default=0,
        ),
        "cells": cells,
        "scope": (
            "synthetic instances; seed means within instance; "
            "unequal component budgets; no speedup claim"
        ),
    }
    write_json(source / "aggregate.json", aggregate)
    text = [
        "# 第二批逐实例结果",
        "",
        "36 个实例 × 10 个种子，两条执行路径；不可行和纯经典传播均保留。",
        "",
        "原始模型只有候选计数，不虚构含无效候选的线路。分解前后预算不同，不能用命中数宣称公平胜率优势。",
        "",
        "| 实例 | 最优值 | 原始→日历→传播→最大组件候选 | "
        "CNOT 日历完整→组件完整→精简 | 直接/化简最优命中（各10） |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        text.append(
            f"| {r['instance']} | {r['optimum']} | "
            f"{r['raw_candidates']}→{r['calendar_qubits']}→"
            f"{r['propagation_qubits']}→{r['max_component_qubits']} | "
            f"{r['cnot_calendar_full']}→{r['cnot_components_full']}→"
            f"{r['cnot_components_auto']} | "
            f"{r['direct_optimal_samples']}/{r['reduced_optimal_samples']} |"
        )
    (source / "SUMMARY.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    matplotlib.rcParams["svg.hashsalt"] = "lab-v2"
    matplotlib.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(len(rows))
    for key, label in (
        ("raw_candidates", "Raw candidates"),
        ("calendar_qubits", "After calendar"),
        ("max_component_qubits", "Largest remaining component"),
    ):
        axes[0].plot(
            x,
            [r[key] if r[key] is not None else np.nan for r in rows],
            ".-",
            label=label,
        )
    axes[0].set(
        xlabel="Registered instance index (0–35)",
        ylabel="Candidate / qubit count",
        title="Missing final points = propagation proved infeasible",
    )
    axes[0].legend(frameon=False, fontsize=8)
    feasible_rows = [r for r in rows if r["feasible"]]
    axes[1].scatter(
        [r["uniform_p_opt"] for r in feasible_rows],
        [r["mean_p_opt"] for r in feasible_rows],
        color="#167d9a",
    )
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(
        xlabel="Uniform optimum probability",
        ylabel="XY optimum probability (mean of 10 seeds)",
        title="Each dot = one feasible instance; training excluded",
    )
    fig.savefig(source / "overview.svg", metadata={"Date": None})
    plt.close(fig)
    print(json.dumps({k: v for k, v in aggregate.items() if k != "cells"}, indent=2))


def main() -> None:
    """Summarize a completed archive in place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    summarize(parser.parse_args().source)


if __name__ == "__main__":
    main()
