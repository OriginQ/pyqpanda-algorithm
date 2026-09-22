"""Archive all frozen-corpus runs and render instance-weighted evidence plots."""

import argparse
import json
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(source: Path, output: Path) -> None:
    """Preserve failures; aggregate optimization seeds within each instance first."""
    provenance = json.loads((source / "provenance.json").read_text())
    records = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in provenance["manifest"]["instances"]:
        if instance["split"] != provenance["split"]:
            continue
        for seed in provenance["manifest"]["optimization_seeds"]:
            name = f"{Path(instance['file']).stem}-seed{seed}.json"
            record = json.loads((source / name).read_text())
            records.append(record)
            groups[instance["file"]].append(record)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("summary.csv", "provenance.json", "execution.json"):
        shutil.copyfile(source / name, output / name)
    with (output / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
                + "\n"
            )
    aggregate = []
    for name, group in groups.items():
        first = group[0]
        p = [r["metrics"]["optimal_probability"] for r in group]
        baseline = first["metrics"]["uniform_optimal_probability"]
        aggregate.append(
            {
                "instance": name,
                "status": first["exact"]["status"],
                "qubits": first["model"]["qubits"],
                "candidate_conflict_density": len(first["model"]["conflicts"])
                / sum(
                    len(a) * len(b)
                    for a, b in combinations(first["model"]["domains"], 2)
                ),
                "mean_p_opt": float(np.mean(p)),
                "min_p_opt": min(p),
                "max_p_opt": max(p),
                "uniform_p_opt": baseline,
                "seeds_better_than_uniform": sum(
                    value > baseline + 1e-12 for value in p
                ),
                "cnot_full": first["resources"]["full"]["cnot"],
                "cnot_auto": first["resources"]["auto"]["cnot"],
                "depth_full": first["resources"]["full"]["depth"],
                "depth_auto": first["resources"]["auto"]["depth"],
                "mean_training_seconds": float(
                    np.mean([r["quantum"]["runtime_seconds"] for r in group])
                ),
                "milp_seconds": first["raw_milp"]["runtime_seconds"],
            }
        )
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    md = [
        "# 冻结语料逐实例汇总",
        "",
        "完整训练记录见 runs.jsonl，每行一次训练；包含全部不可行实例。",
        "",
        "| 实例 | 状态 | 平均/最差 p_opt | 均匀 p_opt | 胜出种子/10 | CNOT 原始→精简 |",
        "|---|---|---|---|---|---|",
    ]
    for row in aggregate:
        md.append(
            f"| {row['instance']} | {row['status']} | "
            f"{row['mean_p_opt']:.6f} / {row['min_p_opt']:.6f} | "
            f"{row['uniform_p_opt']:.6f} | {row['seeds_better_than_uniform']} | "
            f"{row['cnot_full']} → {row['cnot_auto']} |"
        )
    (output / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    matplotlib.rcParams["svg.hashsalt"] = "lab-enhancement-v1"
    matplotlib.rcParams["svg.fonttype"] = "none"
    labels = [f"q{r['qubits']}/d{i % 3}" for i, r in enumerate(aggregate)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.bar(
        x - 0.2,
        [r["cnot_full"] for r in aggregate],
        0.4,
        label="Full phase",
        color="#b47750",
    )
    ax.bar(
        x + 0.2,
        [r["cnot_auto"] for r in aggregate],
        0.4,
        label="Simplified XY",
        color="#167d9a",
    )
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Native CNOT operations",
        title="Same trained parameters; p=1; all 12 registered instances",
    )
    ax.legend(frameon=False)
    fig.savefig(output / "resources.svg", metadata={"Date": None})
    plt.close(fig)
    feasible_groups = [
        g for g in groups.values() if g[0]["exact"]["status"] == "optimal"
    ]
    budgets = provenance["manifest"]["shots"]
    quantum, uniform, observed_q, observed_u = [], [], [], []
    for i, _shots in enumerate(budgets):
        quantum.append(
            np.mean(
                [
                    np.mean([r["metrics"]["hit_curve"][i]["quantum"] for r in g])
                    for g in feasible_groups
                ]
            )
        )
        uniform.append(
            np.mean(
                [g[0]["metrics"]["hit_curve"][i]["uniform"] for g in feasible_groups]
            )
        )
        for target, key in (
            (observed_q, "quantum_objective"),
            (observed_u, "uniform_objective"),
        ):
            target.append(
                np.mean(
                    [
                        np.mean(
                            [
                                r["samples"][i][key] == r["metrics"]["exact_objective"]
                                for r in g
                            ]
                        )
                        for g in feasible_groups
                    ]
                )
            )
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(
        budgets, quantum, "o-", label="XY: analytic after training", color="#167d9a"
    )
    ax.plot(budgets, uniform, "o--", label="Uniform one-hot: analytic", color="#b47750")
    ax.scatter(
        budgets,
        observed_q,
        marker="x",
        label="XY: independent sample runs",
        color="#167d9a",
    )
    ax.scatter(
        budgets, observed_u, marker="+", label="Uniform: sample runs", color="#b47750"
    )
    ax.set(
        xscale="log",
        xticks=budgets,
        xticklabels=budgets,
        ylim=(0, 1.03),
        xlabel="Shots (training cost excluded)",
        ylabel="At least one optimum",
        title=(
            f"Equal weight over {len(feasible_groups)} feasible instances; "
            "10 seeds each"
        ),
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    fig.savefig(output / "hit-curves.svg", metadata={"Date": None})
    plt.close(fig)
    (output / "hit-curves.json").write_text(
        json.dumps(
            {
                "shots": budgets,
                "analytic_xy": quantum,
                "analytic_uniform": uniform,
                "observed_xy": observed_q,
                "observed_uniform": observed_u,
                "aggregation": "seed mean within instance, then equal instance mean",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Produce reviewable plots and a single lossless JSONL archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/lab-evidence"))
    args = parser.parse_args()
    summarize(args.source, args.output)


if __name__ == "__main__":
    main()
