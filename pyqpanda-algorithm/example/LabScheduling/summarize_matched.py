"""Summarize every matched-budget record, keeping classical-only cases separate."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from stress_benchmark import write_json

PATHS = ("direct", "reduced", "uniform_direct", "uniform_reduced")


def without_timing(value: Any) -> Any:
    """Exclude only timing fields; preserve parameters, counts and all failures."""
    if isinstance(value, dict):
        return {
            key: without_timing(item)
            for key, item in value.items()
            if not key.endswith("_seconds")
        }
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def verify_replay(original: Path, replay: Path) -> None:
    """Reject environment/source changes or any non-timing record discrepancy."""
    left_provenance = json.loads((original / "provenance.json").read_text())
    right_provenance = json.loads((replay / "provenance.json").read_text())
    if left_provenance != right_provenance:
        raise ValueError("replay provenance changed")
    left = [
        json.loads(line) for line in (original / "runs.jsonl").read_text().splitlines()
    ]
    right = [
        json.loads(line) for line in (replay / "runs.jsonl").read_text().splitlines()
    ]
    if without_timing(left) != without_timing(right):
        raise ValueError("replay records differ beyond timing fields")
    write_json(
        original / "replay.json",
        {
            "records_compared": len(left),
            "identical_except_timing": True,
            "excluded_fields": "only keys ending in _seconds, recursively",
            "replay_execution": json.loads((replay / "execution.json").read_text()),
        },
    )


def summarize(folder: Path) -> None:
    """Write instance-first summaries and conditional total-shot hit curves."""
    records = [
        json.loads(line) for line in (folder / "runs.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((folder / "provenance.json").read_text())["manifest"]
    expected = {
        (item["file"], seed)
        for item in manifest["instances"]
        for seed in manifest["optimization_seeds"]
    }
    if (
        len(records) != len(expected)
        or {(r["instance"], r["seed"]) for r in records} != expected
    ):
        raise ValueError("missing, duplicate or unexpected instance/seed record")
    rows = []
    quantum_groups = []
    for item in manifest["instances"]:
        group = [r for r in records if r["instance"] == item["file"]]
        first = group[0]
        category = (
            "infeasible"
            if first["exact_objective"] is None
            else "classical_only"
            if first["paths"]["reduced"]["status"] == "classical_propagation"
            else "quantum_required"
        )
        if category == "quantum_required":
            quantum_groups.append(group)
        row: dict[str, Any] = {
            "instance": item["file"],
            "category": category,
            "optimum": first["exact_objective"],
        }
        for path in PATHS:
            row[f"{path}_optimal_seeds"] = sum(
                r["absolute_gaps"][path] == 0 for r in group
            )
            row[f"{path}_feasible_seeds"] = sum(
                bool(r["paths"][path]["best"] and r["paths"][path]["best"]["feasible"])
                for r in group
            )
            for field in ("circuit_evaluations", "shots", "runtime_seconds"):
                row[f"{path}_mean_{field}"] = mean(
                    r["paths"][path].get(field, 0) for r in group
                )
        rows.append(row)
    with (folder / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    curves = []
    for index, shots in enumerate((8, 16, 32, 64, 128, 512)):
        row = {"total_shots": shots}
        for path in PATHS:
            row[path] = mean(
                mean(r["conditional_hit_curve"][index][path] for r in group)
                for group in quantum_groups
            )
        deltas = [
            mean(
                r["conditional_hit_curve"][index]["reduced"]
                - r["conditional_hit_curve"][index]["uniform_reduced"]
                for r in group
            )
            for group in quantum_groups
        ]
        row["reduced_vs_uniform_instance_delta_min"] = min(deltas)
        row["reduced_vs_uniform_instance_delta_max"] = max(deltas)
        row["instances_reduced_better_than_uniform"] = sum(d > 1e-12 for d in deltas)
        row["instances_reduced_worse_than_uniform"] = sum(d < -1e-12 for d in deltas)
        curves.append(row)
    aggregate = {
        "paired_records": len(records),
        "instance_categories": dict(Counter(row["category"] for row in rows)),
        "statuses": {
            path: dict(Counter(r["paths"][path]["status"] for r in records))
            for path in PATHS
        },
        "quantum_required_optimal_seeds": {
            path: sum(
                r["absolute_gaps"][path] == 0 for group in quantum_groups for r in group
            )
            for path in PATHS
        },
        "all_feasible_optimal_seeds": {
            path: sum(
                r["absolute_gaps"][path] == 0
                for r in records
                if r["exact_objective"] is not None
            )
            for path in PATHS
        },
        "quantum_required_mean_evaluations": {
            path: mean(
                row[f"{path}_mean_circuit_evaluations"]
                for row in rows
                if row["category"] == "quantum_required"
            )
            for path in PATHS
        },
        "conditional_hit_curves_quantum_required": curves,
        "interpretation": (
            "Equal caps, not equal actual work. Instance-first means; curves "
            "condition on trained states. Known synthetic corpus, not a new holdout."
        ),
    }
    write_json(folder / "aggregate.json", aggregate)
    lines = [
        "# 同总预算对照结果",
        "",
        "全部 36 个实例、每实例 10 种子；Q=直接量子，R=分量量子，"
        "U=直接均匀随机，S=分量均匀随机。",
        "",
        "下表为实际 512 总 shots 下采到精确最优的种子数；不可行行无最优值，记 0。",
        "",
        "| 实例 | 类别 | 最优值 | Q | R | U | S |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = [
            row["instance"],
            row["category"],
            row["optimum"],
            *[row[f"{p}_optimal_seeds"] for p in PATHS],
        ]
        lines.append(
            "| " + " | ".join(str(v) if v is not None else "—" for v in cells) + " |"
        )
    lines += [
        "",
        "## 同总 shots 的条件命中概率",
        "",
        "仅 22 个仍需量子的可行实例，先平均每实例的十种子，再等权平均实例。",
        "这是训练后理论曲线，低 shots 未重新训练或实际抽样，不包括训练成本。",
        "",
        "| 总 shots | Q | R | U | S | R>S 实例 | R<S 实例 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curves:
        cells = [
            str(row["total_shots"]),
            *[f"{row[p]:.6f}" for p in PATHS],
            str(row["instances_reduced_better_than_uniform"]),
            str(row["instances_reduced_worse_than_uniform"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "R/S 包含精确化简与组合收益；R 与 S 才是相同结构上的量子/随机对照。",
        "不能将 R 相对 U 的全部改善归于量子，"
        "也不能用相等预算上限声称相等实际计算成本。",
        "实际评价数、采样数、串行运行时间见 summary.csv；"
        "完整失败与 counts 见 runs.jsonl。",
        "",
    ]
    (folder / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Summarize a complete frozen 360-record evaluation directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--replay", type=Path, help="also verify an independent rerun")
    args = parser.parse_args()
    summarize(args.folder)
    if args.replay:
        verify_replay(args.folder, args.replay)


if __name__ == "__main__":
    main()
