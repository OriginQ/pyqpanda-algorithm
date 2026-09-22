"""Run the 24-hour case5 day-ahead scheduling study locally.

The full classical reference is a SciPy MILP with hourly commitment,
start-up, reserve, network, and dispatch constraints.  The optional quantum
comparison uses one five-qubit QAOA commitment solve per hour and evaluates
the resulting trajectory with the same 24-hour dispatch routine.  No cloud
service or real quantum processor is contacted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqpanda_alg.UnitCommitment import (  # noqa: E402
    load_case5_day_ahead,
    solve_hourly_qaoa,
    solve_milp_uc,
)


def _reference_hourly(instance, reference) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hour, dispatch in enumerate(reference.schedule.dispatch):
        rows.append(
            {
                "hour": hour,
                "demand_mw": float(instance.demand_mw[hour]),
                "reserve_mw": float(instance.reserve_mw[hour]),
                "commitments": list(reference.commitments[hour]),
                "generation_mw": list(dispatch.generation_mw),
                "variable_cost": float(dispatch.variable_cost),
                "fixed_cost": float(dispatch.fixed_cost),
                "total_cost": float(dispatch.total_cost),
                "branch_flows_mw": list(dispatch.branch_flows_mw),
                "max_line_violation_mw": float(dispatch.max_line_violation_mw),
            }
        )
    return rows


def _quantum_hourly(instance, quantum) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hour, dispatch in enumerate(quantum.schedule.dispatch):
        item = quantum.hourly_results[hour]
        rows.append(
            {
                "hour": hour,
                "demand_mw": float(instance.demand_mw[hour]),
                "reserve_mw": float(instance.reserve_mw[hour]),
                "commitments": list(quantum.commitments[hour]),
                "generation_mw": list(dispatch.generation_mw),
                "variable_cost": float(dispatch.variable_cost),
                "fixed_cost": float(dispatch.fixed_cost),
                "total_cost": float(dispatch.total_cost),
                "max_line_violation_mw": float(dispatch.max_line_violation_mw),
                "qaoa_feasible_probability": float(item.qaoa.feasible_probability),
                "qaoa_best_cost": item.best_cost,
                "fallback_to_classical": bool(item.fallback_to_classical),
            }
        )
    return rows


def _write_figures(output: Path, instance, reference_rows, quantum_rows) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (output / "plot_note.txt").write_text(f"Plots not generated: {exc}\n", encoding="utf-8")
        return []

    hours = np.arange(1, 25)
    demand = np.asarray(instance.demand_mw, dtype=float)
    reserve_target = demand + np.asarray(instance.reserve_mw, dtype=float)
    reference_generation = np.asarray([row["generation_mw"] for row in reference_rows], dtype=float)
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2")

    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True, height_ratios=(1, 1.6))
    axes[0].plot(hours, demand, color="#1f2937", linewidth=2.2, marker="o", label="load")
    axes[0].plot(hours, reserve_target, color="#b23a48", linestyle="--", linewidth=1.6, label="load + reserve")
    axes[0].fill_between(hours, demand, reserve_target, color="#b23a48", alpha=0.12)
    axes[0].set_ylabel("MW")
    axes[0].set_title("PJM five-bus 24-hour day-ahead load profile")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    bottom = np.zeros(24, dtype=float)
    for index in range(reference_generation.shape[1]):
        axes[1].bar(
            hours,
            reference_generation[:, index],
            bottom=bottom,
            color=colors[index],
            width=0.78,
            label=f"G{index + 1}",
        )
        bottom += reference_generation[:, index]
    axes[1].plot(hours, demand, color="#111827", linewidth=1.8, marker=".", label="load")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Generation (MW)")
    axes[1].set_xticks(hours)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, ncol=6, loc="upper left")
    figure.tight_layout()
    dispatch_path = output / "case5_day_ahead_dispatch.png"
    figure.savefig(dispatch_path, dpi=180)
    plt.close(figure)

    reference_costs = np.asarray([row["total_cost"] for row in reference_rows], dtype=float)
    figure, axis = plt.subplots(figsize=(11.0, 4.4))
    width = 0.38
    axis.bar(hours - width / 2, reference_costs, width=width, color="#4c78a8", label="MILP reference")
    if quantum_rows is not None:
        quantum_costs = np.asarray([row["total_cost"] for row in quantum_rows], dtype=float)
        axis.bar(hours + width / 2, quantum_costs, width=width, color="#f58518", label="hourly QAOA")
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Hourly cost")
    axis.set_title("Day-ahead hourly dispatch cost comparison")
    axis.set_xticks(hours)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    cost_path = output / "case5_day_ahead_hourly_costs.png"
    figure.savefig(cost_path, dpi=180)
    plt.close(figure)
    return [str(dispatch_path), str(cost_path)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "day_ahead",
    )
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--maxiter", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--backend", choices=("statevector", "local_fake", "local_noisy"), default="statevector")
    parser.add_argument("--skip-quantum", action="store_true")
    parser.add_argument("--reserve-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.shots < 1 or args.maxiter < 1 or args.reserve_fraction < 0.0:
        raise SystemExit("shots and maxiter must be positive; reserve-fraction must be non-negative")

    instance = load_case5_day_ahead(reserve_fraction=args.reserve_fraction)
    reference = solve_milp_uc(instance, evaluate_method="linprog", enforce_network=True)
    if not reference.success or not reference.schedule.success:
        raise RuntimeError(f"24-hour MILP reference failed: {reference.message}")
    reference_rows = _reference_hourly(instance, reference)

    quantum = None
    quantum_rows = None
    if not args.skip_quantum:
        quantum = solve_hourly_qaoa(
            instance,
            backend=args.backend,
            shots=args.shots,
            maxiter=args.maxiter,
            seed=args.seed,
            top_k=8,
            evaluate_method="linprog",
            enforce_network=True,
        )
        quantum_rows = _quantum_hourly(instance, quantum)

    reference_cost = float(reference.schedule.total_cost)
    quantum_cost = None if quantum is None else float(quantum.schedule.total_cost)
    payload: dict[str, object] = {
        "schema_version": "case5_day_ahead_v1",
        "scope": {
            "hours": 24,
            "real_qpu_submitted": False,
            "remote_quantum_task_submitted": False,
            "execution_environment": "local CPUQVM and SciPy",
        },
        "profile": {
            "name": instance.name,
            "load_scales": list(instance.demand_scales),
            "demand_mw": list(instance.demand_mw),
            "reserve_mw": list(instance.reserve_mw),
            "reserve_fraction": args.reserve_fraction,
        },
        "classical_reference": {
            "solver": reference.solver,
            "total_cost": reference_cost,
            "commitments": [list(row) for row in reference.commitments],
            "hourly": reference_rows,
        },
        "hourly_qaoa": None
        if quantum is None
        else {
            "backend": quantum.backend,
            "decomposition": quantum.decomposition,
            "total_cost": quantum_cost,
            "gap_percent": 100.0 * (quantum_cost - reference_cost) / reference_cost,
            "commitments": [list(row) for row in quantum.commitments],
            "hourly": quantum_rows,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "case5_day_ahead.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figures = _write_figures(args.output, instance, reference_rows, quantum_rows)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "figures": figures,
                "reference_cost": reference_cost,
                "hourly_qaoa_cost": quantum_cost,
                "hourly_qaoa_gap_percent": None
                if quantum_cost is None
                else 100.0 * (quantum_cost - reference_cost) / reference_cost,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

