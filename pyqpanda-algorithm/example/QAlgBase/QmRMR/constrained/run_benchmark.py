"""Run the reproducible QmRMR feature-selection benchmark.

The benchmark evaluates a fixed-cardinality problem with an exact classical
oracle, a greedy selector, an exchange-ansatz baseline, and constrained QAOA.
All quantum results use the pyQPanda3 CPU state-vector simulator, so the
reported probabilities can be inspected without sampling noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqpanda_alg.QmRMR.constrained import (  # noqa: E402
    ConstrainedQAOAQmRMR,
    ExchangeAnsatzSelector,
    QmRMRProblem,
    greedy_forward_selection,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_instance(
    seed: int,
    n_features: int,
    select_num: int,
) -> tuple[QmRMRProblem, np.ndarray]:
    """Generate one deterministic symmetric QmRMR instance."""

    rng = np.random.RandomState(int(seed))
    relevance = rng.random(int(n_features))
    quadratic = rng.random((int(n_features), int(n_features)))
    quadratic = 0.5 * (quadratic + quadratic.T)
    problem = QmRMRProblem.from_mrmr(
        quadratic,
        relevance,
        int(select_num),
        convention="canonical_mrmr",
    )
    exchange_initial = (
        rng.random((int(n_features) // 2) * int(n_features)) * np.pi
    )
    return problem, exchange_initial


def _state_key(state: tuple[int, ...] | list[int]) -> str:
    return "".join(str(int(bit)) for bit in state)


def _summary_row(
    summary: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Convert a solver summary to the common CSV schema."""

    metadata = summary.get("metadata", {})
    return {
        "instance": spec["name"],
        "seed": int(spec["seed"]),
        "n_features": int(spec["n_features"]),
        "select_num": int(spec["select_num"]),
        "method": summary["method"],
        "selected_state": summary["selected_state"],
        "selected_value": summary["selected_value"],
        "expected_value": summary.get("expected_value", ""),
        "expected_gap": summary.get("expected_gap", ""),
        "optimization_value": summary.get("optimization_value", ""),
        "optimization_objective": summary.get("optimization_objective", ""),
        "cvar_alpha": summary.get("cvar_alpha", ""),
        "selected_probability": summary["selected_probability"],
        "feasible_probability": summary["feasible_probability"],
        "optimum_probability": summary["optimum_probability"],
        "optimum_value": summary["optimum_value"],
        "optimum_gap": summary["optimum_gap"],
        "evaluations": summary["evaluations"],
        "parameter_count": summary["parameter_count"],
        "depth": summary["depth"],
        "optimizer": metadata.get("optimizer", ""),
        "layers": metadata.get("layers", ""),
        "restarts": metadata.get("restart_count", ""),
        "warm_start_used": metadata.get("warm_start_used", ""),
    }


def _classical_row(
    method: str,
    state: tuple[int, ...],
    value: float,
    gap: float,
    evaluations: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Put a non-variational reference in the common CSV schema."""

    return {
        "instance": spec["name"],
        "seed": int(spec["seed"]),
        "n_features": int(spec["n_features"]),
        "select_num": int(spec["select_num"]),
        "method": method,
        "selected_state": _state_key(state),
        "selected_value": float(value),
        "expected_value": "",
        "expected_gap": "",
        "optimization_value": "",
        "optimization_objective": "",
        "cvar_alpha": "",
        "selected_probability": "",
        "feasible_probability": "",
        "optimum_probability": "",
        "optimum_value": "",
        "optimum_gap": float(gap),
        "evaluations": int(evaluations),
        "parameter_count": 0,
        "depth": 0,
        "optimizer": "",
        "layers": "",
        "restarts": "",
        "warm_start_used": "",
    }


def _run_case(
    spec: dict[str, Any],
    *,
    optimizer: str,
    maxiter: int,
    max_layers: int,
    initializer: str,
    initializer_strength: float,
    ring: bool,
    restarts: int,
    warm_start: bool,
    initial_scale: float,
    optimization_objective: str,
    cvar_alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]], dict[str, float]]:
    problem, exchange_initial = _build_instance(
        int(spec["seed"]),
        int(spec["n_features"]),
        int(spec["select_num"]),
    )
    exact = problem.exact_solution()
    greedy = greedy_forward_selection(problem)
    rows = [
        _classical_row(
            "exact_enumeration",
            exact.optimal_states[0],
            exact.optimal_value,
            0.0,
            len(exact.values),
            spec,
        ),
        _classical_row(
            "greedy_forward",
            greedy.state,
            greedy.value,
            greedy.value - exact.optimal_value,
            greedy.evaluations,
            spec,
        ),
    ]
    histories: dict[str, list[float]] = {}
    methods: list[dict[str, Any]] = []

    exchange = ExchangeAnsatzSelector(problem, bit_order="qubit").optimize(
        initial_parameters=exchange_initial,
        optimizer=optimizer,  # type: ignore[arg-type]
        maxiter=maxiter,
        seed=int(spec["seed"]) + 1,
        restarts=restarts,
    )
    exchange_summary = exchange.summary()
    exchange_summary["metadata"]["run_seed"] = int(spec["seed"]) + 1
    methods.append(exchange_summary)
    rows.append(_summary_row(exchange_summary, spec))
    histories["exchange baseline"] = exchange.history

    if warm_start:
        stages = ConstrainedQAOAQmRMR(
            problem,
            layers=max_layers,
            initializer=initializer,
            initializer_strength=initializer_strength,
            ring=ring,
            objective_mode=optimization_objective,
            cvar_alpha=cvar_alpha,
        ).optimize_progressive(
            optimizer=optimizer,  # type: ignore[arg-type]
            maxiter=maxiter,
            seed=int(spec["seed"]) + 100,
            restarts=restarts,
            initial_scale=initial_scale,
        )
    else:
        stages = []
        for layer in range(1, max_layers + 1):
            stages.append(
                ConstrainedQAOAQmRMR(
                    problem,
                    layers=layer,
                    initializer=initializer,
                    initializer_strength=initializer_strength,
                    ring=ring,
                    objective_mode=optimization_objective,
                    cvar_alpha=cvar_alpha,
                ).optimize(
                    optimizer=optimizer,  # type: ignore[arg-type]
                    maxiter=maxiter,
                    seed=int(spec["seed"]) + 100 + layer,
                    restarts=restarts,
                    initial_scale=initial_scale,
                )
            )
            metadata = dict(stages[-1].metadata)
            metadata["warm_start_used"] = False
            stages[-1].metadata = metadata

    for layer, result in enumerate(stages, start=1):
        summary = result.summary()
        summary["metadata"]["run_seed"] = int(spec["seed"]) + 100 + layer
        methods.append(summary)
        rows.append(_summary_row(summary, spec))
        suffix = "warm-start" if warm_start else "independent"
        histories[f"constrained QAOA p={layer} ({suffix})"] = result.history

    best_qaoa = min(
        stages,
        key=lambda result: (
            float(result.metadata.get("optimization_value", result.expected_value)),
            result.selected_value,
            -result.optimum_probability,
        ),
    )
    case_payload: dict[str, Any] = {
        "name": spec["name"],
        "seed": int(spec["seed"]),
        "n_features": int(spec["n_features"]),
        "select_num": int(spec["select_num"]),
        "problem": {
            "quadratic": problem.quadratic.tolist(),
            "linear": problem.linear.tolist(),
            "objective_convention": problem.objective_convention,
            "objective_label": problem.objective_label,
        },
        "exact": {
            "optimal_value": float(exact.optimal_value),
            "optimal_states": [_state_key(state) for state in exact.optimal_states],
            "feasible_state_count": len(exact.values),
            "values": exact.values,
        },
        "classical_greedy": {
            "selected_state": _state_key(greedy.state),
            "selected_value": float(greedy.value),
            "optimum_gap": float(greedy.value - exact.optimal_value),
            "evaluations": int(greedy.evaluations),
        },
        "methods": methods,
        "best_qaoa_method": best_qaoa.method,
        "best_qaoa_probabilities": {
            _state_key(tuple(key[::-1])): float(value)
            for key, value in best_qaoa.probabilities.items()
            if key[::-1].count("1") == int(spec["select_num"])
        },
    }
    best_info = {
        "method": best_qaoa.method,
        "selected_value": float(best_qaoa.selected_value),
        "expected_value": float(best_qaoa.expected_value),
        "optimum_gap": float(best_qaoa.optimum_gap),
        "optimum_probability": float(best_qaoa.optimum_probability),
    }
    return case_payload, rows, histories, best_info


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate comparable variational rows across instances."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["method"] in {"exact_enumeration", "greedy_forward"}:
            continue
        method = str(row["method"])
        layers = row.get("layers", "")
        key = f"{method}_p{layers}" if method == "constrained_qaoa_xy" else method
        grouped.setdefault(key, []).append(row)

    aggregate: dict[str, dict[str, Any]] = {}
    metric_names = (
        "selected_value",
        "expected_value",
        "expected_gap",
        "optimum_gap",
        "selected_probability",
        "feasible_probability",
        "optimum_probability",
    )
    for method, method_rows in grouped.items():
        metrics: dict[str, Any] = {"count": len(method_rows)}
        for name in metric_names:
            values = [
                float(row[name])
                for row in method_rows
                if row.get(name) not in (None, "")
            ]
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            metrics[name] = {
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "std": float(np.std(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            }
        gaps = np.asarray([float(row["optimum_gap"]) for row in method_rows])
        metrics["success_rate"] = float(np.mean(gaps <= 1e-8))
        feasible = [
            float(row["feasible_probability"])
            for row in method_rows
            if row.get("feasible_probability") not in (None, "")
        ]
        metrics["feasibility_min"] = float(min(feasible)) if feasible else 0.0
        aggregate[method] = metrics
    return aggregate


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _public_command(root: Path) -> list[str]:
    """Return a portable command description for archived metadata."""

    arguments: list[str] = []
    root_resolved = root.resolve()
    for argument in sys.argv[1:]:
        candidate = Path(argument)
        if candidate.is_absolute():
            try:
                argument = str(candidate.resolve().relative_to(root_resolved))
            except ValueError:
                argument = candidate.name
        arguments.append(argument)
    return ["python", "scripts/run_benchmark.py", *arguments]


def _write_outputs(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    histories: dict[str, list[float]],
    *,
    root: Path,
    run_id: str | None,
    make_plots: bool,
) -> list[Path]:
    result_dirs = [root / "results" / "latest"]
    figure_dirs = [root / "figures" / "latest"]
    if run_id:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        result_dirs.append(root / "results" / "runs" / safe_id)
        figure_dirs.append(root / "figures" / "runs" / safe_id)

    written: list[Path] = []
    for directory in result_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / "benchmark_summary.json"
        summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        csv_path = directory / "benchmark_metrics.csv"
        fieldnames = list(rows[0]) if rows else []
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        metadata_path = directory / "RUN_METADATA.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "run_id": run_id or "latest",
                    "project_version": "0.2.0",
                    "command": _public_command(root),
                    "config": payload.get("config", {}),
                    "instances": [
                        {
                            "name": case.get("name"),
                            "seed": case.get("seed"),
                            "n_features": case.get("n_features"),
                            "select_num": case.get("select_num"),
                        }
                        for case in payload.get("instances", [])
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        written.extend([summary_path, csv_path, metadata_path])

    if not make_plots:
        return written

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary_name = payload["primary_instance"]["name"]
    primary_rows = [row for row in rows if row["instance"] == primary_name]
    for directory in figure_dirs:
        directory.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7.5, 4.4))
        for label, values in histories.items():
            plt.plot(np.arange(1, len(values) + 1), values, label=label)
        plt.axhline(
            float(payload["primary_instance"]["exact"]["optimal_value"]),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="exact optimum",
        )
        plt.xlabel("Objective evaluations")
        plt.ylabel(r"Expected objective $\mathbb{E}[f(x)]$")
        plt.title("Reference instance: objective convergence")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = directory / "objective_convergence.png"
        plt.savefig(path, dpi=180)
        plt.close()
        written.append(path)

        gap_rows = [
            row
            for row in primary_rows
            if row["method"] not in {"exact_enumeration", "greedy_forward"}
        ]
        if gap_rows:
            labels = []
            for row in gap_rows:
                label = str(row["method"]).replace("exchange_baseline", "Exchange")
                if row.get("layers") not in (None, ""):
                    label = f"{label} p={row['layers']}"
                labels.append(label)
            gaps = [float(row["optimum_gap"]) for row in gap_rows]
            plt.figure(figsize=(8.5, 4.5))
            plt.bar(labels, gaps, color="#64748b")
            plt.ylabel("Selected-state gap to exact optimum")
            plt.title("Reference instance: selected-state gap")
            plt.xticks(rotation=25, ha="right")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            path = directory / "canonical_gap_comparison.png"
            plt.savefig(path, dpi=180)
            plt.close()
            written.append(path)

        probability_data = payload["primary_instance"].get("best_qaoa_probabilities", {})
        if probability_data:
            top = sorted(
                probability_data.items(), key=lambda item: item[1], reverse=True
            )[:10]
            plt.figure(figsize=(7.5, 4.4))
            plt.bar([item[0] for item in top], [item[1] for item in top], color="#3b82f6")
            plt.xlabel("Feature-index state")
            plt.ylabel("Probability")
            plt.title("Reference instance: most probable feasible states")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            path = directory / "feasible_state_probabilities.png"
            plt.savefig(path, dpi=180)
            plt.close()
            written.append(path)

        aggregate_items = [
            (key, value)
            for key, value in payload.get("aggregate", {}).items()
            if "optimum_gap" in value
        ]
        if aggregate_items:
            labels = []
            for key, _ in aggregate_items:
                label = key.replace("exchange_baseline", "Exchange")
                label = label.replace("constrained_qaoa_xy_", "QAOA ")
                labels.append(label)
            means = [value["optimum_gap"]["mean"] for _, value in aggregate_items]
            stds = [value["optimum_gap"]["std"] for _, value in aggregate_items]
            plt.figure(figsize=(8.5, 4.5))
            plt.bar(labels, means, yerr=stds, capsize=4, color="#8b5cf6")
            plt.ylabel("Mean selected-state gap")
            plt.title("Multi-instance gap summary")
            plt.xticks(rotation=25, ha="right")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            path = directory / "multi_instance_gap_summary.png"
            plt.savefig(path, dpi=180)
            plt.close()
            written.append(path)

            success = [value["success_rate"] for _, value in aggregate_items]
            plt.figure(figsize=(8.5, 4.5))
            plt.bar(labels, success, color="#10b981")
            plt.ylim(0.0, 1.05)
            plt.ylabel("Exact-state success rate")
            plt.title("Multi-instance exact-state success rate")
            plt.xticks(rotation=25, ha="right")
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            path = directory / "multi_instance_success_rate.png"
            plt.savefig(path, dpi=180)
            plt.close()
            written.append(path)
    return written


def _instance_specs(config: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    configured = config.get("instances")
    if configured is None:
        configured = [
            {
                "name": "reference_n6_k3_seed42",
                "seed": config["seed"],
                "n_features": config["n_features"],
                "select_num": config["select_num"],
            }
        ]
    specs: list[dict[str, Any]] = []
    for index, item in enumerate(configured):
        spec = dict(item)
        spec.setdefault(
            "name",
            f"case_{index + 1}_n{spec['n_features']}_k{spec['select_num']}_seed{spec['seed']}",
        )
        for key in ("seed", "n_features", "select_num"):
            if key not in spec:
                raise ValueError(f"instance {index} is missing {key}")
            spec[key] = int(spec[key])
        if spec["n_features"] < 1 or not 0 <= spec["select_num"] <= spec["n_features"]:
            raise ValueError(f"invalid instance specification: {spec}")
        specs.append(spec)
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("--instances must be positive")
        specs = specs[: int(limit)]
    if not specs:
        raise ValueError("at least one benchmark instance is required")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "benchmark.json")
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--optimizer", choices=("slsqp", "spsa"), default=None)
    parser.add_argument(
        "--initializer",
        choices=("auto", "amplitude", "relevance", "biased"),
        default=None,
    )
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--initial-scale", type=float, default=None)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--initializer-strength", type=float, default=None)
    parser.add_argument(
        "--optimization-objective",
        choices=("expectation", "cvar"),
        default=None,
    )
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-id", default=None, help="archive under results/runs/<run-id>")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    specs = _instance_specs(config, args.instances)
    maxiter = int(args.maxiter if args.maxiter is not None else config.get("maxiter", 40))
    max_layers = int(args.layers if args.layers is not None else config.get("layers", 3))
    optimizer = str(args.optimizer if args.optimizer is not None else config.get("optimizer", "slsqp"))
    initializer = str(
        args.initializer if args.initializer is not None else config.get("initializer", "amplitude")
    )
    initializer_strength = float(
        args.initializer_strength
        if args.initializer_strength is not None
        else config.get("initializer_strength", 1.0)
    )
    ring = bool(args.ring if args.ring is not None else config.get("ring", False))
    restarts = int(args.restarts if args.restarts is not None else config.get("restarts", 1))
    initial_scale = float(
        args.initial_scale
        if args.initial_scale is not None
        else config.get("initial_scale", 0.5)
    )
    optimization_objective = str(
        args.optimization_objective
        if args.optimization_objective is not None
        else config.get("optimization_objective", "expectation")
    )
    cvar_alpha = float(
        args.cvar_alpha
        if args.cvar_alpha is not None
        else config.get("cvar_alpha", 0.25)
    )
    warm_start = bool(
        args.warm_start if args.warm_start is not None else config.get("warm_start", True)
    )
    if (
        maxiter < 1
        or max_layers < 1
        or restarts < 1
        or initial_scale <= 0.0
        or initializer_strength < 0.0
        or optimization_objective not in {"expectation", "cvar"}
        or not 0.0 < cvar_alpha <= 1.0
    ):
        raise ValueError(
            "maxiter, layers, restarts, and initial_scale must be positive; "
            "initializer_strength must be non-negative; cvar_alpha must "
            "satisfy 0 < cvar_alpha <= 1"
        )

    all_rows: list[dict[str, Any]] = []
    all_histories: dict[str, list[float]] = {}
    cases: list[dict[str, Any]] = []
    primary_best: dict[str, float] = {}
    for index, spec in enumerate(specs):
        case, rows, histories, best_info = _run_case(
            spec,
            optimizer=optimizer,
            maxiter=maxiter,
            max_layers=max_layers,
            initializer=initializer,
            initializer_strength=initializer_strength,
            ring=ring,
            restarts=restarts,
            warm_start=warm_start,
            initial_scale=initial_scale,
            optimization_objective=optimization_objective,
            cvar_alpha=cvar_alpha,
        )
        cases.append(case)
        all_rows.extend(rows)
        if index == 0:
            all_histories = histories
            primary_best = best_info

    primary = cases[0]
    aggregate = _aggregate_rows(all_rows)
    payload: dict[str, Any] = {
        "project": "Constraint-preserving QAOA for QmRMR feature selection",
        "objective": "f(x) = x.T @ Q @ x - relevance.T @ x",
        "constraint": "each instance uses sum(x) = select_num",
        "backend": "pyqpanda3 CPUQVM state-vector probabilities",
        "config": {
            "optimizer": optimizer,
            "maxiter": maxiter,
            "layers": max_layers,
            "initializer": initializer,
            "initializer_strength": initializer_strength,
            "ring": ring,
            "restarts": restarts,
            "initial_scale": initial_scale,
            "optimization_objective": optimization_objective,
            "cvar_alpha": cvar_alpha,
            "warm_start": warm_start,
            "instance_count": len(specs),
        },
        "comparison_conventions": {
            "state_order": "feature-index order; pyQPanda3 keys are reversed before evaluation",
            "objective": "selected, expected, and gap metrics use the canonical QmRMR objective",
            "restart_selection": "minimum variational objective, then selected-state value, then maximum optimum probability",
        },
        "primary_instance": primary,
        "instances": cases,
        "aggregate": aggregate,
        "primary_best_qaoa": primary_best,
    }
    written = _write_outputs(
        payload,
        all_rows,
        all_histories,
        root=PROJECT_DIR,
        run_id=args.run_id,
        make_plots=not args.no_plot,
    )

    print(
        json.dumps(
            {
                "instance_count": len(cases),
                "primary_exact": primary["exact"],
                "primary_best_qaoa": primary_best,
                "aggregate": aggregate,
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )
    print("Wrote:")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

