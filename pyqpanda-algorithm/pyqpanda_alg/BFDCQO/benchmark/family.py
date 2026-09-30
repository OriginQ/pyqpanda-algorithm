# Licensed under the Apache License, Version 2.0.
"""BF-DCQO-family benchmark: Basic vs CVaR vs Adaptive shots.

The benchmark deliberately uses one QUBO application (weighted MaxCut) and one
native HUBO application (frustrated MAX-3-SAT), under identical total shot
budgets.  This isolates the feedback/sampling changes from the application
with QUBO and native-HUBO application benchmarks.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import numpy as np

from .instances import default_maxcut_instances
from .max3sat import default_max3sat_instances
from ..solvers import BFDCQO, CVaRBFDCQO, AdaptiveShotBFDCQO


def _problems():
    maxcut = next(p for p in default_maxcut_instances() if p.name == "weighted6")
    max3sat = next(p for p in default_max3sat_instances() if p.name == "frustrated5_unsat")
    return [
        {
            "name": "weighted6_maxcut",
            "kind": "QUBO",
            "app": maxcut,
            "problem": maxcut.to_ising(),
            "optimum": maxcut.exact_maxcut()[0],
        },
        {
            "name": "frustrated5_max3sat",
            "kind": "HUBO-3",
            "app": max3sat,
            "problem": max3sat.to_native_ising(),
            "optimum": max3sat.exact_solution()[0],
        },
    ]


def _metrics(spec, result):
    last = result.history[-1]
    if spec["name"] == "weighted6_maxcut":
        best = -float(result.best_energy)
        expected = -float(last.expected_energy)
    else:
        app = spec["app"]
        best = float(app.total_weight - result.best_energy)
        expected = float(app.total_weight - last.expected_energy)
    opt = float(spec["optimum"])
    return {
        "best_objective": best,
        "expected_objective": expected,
        "best_approx_ratio": best / opt if opt > 0 else 1.0,
        "expected_approx_ratio": expected / opt if opt > 0 else 1.0,
        "p_opt": float(last.ground_state_probability),
        "success": bool(np.isclose(best, opt, atol=1e-9)),
    }


def _backend_label(backend):
    if backend == "numpy":
        return "numpy_reference"
    if backend == "pyqpanda":
        return "pyqpanda3.CPUQVM"
    return str(backend)


def _make_solver(variant, problem, budget, seed, cvar_alpha, backend):
    iterations = 6
    if budget % iterations != 0:
        raise ValueError("fixed-shot budgets must be divisible by six")
    fixed_shots = budget // iterations
    common = dict(problem=problem, n_trotter=3, dt=0.1, backend=backend, seed=seed)
    if variant == "basic":
        return BFDCQO(**common, iterations=iterations, shots=fixed_shots)
    if variant == "cvar":
        return CVaRBFDCQO(**common, iterations=iterations, shots=fixed_shots, cvar_alpha=cvar_alpha)
    if variant == "adaptive":
        # The minimum is intentionally lower than the uniform allocation. The
        # final iteration consumes the exact remainder, so total shots == budget.
        min_shots = max(32, budget // 24)
        return AdaptiveShotBFDCQO(
            **common,
            total_shot_budget=budget,
            max_iterations=iterations,
            min_shots=min_shots,
            max_shots=2 * fixed_shots,
            initial_shots=fixed_shots,
            early_stop=False,
        )
    raise ValueError(variant)


def run_family_benchmark(
    *,
    budgets=(1536, 3072, 6144),
    seeds=tuple(range(10)),
    cvar_alpha=0.10,
    backend="numpy",
    output_dir=None,
):
    """Run matched-shot Basic/CVaR/Adaptive benchmarks on QUBO and HUBO cases."""
    rows = []
    for spec in _problems():
        for budget in budgets:
            budget = int(budget)
            for seed in seeds:
                for variant in ("basic", "cvar", "adaptive"):
                    solver = _make_solver(
                        variant, spec["problem"], budget, int(seed),
                        cvar_alpha, backend,
                    )
                    t0 = time.perf_counter()
                    result = solver.solve()
                    elapsed = time.perf_counter() - t0
                    if result.total_shots != budget:
                        raise RuntimeError(
                            f"shot budget mismatch: {spec['name']} {variant} "
                            f"used {result.total_shots}, expected {budget}"
                        )
                    met = _metrics(spec, result)
                    rows.append({
                        "problem": spec["name"],
                        "problem_kind": spec["kind"],
                        "degree": int(getattr(spec["problem"], "degree", 2)),
                        "variant": variant,
                        "backend": _backend_label(backend),
                        "seed": int(seed),
                        "shot_budget": budget,
                        "total_shots": int(result.total_shots),
                        "circuit_evaluations": int(result.circuit_evaluations),
                        "shot_history": [int(x) for x in result.shot_history],
                        "uncertainty_history": [float(x) for x in result.uncertainty_history],
                        "final_uncertainty": float(result.history[-1].uncertainty),
                        "final_bias_delta": float(result.history[-1].bias_delta),
                        "feedback_alpha": float(result.history[-1].feedback_alpha),
                        "effective_feedback_shots_last": float(result.history[-1].effective_feedback_shots),
                        "optimum": float(spec["optimum"]),
                        **met,
                        "walltime_s": float(elapsed),
                    })

    summary = []
    for problem in sorted({r["problem"] for r in rows}):
        for budget in sorted({r["shot_budget"] for r in rows}):
            for variant in ("basic", "cvar", "adaptive"):
                rr = [r for r in rows if r["problem"] == problem and r["shot_budget"] == budget and r["variant"] == variant]
                summary.append({
                    "problem": problem,
                    "problem_kind": rr[0]["problem_kind"],
                    "variant": variant,
                    "shot_budget": budget,
                    "runs": len(rr),
                    "mean_p_opt": float(np.mean([r["p_opt"] for r in rr])),
                    "std_p_opt": float(np.std([r["p_opt"] for r in rr])),
                    "mean_expected_approx_ratio": float(np.mean([r["expected_approx_ratio"] for r in rr])),
                    "std_expected_approx_ratio": float(np.std([r["expected_approx_ratio"] for r in rr])),
                    "mean_best_approx_ratio": float(np.mean([r["best_approx_ratio"] for r in rr])),
                    "success_rate": float(np.mean([r["success"] for r in rr])),
                    "mean_final_uncertainty": float(np.mean([r["final_uncertainty"] for r in rr])),
                    "mean_circuit_evaluations": float(np.mean([r["circuit_evaluations"] for r in rr])),
                    "mean_walltime_s": float(np.mean([r["walltime_s"] for r in rr])),
                })

    payload = {
        "config": {
            "budgets": [int(x) for x in budgets],
            "seeds": [int(x) for x in seeds],
            "cvar_alpha": float(cvar_alpha),
            "fixed_iterations": 6,
            "n_trotter": 3,
            "dt": 0.1,
            "backend": _backend_label(backend),
            "note": "alpha=0.10 is used for this low-shot regression; enhanced BF-DCQO literature used alpha=0.01 with much larger shot counts.",
        },
        "rows": rows,
        "summary": summary,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "family_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if rows:
            with (out / "family_raw.csv").open("w", newline="", encoding="utf-8") as f:
                # stringify list columns for CSV portability
                fieldnames = list(rows[0].keys())
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    d = dict(r)
                    d["shot_history"] = json.dumps(d["shot_history"])
                    d["uncertainty_history"] = json.dumps(d["uncertainty_history"])
                    w.writerow(d)
        if summary:
            with (out / "family_summary.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                w.writeheader()
                w.writerows(summary)
    return payload


def run_adaptive_efficiency_benchmark(
    *,
    budget_caps=(1536, 3072, 6144),
    seeds=tuple(range(20)),
    cvar_alpha=0.10,
    backend="numpy",
    output_dir=None,
):
    """Compare fixed-shot CVaR with adaptive-CVaR early stopping.

    This is a *budget-cap* experiment rather than strict matched-spend: the fixed
    CVaR solver uses the full cap, while adaptive-CVaR may stop early once both
    uncertainty and bias change are small.  It therefore quantifies shot savings
    at comparable final solution quality.
    """
    rows = []
    for spec in _problems():
        for budget in budget_caps:
            budget = int(budget)
            if budget % 6 != 0:
                raise ValueError("efficiency budgets must be divisible by six")
            fixed = budget // 6
            for seed in seeds:
                solvers = {
                    "cvar_fixed": CVaRBFDCQO(
                        spec["problem"], n_trotter=3, dt=0.1,
                        iterations=6, shots=fixed, cvar_alpha=cvar_alpha,
                        seed=int(seed), backend=backend,
                    ),
                    "adaptive_cvar": AdaptiveShotBFDCQO(
                        spec["problem"], n_trotter=3, dt=0.1,
                        total_shot_budget=budget, max_iterations=6,
                        min_shots=max(32, budget // 24),
                        max_shots=2 * fixed, initial_shots=fixed,
                        cvar_alpha=cvar_alpha,
                        early_stop=True, min_iterations=2,
                        uncertainty_tol=0.30, bias_delta_tol=0.10,
                        seed=int(seed), backend=backend,
                    ),
                }
                for label, solver in solvers.items():
                    t0 = time.perf_counter()
                    r = solver.solve()
                    elapsed = time.perf_counter() - t0
                    met = _metrics(spec, r)
                    rows.append({
                        "problem": spec["name"],
                        "problem_kind": spec["kind"],
                        "variant": label,
                        "backend": _backend_label(backend),
                        "seed": int(seed),
                        "budget_cap": budget,
                        "total_shots": int(r.total_shots),
                        "shot_fraction_used": float(r.total_shots / budget),
                        "shots_saved": int(budget - r.total_shots),
                        "circuit_evaluations": int(r.circuit_evaluations),
                        "stopped_early": bool(r.stopped_early),
                        "shot_history": [int(x) for x in r.shot_history],
                        "p_opt": float(r.history[-1].ground_state_probability),
                        "expected_approx_ratio": float(met["expected_approx_ratio"]),
                        "best_approx_ratio": float(met["best_approx_ratio"]),
                        "success": bool(met["success"]),
                        "walltime_s": float(elapsed),
                    })

    summary = []
    for problem in sorted({r["problem"] for r in rows}):
        for cap in sorted({r["budget_cap"] for r in rows}):
            for variant in ("cvar_fixed", "adaptive_cvar"):
                rr = [r for r in rows if r["problem"] == problem and r["budget_cap"] == cap and r["variant"] == variant]
                summary.append({
                    "problem": problem,
                    "variant": variant,
                    "budget_cap": cap,
                    "runs": len(rr),
                    "mean_total_shots": float(np.mean([r["total_shots"] for r in rr])),
                    "mean_shot_fraction_used": float(np.mean([r["shot_fraction_used"] for r in rr])),
                    "mean_shots_saved": float(np.mean([r["shots_saved"] for r in rr])),
                    "early_stop_rate": float(np.mean([r["stopped_early"] for r in rr])),
                    "mean_p_opt": float(np.mean([r["p_opt"] for r in rr])),
                    "std_p_opt": float(np.std([r["p_opt"] for r in rr])),
                    "mean_expected_approx_ratio": float(np.mean([r["expected_approx_ratio"] for r in rr])),
                    "success_rate": float(np.mean([r["success"] for r in rr])),
                })

    payload = {
        "config": {
            "budget_caps": [int(x) for x in budget_caps],
            "seeds": [int(x) for x in seeds],
            "cvar_alpha": float(cvar_alpha),
            "adaptive_early_stop": {"uncertainty_tol": 0.30, "bias_delta_tol": 0.10, "min_iterations": 2},
            "backend": _backend_label(backend),
        },
        "rows": rows,
        "summary": summary,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "adaptive_efficiency_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (out / "adaptive_efficiency_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
    return payload
