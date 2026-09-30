# Licensed under the Apache License, Version 2.0.
"""Matched-shot MaxCut benchmark for BF-DCQO and QAOA."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .instances import default_maxcut_instances
from .metrics import maxcut_approximation
from .upstream_qaoa import UpstreamQAOASolver
from ..backends import pyqpanda3_backend
from ..solvers import BFDCQO, ExactSolver, QAOASolver


def _bfdcqo_backend_label(backend: str) -> str:
    if backend == "numpy":
        return "numpy_reference"
    if backend == "pyqpanda":
        return "pyqpanda3.CPUQVM"
    return str(backend)


def _upstream_metrics(problem, exact, result):
    energies = {
        bitstring: float(problem.energy_bitstring(bitstring))
        for bitstring in result.distribution
    }
    best_solution = min(energies, key=lambda key: (energies[key], key))
    ground_states = set(exact.best_solutions)
    return {
        "best_energy": energies[best_solution],
        "expected_energy": float(sum(
            result.distribution[key] * energy
            for key, energy in energies.items()
        )),
        "ground_state_probability": float(sum(
            probability
            for key, probability in result.distribution.items()
            if key in ground_states
        )),
        "success": best_solution in ground_states,
    }


def run_maxcut_benchmark(
    *,
    seeds=(3, 7, 11),
    bfdcqo_backend="numpy",
    qaoa_implementation="reference",
    qaoa_backend="auto",
    qaoa_shots=256,
    qaoa_max_evals=20,
    qaoa_final_shots=1024,
    output_dir=None,
):
    """Run the standard MaxCut Exact/BF-DCQO/QAOA benchmark suite.

    ``qaoa_implementation="reference"`` retains the deterministic local
    reference path. ``"upstream"`` executes the repository's public QAOA on
    CPUQVM and reports the circuit count and shots observed from that solver.
    """

    if qaoa_implementation not in {"reference", "upstream"}:
        raise ValueError(
            "qaoa_implementation must be 'reference' or 'upstream'"
        )
    if qaoa_backend == "auto":
        qaoa_backend = (
            "pyqpanda" if pyqpanda3_backend.available() else "numpy"
        )

    bf_backend_label = _bfdcqo_backend_label(bfdcqo_backend)
    qaoa_backend_label = (
        "pyqpanda_alg.QAOA.CPUQVM"
        if qaoa_implementation == "upstream"
        else qaoa_backend
    )
    rows = []
    for app in default_maxcut_instances():
        problem = app.to_ising()
        exact = ExactSolver().solve(problem)
        opt_cut = -exact.best_energy
        for seed in seeds:
            t0 = time.perf_counter()
            bf = BFDCQO(
                problem,
                n_trotter=3,
                dt=0.1,
                iterations=6,
                shots=1024,
                seed=seed,
                backend=bfdcqo_backend,
            ).solve()
            bf_time = time.perf_counter() - t0
            bf_cut = -bf.best_energy
            rows.append({
                "instance": app.name,
                "n": app.n,
                "solver": "BF-DCQO",
                "backend": bf_backend_label,
                "seed": seed,
                "opt_cut": opt_cut,
                "best_cut": bf_cut,
                "approx_ratio": maxcut_approximation(bf_cut, opt_cut),
                "p_opt": bf.history[-1].ground_state_probability,
                "expected_cut": -bf.history[-1].expected_energy,
                "expected_approx_ratio": (
                    -bf.history[-1].expected_energy / opt_cut
                ),
                "target_shot_budget": 6144,
                "total_shots": bf.total_shots,
                "budget_delta": bf.total_shots - 6144,
                "circuit_evaluations": bf.circuit_evaluations,
                "walltime_s": bf_time,
                "success": bf.success,
            })

            t0 = time.perf_counter()
            if qaoa_implementation == "reference":
                target_budget = qaoa_max_evals * qaoa_shots + qaoa_final_shots
                q = QAOASolver(
                    problem,
                    p=1,
                    backend=qaoa_backend,
                    shots_per_eval=qaoa_shots,
                    max_evals=qaoa_max_evals,
                    final_shots=qaoa_final_shots,
                    total_shot_budget=target_budget,
                    seed=seed,
                    method="COBYLA",
                ).solve()
                q_metrics = {
                    "best_energy": q.best_energy,
                    "expected_energy": q.expected_energy,
                    "ground_state_probability": q.ground_state_probability,
                    "success": q.success,
                }
                q_total_shots = q.total_shots
                q_circuit_evaluations = q.circuit_evaluations
                q_label = "QAOA-p1-reference"
            else:
                target_budget = (qaoa_max_evals + 2) * qaoa_shots
                q = UpstreamQAOASolver(
                    problem,
                    p=1,
                    shots=qaoa_shots,
                    max_evals=qaoa_max_evals,
                    initial_parameter_seed=seed,
                ).run()
                q_metrics = _upstream_metrics(problem, exact, q)
                q_total_shots = q.total_shots
                q_circuit_evaluations = q.circuit_evaluations
                q_label = "QAOA-p1-upstream"
            q_time = time.perf_counter() - t0
            q_cut = -q_metrics["best_energy"]
            rows.append({
                "instance": app.name,
                "n": app.n,
                "solver": q_label,
                "backend": qaoa_backend_label,
                "seed": seed,
                "opt_cut": opt_cut,
                "best_cut": q_cut,
                "approx_ratio": maxcut_approximation(q_cut, opt_cut),
                "p_opt": q_metrics["ground_state_probability"],
                "expected_cut": -q_metrics["expected_energy"],
                "expected_approx_ratio": (
                    -q_metrics["expected_energy"] / opt_cut
                ),
                "target_shot_budget": target_budget,
                "total_shots": q_total_shots,
                "budget_delta": q_total_shots - target_budget,
                "circuit_evaluations": q_circuit_evaluations,
                "walltime_s": q_time,
                "success": q_metrics["success"],
            })

    summary = []
    solver_order = ("BF-DCQO", f"QAOA-p1-{qaoa_implementation}")
    for inst in sorted({row["instance"] for row in rows}):
        for solver in solver_order:
            selected = [
                row for row in rows
                if row["instance"] == inst and row["solver"] == solver
            ]
            summary.append({
                "instance": inst,
                "solver": solver,
                "backend": selected[0]["backend"],
                "runs": len(selected),
                "opt_cut": selected[0]["opt_cut"],
                "mean_best_cut": float(np.mean([
                    row["best_cut"] for row in selected
                ])),
                "mean_approx_ratio": float(np.mean([
                    row["approx_ratio"] for row in selected
                ])),
                "mean_p_opt": float(np.mean([
                    row["p_opt"] for row in selected
                ])),
                "mean_expected_approx_ratio": float(np.mean([
                    row["expected_approx_ratio"] for row in selected
                ])),
                "std_p_opt": float(np.std([
                    row["p_opt"] for row in selected
                ])),
                "mean_expected_cut": float(np.mean([
                    row["expected_cut"] for row in selected
                ])),
                "mean_total_shots": float(np.mean([
                    row["total_shots"] for row in selected
                ])),
                "mean_circuit_evaluations": float(np.mean([
                    row["circuit_evaluations"] for row in selected
                ])),
                "success_rate": float(np.mean([
                    row["success"] for row in selected
                ])),
                "mean_walltime_s": float(np.mean([
                    row["walltime_s"] for row in selected
                ])),
            })

    result = {
        "config": {
            "bfdcqo_backend": bf_backend_label,
            "qaoa_implementation": qaoa_implementation,
            "qaoa_backend": qaoa_backend_label,
            "seeds": [int(seed) for seed in seeds],
            "qaoa_shots": int(qaoa_shots),
            "qaoa_max_evals": int(qaoa_max_evals),
            "qaoa_final_shots": (
                int(qaoa_final_shots)
                if qaoa_implementation == "reference"
                else None
            ),
        },
        # Retained for consumers of the original benchmark schema.
        "qaoa_backend": qaoa_backend_label,
        "pyqpanda3_available": pyqpanda3_backend.available(),
        "budget": {
            "bf": "6 iterations x 1024 shots = 6144",
            "qaoa": (
                "observed CPUQVM circuit evaluations x shots"
                if qaoa_implementation == "upstream"
                else "objective evaluations x shots + final sample"
            ),
        },
        "rows": rows,
        "summary": summary,
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_raw.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        with (out / "benchmark_raw.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (out / "benchmark_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return result
