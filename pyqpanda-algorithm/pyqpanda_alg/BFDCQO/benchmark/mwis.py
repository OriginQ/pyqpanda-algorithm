# Licensed under the Apache License, Version 2.0.
"""Low-shot MWIS benchmark for hybrid BF-DCQO."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Literal
import numpy as np

from ..applications.mwis import MWISProblem
from ..solvers import ExactSolver, BFDCQO, HybridBFDCQO, QAOASolver
from ..backends import pyqpanda3_backend
from .upstream_qaoa import UpstreamQAOASolver


def default_mwis_instances() -> list[MWISProblem]:
    """Three fixed 8-vertex weighted graphs used by the low-shot benchmark.

    The instances are intentionally small enough for exact verification but have
    distinct graph densities / weighted optima. They were fixed before the final
    These instances are deterministic and are not generated during benchmarking.
    """
    return [
        MWISProblem(
            weights=np.array([1.06, 1.66, 1.51, 1.56, 1.15, 2.00, 2.00, 1.92]),
            edges=((0,1),(0,7),(1,3),(1,5),(1,6),(2,6),(3,6),(3,7),(4,7),(5,6),(5,7)),
            name="mwis8_a",
        ),
        MWISProblem(
            weights=np.array([1.66, 1.16, 1.79, 1.55, 0.64, 1.96, 1.64, 1.68]),
            edges=((0,1),(0,3),(1,2),(1,4),(2,3),(2,7),(3,5),(3,6),(4,7),(5,6),(6,7)),
            name="mwis8_b",
        ),
        MWISProblem(
            weights=np.array([1.08, 1.39, 1.27, 1.54, 1.48, 1.79, 1.38, 1.29]),
            edges=((0,2),(0,3),(0,4),(0,6),(0,7),(1,2),(1,6),(1,7),(2,6),(3,5),(3,6),(3,7),(4,6),(5,6)),
            name="mwis8_c",
        ),
    ]


def _mwis_metrics(app: MWISProblem, bitstring: str, opt_weight: float) -> dict:
    feasible = app.is_independent_bitstring(bitstring)
    weight = app.weight_bitstring(bitstring) if feasible else 0.0
    return {
        "best_solution": bitstring,
        "feasible": bool(feasible),
        "best_weight": float(weight),
        "weight_ratio": float(weight / opt_weight if opt_weight > 0 else 1.0),
        "violations": int(app.violations_bitstring(bitstring)),
    }


def _backend_label(backend):
    if backend == "numpy":
        return "numpy_reference"
    if backend == "pyqpanda":
        return "pyqpanda3.CPUQVM"
    return str(backend)


def run_mwis_hybrid_benchmark(
    *,
    seeds=tuple(range(10)),
    instances=None,
    quantum_budget: int = 72,
    local_neighborhood_size: int = 4,
    local_rounds: int = 1,
    bfdcqo_backend: Literal["numpy", "pyqpanda"] = "numpy",
    qaoa_implementation: str = "reference",
    qaoa_backend: Literal["auto", "numpy", "pyqpanda"] = "auto",
    output_dir=None,
):
    """Low-shot MWIS benchmark for Exact, QAOA, Basic BF and Hybrid BF.

    The benchmark intentionally uses a severe low-shot regime so a small local
    refinement can be measured rather than being hidden by near-certain quantum
    optimum sampling.  With the default budget:

      BF:   3 iterations x 24 shots = 72
      Reference QAOA: 4 objective evaluations x 12 shots + 24 final
      shots = 72. The upstream target includes the same four optimizer calls
      plus its mandatory final distribution and final loss circuits; observed
      CPUQVM counts are still reported separately.

    Both hybrid variants use the *same* BF quantum stage and exactly
    ``2**local_neighborhood_size * local_rounds`` classical energy evaluations.
    """
    if qaoa_implementation not in {"reference", "upstream"}:
        raise ValueError(
            "qaoa_implementation must be 'reference' or 'upstream'"
        )
    quantum_budget = int(quantum_budget)
    if quantum_budget % 6 != 0:
        raise ValueError(
            "quantum_budget must be divisible by 6 for the BF/QAOA schedules"
        )
    bf_shots = quantum_budget // 3
    # Keep the final sampling stage at one third; split the remaining two thirds
    # over four low-shot objective evaluations.
    q_shots_eval = quantum_budget // 6
    q_final = quantum_budget - 4 * q_shots_eval
    if qaoa_backend == "auto":
        qaoa_backend = "pyqpanda" if pyqpanda3_backend.available() else "numpy"
    benchmark_instances = (
        tuple(default_mwis_instances())
        if instances is None
        else tuple(instances)
    )
    bf_backend_label = _backend_label(bfdcqo_backend)
    qaoa_backend_label = (
        "pyqpanda_alg.QAOA.CPUQVM"
        if qaoa_implementation == "upstream"
        else qaoa_backend
    )

    rows = []
    for app in benchmark_instances:
        problem = app.to_ising()
        exact = ExactSolver().solve(problem)
        opt_weight, opt_states = app.exact_mwis()
        if not np.isclose(exact.best_energy, -opt_weight, atol=1e-10):
            raise RuntimeError(f"MWIS/QUBO exact mismatch on {app.name}")

        for seed in seeds:
            seed = int(seed)
            # Basic BF quantum-only baseline.
            t0 = time.perf_counter()
            bf = BFDCQO(
                problem, n_trotter=2, dt=0.1,
                iterations=3, shots=bf_shots, seed=seed,
                backend=bfdcqo_backend,
            ).solve()
            bf_t = time.perf_counter() - t0
            met = _mwis_metrics(app, bf.best_solution, opt_weight)
            rows.append({
                "instance": app.name, "n": app.n, "edges": len(app.edges),
                "solver": "BF-basic", "backend": bf_backend_label, "seed": seed,
                "opt_weight": opt_weight, **met,
                "p_opt": float(bf.history[-1].ground_state_probability),
                "total_shots": int(bf.total_shots),
                "target_shot_budget": quantum_budget,
                "budget_delta": int(bf.total_shots - quantum_budget),
                "circuit_evaluations": int(bf.circuit_evaluations),
                "local_evaluations": 0, "local_improvement_energy": 0.0,
                "success": bool(bf.success), "walltime_s": float(bf_t),
            })

            # Equal-cost random block ablation.
            t0 = time.perf_counter()
            hr = HybridBFDCQO(
                problem, base_variant="basic", local_method="random",
                neighborhood_size=local_neighborhood_size, local_rounds=local_rounds,
                local_seed=seed + 321, n_trotter=2, dt=0.1,
                max_iterations=3, shots=bf_shots, seed=seed,
                backend=bfdcqo_backend,
            ).solve()
            hr_t = time.perf_counter() - t0
            met = _mwis_metrics(app, hr.best_solution, opt_weight)
            rows.append({
                "instance": app.name, "n": app.n, "edges": len(app.edges),
                "solver": "Hybrid-random", "backend": bf_backend_label, "seed": seed,
                "opt_weight": opt_weight, **met,
                "p_opt": float(hr.quantum_result.history[-1].ground_state_probability),
                "total_shots": int(hr.total_shots),
                "target_shot_budget": quantum_budget,
                "budget_delta": int(hr.total_shots - quantum_budget),
                "circuit_evaluations": int(hr.circuit_evaluations),
                "local_evaluations": int(hr.local_evaluations),
                "local_improvement_energy": float(hr.local_improvement),
                "success": bool(hr.success), "walltime_s": float(hr_t),
            })

            # Confidence-guided block refinement, same quantum + classical budgets.
            t0 = time.perf_counter()
            hc = HybridBFDCQO(
                problem, base_variant="basic", local_method="confidence",
                neighborhood_size=local_neighborhood_size, local_rounds=local_rounds,
                n_trotter=2, dt=0.1,
                max_iterations=3, shots=bf_shots, seed=seed,
                backend=bfdcqo_backend,
            ).solve()
            hc_t = time.perf_counter() - t0
            met = _mwis_metrics(app, hc.best_solution, opt_weight)
            rows.append({
                "instance": app.name, "n": app.n, "edges": len(app.edges),
                "solver": "Hybrid-confidence", "backend": bf_backend_label, "seed": seed,
                "opt_weight": opt_weight, **met,
                "p_opt": float(hc.quantum_result.history[-1].ground_state_probability),
                "total_shots": int(hc.total_shots),
                "target_shot_budget": quantum_budget,
                "budget_delta": int(hc.total_shots - quantum_budget),
                "circuit_evaluations": int(hc.circuit_evaluations),
                "local_evaluations": int(hc.local_evaluations),
                "local_improvement_energy": float(hc.local_improvement),
                "success": bool(hc.success), "walltime_s": float(hc_t),
            })

            # Selectable local-reference or public upstream QAOA baseline.
            t0 = time.perf_counter()
            if qaoa_implementation == "reference":
                q = QAOASolver(
                    problem, p=1, backend=qaoa_backend,
                    shots_per_eval=q_shots_eval, max_evals=4,
                    final_shots=q_final, total_shot_budget=quantum_budget,
                    seed=seed, method="COBYLA",
                ).solve()
                q_best_solution = q.best_solution
                q_p_opt = q.ground_state_probability
                q_total_shots = q.total_shots
                q_circuit_evaluations = q.circuit_evaluations
                q_success = q.success
                q_target_budget = quantum_budget
                q_label = "QAOA-p1-reference"
            else:
                q_upstream = UpstreamQAOASolver(
                    problem, p=1, shots=q_shots_eval, max_evals=4,
                    initial_parameter_seed=seed,
                ).run()
                q_best_solution = min(
                    q_upstream.distribution,
                    key=lambda bitstring: (
                        problem.energy_bitstring(bitstring), bitstring
                    ),
                )
                ground_states = set(exact.best_solutions)
                q_p_opt = float(sum(
                    probability
                    for bitstring, probability in q_upstream.distribution.items()
                    if bitstring in ground_states
                ))
                q_total_shots = q_upstream.total_shots
                q_circuit_evaluations = q_upstream.circuit_evaluations
                q_success = q_best_solution in ground_states
                q_target_budget = (4 + 2) * q_shots_eval
                q_label = "QAOA-p1-upstream"
            q_t = time.perf_counter() - t0
            met = _mwis_metrics(app, q_best_solution, opt_weight)
            rows.append({
                "instance": app.name, "n": app.n, "edges": len(app.edges),
                "solver": q_label, "backend": qaoa_backend_label, "seed": seed,
                "opt_weight": opt_weight, **met,
                "p_opt": float(q_p_opt),
                "total_shots": int(q_total_shots),
                "target_shot_budget": int(q_target_budget),
                "budget_delta": int(q_total_shots - q_target_budget),
                "circuit_evaluations": int(q_circuit_evaluations),
                "local_evaluations": 0, "local_improvement_energy": 0.0,
                "success": bool(q_success), "walltime_s": float(q_t),
            })

    summary = []
    solver_order = (
        "BF-basic", "Hybrid-random", "Hybrid-confidence",
        f"QAOA-p1-{qaoa_implementation}",
    )
    for instance in [x.name for x in benchmark_instances]:
        for solver in solver_order:
            rr = [r for r in rows if r["instance"] == instance and r["solver"] == solver]
            summary.append({
                "instance": instance,
                "solver": solver,
                "runs": len(rr),
                "opt_weight": rr[0]["opt_weight"],
                "mean_best_weight": float(np.mean([r["best_weight"] for r in rr])),
                "mean_weight_ratio": float(np.mean([r["weight_ratio"] for r in rr])),
                "success_rate": float(np.mean([r["success"] for r in rr])),
                "feasible_rate": float(np.mean([r["feasible"] for r in rr])),
                "mean_p_opt": float(np.mean([r["p_opt"] for r in rr])),
                "mean_total_shots": float(np.mean([r["total_shots"] for r in rr])),
                "mean_local_evaluations": float(np.mean([r["local_evaluations"] for r in rr])),
                "mean_local_improvement_energy": float(np.mean([r["local_improvement_energy"] for r in rr])),
                "mean_walltime_s": float(np.mean([r["walltime_s"] for r in rr])),
            })

    payload = {
        "config": {
            "seeds": [int(s) for s in seeds],
            "quantum_budget": quantum_budget,
            "bf": {"iterations": 3, "shots_per_iteration": bf_shots, "n_trotter": 2},
            "qaoa": {
                "p": 1,
                "objective_evaluations_cap": 4,
                "shots_per_eval": q_shots_eval,
                "final_shots": (
                    q_final if qaoa_implementation == "reference" else None
                ),
                "target_shot_budget": (
                    quantum_budget
                    if qaoa_implementation == "reference"
                    else (4 + 2) * q_shots_eval
                ),
            },
            "local": {"neighborhood_size": int(local_neighborhood_size), "rounds": int(local_rounds), "evaluations_per_run": int((2 ** min(local_neighborhood_size, 8)) * local_rounds)},
            "bfdcqo_backend": bf_backend_label,
            "qaoa_implementation": qaoa_implementation,
            "qaoa_backend": qaoa_backend_label,
            "qaoa_final_shots": (
                q_final if qaoa_implementation == "reference" else None
            ),
            "pyqpanda3_available": pyqpanda3_backend.available(),
        },
        "rows": rows,
        "summary": summary,
    }
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "mwis_hybrid_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (out / "mwis_hybrid_raw.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        with (out / "mwis_hybrid_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    return payload
