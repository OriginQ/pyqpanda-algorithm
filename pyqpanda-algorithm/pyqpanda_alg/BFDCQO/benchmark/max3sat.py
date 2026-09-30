# Licensed under the Apache License, Version 2.0.
"""Native-HUBO MAX-3-SAT benchmark against a quadratized QAOA baseline."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import numpy as np

from ..applications.max3sat import Max3SATProblem, max3sat_to_qubo
from ..solvers import BFDCQO, QAOASolver, ExactSolver
from ..backends import pyqpanda3_backend
from ..backends.pyqpanda3_backend import pauli_rotation_resources
from .upstream_qaoa import UpstreamQAOASolver


def default_max3sat_instances() -> list[Max3SATProblem]:
    """Small deterministic MAX-3-SAT instances chosen for exact verification.

    Cubic supports are intentionally reused so HUBO->QUBO ancilla overhead is
    visible but still small enough for repeatable statevector regression tests.
    """
    return [
        Max3SATProblem.from_clauses(
            5,
            [
                (1, 2, 3), (-1, 2, 3), (1, -2, 3),
                (-1, 4, 5), (1, -4, 5), (1, 4, -5),
                (2, -3, 5), (-2, 3, 5),
            ],
            name="sat5_reused",
        ),
        Max3SATProblem.from_clauses(
            5,
            [
                ((1, 2, 4), 1.0), ((-1, 2, 4), 1.4), ((1, -2, 4), 0.8),
                ((1, 3, 5), 1.2), ((-1, 3, 5), 0.9),
                ((2, 3, -5), 1.6), ((-2, -3, 5), 1.1),
                ((2, 4, 5), 0.7), ((-2, 4, -5), 1.3),
            ],
            name="weighted5",
        ),
        Max3SATProblem.from_clauses(
            6,
            [
                (1, 2, 3), (-1, 2, 3),
                (1, 4, 6), (1, -4, 6),
                (2, 4, 5), (-2, 4, 5),
                (3, 5, 6), (3, -5, 6),
                (-1, -2, 3), (1, 4, -6),
            ],
            name="sat6_sparsecubic",
        ),
        Max3SATProblem.from_clauses(
            5,
            [
                (2, 3, 4), (-2, -3, -4), (-2, 4, 5), (2, 4, -5),
                (1, -4, 5), (-1, 4, -5), (-2, 4, -5), (2, -3, -4),
                (-1, -2, -3), (2, 4, 5), (-1, -4, -5), (1, -2, 3),
                (1, -4, -5), (-1, -4, 5), (2, 3, -4), (2, -3, 4),
            ],
            name="frustrated5_unsat",
        ),
    ]




def _bfdcqo_resource_summary(solver: BFDCQO, result):
    """Logical gate-count estimate for the circuits actually represented by a BF run."""
    biases = [np.zeros(solver.problem.n)] + [r.bias for r in result.history[:-1]]
    per = []
    for bias in biases:
        layers, _ = solver._layers(np.asarray(bias, dtype=float))
        cnot = rz = basis = 0
        max_w = 0
        for layer in layers:
            for ps, theta in layer:
                if abs(theta) < 1e-15:
                    continue
                rr = pauli_rotation_resources(ps)
                cnot += rr["cnot"]; rz += rr["rz"]; basis += rr["basis_change_gates"]
                max_w = max(max_w, rr["weight"])
        per.append({"cnot": cnot, "rz": rz, "basis": basis, "max_weight": max_w})
    shots = 1 if solver.shots is None else solver.shots
    return {
        "mean_cnot_per_circuit": float(np.mean([x["cnot"] for x in per])),
        "max_cnot_per_circuit": int(max(x["cnot"] for x in per)),
        "cnot_shot_executions": int(sum(x["cnot"] * shots for x in per)),
        "max_pauli_weight": int(max(x["max_weight"] for x in per)),
    }


def _qaoa_resource_summary(problem, result, p: int = 1):
    pair_count = sum(1 for i in range(problem.n) for j in range(i + 1, problem.n) if abs(problem.J[i, j]) > 1e-15)
    cnot_per = 2 * pair_count * p
    return {
        "mean_cnot_per_circuit": float(cnot_per),
        "max_cnot_per_circuit": int(cnot_per),
        "cnot_shot_executions": int(cnot_per * result.total_shots),
        "max_pauli_weight": 2 if pair_count else (1 if np.any(np.abs(problem.h) > 1e-15) else 0),
    }

def _logical_metrics_from_distribution(app: Max3SATProblem, prob: np.ndarray, full_n: int):
    if prob.shape != (1 << full_n,):
        raise ValueError("probability length mismatch")
    opt, opt_states = app.exact_solution()
    opt_idx = {int(s, 2) for s in opt_states}
    mask = (1 << app.n) - 1
    p_opt = 0.0
    expected_sat = 0.0
    for idx, p in enumerate(prob):
        if p == 0:
            continue
        logical_idx = idx & mask
        bits = [(logical_idx >> q) & 1 for q in range(app.n)]
        sat = app.satisfied_weight_bits(bits)
        expected_sat += float(p) * sat
        if logical_idx in opt_idx:
            p_opt += float(p)
    return {
        "p_opt_logical": float(p_opt),
        "expected_satisfied_weight": float(expected_sat),
        "expected_approx_ratio": float(expected_sat / opt) if opt > 0 else 1.0,
    }


def _backend_label(backend):
    if backend == "numpy":
        return "numpy_reference"
    if backend == "pyqpanda":
        return "pyqpanda3.CPUQVM"
    return str(backend)


def _dense_distribution(distribution, n):
    prob = np.zeros(1 << n, dtype=float)
    for bitstring, probability in distribution.items():
        prob[int(bitstring, 2)] = float(probability)
    return prob


def run_max3sat_benchmark(
    *,
    seeds=(0, 1, 2),
    instances=None,
    bfdcqo_backend="numpy",
    qaoa_implementation="reference",
    qaoa_backend="auto",
    qaoa_shots=256,
    qaoa_max_evals=20,
    qaoa_final_shots=1024,
    output_dir=None,
):
    """Run native-HUBO BF-DCQO versus quadratized QAOA MAX-3-SAT benchmarks."""
    if qaoa_implementation not in {"reference", "upstream"}:
        raise ValueError(
            "qaoa_implementation must be 'reference' or 'upstream'"
        )
    if qaoa_backend == "auto":
        qaoa_backend = "pyqpanda" if pyqpanda3_backend.available() else "numpy"
    benchmark_instances = (
        tuple(default_max3sat_instances())
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
    reductions = []
    for app in benchmark_instances:
        native = app.to_native_ising()
        exact_native = ExactSolver().solve(native)
        opt_sat, opt_states = app.exact_solution()
        # Native energy is weighted number of unsatisfied clauses.
        if not np.isclose(app.total_weight - exact_native.best_energy, opt_sat, atol=1e-10):
            raise RuntimeError("native HUBO exact energy disagrees with MAX-3-SAT objective")

        red = max3sat_to_qubo(app, penalty_margin=1.0)
        qproblem = red.to_ising()
        exact_q = ExactSolver().solve(qproblem)
        q_logical = exact_q.best_solutions[0][-app.n:]
        q_best_sat = app.satisfied_weight_bitstring(q_logical)
        if not np.isclose(q_best_sat, opt_sat, atol=1e-10):
            raise RuntimeError("quadratized QUBO ground state does not map to MAX-3-SAT optimum")
        reductions.append({
            "instance": app.name,
            "logical_n": app.n,
            "native_degree": native.degree,
            "native_terms": native.num_terms,
            "cubic_binary_terms": sum(1 for t in app.binary_hubo()[1] if len(t) == 3),
            "qubo_n": red.total_n,
            "ancillas": red.ancilla_count,
            "qubo_terms": len(red.terms),
            "opt_satisfied_weight": opt_sat,
            "total_clause_weight": app.total_weight,
            "native_exact_energy": exact_native.best_energy,
            "qubo_exact_energy": exact_q.best_energy,
        })

        for seed in seeds:
            t0 = time.perf_counter()
            bf_solver = BFDCQO(
                native, n_trotter=3, dt=0.1, iterations=6,
                shots=1024, seed=seed, backend=bfdcqo_backend,
            )
            bf = bf_solver.solve()
            bf_time = time.perf_counter() - t0
            bf_last = bf.history[-1]
            bf_res = _bfdcqo_resource_summary(bf_solver, bf)
            bf_best_sat = app.total_weight - bf.best_energy
            rows.append({
                "instance": app.name,
                "solver": "BF-DCQO-native-HUBO",
                "backend": bf_backend_label,
                "seed": seed,
                "logical_n": app.n,
                "solver_n": app.n,
                "ancillas": 0,
                "opt_satisfied_weight": opt_sat,
                "best_satisfied_weight": bf_best_sat,
                "best_logical_solution": bf.best_solution,
                "best_approx_ratio": bf_best_sat / opt_sat if opt_sat > 0 else 1.0,
                "p_opt_logical": bf_last.ground_state_probability,
                "expected_satisfied_weight": app.total_weight - bf_last.expected_energy,
                "expected_approx_ratio": (app.total_weight - bf_last.expected_energy) / opt_sat if opt_sat > 0 else 1.0,
                "formulation": "native-HUBO",
                "target_shot_budget": 6144,
                "total_shots": bf.total_shots,
                "budget_delta": bf.total_shots - 6144,
                "circuit_evaluations": bf.circuit_evaluations,
                **bf_res,
                "success": bool(np.isclose(bf_best_sat, opt_sat, atol=1e-9)),
                "walltime_s": bf_time,
            })

            t0 = time.perf_counter()
            if qaoa_implementation == "reference":
                target_budget = (
                    qaoa_max_evals * qaoa_shots + qaoa_final_shots
                )
                qsolver = QAOASolver(
                    qproblem, p=1, backend=qaoa_backend,
                    shots_per_eval=qaoa_shots, max_evals=qaoa_max_evals,
                    final_shots=qaoa_final_shots,
                    total_shot_budget=target_budget,
                    seed=seed, method="COBYLA",
                )
                q = qsolver.solve()
                qprob = qsolver.probabilities(q.parameters)
                q_best_solution = q.best_solution
                q_total_shots = q.total_shots
                q_circuit_evaluations = q.circuit_evaluations
                q_success = q.success
                q_label = "QAOA-p1-reference"
            else:
                target_budget = (qaoa_max_evals + 2) * qaoa_shots
                q = UpstreamQAOASolver(
                    qproblem, p=1, shots=qaoa_shots,
                    max_evals=qaoa_max_evals,
                    initial_parameter_seed=seed,
                ).run()
                qprob = _dense_distribution(q.distribution, qproblem.n)
                q_best_solution = min(
                    q.distribution,
                    key=lambda bitstring: (
                        qproblem.energy_bitstring(bitstring), bitstring
                    ),
                )
                q_total_shots = q.total_shots
                q_circuit_evaluations = q.circuit_evaluations
                q_success = np.isclose(
                    qproblem.energy_bitstring(q_best_solution),
                    exact_q.best_energy,
                    atol=1e-9,
                )
                q_label = "QAOA-p1-upstream"
            q_time = time.perf_counter() - t0
            q_res = _qaoa_resource_summary(qproblem, q, p=1)
            lm = _logical_metrics_from_distribution(app, qprob, qproblem.n)
            logical_best = q_best_solution[-app.n:]
            q_best_sat = app.satisfied_weight_bitstring(logical_best)
            rows.append({
                "instance": app.name,
                "solver": q_label,
                "backend": qaoa_backend_label,
                "seed": seed,
                "logical_n": app.n,
                "solver_n": qproblem.n,
                "ancillas": red.ancilla_count,
                "opt_satisfied_weight": opt_sat,
                "best_satisfied_weight": q_best_sat,
                "best_logical_solution": logical_best,
                "best_approx_ratio": q_best_sat / opt_sat if opt_sat > 0 else 1.0,
                "p_opt_logical": lm["p_opt_logical"],
                "expected_satisfied_weight": lm["expected_satisfied_weight"],
                "expected_approx_ratio": lm["expected_approx_ratio"],
                "formulation": "Rosenberg-quadratized-QUBO",
                "target_shot_budget": target_budget,
                "total_shots": q_total_shots,
                "budget_delta": q_total_shots - target_budget,
                "circuit_evaluations": q_circuit_evaluations,
                **q_res,
                "success": bool(q_success and np.isclose(q_best_sat, opt_sat, atol=1e-9)),
                "walltime_s": q_time,
            })

    summary = []
    for inst in sorted({r["instance"] for r in rows}):
        for solver in (
            "BF-DCQO-native-HUBO",
            f"QAOA-p1-{qaoa_implementation}",
        ):
            rr = [r for r in rows if r["instance"] == inst and r["solver"] == solver]
            summary.append({
                "instance": inst,
                "solver": solver,
                "backend": rr[0]["backend"],
                "runs": len(rr),
                "logical_n": rr[0]["logical_n"],
                "solver_n": rr[0]["solver_n"],
                "ancillas": rr[0]["ancillas"],
                "opt_satisfied_weight": rr[0]["opt_satisfied_weight"],
                "mean_best_satisfied_weight": float(np.mean([r["best_satisfied_weight"] for r in rr])),
                "mean_best_approx_ratio": float(np.mean([r["best_approx_ratio"] for r in rr])),
                "mean_p_opt_logical": float(np.mean([r["p_opt_logical"] for r in rr])),
                "std_p_opt_logical": float(np.std([r["p_opt_logical"] for r in rr])),
                "mean_expected_satisfied_weight": float(np.mean([r["expected_satisfied_weight"] for r in rr])),
                "mean_expected_approx_ratio": float(np.mean([r["expected_approx_ratio"] for r in rr])),
                "mean_total_shots": float(np.mean([r["total_shots"] for r in rr])),
                "mean_circuit_evaluations": float(np.mean([r["circuit_evaluations"] for r in rr])),
                "mean_cnot_per_circuit": float(np.mean([r["mean_cnot_per_circuit"] for r in rr])),
                "mean_cnot_shot_executions": float(np.mean([r["cnot_shot_executions"] for r in rr])),
                "max_pauli_weight": int(max(r["max_pauli_weight"] for r in rr)),
                "success_rate": float(np.mean([r["success"] for r in rr])),
                "mean_walltime_s": float(np.mean([r["walltime_s"] for r in rr])),
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
        "comparison": "native degree-3 HUBO BF-DCQO vs Rosenberg-quadratized QUBO p=1 QAOA",
        "reductions": reductions,
        "rows": rows,
        "summary": summary,
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "max3sat_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (out / "max3sat_raw.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with (out / "max3sat_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
        with (out / "max3sat_reduction_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(reductions[0].keys()))
            w.writeheader(); w.writerows(reductions)
    return result
