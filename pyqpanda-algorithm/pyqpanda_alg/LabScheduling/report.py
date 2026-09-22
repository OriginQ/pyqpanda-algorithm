"""End-to-end reports and compact, machine-readable CLI output."""

from __future__ import annotations

import hashlib
import platform
from importlib.metadata import version
from math import prod
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .classical import solve_exact
from .metrics import circuit_resources, distribution_metrics
from .milp import MAX_RAW_CANDIDATES, solve_milp, validate_assignment
from .model import compile_qubo, evaluate
from .problem import parse_problem
from .quantum import circuit_probabilities, solve_qaoa
from .reduction import solve_reduced


def run_experiment(
    path: str | Path,
    seed: int = 7,
    layers: int = 1,
    shots: int = 512,
    maxiter: int = 60,
    restarts: int = 2,
    mixer: str = "xy",
    *,
    reduce: bool = False,
    pruning: str = "singleton",
) -> dict[str, Any]:
    """Solve a snapshot; report unavailable baseline budgets without losing results.

    Optional arc pruning requires reduce=True. A skipped MILP contributes no
    certificate; absolute_gap is unknown unless enumeration proves optimality.
    """
    if pruning not in ("singleton", "arc"):
        raise ValueError("pruning must be singleton or arc")
    if pruning != "singleton" and not reduce:
        raise ValueError("arc pruning requires reduce=True (--reduce)")
    started = perf_counter()
    path = Path(path)
    snapshot = path.read_bytes()
    problem = parse_problem(snapshot.decode("utf-8"))
    model = compile_qubo(problem)
    quantum = (
        solve_reduced(
            model, seed, layers, shots, maxiter, restarts, mixer, pruning=pruning
        )
        if reduce
        else solve_qaoa(model, seed, layers, shots, maxiter, restarts, mixer)
    )
    exact: dict[str, Any] = (
        solve_exact(model)
        if prod(map(len, model.domains)) <= 1_000_000
        else {
            "status": "budget_exceeded",
            "best": None,
            "combinations": prod(map(len, model.domains)),
            "feasible_count": None,
        }
    )
    raw_candidates = sum(len(task.options) for task in problem.tasks)
    milp: dict[str, Any] = (
        solve_milp(problem)
        if raw_candidates <= MAX_RAW_CANDIDATES
        else {
            "status": "budget_exceeded",
            "variables": raw_candidates,
            "max_variables": MAX_RAW_CANDIDATES,
            "reason": "raw candidate count exceeds the independent MILP budget",
            "best": None,
            "dual_bound": None,
            "gap": None,
        }
    )
    if exact["best"] and milp["status"] == "optimal":
        if exact["best"]["objective"] != milp["best"]["objective"]:
            raise RuntimeError("enumeration and raw-input MILP disagree")
    if milp["status"] == "infeasible" and (
        exact["best"] or quantum["best"] and quantum["best"]["feasible"]
    ):
        raise RuntimeError("raw-input MILP rejected a feasible witness")
    if exact["status"] == "infeasible" and milp["best"]:
        raise RuntimeError("enumeration rejected a raw-input MILP witness")
    if quantum["best"] and quantum["best"]["feasible"]:
        choices = []
        for ti, task in enumerate(problem.tasks):
            chosen = next(
                v.option
                for v, bit in zip(model.variables, quantum["best"]["bits"], strict=True)
                if bit and v.task == ti
            )
            choices.append(task.options.index(chosen))
        independent = validate_assignment(problem, choices)
        if (
            not independent["feasible"]
            or independent["objective"] != quantum["best"]["objective"]
        ):
            raise RuntimeError("quantum schedule failed raw-input validation")
        quantum["business_metrics"] = independent
    if "parameters" in quantum:
        quantum["resources"] = circuit_resources(model, quantum["parameters"], mixer)
        quantum["distribution"] = distribution_metrics(
            model, circuit_probabilities(model, quantum["parameters"], mixer)
        )
    uniform_started = perf_counter()
    random = np.random.default_rng(seed)
    uniform_best = None
    feasible_samples = 0
    if all(model.domains):
        for _ in range(shots):
            bits = [0] * len(model.variables)
            for domain in model.domains:
                bits[int(random.choice(domain))] = 1
            candidate = evaluate(model, bits)
            feasible_samples += candidate["feasible"]
            if uniform_best is None or (
                not candidate["feasible"],
                candidate["qubo_energy"],
            ) < (not uniform_best["feasible"], uniform_best["qubo_energy"]):
                uniform_best = candidate
    uniform = {
        "best": uniform_best,
        "shots": shots,
        "seed": seed,
        "feasible_probability": (
            exact["feasible_count"] / exact["combinations"]
            if exact["combinations"] and exact["feasible_count"] is not None
            else 0.0
            if exact["combinations"] == 0
            else None
        ),
        "sampled_feasible_fraction": feasible_samples / shots,
        "runtime_seconds": perf_counter() - uniform_started,
    }
    quantum_best = quantum["best"]
    exact_best = exact["best"] or (
        milp["best"] if milp["status"] == "optimal" else None
    )
    gap = (
        quantum_best["objective"] - exact_best["objective"]
        if quantum_best and quantum_best["feasible"] and exact_best
        else None
    )
    if gap is not None and gap < 0:
        raise RuntimeError(
            "quantum result contradicts the independent exact certificate"
        )
    return {
        "schema_version": 2,
        "problem": problem.name,
        "input_sha256": hashlib.sha256(snapshot).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            **{
                name: version(name)
                for name in ("pyqpanda3", "pyqpanda_alg", "numpy", "scipy")
            },
        },
        "model": model.audit(),
        "quantum": quantum,
        "exact": exact,
        "milp": milp,
        "uniform_one_hot": uniform,
        "absolute_gap": gap,
        "status": (
            "infeasible"
            if exact["status"] == "infeasible"
            or milp["status"] == "infeasible"
            or quantum["status"] == "infeasible_propagation"
            else "resource_limit"
            if quantum["status"] == "resource_limit"
            else "feasible"
            if quantum_best and quantum_best["feasible"]
            else "no_feasible_sample"
        ),
        "runtime_seconds": perf_counter() - started,
    }
