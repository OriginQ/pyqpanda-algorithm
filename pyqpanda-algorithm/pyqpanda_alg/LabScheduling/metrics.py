"""Post-training diagnostics: optimum probability and native circuit resources."""

from __future__ import annotations

from itertools import product
from math import ceil, log, log1p, prod
from typing import Any

import numpy as np

from .model import Model, evaluate
from .quantum import build_circuit

SHOT_BUDGETS = (8, 16, 32, 64, 128, 512)


def hit_probability(probability: float, shots: int) -> float:
    """Probability of at least one hit in independent draws from a fixed state."""
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if type(shots) is not int or shots < 1:
        raise ValueError("shots must be a positive integer")
    return float(-np.expm1(shots * np.log1p(-probability))) if probability < 1 else 1.0


def distribution_metrics(model: Model, probabilities: np.ndarray) -> dict[str, Any]:
    """Enumerate schedules AFTER training; count all tied optima, not one witness.

    Hit curves condition on this already trained noiseless state. They exclude
    training and statevector diagnostic costs and are not end-to-end speedups.
    """
    if (
        probabilities.shape != (1 << len(model.variables),)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise ValueError("expected a normalized finite nonnegative probability vector")
    combinations = prod(map(len, model.domains))
    if combinations > 1_000_000:
        raise ValueError("distribution certification exceeds 1000000 combinations")
    optimum = None
    optimal_indices: list[int] = []
    feasible_count = 0
    feasible_probability = 0.0
    for choice in product(*model.domains):
        bits = [0] * len(model.variables)
        for i in choice:
            bits[i] = 1
        row = evaluate(model, bits)
        if not row["feasible"]:
            continue
        index = sum(1 << i for i in choice)
        feasible_count += 1
        feasible_probability += float(probabilities[index])
        cost = row["objective"]
        if optimum is None or cost < optimum:
            optimum, optimal_indices = cost, [index]
        elif cost == optimum:
            optimal_indices.append(index)
    probability = min(1.0, float(probabilities[optimal_indices].sum()))
    uniform = len(optimal_indices) / combinations if combinations else 0.0
    return {
        "exact_objective": optimum,
        "optimal_count": len(optimal_indices),
        "combinations": combinations,
        "optimal_probability": probability,
        "feasible_probability": min(1.0, feasible_probability),
        "uniform_optimal_probability": uniform,
        "uniform_feasible_probability": feasible_count / combinations
        if combinations
        else 0,
        "shots_for_95_percent": (
            1
            if probability == 1
            else ceil(log(0.05) / log1p(-probability))
            if probability > 0
            else None
        ),
        "hit_curve": [
            {
                "shots": s,
                "quantum": hit_probability(probability, s),
                "uniform": hit_probability(uniform, s),
            }
            for s in SHOT_BUDGETS
        ],
        "interpretation": "conditional on trained state; training cost excluded",
    }


def circuit_resources(
    model: Model,
    parameters: list[float],
    mixer: str = "xy",
) -> dict[str, Any]:
    """Compare native CPU circuit operations; CRY is not decomposed into CNOTs."""
    result = {}
    for mode in ("full", "auto"):
        program = build_circuit(model, parameters, mixer, phase_mode=mode)
        operations = dict(program.count_ops())
        result[mode] = {
            "depth": program.depth(),
            "operations": operations,
            "cnot": operations.get("CNOT", 0),
            "cry": operations.get("CRY", 0),
        }
    return result
