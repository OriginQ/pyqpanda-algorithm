"""Reproducible offline benchmark for QShadow planning and shot budgets.

This script compares uniform and observable-aware local-Pauli plans on the same
state, observables, shot counts, confidence level, and fixed seed blocks.  It
uses CPUQVM once to obtain the exact statevector and NumPy sampling afterwards;
it never connects to a cloud backend.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pyqpanda3.core import CNOT, CPUQVM, QCircuit, QProg, RY

from pyqpanda_alg.QShadow import (
    PauliObservable,
    ShadowBudgetController,
    ShadowDataset,
    estimate_observables,
    optimize_basis_probabilities,
    planning_loss,
    sample_measurement_plan,
    simulate_pauli_measurements,
    uniform_probabilities,
)


def prepare(qubits):
    """Prepare the same three-qubit state used by the QShadow example."""

    return (
        QCircuit()
        << RY(qubits[0], 0.7)
        << RY(qubits[1], -0.4)
        << RY(qubits[2], 0.3)
        << CNOT(qubits[0], qubits[1])
        << CNOT(qubits[1], qubits[2])
    )


def exact_statevector() -> list[complex]:
    """Return the CPUQVM statevector before measurement."""

    program = QProg(3)
    program << prepare(program.qubits())
    machine = CPUQVM()
    machine.run(program, shots=1)
    return [complex(value) for value in machine.result().get_state_vector()]


def benchmark_observables() -> list[PauliObservable]:
    """Return the fixed observable family used by this benchmark."""

    return [
        PauliObservable.from_terms(
            {
                "ZZI": -1.0,
                "IZZ": -0.8,
                "XII": 0.20,
                "IXI": 0.15,
                "IIX": 0.10,
            },
            name="structured_energy",
        ),
        PauliObservable.from_terms(
            {"ZIZ": 1.0, "XXI": 0.25}, name="long_range_order"
        ),
    ]


def _fixed_shot_statistics(
    *,
    state: Sequence[complex],
    observables: list[PauliObservable],
    probabilities: dict[str, list[list[float]]],
    trials: int,
    shots: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    exact = np.asarray(
        [observable.exact_expectation(state) for observable in observables]
    )
    results: dict[str, Any] = {}
    for method_index, (method, method_probabilities) in enumerate(
        probabilities.items()
    ):
        values: list[list[float]] = []
        half_widths: list[list[float]] = []
        covered: list[list[bool]] = []
        for trial in range(trials):
            offset = method_index * 200_000 + trial
            plan = sample_measurement_plan(
                method_probabilities, shots, seed=seed + offset
            )
            dataset = simulate_pauli_measurements(
                state, plan, seed=seed + 500_000 + offset
            )
            estimates = estimate_observables(
                dataset, observables, family_confidence=confidence
            )
            values.append(
                [estimates[observable.name].value for observable in observables]
            )
            half_widths.append(
                [
                    estimates[observable.name].half_width
                    for observable in observables
                ]
            )
            covered.append(
                [
                    estimates[observable.name].lower
                    <= exact_value
                    <= estimates[observable.name].upper
                    for observable, exact_value in zip(observables, exact)
                ]
            )

        value_array = np.asarray(values)
        width_array = np.asarray(half_widths)
        covered_array = np.asarray(covered, dtype=bool)
        errors = value_array - exact
        per_observable: dict[str, Any] = {}
        for index, observable in enumerate(observables):
            bias = float(np.mean(errors[:, index]))
            bias_standard_error = float(
                np.std(errors[:, index], ddof=1) / np.sqrt(trials)
            )
            per_observable[observable.name] = {
                "bias": bias,
                "bias_standard_error": bias_standard_error,
                "bias_z_score": bias / bias_standard_error,
                "rmse": float(np.sqrt(np.mean(errors[:, index] ** 2))),
                "mean_absolute_error": float(np.mean(np.abs(errors[:, index]))),
                "mean_half_width": float(np.mean(width_array[:, index])),
                "empirical_coverage": float(np.mean(covered_array[:, index])),
            }
        results[method] = {
            "per_observable": per_observable,
            "aggregate_rmse": float(np.sqrt(np.mean(errors**2))),
            "family_wise_empirical_coverage": float(
                np.mean(np.all(covered_array, axis=1))
            ),
        }
    return {"exact_values": dict(zip((o.name for o in observables), exact)), **results}


def _sequential_budget_statistics(
    *,
    state: Sequence[complex],
    observables: list[PauliObservable],
    probabilities: dict[str, list[list[float]]],
    trials: int,
    target_half_width: float,
    confidence: float,
    min_shots: int,
    max_shots: int,
    batch_shots: int,
    seed: int,
) -> dict[str, Any]:
    exact = {
        observable.name: observable.exact_expectation(state)
        for observable in observables
    }
    results: dict[str, Any] = {}
    for method_index, (method, method_probabilities) in enumerate(
        probabilities.items()
    ):
        final_shots: list[int] = []
        rounds: list[int] = []
        completed: list[bool] = []
        family_covered: list[bool] = []
        hard_max_violations = 0
        width_rule_violations = 0
        for trial in range(trials):
            controller = ShadowBudgetController(
                observables,
                target_half_width=target_half_width,
                family_confidence=confidence,
                min_shots=min_shots,
                max_shots=max_shots,
                batch_shots=batch_shots,
            )
            datasets: list[ShadowDataset] = []
            next_shots = min_shots
            step = 0
            while True:
                offset = method_index * 500_000 + trial * 50 + step
                plan = sample_measurement_plan(
                    method_probabilities,
                    next_shots,
                    seed=seed + 2_000_000 + offset,
                )
                datasets.append(
                    simulate_pauli_measurements(
                        state,
                        plan,
                        seed=seed + 3_000_000 + offset,
                    )
                )
                dataset = ShadowDataset.concatenate(*datasets)
                report = controller.evaluate(dataset)
                step += 1
                if dataset.shots > max_shots:
                    hard_max_violations += 1
                if report.complete:
                    if any(
                        estimate.half_width > target_half_width + 1e-12
                        for estimate in report.estimates.values()
                    ):
                        width_rule_violations += 1
                    break
                if report.exhausted:
                    break
                next_shots = report.next_batch_shots
                if step > 20:
                    raise RuntimeError("budget controller did not terminate")

            final_shots.append(dataset.shots)
            rounds.append(step)
            completed.append(report.complete)
            family_covered.append(
                all(
                    estimate.lower <= exact[name] <= estimate.upper
                    for name, estimate in report.estimates.items()
                )
            )

        shot_array = np.asarray(final_shots)
        results[method] = {
            "target_reached_rate": float(np.mean(completed)),
            "family_wise_empirical_coverage": float(np.mean(family_covered)),
            "hard_max_violations": hard_max_violations,
            "width_rule_violations": width_rule_violations,
            "final_shots": {
                "mean": float(np.mean(shot_array)),
                "median": float(np.median(shot_array)),
                "p10": float(np.quantile(shot_array, 0.1)),
                "p90": float(np.quantile(shot_array, 0.9)),
                "minimum": int(np.min(shot_array)),
                "maximum": int(np.max(shot_array)),
            },
            "mean_rounds": float(np.mean(rounds)),
        }
    return results


def run_benchmark(
    *,
    trials: int = 200,
    shots: int = 2_000,
    budget_trials: int = 100,
    target_half_width: float = 0.22,
    confidence: float = 0.95,
    min_shots: int = 500,
    max_shots: int = 12_000,
    batch_shots: int = 250,
    seed: int = 2_000_000,
) -> dict[str, Any]:
    """Run the fixed-shot and sequential-budget benchmark."""

    if trials <= 1 or budget_trials <= 0 or shots <= 0:
        raise ValueError("trials must exceed one; budget_trials and shots must be positive")
    state = exact_statevector()
    observables = benchmark_observables()
    uniform = uniform_probabilities(3).tolist()
    optimized = optimize_basis_probabilities(
        observables, probability_floor=0.04
    ).probabilities.tolist()
    probabilities = {"uniform": uniform, "optimized": optimized}
    fixed = _fixed_shot_statistics(
        state=state,
        observables=observables,
        probabilities=probabilities,
        trials=trials,
        shots=shots,
        confidence=confidence,
        seed=seed,
    )
    budget = _sequential_budget_statistics(
        state=state,
        observables=observables,
        probabilities=probabilities,
        trials=budget_trials,
        target_half_width=target_half_width,
        confidence=confidence,
        min_shots=min_shots,
        max_shots=max_shots,
        batch_shots=batch_shots,
        seed=seed,
    )

    uniform_fixed = fixed["uniform"]
    optimized_fixed = fixed["optimized"]
    rmse_reduction = {
        observable.name: 1.0
        - optimized_fixed["per_observable"][observable.name]["rmse"]
        / uniform_fixed["per_observable"][observable.name]["rmse"]
        for observable in observables
    }
    width_reduction = {
        observable.name: 1.0
        - optimized_fixed["per_observable"][observable.name]["mean_half_width"]
        / uniform_fixed["per_observable"][observable.name]["mean_half_width"]
        for observable in observables
    }
    shot_reduction = 1.0 - (
        budget["optimized"]["final_shots"]["mean"]
        / budget["uniform"]["final_shots"]["mean"]
    )
    return {
        "configuration": {
            "fixed_shot_trials": trials,
            "shots_per_fixed_trial": shots,
            "budget_trials": budget_trials,
            "target_half_width": target_half_width,
            "family_confidence": confidence,
            "min_shots": min_shots,
            "max_shots": max_shots,
            "batch_shots": batch_shots,
            "seed": seed,
        },
        "planning_loss": {
            "uniform": planning_loss(uniform, observables),
            "optimized": planning_loss(optimized, observables),
        },
        "fixed_shot_statistics": fixed,
        "sequential_budget_statistics": budget,
        "optimized_reduction_fraction": {
            "aggregate_rmse": 1.0
            - optimized_fixed["aggregate_rmse"]
            / uniform_fixed["aggregate_rmse"],
            "rmse_by_observable": rmse_reduction,
            "mean_half_width_by_observable": width_reduction,
            "mean_final_shots_to_target": shot_reduction,
        },
        "interpretation": (
            "Observed coverage is an empirical diagnostic, not proof of exact "
            "95% calibration. The comparison is specific to this fixed state, "
            "observable family, probability floor, and seed schedule."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--shots", type=int, default=2_000)
    parser.add_argument("--budget-trials", type=int, default=100)
    parser.add_argument("--target-half-width", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_benchmark(
        trials=args.trials,
        shots=args.shots,
        budget_trials=args.budget_trials,
        target_half_width=args.target_half_width,
        seed=args.seed,
    )
    payload = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
