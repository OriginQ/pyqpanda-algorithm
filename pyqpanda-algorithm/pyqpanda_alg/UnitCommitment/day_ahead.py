"""Twenty-four-hour day-ahead scheduling helpers for the case5 benchmark.

The full day-ahead reference keeps the 24-hour commitment and dispatch
variables in one SciPy MILP.  The quantum companion deliberately solves one
five-generator commitment QUBO per hour and then evaluates the resulting
24-hour commitment trajectory with the same dispatch and start-up-cost
routine.  This decomposition keeps each local QAOA circuit at five qubits;
constructing a 120-qubit state-vector circuit would not be a useful local
validation experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .case5 import Case5UCInstance, load_case5_uc
from .case5_classical import ScheduleEvaluation, solve_milp_uc, evaluate_schedule
from .case5_solver import Case5QAOA, Case5QAOAResult


# Normalized to the 1,000 MW MATPOWER case5 base load.  The shape represents
# a reproducible residential/commercial day: overnight valley, morning ramp,
# midday plateau, evening peak, and a late-evening decline.
DEFAULT_DAY_AHEAD_LOAD_SCALES: tuple[float, ...] = (
    0.62,
    0.60,
    0.58,
    0.58,
    0.60,
    0.66,
    0.75,
    0.84,
    0.92,
    0.96,
    0.98,
    1.00,
    0.99,
    0.97,
    0.95,
    0.97,
    1.03,
    1.10,
    1.08,
    1.02,
    0.94,
    0.84,
    0.75,
    0.68,
)


def default_day_ahead_load_scales() -> tuple[float, ...]:
    """Return the immutable 24-hour load-scale profile used in the examples."""

    return DEFAULT_DAY_AHEAD_LOAD_SCALES


def load_case5_day_ahead(
    *,
    load_scales: Sequence[float] | None = None,
    reserve_fraction: float = 0.05,
    startup_costs: Sequence[float] | None = None,
    no_load_costs: Sequence[float] | None = None,
    name: str = "matpower_case5_day_ahead_24h",
) -> Case5UCInstance:
    """Create a 24-hour case5 day-ahead scheduling instance.

    ``load_scales`` is expressed relative to the 1,000 MW case5 base load.
    A spinning-reserve requirement equal to ``reserve_fraction`` of each
    hourly demand is added by default.  The returned instance is compatible
    with :func:`solve_milp_uc`, :func:`evaluate_schedule`, and the existing
    commitment QUBO builder.
    """

    scales = tuple(
        float(value)
        for value in (DEFAULT_DAY_AHEAD_LOAD_SCALES if load_scales is None else load_scales)
    )
    if len(scales) != 24:
        raise ValueError("a day-ahead profile must contain exactly 24 hourly scales")
    if any(not np.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("hourly load scales must be finite and positive")
    reserve_fraction = float(reserve_fraction)
    if not np.isfinite(reserve_fraction) or reserve_fraction < 0.0:
        raise ValueError("reserve_fraction must be finite and non-negative")
    case5 = load_case5_uc(
        demand_scales=scales,
        reserve_mw=tuple(1000.0 * reserve_fraction * value for value in scales),
        startup_costs=startup_costs,
        no_load_costs=no_load_costs,
        name=name,
    )
    return case5


@dataclass(frozen=True)
class HourlyQAOAResult:
    """One five-qubit QAOA result used by the hourly decomposition."""

    hour: int
    qaoa: Case5QAOAResult
    commitments: tuple[int, ...]
    fallback_to_classical: bool

    @property
    def best_cost(self) -> float | None:
        if self.qaoa.best_candidate is None or self.qaoa.best_candidate.schedule is None:
            return None
        return float(self.qaoa.best_candidate.schedule.total_cost)

    def as_dict(self) -> dict[str, object]:
        return {
            "hour": self.hour,
            "commitments": list(self.commitments),
            "fallback_to_classical": self.fallback_to_classical,
            "qaoa": self.qaoa.as_dict(),
        }


@dataclass(frozen=True)
class DayAheadQuantumResult:
    """Aggregated hourly-QAOA commitment trajectory and day dispatch."""

    backend: str
    hourly_results: tuple[HourlyQAOAResult, ...]
    commitments: tuple[tuple[int, ...], ...]
    schedule: ScheduleEvaluation
    decomposition: str = "24 independent five-qubit QAOA solves"

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "decomposition": self.decomposition,
            "commitments": [list(row) for row in self.commitments],
            "schedule": self.schedule.as_dict(),
            "hourly_results": [item.as_dict() for item in self.hourly_results],
        }


def solve_hourly_qaoa(
    instance: Case5UCInstance,
    *,
    backend: str = "statevector",
    shots: int = 256,
    maxiter: int = 4,
    seed: int = 11,
    top_k: int = 8,
    evaluate_method: str = "linprog",
    enforce_network: bool = True,
) -> DayAheadQuantumResult:
    """Solve each hourly commitment QUBO and evaluate the full day schedule.

    The QAOA result is used whenever it contains a physically feasible hourly
    candidate.  A one-hour MILP fallback is recorded explicitly if a sampled
    run fails to produce one, keeping the day-ahead pipeline total and
    auditable instead of silently fabricating a commitment vector.
    """

    if instance.time_periods != 24:
        raise ValueError("solve_hourly_qaoa requires a 24-hour instance")
    if maxiter < 1 or shots < 1 or top_k < 1:
        raise ValueError("maxiter, shots, and top_k must be positive")

    hourly: list[HourlyQAOAResult] = []
    commitments: list[tuple[int, ...]] = []
    for hour in range(24):
        one_hour = load_case5_uc(
            demand_scales=(instance.demand_scales[hour],),
            reserve_mw=(instance.reserve_mw[hour],),
            startup_costs=tuple(unit.startup_cost for unit in instance.units),
            no_load_costs=tuple(unit.no_load_cost for unit in instance.units),
            name=f"{instance.name}_hour_{hour:02d}",
        )
        qaoa = Case5QAOA(
            one_hour,
            backend=backend,
            layers=1,
            shots=None if backend == "statevector" else shots,
            seed=int(seed) + hour,
            optimizer_options={"options": {"maxiter": int(maxiter), "disp": False}},
        ).solve(
            top_k=top_k,
            evaluate_method=evaluate_method,
            enforce_network=enforce_network,
        )
        fallback = qaoa.best_candidate is None
        if fallback:
            classical = solve_milp_uc(
                one_hour,
                evaluate_method=evaluate_method,
                enforce_network=enforce_network,
            )
            if not classical.success or not classical.commitments:
                raise RuntimeError(f"hour {hour} produced no feasible commitment")
            chosen = tuple(int(value) for value in classical.commitments[0])
        else:
            chosen = tuple(int(value) for value in qaoa.best_candidate.commitments[0])
        commitments.append(chosen)
        hourly.append(
            HourlyQAOAResult(
                hour=hour,
                qaoa=qaoa,
                commitments=chosen,
                fallback_to_classical=fallback,
            )
        )

    schedule = evaluate_schedule(
        instance,
        commitments,
        method=evaluate_method,
        enforce_network=enforce_network,
    )
    if not schedule.success:
        raise RuntimeError(f"hourly QAOA schedule failed physical evaluation: {schedule.message}")
    return DayAheadQuantumResult(
        backend=backend,
        hourly_results=tuple(hourly),
        commitments=tuple(commitments),
        schedule=schedule,
    )


__all__ = [
    "DEFAULT_DAY_AHEAD_LOAD_SCALES",
    "DayAheadQuantumResult",
    "HourlyQAOAResult",
    "default_day_ahead_load_scales",
    "load_case5_day_ahead",
    "solve_hourly_qaoa",
]

