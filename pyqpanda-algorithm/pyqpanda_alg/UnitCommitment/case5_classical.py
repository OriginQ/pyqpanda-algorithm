"""Classical optimization baselines for the MATPOWER case5 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .case5 import Case5UCInstance


def _as_commitment_matrix(
    instance: Case5UCInstance,
    commitments: Sequence[Sequence[int]],
) -> np.ndarray:
    matrix = np.asarray(commitments, dtype=int)
    expected = (instance.time_periods, len(instance.units))
    if matrix.shape != expected:
        raise ValueError(f"commitments must have shape {expected}")
    if np.any((matrix != 0) & (matrix != 1)):
        raise ValueError("commitments must be binary")
    return matrix


@dataclass(frozen=True)
class EconomicDispatchResult:
    """Continuous dispatch result for one period."""

    method: str
    success: bool
    generation_mw: tuple[float, ...]
    variable_cost: float
    fixed_cost: float
    branch_flows_mw: tuple[float, ...]
    equality_residual_mw: float
    max_line_violation_mw: float
    message: str

    @property
    def total_cost(self) -> float:
        return float(self.variable_cost + self.fixed_cost)

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "success": self.success,
            "generation_mw": list(self.generation_mw),
            "variable_cost": self.variable_cost,
            "fixed_cost": self.fixed_cost,
            "total_cost": self.total_cost,
            "branch_flows_mw": list(self.branch_flows_mw),
            "equality_residual_mw": self.equality_residual_mw,
            "max_line_violation_mw": self.max_line_violation_mw,
            "message": self.message,
        }


@dataclass(frozen=True)
class ScheduleEvaluation:
    """Dispatch and physical cost for a complete commitment schedule."""

    success: bool
    method: str
    total_cost: float
    commitment_cost: float
    variable_cost: float
    dispatch: tuple[EconomicDispatchResult, ...]
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "method": self.method,
            "total_cost": self.total_cost,
            "commitment_cost": self.commitment_cost,
            "variable_cost": self.variable_cost,
            "message": self.message,
            "dispatch": [item.as_dict() for item in self.dispatch],
        }


@dataclass(frozen=True)
class ClassicalUCResult:
    """MILP commitment result and its continuous dispatch evaluation."""

    success: bool
    message: str
    commitments: tuple[tuple[int, ...], ...]
    objective: float
    schedule: ScheduleEvaluation
    solver: str = "scipy.optimize.milp"

    def as_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "message": self.message,
            "solver": self.solver,
            "commitments": [list(row) for row in self.commitments],
            "objective": self.objective,
            "schedule": self.schedule.as_dict(),
        }


def _network_constraint_data(
    instance: Case5UCInstance,
    period: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, float, float]]]:
    case = instance.case
    load = np.asarray(instance.load_profiles_mw[period], dtype=float)
    generator_bus = case.generator_bus_matrix()
    ptdf = case.ptdf()
    generation_flow = ptdf @ generator_bus
    offset = -(ptdf @ load)
    constrained: list[tuple[np.ndarray, float, float]] = []
    for index, branch in enumerate(case.branches):
        if branch.rate_a_mva <= 0.0:
            continue
        constrained.append(
            (
                np.asarray(generation_flow[index], dtype=float),
                float(offset[index]),
                float(branch.rate_a_mva),
            )
        )
    return generation_flow, offset, constrained


def _initial_dispatch(
    instance: Case5UCInstance,
    commitment: np.ndarray,
    demand: float,
) -> np.ndarray | None:
    lower = np.array([unit.p_min_mw * commitment[index] for index, unit in enumerate(instance.units)])
    upper = np.array([unit.p_max_mw * commitment[index] for index, unit in enumerate(instance.units)])
    if demand < float(np.sum(lower)) - 1.0e-8 or demand > float(np.sum(upper)) + 1.0e-8:
        return None
    dispatch = lower.copy()
    remaining = float(demand - np.sum(dispatch))
    order = sorted(
        range(len(instance.units)),
        key=lambda index: (instance.units[index].linear_cost, instance.units[index].name),
    )
    for index in order:
        amount = min(remaining, upper[index] - dispatch[index])
        if amount > 0.0:
            dispatch[index] += amount
            remaining -= amount
        if remaining <= 1.0e-8:
            break
    return dispatch if remaining <= 1.0e-7 else None


def economic_dispatch(
    instance: Case5UCInstance,
    commitment: Sequence[int],
    period: int,
    *,
    method: str = "slsqp",
    enforce_network: bool = True,
    tolerance: float = 1.0e-6,
) -> EconomicDispatchResult:
    """Solve one period of convex economic dispatch.

    ``linprog`` is used for the linear MATPOWER cost rows; ``slsqp`` also
    handles optional quadratic coefficients and is used by default for the
    hybrid QAOA pipeline.  Both methods enforce generator bounds, power
    balance, and the two finite thermal line limits in case5.
    """

    if not 0 <= int(period) < instance.time_periods:
        raise ValueError("period is outside the instance horizon")
    commitment_array = np.asarray(tuple(int(value) for value in commitment), dtype=int)
    if commitment_array.shape != (len(instance.units),) or np.any(
        (commitment_array != 0) & (commitment_array != 1)
    ):
        raise ValueError("commitment must be a binary vector with one entry per generator")
    demand = float(instance.demand_mw[period])
    lower = np.array(
        [unit.p_min_mw * commitment_array[index] for index, unit in enumerate(instance.units)],
        dtype=float,
    )
    upper = np.array(
        [unit.p_max_mw * commitment_array[index] for index, unit in enumerate(instance.units)],
        dtype=float,
    )
    if demand < float(np.sum(lower)) - tolerance or demand > float(np.sum(upper)) + tolerance:
        return EconomicDispatchResult(
            method=str(method).lower(),
            success=False,
            generation_mw=tuple(float(value) for value in lower),
            variable_cost=float("inf"),
            fixed_cost=float(
                sum(unit.no_load_cost * commitment_array[index] for index, unit in enumerate(instance.units))
            ),
            branch_flows_mw=tuple(),
            equality_residual_mw=float(np.sum(lower) - demand),
            max_line_violation_mw=float("inf"),
            message="committed capacity cannot satisfy the period demand",
        )

    _, offset, constrained = _network_constraint_data(instance, period)
    method_name = str(method).strip().lower()
    if method_name not in {"slsqp", "linprog"}:
        raise ValueError("method must be 'slsqp' or 'linprog'")
    if method_name == "linprog" and any(abs(unit.quadratic_cost) > 1.0e-12 for unit in instance.units):
        raise ValueError("linprog is only valid when all quadratic costs are zero")

    generation: np.ndarray | None = None
    success = False
    message = ""
    if method_name == "linprog":
        from scipy.optimize import linprog

        a_ub: list[np.ndarray] = []
        b_ub: list[float] = []
        if enforce_network:
            for row, line_offset, limit in constrained:
                a_ub.extend((row, -row))
                b_ub.extend((limit - line_offset, limit + line_offset))
        result = linprog(
            c=np.array([unit.linear_cost for unit in instance.units], dtype=float),
            A_ub=np.asarray(a_ub) if a_ub else None,
            b_ub=np.asarray(b_ub) if b_ub else None,
            A_eq=np.ones((1, len(instance.units)), dtype=float),
            b_eq=np.array([demand], dtype=float),
            bounds=list(zip(lower, upper)),
            method="highs",
        )
        success = bool(result.success)
        message = str(result.message)
        if success:
            generation = np.asarray(result.x, dtype=float)
    else:
        from scipy.optimize import minimize

        initial = _initial_dispatch(instance, commitment_array, demand)
        if initial is None:
            return EconomicDispatchResult(
                method=method_name,
                success=False,
                generation_mw=tuple(float(value) for value in lower),
                variable_cost=float("inf"),
                fixed_cost=float(
                    sum(unit.no_load_cost * commitment_array[index] for index, unit in enumerate(instance.units))
                ),
                branch_flows_mw=tuple(),
                equality_residual_mw=float(np.sum(lower) - demand),
                max_line_violation_mw=float("inf"),
                message="could not construct an initial dispatch",
            )
        quadratic = np.array([unit.quadratic_cost for unit in instance.units], dtype=float)
        linear = np.array([unit.linear_cost for unit in instance.units], dtype=float)
        constraints: list[dict[str, object]] = [
            {
                "type": "eq",
                "fun": lambda values: float(np.sum(values) - demand),
                "jac": lambda values: np.ones(len(instance.units), dtype=float),
            }
        ]
        if enforce_network:
            for row, line_offset, limit in constrained:
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda values, row=row, line_offset=line_offset, limit=limit: float(
                            limit - (np.dot(row, values) + line_offset)
                        ),
                        "jac": lambda values, row=row: -row,
                    }
                )
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda values, row=row, line_offset=line_offset, limit=limit: float(
                            limit + (np.dot(row, values) + line_offset)
                        ),
                        "jac": lambda values, row=row: row,
                    }
                )
        result = minimize(
            lambda values: float(np.dot(quadratic, values * values) + np.dot(linear, values)),
            initial,
            jac=lambda values: 2.0 * quadratic * values + linear,
            bounds=list(zip(lower, upper)),
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1.0e-9, "maxiter": 500, "disp": False},
        )
        success = bool(result.success)
        message = str(result.message)
        if success:
            generation = np.asarray(result.x, dtype=float)

    if generation is None:
        return EconomicDispatchResult(
            method=method_name,
            success=False,
            generation_mw=tuple(float(value) for value in lower),
            variable_cost=float("inf"),
            fixed_cost=float(
                sum(unit.no_load_cost * commitment_array[index] for index, unit in enumerate(instance.units))
            ),
            branch_flows_mw=tuple(),
            equality_residual_mw=float(np.sum(lower) - demand),
            max_line_violation_mw=float("inf"),
            message=message,
        )

    generation = np.clip(generation, lower, upper)
    load = instance.load_profiles_mw[period]
    try:
        flows = instance.case.branch_flows(generation, load)
    except ValueError as error:
        flows = np.array([], dtype=float)
        success = False
        message = str(error)
    limits = np.asarray(instance.case.line_limits_mw(), dtype=float)
    finite_limits = limits > 0.0
    if flows.size and enforce_network:
        max_violation = float(
            np.max(np.maximum(np.abs(flows[finite_limits]) - limits[finite_limits], 0.0))
            if np.any(finite_limits)
            else 0.0
        )
    elif flows.size:
        max_violation = 0.0
    else:
        max_violation = float("inf")
    equality_residual = float(np.sum(generation) - demand)
    if abs(equality_residual) > tolerance or max_violation > tolerance:
        success = False
        message = message or "dispatch violates balance or a thermal line limit"
    quadratic = np.array([unit.quadratic_cost for unit in instance.units], dtype=float)
    linear = np.array([unit.linear_cost for unit in instance.units], dtype=float)
    variable_cost = float(np.dot(quadratic, generation * generation) + np.dot(linear, generation))
    fixed_cost = float(
        sum(unit.no_load_cost * commitment_array[index] for index, unit in enumerate(instance.units))
    )
    return EconomicDispatchResult(
        method=method_name,
        success=success,
        generation_mw=tuple(float(value) for value in generation),
        variable_cost=variable_cost if success else float("inf"),
        fixed_cost=fixed_cost,
        branch_flows_mw=tuple(float(value) for value in flows),
        equality_residual_mw=equality_residual,
        max_line_violation_mw=max_violation,
        message=message,
    )


def evaluate_schedule(
    instance: Case5UCInstance,
    commitments: Sequence[Sequence[int]],
    *,
    method: str = "slsqp",
    enforce_network: bool = True,
) -> ScheduleEvaluation:
    """Run economic dispatch for every period and add commitment costs."""

    matrix = _as_commitment_matrix(instance, commitments)
    dispatch_results: list[EconomicDispatchResult] = []
    variable_cost = 0.0
    commitment_cost = 0.0
    for period in instance.periods:
        result = economic_dispatch(
            instance,
            matrix[period],
            period,
            method=method,
            enforce_network=enforce_network,
        )
        dispatch_results.append(result)
        if not result.success:
            return ScheduleEvaluation(
                success=False,
                method=str(method).lower(),
                total_cost=float("inf"),
                commitment_cost=commitment_cost,
                variable_cost=float("inf"),
                dispatch=tuple(dispatch_results),
                message=f"period {period}: {result.message}",
            )
        variable_cost += result.variable_cost
        commitment_cost += result.fixed_cost
        for index, unit in enumerate(instance.units):
            previous = unit.initial_on if period == 0 else int(matrix[period - 1, index])
            commitment_cost += unit.startup_cost * int(matrix[period, index]) * (1 - int(previous))
    return ScheduleEvaluation(
        success=True,
        method=str(method).lower(),
        total_cost=float(variable_cost + commitment_cost),
        commitment_cost=float(commitment_cost),
        variable_cost=float(variable_cost),
        dispatch=tuple(dispatch_results),
        message="optimal dispatch found for every period",
    )


def _proxy_cost(instance: Case5UCInstance, index: int, fraction: float) -> float:
    unit = instance.units[index]
    return float(
        fraction
        * (unit.quadratic_cost * unit.p_max_mw**2 + unit.linear_cost * unit.p_max_mw)
        / instance.case.base_mva
    )


def solve_milp_uc(
    instance: Case5UCInstance,
    *,
    fuel_proxy_fraction: float = 0.0,
    time_limit: float | None = None,
    evaluate_method: str = "slsqp",
    enforce_network: bool = True,
) -> ClassicalUCResult:
    """Solve commitment and linear economic dispatch jointly with MILP.

    This is a genuine optimization baseline, not enumeration.  Binary
    commitment/startup variables and continuous generator outputs are solved
    in one model.  The model contains power balance, generator bounds,
    spinning-reserve adequacy, DC line limits, and the linear MATPOWER energy
    costs.  A separate dispatch call below re-evaluates the returned schedule
    so that the QAOA and MILP results use the same reporting path.
    """

    if fuel_proxy_fraction < 0.0 or not np.isfinite(fuel_proxy_fraction):
        raise ValueError("fuel_proxy_fraction must be finite and non-negative")
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as error:  # pragma: no cover - depends on SciPy version
        raise RuntimeError("SciPy >= 1.9 with scipy.optimize.milp is required") from error

    generators = instance.units
    period_values = instance.periods
    n_periods = instance.time_periods
    count = len(generators)
    if any(abs(unit.quadratic_cost) > 1.0e-12 for unit in generators):
        raise ValueError("the MILP baseline requires zero quadratic costs; use SLSQP for a quadratic case")
    n_commitment = count * n_periods
    n_startup = n_commitment
    n_dispatch = n_commitment
    n_variables = n_commitment + n_startup + n_dispatch

    def u_index(period: int, generator: int) -> int:
        return period * count + generator

    def v_index(period: int, generator: int) -> int:
        return n_commitment + period * count + generator

    def p_index(period: int, generator: int) -> int:
        return n_commitment + n_startup + period * count + generator

    objective = np.zeros(n_variables, dtype=float)
    for period in period_values:
        for generator_index, unit in enumerate(generators):
            objective[u_index(period, generator_index)] = unit.no_load_cost + _proxy_cost(
                instance, generator_index, fuel_proxy_fraction
            )
            objective[v_index(period, generator_index)] = unit.startup_cost
            objective[p_index(period, generator_index)] = unit.linear_cost

    rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for period, demand in enumerate(instance.demand_mw):
        # Exact active-power balance.
        row = np.zeros(n_variables, dtype=float)
        row[[p_index(period, index) for index in range(count)]] = 1.0
        rows.append(row)
        lower_bounds.append(float(demand))
        upper_bounds.append(float(demand))
        # Spinning-reserve adequacy: committed headroom must cover reserve.
        reserve = float(instance.reserve_mw[period])
        if reserve > 0.0:
            reserve_row = np.zeros(n_variables, dtype=float)
            reserve_row[[u_index(period, index) for index in range(count)]] = [
                unit.p_max_mw for unit in generators
            ]
            reserve_row[[p_index(period, index) for index in range(count)]] = -1.0
            rows.append(reserve_row)
            lower_bounds.append(reserve)
            upper_bounds.append(float("inf"))
        # Generator on/off linking bounds.
        for generator_index, unit in enumerate(generators):
            upper_row = np.zeros(n_variables, dtype=float)
            upper_row[p_index(period, generator_index)] = 1.0
            upper_row[u_index(period, generator_index)] = -unit.p_max_mw
            rows.append(upper_row)
            lower_bounds.append(-float("inf"))
            upper_bounds.append(0.0)
            lower_row = np.zeros(n_variables, dtype=float)
            lower_row[p_index(period, generator_index)] = 1.0
            lower_row[u_index(period, generator_index)] = -unit.p_min_mw
            rows.append(lower_row)
            lower_bounds.append(0.0)
            upper_bounds.append(float("inf"))
        if enforce_network:
            _, offset, constrained = _network_constraint_data(instance, period)
            generator_flow = instance.case.ptdf() @ instance.case.generator_bus_matrix()
            for branch_index, branch in enumerate(instance.case.branches):
                if branch.rate_a_mva <= 0.0:
                    continue
                flow_row = np.zeros(n_variables, dtype=float)
                flow_row[[p_index(period, index) for index in range(count)]] = generator_flow[branch_index]
                rows.append(flow_row)
                lower_bounds.append(-float(branch.rate_a_mva) - float(offset[branch_index]))
                upper_bounds.append(float(branch.rate_a_mva) - float(offset[branch_index]))
    for period in period_values:
        for generator_index, unit in enumerate(generators):
            row = np.zeros(n_variables, dtype=float)
            row[v_index(period, generator_index)] = 1.0
            row[u_index(period, generator_index)] = -1.0
            if period == 0:
                rows.append(row)
                lower_bounds.append(-float(unit.initial_on))
                upper_bounds.append(float("inf"))
                # If initially on, a startup bit is not required at t=0.
                row_upper = np.zeros(n_variables, dtype=float)
                row_upper[v_index(period, generator_index)] = 1.0
                rows.append(row_upper)
                lower_bounds.append(-float("inf"))
                upper_bounds.append(1.0 - float(unit.initial_on))
            else:
                row[u_index(period - 1, generator_index)] = 1.0
                rows.append(row)
                lower_bounds.append(0.0)
                upper_bounds.append(float("inf"))
                row_upper = np.zeros(n_variables, dtype=float)
                row_upper[v_index(period, generator_index)] = 1.0
                row_upper[u_index(period, generator_index)] = -1.0
                rows.append(row_upper)
                lower_bounds.append(-float("inf"))
                upper_bounds.append(0.0)
                row_prev = np.zeros(n_variables, dtype=float)
                row_prev[v_index(period, generator_index)] = 1.0
                row_prev[u_index(period - 1, generator_index)] = 1.0
                rows.append(row_prev)
                lower_bounds.append(-float("inf"))
                upper_bounds.append(1.0)

    constraints = LinearConstraint(np.asarray(rows), np.asarray(lower_bounds), np.asarray(upper_bounds))
    options: dict[str, object] = {"disp": False}
    if time_limit is not None:
        if time_limit <= 0.0 or not np.isfinite(time_limit):
            raise ValueError("time_limit must be positive and finite")
        options["time_limit"] = float(time_limit)
    variable_lower = np.zeros(n_variables, dtype=float)
    variable_upper = np.ones(n_variables, dtype=float)
    for period in period_values:
        for generator_index, unit in enumerate(generators):
            variable_lower[p_index(period, generator_index)] = 0.0
            variable_upper[p_index(period, generator_index)] = unit.p_max_mw
    result = milp(
        c=objective,
        # Commitment and startup bits are binary; dispatch variables remain
        # continuous (integrality=0) in the mixed-integer linear model.
        integrality=np.concatenate(
            (np.ones(n_commitment + n_startup, dtype=int), np.zeros(n_dispatch, dtype=int))
        ),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=constraints,
        options=options,
    )
    if not bool(result.success) or result.x is None:
        empty = tuple(tuple(0 for _ in generators) for _ in period_values)
        schedule = evaluate_schedule(instance, empty, method=evaluate_method, enforce_network=enforce_network)
        return ClassicalUCResult(False, str(result.message), empty, float("inf"), schedule)
    raw_solution = np.asarray(result.x, dtype=float)
    solution = np.rint(raw_solution[:n_commitment]).astype(int)
    commitments = tuple(
        tuple(int(solution[u_index(period, generator)]) for generator in range(count))
        for period in period_values
    )
    schedule = evaluate_schedule(
        instance,
        commitments,
        method=evaluate_method,
        enforce_network=enforce_network,
    )
    return ClassicalUCResult(
        success=True,
        message=str(result.message),
        commitments=commitments,
        objective=float(result.fun),
        schedule=schedule,
    )


__all__ = [
    "ClassicalUCResult",
    "EconomicDispatchResult",
    "ScheduleEvaluation",
    "economic_dispatch",
    "evaluate_schedule",
    "solve_milp_uc",
]

