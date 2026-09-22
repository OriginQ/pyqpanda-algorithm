"""Compact unit-commitment QUBO construction.

The builder only assembles the model.  Sampling and optimization are
delegated to the repository's existing ``QAOA`` implementation so that this
application remains easy to merge upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import sympy as sp

from .data_model import UnitCommitmentInstance
from .registry import VariableKey, VariableRegistry


@dataclass
class QuboAccumulator:
    """Sparse upper-triangular QUBO accumulator for binary variables."""

    constant: float = 0.0
    linear: dict[int, float] = field(default_factory=dict)
    quadratic: dict[tuple[int, int], float] = field(default_factory=dict)

    def add_constant(self, value: float) -> None:
        self.constant += float(value)

    def add_linear(self, index: int, value: float) -> None:
        index = int(index)
        self.linear[index] = self.linear.get(index, 0.0) + float(value)

    def add_product(self, first: int, second: int, value: float) -> None:
        first, second = int(first), int(second)
        if first == second:
            self.add_linear(first, value)
            return
        pair = tuple(sorted((first, second)))
        self.quadratic[pair] = self.quadratic.get(pair, 0.0) + float(value)

    def add_square(self, coefficients: Mapping[int, float], rhs: float, weight: float) -> None:
        """Add ``weight * (sum(coefficients*x) - rhs)**2``."""

        weight = float(weight)
        rhs = float(rhs)
        self.add_constant(weight * rhs * rhs)
        items = [(int(index), float(coefficient)) for index, coefficient in coefficients.items()]
        for index, coefficient in items:
            self.add_linear(index, weight * (coefficient * coefficient - 2.0 * rhs * coefficient))
        for position, (first, first_coefficient) in enumerate(items):
            for second, second_coefficient in items[position + 1 :]:
                self.add_product(first, second, 2.0 * weight * first_coefficient * second_coefficient)

    def energy(self, bits: tuple[int, ...] | list[int] | np.ndarray) -> float:
        values = tuple(int(value) for value in bits)
        if any(value not in (0, 1) for value in values):
            raise ValueError("QUBO bits must be binary")
        result = self.constant
        result += sum(coefficient * values[index] for index, coefficient in self.linear.items())
        result += sum(
            coefficient * values[first] * values[second]
            for (first, second), coefficient in self.quadratic.items()
        )
        return float(result)

    def validate(self, n_variables: int) -> None:
        """Check sparse indices and coefficients before exporting a QUBO."""

        if not np.isfinite(self.constant):
            raise ValueError("QUBO constant must be finite")
        for index, coefficient in self.linear.items():
            if not 0 <= index < n_variables or not np.isfinite(coefficient):
                raise ValueError("QUBO linear term has an invalid index or coefficient")
        for (first, second), coefficient in self.quadratic.items():
            if not 0 <= first < second < n_variables or not np.isfinite(coefficient):
                raise ValueError("QUBO quadratic term has an invalid pair or coefficient")

    def to_expression(self, registry: VariableRegistry) -> sp.Expr:
        symbols = [sp.Symbol(name) for name in registry.names]
        expression: sp.Expr = sp.Float(self.constant)
        for index, coefficient in self.linear.items():
            expression += sp.Float(coefficient) * symbols[index]
        for (first, second), coefficient in self.quadratic.items():
            expression += sp.Float(coefficient) * symbols[first] * symbols[second]
        return sp.expand(expression)


@dataclass
class QuboModel:
    """A QUBO expression plus its stable variable registry."""

    registry: VariableRegistry
    accumulator: QuboAccumulator
    expression: sp.Expr
    components: dict[str, float] = field(default_factory=dict)
    instance: UnitCommitmentInstance | None = None

    @property
    def variable_names(self) -> tuple[str, ...]:
        return self.registry.names

    @property
    def n_variables(self) -> int:
        return len(self.registry)

    def energy(self, bits: tuple[int, ...] | list[int] | np.ndarray) -> float:
        return self.accumulator.energy(bits)

    def as_quadratic_binary(self):
        """Return a small compatibility object with ``function_value``.

        The upstream pyqpanda-algorithm package offers a richer
        ``QuadraticBinary`` class.  Keeping this adapter local means the
        standalone case5 project remains runnable without importing the whole
        upstream distribution.
        """

        symbols = [sp.Symbol(name) for name in self.registry.names]
        evaluator = sp.lambdify(symbols, self.expression, "numpy")

        class _QuadraticBinaryAdapter:
            def function_value(self, values):
                return float(evaluator(*tuple(int(value) for value in values)))

        return _QuadraticBinaryAdapter()

    def to_quadratic_binary(self):
        return self.as_quadratic_binary()


class UnitCommitmentQuboBuilder:
    """Build the compact binary dispatch formulation.

    For each unit and period, ``u`` is the on/off bit and ``z`` is a dispatch
    increment bit.  Dispatch is ``p = p_min*u + (p_max-p_min)*z`` and the
    linking penalty enforces ``z <= u``.  Demand balance and startup terms are
    quadratic, so no additional ancilla or custom optimizer is required.
    """

    def __init__(
        self,
        penalty_weight: float = 50.0,
        balance_weight: float | None = None,
        *,
        profile: str = "compact",
        dispatch_encoding: str = "binary_increment",
        penalty_policy: object | None = None,
    ):
        if profile != "compact":
            raise NotImplementedError(
                "v0.1 implements only the compact-full-QUBO profile"
            )
        if dispatch_encoding not in {"binary_increment", "compact"}:
            raise NotImplementedError(
                "v0.1 implements only binary_increment dispatch encoding"
            )
        policy_balance = None
        if penalty_policy is not None and hasattr(penalty_policy, "weight"):
            penalty_weight = float(penalty_policy.weight)
            policy_balance = getattr(penalty_policy, "balance_weight", None)
        if penalty_weight <= 0.0 or not np.isfinite(penalty_weight):
            raise ValueError("penalty_weight must be positive and finite")
        self.penalty_weight = float(penalty_weight)
        self.profile = profile
        self.dispatch_encoding = dispatch_encoding
        if balance_weight is None:
            balance_weight = policy_balance
        self.balance_weight = self.penalty_weight if balance_weight is None else float(balance_weight)
        if self.balance_weight <= 0.0 or not np.isfinite(self.balance_weight):
            raise ValueError("balance_weight must be positive and finite")

    def build(self, instance: UnitCommitmentInstance) -> QuboModel:
        if any(value > 0.0 for value in instance.reserve):
            raise NotImplementedError(
                "v0.1 compact encoding does not include spinning-reserve constraints"
            )
        registry = VariableRegistry.for_instance(instance)
        accumulator = QuboAccumulator()
        component_totals = {
            "fuel_and_fixed": 0.0,
            "startup": 0.0,
            "link_penalty": 0.0,
            "balance_penalty": 0.0,
        }

        def index(kind: str, unit_name: str, period: int) -> int:
            return registry.index(VariableKey(kind, unit_name, period))

        for unit in instance.units:
            increment = unit.p_max - unit.p_min
            for period in instance.periods:
                u_index = index("u", unit.name, period)
                z_index = index("z", unit.name, period)
                dispatch_form = {u_index: unit.p_min, z_index: increment}

                # a*p^2 + b*p + c*u, with p represented by the two bits above.
                accumulator.add_square(dispatch_form, rhs=0.0, weight=unit.quadratic_cost)
                accumulator.add_linear(u_index, unit.linear_cost * unit.p_min + unit.no_load_cost)
                accumulator.add_linear(z_index, unit.linear_cost * increment)
                component_totals["fuel_and_fixed"] += unit.no_load_cost

                # z*(1-u) is zero for all valid dispatch states.
                accumulator.add_linear(z_index, self.penalty_weight)
                accumulator.add_product(z_index, u_index, -self.penalty_weight)
                component_totals["link_penalty"] += self.penalty_weight

                if period == 0:
                    if not unit.initial_on:
                        accumulator.add_linear(u_index, unit.startup_cost)
                        component_totals["startup"] += unit.startup_cost
                else:
                    previous = index("u", unit.name, period - 1)
                    accumulator.add_linear(u_index, unit.startup_cost)
                    accumulator.add_product(u_index, previous, -unit.startup_cost)

        for period, demand in enumerate(instance.demand):
            dispatch_form = {}
            for unit in instance.units:
                dispatch_form[index("u", unit.name, period)] = unit.p_min
                dispatch_form[index("z", unit.name, period)] = unit.p_max - unit.p_min
            accumulator.add_square(dispatch_form, rhs=demand, weight=self.balance_weight)
            component_totals["balance_penalty"] += self.balance_weight * demand * demand

        accumulator.validate(len(registry))
        expression = accumulator.to_expression(registry)
        return QuboModel(registry, accumulator, expression, component_totals, instance)


__all__ = ["QuboAccumulator", "QuboModel", "UnitCommitmentQuboBuilder"]

