"""Commitment-only QUBO for the MATPOWER 5-bus benchmark.

The dispatch variables are intentionally eliminated from this QUBO.  A bit
``u[g,t]`` means that generator ``g`` is available in period ``t``.  The
continuous dispatch is recovered by :mod:`case5_classical` after QAOA.  This
keeps the five-generator, two-period experiment at ten logical qubits while
retaining a physically meaningful economic-dispatch check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .case5 import Case5UCInstance
from .qubo import QuboAccumulator, QuboModel
from .registry import VariableKey, VariableRegistry


@dataclass
class Case5CommitmentQuboBuilder:
    """Build a normalized QUBO for case5 generator commitment.

    The squared capacity term is a compact penalty surrogate,

    ``W * (sum(Pmax*u)/baseMVA - (demand+reserve)/baseMVA)**2``.

    It favors schedules whose committed capacity tracks the demand target.
    The independent MILP comparator enforces capacity adequacy as a hard
    inequality; therefore reported comparisons use physical feasibility and
    economic-dispatch cost rather than raw QUBO energy.
    """

    capacity_weight: float = 200.0
    objective_scale: float = 1.0
    fuel_proxy_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("capacity_weight", self.capacity_weight),
            ("objective_scale", self.objective_scale),
            ("fuel_proxy_fraction", self.fuel_proxy_fraction),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.capacity_weight <= 0.0:
            raise ValueError("capacity_weight must be positive")
        if self.objective_scale <= 0.0:
            raise ValueError("objective_scale must be positive")
        if self.fuel_proxy_fraction < 0.0:
            raise ValueError("fuel_proxy_fraction must be non-negative")

    def build(self, instance: Case5UCInstance) -> QuboModel:
        registry = VariableRegistry.for_commitment(
            instance.generator_names,
            instance.time_periods,
        )
        accumulator = QuboAccumulator()
        components = {
            "commitment_cost": 0.0,
            "startup_cost": 0.0,
            "capacity_penalty": 0.0,
            "fuel_proxy": 0.0,
        }

        def index(generator_name: str, period: int) -> int:
            return registry.index(VariableKey("u", generator_name, period))

        for generator in instance.units:
            for period in instance.periods:
                variable = index(generator.name, period)
                # No-load and optional fuel proxy terms are scaled by baseMVA
                # so that the QUBO coefficients remain numerically moderate.
                no_load = self.objective_scale * generator.no_load_cost / instance.case.base_mva
                proxy = (
                    self.objective_scale
                    * self.fuel_proxy_fraction
                    * (generator.quadratic_cost * generator.p_max_mw**2
                       + generator.linear_cost * generator.p_max_mw)
                    / instance.case.base_mva
                )
                accumulator.add_linear(variable, no_load + proxy)
                components["commitment_cost"] += no_load
                components["fuel_proxy"] += proxy
                if period == 0:
                    if not generator.initial_on:
                        startup = self.objective_scale * generator.startup_cost / instance.case.base_mva
                        accumulator.add_linear(variable, startup)
                        components["startup_cost"] += startup
                else:
                    previous = index(generator.name, period - 1)
                    startup = self.objective_scale * generator.startup_cost / instance.case.base_mva
                    # startup * u_t * (1 - u_{t-1})
                    accumulator.add_linear(variable, startup)
                    accumulator.add_product(variable, previous, -startup)
                    components["startup_cost"] += startup

        for period, demand in enumerate(instance.demand_mw):
            target = (demand + instance.reserve_mw[period]) / instance.case.base_mva
            coefficients = {
                index(generator.name, period): generator.p_max_mw / instance.case.base_mva
                for generator in instance.units
            }
            accumulator.add_square(coefficients, rhs=target, weight=self.capacity_weight)
            components["capacity_penalty"] += self.capacity_weight * target * target

        accumulator.validate(len(registry))
        expression = accumulator.to_expression(registry)
        return QuboModel(registry, accumulator, expression, components, instance)


@dataclass(frozen=True)
class Case5QuboAudit:
    """Lightweight sampled consistency audit for the ten-bit QUBO."""

    degree: int
    samples_checked: int
    energy_identity_max_error: float
    bit_order_ok: bool

    @property
    def passed(self) -> bool:
        return (
            self.degree <= 2
            and self.energy_identity_max_error <= 1.0e-8
            and self.bit_order_ok
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "degree": self.degree,
            "samples_checked": self.samples_checked,
            "energy_identity_max_error": self.energy_identity_max_error,
            "bit_order_ok": self.bit_order_ok,
        }


def audit_case5_qubo(
    model: QuboModel,
    *,
    samples: int = 128,
    seed: int = 2026,
) -> Case5QuboAudit:
    """Compare the sparse accumulator with the repository QUBO bridge.

    The audit deliberately samples binary states instead of using a classical
    exhaustive solver.  The complete state-vector distribution is still
    produced by QAOA when the statevector backend is selected.
    """

    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(int(seed))
    # Keep this audit self-contained.  The full pyqpanda-algorithm package
    # exposes a QuadraticBinary bridge, but a standalone case5 checkout should
    # not need that optional package merely to verify the sparse coefficients.
    symbols = [sp.Symbol(name) for name in model.registry.names]
    expression_value = sp.lambdify(symbols, model.expression, "numpy")
    max_error = 0.0
    bit_order_ok = True
    for _ in range(int(samples)):
        bits = tuple(int(value) for value in rng.integers(0, 2, size=model.n_variables))
        max_error = max(max_error, abs(float(expression_value(*bits)) - model.energy(bits)))
        encoded = model.registry.bitstring_from_bits(bits)
        bit_order_ok = bit_order_ok and model.registry.bits_from_bitstring(encoded) == bits
    return Case5QuboAudit(
        degree=2 if model.accumulator.quadratic else (1 if model.accumulator.linear else 0),
        samples_checked=int(samples),
        energy_identity_max_error=float(max_error),
        bit_order_ok=bool(bit_order_ok),
    )


def decode_commitment(
    instance: Case5UCInstance,
    registry: VariableRegistry,
    bitstring: str,
) -> tuple[tuple[int, ...], ...]:
    """Decode a QAOA bit string as ``(period, generator)`` commitments."""

    bits = registry.bits_from_bitstring(bitstring)
    return tuple(
        tuple(
            int(bits[registry.index(VariableKey("u", generator.name, period))])
            for generator in instance.units
        )
        for period in instance.periods
    )


def commitment_capacity(
    instance: Case5UCInstance,
    commitments: tuple[tuple[int, ...], ...],
    period: int,
) -> float:
    """Return committed MW capacity for one period."""

    return float(
        sum(
            generator.p_max_mw * int(commitments[period][index])
            for index, generator in enumerate(instance.units)
        )
    )


__all__ = [
    "Case5CommitmentQuboBuilder",
    "Case5QuboAudit",
    "audit_case5_qubo",
    "commitment_capacity",
    "decode_commitment",
]

