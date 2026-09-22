"""Validated data structures for the compact unit-commitment example."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ThermalUnit:
    """One thermal unit with a linear binary dispatch encoding."""

    name: str
    p_min: float
    p_max: float
    quadratic_cost: float
    linear_cost: float
    no_load_cost: float
    startup_cost: float
    initial_on: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("unit name must be non-empty")
        values = (
            self.p_min,
            self.p_max,
            self.quadratic_cost,
            self.linear_cost,
            self.no_load_cost,
            self.startup_cost,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"unit {self.name!r} contains a non-finite parameter")
        if self.p_min < 0.0 or self.p_max < self.p_min:
            raise ValueError("unit limits must satisfy 0 <= p_min <= p_max")
        if self.quadratic_cost < 0.0 or self.linear_cost < 0.0:
            raise ValueError("fuel costs must be non-negative")
        if self.no_load_cost < 0.0 or self.startup_cost < 0.0:
            raise ValueError("fixed and startup costs must be non-negative")


@dataclass(frozen=True)
class UnitCommitmentInstance:
    """A small finite-horizon unit-commitment instance."""

    demand: tuple[float, ...]
    units: tuple[ThermalUnit, ...]
    reserve: tuple[float, ...] = ()
    name: str = "unit_commitment"

    def __post_init__(self) -> None:
        demand = tuple(float(value) for value in self.demand)
        units = tuple(self.units)
        reserve = tuple(float(value) for value in self.reserve)
        if not demand:
            raise ValueError("at least one demand period is required")
        if not units:
            raise ValueError("at least one thermal unit is required")
        if not all(np.isfinite(value) and value >= 0.0 for value in demand):
            raise ValueError("demand must be finite and non-negative")
        if reserve and len(reserve) != len(demand):
            raise ValueError("reserve must have one value per demand period")
        if reserve and not all(np.isfinite(value) and value >= 0.0 for value in reserve):
            raise ValueError("reserve must be finite and non-negative")
        names = [unit.name for unit in units]
        if len(names) != len(set(names)):
            raise ValueError("unit names must be unique")
        if not self.name or not self.name.strip():
            raise ValueError("instance name must be non-empty")
        object.__setattr__(self, "demand", demand)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "reserve", reserve or (0.0,) * len(demand))

    @property
    def periods(self) -> range:
        return range(len(self.demand))

    @property
    def time_periods(self) -> int:
        return len(self.demand)

    @property
    def reserves(self) -> tuple[float, ...]:
        return self.reserve

    @property
    def unit_names(self) -> tuple[str, ...]:
        return tuple(unit.name for unit in self.units)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "UnitCommitmentInstance":
        raw_units = payload.get("units")
        if raw_units is None:
            # Accept the compacted PGLib-UC shape as an input contract while
            # keeping the internal representation small and explicit.
            raw_units = [
                dict(item, name=name)
                for name, item in payload.get("thermal_generators", {}).items()
            ]
        elif isinstance(raw_units, Mapping):
            raw_units = [dict(item, name=name) for name, item in raw_units.items()]
        normalized_units = []
        for item in raw_units:
            cost = item.get("quadratic_cost", {})
            if isinstance(cost, Mapping):
                quadratic = cost.get("a", item.get("a", 0.0))
                linear = cost.get("b", item.get("b", 0.0))
                fixed = cost.get("c", item.get("c", 0.0))
            else:
                quadratic = cost
                linear = item.get("linear_cost", item.get("b", 0.0))
                fixed = item.get("no_load_cost", item.get("c", 0.0))
            normalized_units.append(
                ThermalUnit(
                    name=str(item["name"]),
                    p_min=float(item.get("p_min", item.get("power_output_minimum"))),
                    p_max=float(item.get("p_max", item.get("power_output_maximum"))),
                    quadratic_cost=float(quadratic),
                    linear_cost=float(item.get("linear_cost", linear)),
                    no_load_cost=float(item.get("no_load_cost", fixed)),
                    startup_cost=float(item.get("startup_cost", 0.0)),
                    initial_on=bool(item.get("initial_on", item.get("unit_on_t0", False))),
                )
            )
        units = tuple(normalized_units)
        reserves = payload.get("reserve", payload.get("reserves", ()))
        return cls(
            demand=tuple(float(value) for value in payload["demand"]),
            units=units,
            reserve=tuple(float(value) for value in reserves),
            name=str(payload.get("name", "unit_commitment")),
        )


@dataclass(frozen=True)
class CommitmentSolution:
    """Decoded binary commitment and dispatch values for all periods."""

    commitment: dict[tuple[str, int], int]
    dispatch_level: dict[tuple[str, int], float]
    dispatch_bit: dict[tuple[str, int], int]


def load_instance(path: str | Path) -> UnitCommitmentInstance:
    """Load a JSON instance without embedding credentials or runtime state."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return UnitCommitmentInstance.from_mapping(payload)


__all__ = [
    "CommitmentSolution",
    "ThermalUnit",
    "UnitCommitmentInstance",
    "load_instance",
]

