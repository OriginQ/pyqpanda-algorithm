"""MATPOWER case5 data and a small DC-network model.

The numerical data in :func:`load_matpower_case5` are the public ``case5``
benchmark distributed with MATPOWER.  The loader is intentionally read-only:
it stores the data in ordinary Python tuples instead of depending on MATLAB or
on a local MATPOWER installation.  MATPOWER's network model is used for the
DC line-flow checks in the economic-dispatch stage; the commitment QUBO keeps
only the five generator on/off variables per period.

MATPOWER source and attribution
--------------------------------
``case5.m`` is described as a modified 5-bus, 5-generator PJM case based on
Li and Bo (2010), and is distributed by MATPOWER with permission.  The source
URL is retained in :class:`MatpowerCase` so an experiment can report its
provenance without copying the MATLAB file into this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .data_model import ThermalUnit


MATPOWER_CASE5_URL = "https://github.com/MATPOWER/matpower/blob/master/data/case5.m"
MATPOWER_CASE5_REFERENCE = (
    "F. Li and R. Bo, 'DCOPF-based LMP simulation: Algorithm, implementation "
    "and example', IEEE PES General Meeting, 2010."
)


@dataclass(frozen=True)
class BusSpec:
    """One MATPOWER bus row reduced to the fields used by this benchmark."""

    bus: int
    load_mw: float
    load_mvar: float = 0.0


@dataclass(frozen=True)
class GeneratorSpec:
    """Generator limits and MATPOWER polynomial cost coefficients."""

    name: str
    bus: int
    p_min_mw: float
    p_max_mw: float
    quadratic_cost: float
    linear_cost: float
    no_load_cost: float = 0.0
    startup_cost: float = 0.0
    initial_on: bool = False

    def __post_init__(self) -> None:
        values = (
            self.p_min_mw,
            self.p_max_mw,
            self.quadratic_cost,
            self.linear_cost,
            self.no_load_cost,
            self.startup_cost,
        )
        if not self.name.strip():
            raise ValueError("generator name must be non-empty")
        if self.bus < 1:
            raise ValueError("MATPOWER bus numbers start at one")
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"generator {self.name!r} contains a non-finite value")
        if self.p_min_mw < 0.0 or self.p_max_mw < self.p_min_mw:
            raise ValueError("generator limits must satisfy 0 <= p_min <= p_max")
        if self.quadratic_cost < 0.0 or self.linear_cost < 0.0:
            raise ValueError("cost coefficients must be non-negative")
        if self.no_load_cost < 0.0 or self.startup_cost < 0.0:
            raise ValueError("fixed and startup costs must be non-negative")

    def as_thermal_unit(self) -> ThermalUnit:
        """Return the generator in the compact package's common unit type."""

        return ThermalUnit(
            name=self.name,
            p_min=self.p_min_mw,
            p_max=self.p_max_mw,
            quadratic_cost=self.quadratic_cost,
            linear_cost=self.linear_cost,
            no_load_cost=self.no_load_cost,
            startup_cost=self.startup_cost,
            initial_on=self.initial_on,
        )


@dataclass(frozen=True)
class BranchSpec:
    """Reduced MATPOWER branch row for a lossless DC flow calculation."""

    from_bus: int
    to_bus: int
    resistance_pu: float
    reactance_pu: float
    charging_pu: float
    rate_a_mva: float = 0.0

    def __post_init__(self) -> None:
        if self.from_bus < 1 or self.to_bus < 1 or self.from_bus == self.to_bus:
            raise ValueError("a branch must connect two distinct positive bus IDs")
        if not np.isfinite(self.reactance_pu) or self.reactance_pu <= 0.0:
            raise ValueError("DC branch reactance must be positive and finite")
        if not all(np.isfinite(value) for value in (self.resistance_pu, self.charging_pu)):
            raise ValueError("branch electrical parameters must be finite")
        if not np.isfinite(self.rate_a_mva) or self.rate_a_mva < 0.0:
            raise ValueError("branch rate must be finite and non-negative")


@dataclass(frozen=True)
class MatpowerCase:
    """A small, immutable network representation derived from MATPOWER."""

    name: str
    base_mva: float
    buses: tuple[BusSpec, ...]
    generators: tuple[GeneratorSpec, ...]
    branches: tuple[BranchSpec, ...]
    source_url: str = MATPOWER_CASE5_URL
    source_reference: str = MATPOWER_CASE5_REFERENCE

    def __post_init__(self) -> None:
        if self.base_mva <= 0.0 or not np.isfinite(self.base_mva):
            raise ValueError("base_mva must be positive and finite")
        bus_ids = tuple(bus.bus for bus in self.buses)
        if not bus_ids or len(set(bus_ids)) != len(bus_ids):
            raise ValueError("bus IDs must be non-empty and unique")
        if any(generator.bus not in bus_ids for generator in self.generators):
            raise ValueError("every generator must be attached to a known bus")
        if any(
            branch.from_bus not in bus_ids or branch.to_bus not in bus_ids
            for branch in self.branches
        ):
            raise ValueError("every branch must use known bus IDs")

    @property
    def bus_ids(self) -> tuple[int, ...]:
        return tuple(bus.bus for bus in self.buses)

    @property
    def total_load_mw(self) -> float:
        return float(sum(bus.load_mw for bus in self.buses))

    @property
    def load_mw(self) -> tuple[float, ...]:
        return tuple(float(bus.load_mw) for bus in self.buses)

    def scaled_load(self, scale: float) -> tuple[float, ...]:
        """Scale active and reactive loads uniformly for a time period."""

        scale = float(scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("load scale must be finite and non-negative")
        return tuple(float(bus.load_mw * scale) for bus in self.buses)

    def generator_bus_matrix(self) -> np.ndarray:
        """Map generator outputs to bus injections (bus-by-generator)."""

        matrix = np.zeros((len(self.buses), len(self.generators)), dtype=float)
        index = {bus: position for position, bus in enumerate(self.bus_ids)}
        for generator_index, generator in enumerate(self.generators):
            matrix[index[generator.bus], generator_index] = 1.0
        return matrix

    def ptdf(self) -> np.ndarray:
        """Return a DC PTDF in MW/MW with the first bus as the reference.

        The matrix maps a balanced vector of bus injections to branch flows.
        Branch charging and resistance are retained in the data object for
        provenance but are not used by the lossless DC approximation.
        """

        bus_index = {bus: position for position, bus in enumerate(self.bus_ids)}
        n_bus = len(self.buses)
        reference = 0
        susceptance = np.zeros((n_bus, n_bus), dtype=float)
        for branch in self.branches:
            first = bus_index[branch.from_bus]
            second = bus_index[branch.to_bus]
            value = 1.0 / branch.reactance_pu
            susceptance[first, first] += value
            susceptance[second, second] += value
            susceptance[first, second] -= value
            susceptance[second, first] -= value
        reduced = np.delete(np.delete(susceptance, reference, axis=0), reference, axis=1)
        if np.linalg.matrix_rank(reduced) < n_bus - 1:
            raise ValueError("case network is disconnected; DC PTDF is undefined")
        ptdf = np.zeros((len(self.branches), n_bus), dtype=float)
        for injection_bus in range(n_bus):
            if injection_bus == reference:
                continue
            injection = np.zeros(n_bus, dtype=float)
            injection[injection_bus] = 1.0 / self.base_mva
            injection[reference] = -1.0 / self.base_mva
            angles = np.zeros(n_bus, dtype=float)
            angles[1:] = np.linalg.solve(reduced, injection[1:])
            for branch_index, branch in enumerate(self.branches):
                first = bus_index[branch.from_bus]
                second = bus_index[branch.to_bus]
                ptdf[branch_index, injection_bus] = (
                    (angles[first] - angles[second]) / branch.reactance_pu * self.base_mva
                )
        return ptdf

    def branch_flows(
        self,
        generation_mw: Sequence[float],
        load_mw: Sequence[float],
        *,
        tolerance: float = 1.0e-7,
    ) -> np.ndarray:
        """Compute branch flows for a balanced generator/load dispatch."""

        generation = np.asarray(tuple(generation_mw), dtype=float)
        load = np.asarray(tuple(load_mw), dtype=float)
        if generation.shape != (len(self.generators),):
            raise ValueError("generation_mw must have one entry per generator")
        if load.shape != (len(self.buses),):
            raise ValueError("load_mw must have one entry per bus")
        if not np.all(np.isfinite(generation)) or not np.all(np.isfinite(load)):
            raise ValueError("generation and load must be finite")
        injection = self.generator_bus_matrix() @ generation - load
        if abs(float(np.sum(injection))) > tolerance:
            raise ValueError("DC branch flows require balanced generation and load")
        return self.ptdf() @ injection

    def line_limits_mw(self) -> tuple[float, ...]:
        return tuple(float(branch.rate_a_mva) for branch in self.branches)


@dataclass(frozen=True)
class Case5UCInstance:
    """Multi-period commitment scenario built from MATPOWER case5.

    ``demand_scales`` scales the original bus loads uniformly.  The default
    profile (0.8, 1.0) produces 800 MW and 1000 MW total demand while keeping
    the QUBO at ten commitment qubits.  Startup and no-load costs are explicit
    scenario assumptions because MATPOWER's case5 gencost rows do not supply
    unit-commitment costs.
    """

    case: MatpowerCase
    demand_scales: tuple[float, ...] = (0.8, 1.0)
    reserve_mw: tuple[float, ...] = (0.0, 0.0)
    units: tuple[GeneratorSpec, ...] | None = None
    name: str = "matpower_case5_unit_commitment"

    def __post_init__(self) -> None:
        scales = tuple(float(value) for value in self.demand_scales)
        reserve = tuple(float(value) for value in self.reserve_mw)
        if not self.name.strip():
            raise ValueError("scenario name must be non-empty")
        if not scales or any(not np.isfinite(value) or value < 0.0 for value in scales):
            raise ValueError("demand_scales must be finite and non-negative")
        if reserve and len(reserve) != len(scales):
            raise ValueError("reserve_mw must have one value per period")
        reserve = reserve or (0.0,) * len(scales)
        if any(not np.isfinite(value) or value < 0.0 for value in reserve):
            raise ValueError("reserve_mw must be finite and non-negative")
        units = tuple(self.units) if self.units is not None else self.case.generators
        if len(units) != len(self.case.generators):
            raise ValueError("units must match the MATPOWER generator count")
        if tuple(unit.name for unit in units) != tuple(generator.name for generator in self.case.generators):
            raise ValueError("units must preserve MATPOWER generator names and ordering")
        object.__setattr__(self, "demand_scales", scales)
        object.__setattr__(self, "reserve_mw", reserve)
        object.__setattr__(self, "units", units)

    @property
    def periods(self) -> range:
        return range(len(self.demand_scales))

    @property
    def time_periods(self) -> int:
        return len(self.demand_scales)

    @property
    def demand_mw(self) -> tuple[float, ...]:
        return tuple(float(self.case.total_load_mw * scale) for scale in self.demand_scales)

    @property
    def load_profiles_mw(self) -> tuple[tuple[float, ...], ...]:
        return tuple(self.case.scaled_load(scale) for scale in self.demand_scales)

    @property
    def generator_names(self) -> tuple[str, ...]:
        return tuple(generator.name for generator in self.units)

    @property
    def reserves(self) -> tuple[float, ...]:
        return self.reserve_mw

    def thermal_units(self) -> tuple[ThermalUnit, ...]:
        return tuple(unit.as_thermal_unit() for unit in self.units)


def load_matpower_case5(
    *,
    startup_costs: Iterable[float] | None = None,
    no_load_costs: Iterable[float] | None = None,
) -> MatpowerCase:
    """Load the official MATPOWER ``case5`` benchmark data.

    MATPOWER uses zero startup and shutdown fields for this case.  Supplying
    ``startup_costs`` or ``no_load_costs`` adds optional UC scenario costs while
    leaving all network, limits, and energy-cost coefficients untouched.
    """

    default_startup = (8.0, 8.0, 12.0, 10.0, 6.0)
    default_no_load = (2.0, 3.0, 5.0, 4.0, 2.0)
    startup = tuple(default_startup if startup_costs is None else startup_costs)
    no_load = tuple(default_no_load if no_load_costs is None else no_load_costs)
    if len(startup) != 5 or len(no_load) != 5:
        raise ValueError("case5 requires five startup and no-load cost values")
    buses = (
        BusSpec(1, 0.0, 0.0),
        BusSpec(2, 300.0, 98.61),
        BusSpec(3, 300.0, 98.61),
        BusSpec(4, 400.0, 131.47),
        BusSpec(5, 0.0, 0.0),
    )
    # MATPOWER gencost rows are 14, 15, 30, 40 and 10 $/MWh linear terms.
    generators = tuple(
        GeneratorSpec(
            name=f"G{index + 1}",
            bus=bus,
            p_min_mw=0.0,
            p_max_mw=p_max,
            quadratic_cost=0.0,
            linear_cost=linear,
            no_load_cost=float(no_load[index]),
            startup_cost=float(startup[index]),
            initial_on=False,
        )
        for index, (bus, p_max, linear) in enumerate(
            ((1, 40.0, 14.0), (1, 170.0, 15.0), (3, 520.0, 30.0), (4, 200.0, 40.0), (5, 600.0, 10.0))
        )
    )
    branches = (
        BranchSpec(1, 2, 0.00281, 0.0281, 0.00712, 400.0),
        BranchSpec(1, 4, 0.00304, 0.0304, 0.00658, 0.0),
        BranchSpec(1, 5, 0.00064, 0.0064, 0.03126, 0.0),
        BranchSpec(2, 3, 0.00108, 0.0108, 0.01852, 0.0),
        BranchSpec(3, 4, 0.00297, 0.0297, 0.00674, 0.0),
        BranchSpec(4, 5, 0.00297, 0.0297, 0.00674, 240.0),
    )
    return MatpowerCase(
        name="case5",
        base_mva=100.0,
        buses=buses,
        generators=generators,
        branches=branches,
    )


def load_case5_uc(
    *,
    demand_scales: Sequence[float] = (0.8, 1.0),
    reserve_mw: Sequence[float] = (0.0, 0.0),
    startup_costs: Iterable[float] | None = None,
    no_load_costs: Iterable[float] | None = None,
    name: str = "matpower_case5_unit_commitment",
) -> Case5UCInstance:
    """Convenience constructor for a reproducible case5 UC scenario."""

    case = load_matpower_case5(
        startup_costs=startup_costs,
        no_load_costs=no_load_costs,
    )
    return Case5UCInstance(
        case=case,
        demand_scales=tuple(float(value) for value in demand_scales),
        reserve_mw=tuple(float(value) for value in reserve_mw),
        name=name,
    )


__all__ = [
    "MATPOWER_CASE5_REFERENCE",
    "MATPOWER_CASE5_URL",
    "BranchSpec",
    "BusSpec",
    "Case5UCInstance",
    "GeneratorSpec",
    "MatpowerCase",
    "load_case5_uc",
    "load_matpower_case5",
]

