"""Linear soil-water balance and QUBO irrigation scheduling.

Supports wheat growth-stage parameters and gross water-supply accounting.

Days in input files are one-based. Internal arrays are zero-based; decision
variables are ordered by zone, then by open day. QUBO pair coefficients occur
once, in the strict upper triangle (not a symmetric x.T @ Q @ x matrix).
"""

from dataclasses import dataclass
from datetime import date, timedelta
import json
import math
from pathlib import Path

import numpy as np


def _number(value, name, minimum=0.0, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(value) or value < minimum or (strictly_positive and value == 0):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _series(value, days, name, minimum=0.0):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return np.full(days, _number(value, name, minimum=minimum))
    if not isinstance(value, list) or len(value) != days:
        raise ValueError(f"{name} must be a number or a list of {days} numbers")
    return np.array([_number(v, name, minimum=minimum) for v in value])


def _stages(value, days):
    values = [value] * days if isinstance(value, str) else value
    if (not isinstance(values, list) or len(values) != days
            or any(not isinstance(v, str) or not v.strip() for v in values)):
        raise ValueError(f"growth_stage must be a nonempty string or a list of {days} strings")
    return tuple(values)


def _volume_event_limit(water_budget, event_volume, n_decisions):
    """Floor the gross supply with only an ULP-sized allowance at boundaries.

    Decimal inputs such as 0.3 / 0.1 may otherwise lose one whole event to
    floating-point rounding. The tolerance covers a few arithmetic rounding
    steps, rather than a relative engineering tolerance on the water supply.
    """
    if water_budget == 0:
        return 0
    total = event_volume * n_decisions
    if water_budget >= total:
        return n_decisions
    limit = min(n_decisions, math.floor(water_budget / event_volume))
    if limit < n_decisions:
        next_volume = (limit + 1) * event_volume
        tolerance = 4 * max(math.ulp(water_budget), math.ulp(next_volume))
        if math.isfinite(next_volume) and next_volume - water_budget <= tolerance:
            limit += 1
    return limit


def bounded_slack_weights(budget):
    """Appendix A bounded binary encoding: all integers 0..K, no larger ones."""
    _integer(budget, "budget")
    if budget == 0:
        return np.empty(0, dtype=int)
    m = budget.bit_length()
    return np.array([1 << k for k in range(m - 1)] + [budget - (1 << (m - 1)) + 1])


@dataclass(frozen=True)
class Zone:
    name: str
    taw_mm: float
    initial_moisture_mm: float
    window_days: tuple
    target_mm: np.ndarray
    stress_weights: np.ndarray
    et0_mm: np.ndarray
    crop_coefficient: np.ndarray
    rainfall_mm: np.ndarray
    effective_rain_fraction: float
    capillary_rise_mm: np.ndarray
    growth_stage: tuple
    allowable_depletion_fraction: np.ndarray
    forecast_adjustment_mm: np.ndarray = 0.0

    @property
    def forcing_mm(self):
        return (self.effective_rain_fraction * self.rainfall_mm
                + self.capillary_rise_mm - self.crop_coefficient * self.et0_mm
                + self.forecast_adjustment_mm)

    @property
    def moisture_threshold_mm(self):
        """Available root-zone water below which the depletion trigger fires."""
        return (1 - self.allowable_depletion_fraction) * self.taw_mm


@dataclass(frozen=True)
class IrrigationInstance:
    name: str
    source_note: str
    days: int
    budget_events: int
    dose_mm: float
    water_cost: float
    spatial_weight: float
    timing_weight: float
    zone_area_ha: float
    zones: tuple
    adjacency: tuple
    region: str = "generic"
    crop: str = "winter_wheat"
    start_date: str = None
    irrigation_efficiency: float = 1.0
    water_budget_m3: float = None

    def __post_init__(self):
        if self.water_budget_m3 is None:
            object.__setattr__(self, "water_budget_m3", self.effective_budget_gross_water_m3)

    @property
    def event_net_water_m3(self):
        """Root-zone application: one hectare-millimetre equals ten m³."""
        return float(self.dose_mm * self.zone_area_ha * 10)

    @property
    def event_gross_water_m3(self):
        return self.event_net_water_m3 / self.irrigation_efficiency

    @property
    def effective_budget_gross_water_m3(self):
        return float(self.budget_events * self.event_gross_water_m3)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Input JSON must contain an object")
        try:
            return cls._parse(data)
        except KeyError as exc:
            raise ValueError(f"Missing input field: {exc.args[0]}") from exc

    @classmethod
    def _parse(cls, data):
        days = _integer(data["days"], "days", 1)
        budget = (_integer(data["budget_events"], "budget_events")
                  if "budget_events" in data else None)
        volume_budget = (_number(data["water_budget_m3"], "water_budget_m3")
                         if "water_budget_m3" in data else None)
        if budget is None and volume_budget is None:
            raise ValueError("Specify budget_events or water_budget_m3")
        dose = _number(data["dose_mm"], "dose_mm", strictly_positive=True)
        area = _number(data["zone_area_ha"], "zone_area_ha", strictly_positive=True)
        efficiency = _number(data.get("irrigation_efficiency", 1.0),
                             "irrigation_efficiency", strictly_positive=True)
        if efficiency > 1:
            raise ValueError("irrigation_efficiency must lie within (0, 1]")
        event_net = _number(dose * area * 10, "event_net_water_m3", strictly_positive=True)
        event_gross = _number(event_net / efficiency, "event_gross_water_m3", strictly_positive=True)
        if not isinstance(data["zones"], list) or not data["zones"]:
            raise ValueError("zones must be a nonempty list")
        zones = []
        for raw in data["zones"]:
            if not isinstance(raw, dict):
                raise ValueError("Each zone must be an object")
            name = raw["name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Zone name must be a nonempty string")
            taw = _number(raw["taw_mm"], "taw_mm", strictly_positive=True)
            initial = _number(raw["initial_moisture_mm"], "initial_moisture_mm")
            if initial > taw:
                raise ValueError("Initial moisture must lie within [0, TAW]")
            window = raw["window_days"]
            if not isinstance(window, list):
                raise ValueError("window_days must be a list of one-based days")
            window = [_integer(d, "window day", 1) for d in window]
            if len(set(window)) != len(window) or any(d > days for d in window):
                raise ValueError("Window days must be unique and within the horizon")
            target = _series(raw["target_mm"], days, "target_mm")
            if np.any(target > taw):
                raise ValueError("Moisture target must lie within [0, TAW]")
            rain_fraction = _number(raw["effective_rain_fraction"], "effective_rain_fraction")
            if rain_fraction > 1:
                raise ValueError("effective_rain_fraction must be within [0, 1]")
            depletion = _series(raw.get("allowable_depletion_fraction", 0.55), days,
                                "allowable_depletion_fraction")
            if np.any(depletion > 1):
                raise ValueError("allowable_depletion_fraction must lie within [0, 1]")
            zones.append(Zone(
                name, taw, initial, tuple(sorted(window)), target,
                _series(raw["stress_weights"], days, "stress_weights"),
                _series(raw["et0_mm"], days, "et0_mm"),
                _series(raw["crop_coefficient"], days, "crop_coefficient"),
                _series(raw["rainfall_mm"], days, "rainfall_mm"), rain_fraction,
                _series(raw["capillary_rise_mm"], days, "capillary_rise_mm"),
                _stages(raw.get("growth_stage", "unspecified"), days), depletion,
                _series(raw.get("forecast_adjustment_mm", 0.0), days,
                        "forecast_adjustment_mm", minimum=-math.inf),
            ))
        names = [z.name for z in zones]
        if len(set(names)) != len(names):
            raise ValueError("Zone names must be unique")
        n_decisions = sum(len(z.window_days) for z in zones)
        if n_decisions == 0:
            raise ValueError("At least one irrigation window must be open")
        if budget is not None and budget > n_decisions:
            raise ValueError("budget_events cannot exceed the number of open zone-days")
        if volume_budget is not None:
            volume_limit = _volume_event_limit(volume_budget, event_gross, n_decisions)
            budget = volume_limit if budget is None else min(budget, volume_limit)
        else:
            volume_budget = _number(budget * event_gross, "water_budget_m3")
        edges = set()
        adjacency = data.get("adjacency", [])
        if not isinstance(adjacency, list):
            raise ValueError("adjacency must be a list of zone-name pairs")
        for edge in adjacency:
            if (not isinstance(edge, list) or len(edge) != 2
                    or any(v not in names for v in edge) or edge[0] == edge[1]):
                raise ValueError("Each adjacency edge must name two distinct existing zones")
            pair = tuple(sorted((names.index(edge[0]), names.index(edge[1]))))
            if pair in edges:
                raise ValueError("Duplicate adjacency edge")
            edges.add(pair)
        name = data.get("name", "custom irrigation instance")
        note = data.get("source_note", "User-supplied inputs; provenance not specified")
        if not isinstance(name, str) or not isinstance(note, str):
            raise ValueError("name and source_note must be strings")
        region = data.get("region", "generic")
        crop = data.get("crop", "winter_wheat")
        if any(not isinstance(v, str) or not v.strip() for v in (region, crop)):
            raise ValueError("region and crop must be nonempty strings")
        start_date = data.get("start_date")
        if start_date is not None:
            try:
                if not isinstance(start_date, str) or date.fromisoformat(start_date).isoformat() != start_date:
                    raise ValueError
                date.fromisoformat(start_date) + timedelta(days=days - 1)
            except (ValueError, OverflowError) as exc:
                raise ValueError("start_date must be an ISO date (YYYY-MM-DD) with a representable planning horizon") from exc
        return cls(
            name, note, days, budget, dose,
            _number(data["water_cost"], "water_cost"),
            _number(data["spatial_weight"], "spatial_weight"),
            _number(data["timing_weight"], "timing_weight"),
            area,
            tuple(zones), tuple(sorted(edges)),
            region, crop, start_date, efficiency, volume_budget,
        )


def load_instance(path):
    return IrrigationInstance.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class IrrigationQUBO:
    instance: IrrigationInstance
    decisions: tuple
    linear: np.ndarray
    quadratic: np.ndarray
    constant: float
    slack_weights: np.ndarray
    budget_penalty: float
    penalty_bound: float

    @property
    def n_decisions(self):
        return len(self.decisions)

    @property
    def n_variables(self):
        return len(self.linear)

    @staticmethod
    def _bits(bits, width):
        bits = np.asarray(bits)
        if bits.ndim == 0 or bits.shape[-1] != width or not np.all((bits == 0) | (bits == 1)):
            raise ValueError(f"Expected binary values with last dimension {width}")
        return bits.astype(float)

    def energy(self, bits):
        values = self._bits(bits, self.n_variables)
        result = values @ self.linear + self.constant
        for i, j in zip(*np.nonzero(self.quadratic)):
            result = result + self.quadratic[i, j] * values[..., i] * values[..., j]
        return result

    def to_ising(self):
        """Return h, upper-triangular J and E0 including both QUBO constants."""
        h = -self.linear / 2 - (self.quadratic.sum(axis=0) + self.quadratic.sum(axis=1)) / 4
        J = self.quadratic / 4
        offset = self.constant + self.linear.sum() / 2 + self.quadratic.sum() / 4
        return h, J, float(offset)

    def decode_schedule(self, decisions):
        x = self._bits(decisions, self.n_decisions)
        if x.ndim != 1:
            raise ValueError("decode_schedule expects a single decision vector")
        schedule = np.zeros((len(self.instance.zones), self.instance.days), dtype=int)
        for bit, (zone, day) in zip(x, self.decisions):
            schedule[zone, day] = int(bit)
        return schedule

    def encode_schedule(self, decisions):
        x = self._bits(decisions, self.n_decisions)
        if x.ndim != 1:
            raise ValueError("encode_schedule expects a single decision vector")
        remaining = self.instance.budget_events - int(x.sum())
        if remaining < 0:
            raise ValueError("Cannot encode a budget-infeasible schedule with zero penalty")
        slack = np.zeros(len(self.slack_weights), dtype=int)
        # Bounded weights need not be sorted, so choose the largest first.
        for k in sorted(range(len(slack)), key=lambda i: self.slack_weights[i], reverse=True):
            if remaining >= self.slack_weights[k]:
                slack[k] = 1
                remaining -= int(self.slack_weights[k])
        if remaining:
            raise ValueError("Slack encoding does not cover the budget residual")
        return np.concatenate((x.astype(int), slack))

    def moisture_trajectory(self, decisions):
        schedule = self.decode_schedule(decisions)
        return np.array([
            z.initial_moisture_mm + np.cumsum(z.forcing_mm + self.instance.dose_mm * schedule[i])
            for i, z in enumerate(self.instance.zones)
        ])

    def evaluate(self, decisions):
        """Evaluate physical costs directly, independently of QUBO expansion.

        Spatial/timing conflicts and moisture excursions are diagnostics, not
        hard feasibility constraints in this scheduling model.
        """
        schedule = self.decode_schedule(decisions)
        moisture = self.moisture_trajectory(decisions)
        events = int(schedule.sum())
        water = self.instance.water_cost * self.instance.dose_mm * events
        stress = sum(float(np.sum(z.stress_weights * (moisture[i] - z.target_mm) ** 2))
                     for i, z in enumerate(self.instance.zones))
        spatial_pairs = sum(int(np.sum(schedule[i] * schedule[j])) for i, j in self.instance.adjacency)
        consecutive_pairs = int(np.sum(schedule[:, :-1] * schedule[:, 1:]))
        spatial = self.instance.spatial_weight * spatial_pairs
        timing = self.instance.timing_weight * consecutive_pairs
        excursions = []
        zone_diagnostics = []
        below_threshold_count = 0
        for i, z in enumerate(self.instance.zones):
            below = moisture[i] < z.moisture_threshold_mm
            below_threshold_count += int(below.sum())
            stage_summary = []
            for stage in dict.fromkeys(z.growth_stage):
                stage_days = np.array([value == stage for value in z.growth_stage])
                stage_summary.append({
                    "stage": stage,
                    "days": (np.flatnonzero(stage_days) + 1).tolist(),
                    "irrigation_events": int(schedule[i, stage_days].sum()),
                    "below_threshold_days": (np.flatnonzero(stage_days & below) + 1).tolist(),
                })
            zone_diagnostics.append({
                "zone": z.name,
                "irrigation_events": int(schedule[i].sum()),
                "minimum_moisture_mm": float(moisture[i].min()),
                "below_threshold_days": (np.flatnonzero(below) + 1).tolist(),
                "stage_summary": stage_summary,
            })
            for day, value in enumerate(moisture[i]):
                if value < 0 or value > z.taw_mm:
                    excursions.append({"zone": z.name, "day": day + 1,
                                       "moisture_mm": float(value), "taw_mm": z.taw_mm})
        net_water = events * self.instance.event_net_water_m3
        gross_water = events * self.instance.event_gross_water_m3
        supply = self.instance.water_budget_m3
        remaining = supply - gross_water
        if remaining < 0 and events <= self.instance.budget_events:
            # A feasible event limit can differ from a decimal input by a few
            # machine ULPs; report that exhausted supply as zero remaining.
            remaining = 0.0
        return {
            "objective": float(water + stress + spatial + timing),
            "water_cost": float(water), "stress_cost": stress,
            "spatial_cost": float(spatial), "timing_cost": float(timing),
            "events": events, "budget_feasible": events <= self.instance.budget_events,
            "net_water_m3": float(net_water), "gross_water_m3": float(gross_water),
            "conveyance_loss_m3": float(gross_water - net_water),
            "remaining_water_m3": float(remaining),
            "water_budget_utilization": float(gross_water / supply) if supply else 0.0,
            "below_threshold_zone_days": below_threshold_count,
            "zone_diagnostics": zone_diagnostics,
            "spatial_pairs": spatial_pairs, "consecutive_pairs": consecutive_pairs,
            "moisture_excursions": excursions,
            "schedule": schedule.tolist(), "moisture_mm": moisture.tolist(),
        }


def build_qubo(instance):
    """Equations (5)-(11), with automatic strictly sufficient budget penalty."""
    decisions = tuple((i, d - 1) for i, z in enumerate(instance.zones) for d in z.window_days)
    slack_weights = bounded_slack_weights(instance.budget_events)
    nx = len(decisions)
    n = nx + len(slack_weights)
    # Fail before allocating a dense matrix for an accidentally enormous input.
    if n > 2048:
        raise ValueError("This dense educational QUBO builder supports at most 2048 variables")
    linear = np.zeros(n)
    quadratic = np.zeros((n, n))
    constant = 0.0
    a = instance.dose_mm
    lookup = {pair: i for i, pair in enumerate(decisions)}
    for z_index, zone in enumerate(instance.zones):
        delta = zone.initial_moisture_mm + np.cumsum(zone.forcing_mm) - zone.target_mm
        S = np.cumsum(zone.stress_weights[::-1])[::-1]
        V = np.cumsum((zone.stress_weights * delta)[::-1])[::-1]
        constant += float(np.sum(zone.stress_weights * delta ** 2))
        for day in zone.window_days:
            d = day - 1
            i = lookup[z_index, d]
            linear[i] = instance.water_cost * a + a * a * S[d] + 2 * a * V[d]
            for later_day in zone.window_days:
                if later_day <= day:
                    continue
                j = lookup[z_index, later_day - 1]
                quadratic[i, j] += 2 * a * a * S[later_day - 1]
                if later_day == day + 1:
                    quadratic[i, j] += instance.timing_weight
    for z, other in instance.adjacency:
        for day in range(instance.days):
            if (z, day) in lookup and (other, day) in lookup:
                i, j = sorted((lookup[z, day], lookup[other, day]))
                quadratic[i, j] += instance.spatial_weight
    bound = max(0.0, -float(np.min(linear[:nx])))
    penalty = 1.1 * max(1.0, bound)
    coefficients = np.concatenate((np.ones(nx), slack_weights))
    K = instance.budget_events
    linear += penalty * (coefficients ** 2 - 2 * K * coefficients)
    quadratic += np.triu(2 * penalty * np.outer(coefficients, coefficients), k=1)
    constant += penalty * K * K
    if not (np.isfinite(linear).all() and np.isfinite(quadratic).all() and np.isfinite(constant)):
        raise ValueError("Input magnitudes overflowed the QUBO coefficients")
    return IrrigationQUBO(instance, decisions, linear, quadratic, constant, slack_weights, penalty, bound)


def greedy_schedule(model, depletion_fraction=None):
    """Daily depletion trigger with at most one irrigation event per day.

    Trigger is evaluated after that day's forecast forcing, before irrigation.
    This extra daily limit belongs only to this heuristic, not to the QUBO.
    By default each zone's growth-stage-dependent allowable depletion is used;
    a scalar override applies the same fraction to every zone and day.
    """
    if depletion_fraction is not None:
        depletion_fraction = _number(depletion_fraction, "depletion_fraction")
        if depletion_fraction > 1:
            raise ValueError("depletion_fraction must lie within [0, 1]")
    instance = model.instance
    moisture = np.array([z.initial_moisture_mm for z in instance.zones])
    x = np.zeros(model.n_decisions, dtype=int)
    lookup = {pair: i for i, pair in enumerate(model.decisions)}
    for day in range(instance.days):
        moisture += np.array([z.forcing_mm[day] for z in instance.zones])
        if int(x.sum()) >= instance.budget_events:
            continue
        candidates = [i for i, z in enumerate(instance.zones)
                      if (i, day) in lookup and moisture[i] < (
                          z.moisture_threshold_mm[day] if depletion_fraction is None
                          else (1 - depletion_fraction) * z.taw_mm)]
        if candidates:
            zone = max(candidates, key=lambda i: instance.zones[i].taw_mm - moisture[i])
            x[lookup[zone, day]] = 1
            moisture[zone] += instance.dose_mm
    return x
