"""Daily reference evapotranspiration and calibrated soil-water conversion.

FAO56 Chapter 2, equation (6), and Chapter 3, equations (8), (11)--(13),
and (17), define the meteorological calculation used here:
https://www.fao.org/4/x0490e/x0490e06.htm
https://www.fao.org/4/x0490e/x0490e07.htm

The soil curve is supplied by the user; this module does not infer a universal
relationship between soil-water tension and root-zone available water.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def daily_fao56(weather: dict) -> float:
    """Return daily short-grass FAO56 ET0 in mm/day from one weather record.

    Temperatures are degrees Celsius, humidity extrema are percentages, wind
    is measured at 2 m in m/s, radiation and heat flux are MJ/m²/day, and
    pressure is kPa. Pressure defaults to 101.3 and soil heat flux to zero.
    The daily coefficients are Cn=900 and Cd=0.34. A negative result is
    reported as zero evaporative demand, rather than a condensation input.
    Net radiation and soil heat flux may both be negative.

    This estimates ET0 conditional on the supplied weather; it does not
    forecast the weather or calculate radiation from incoming solar flux.
    """
    if not isinstance(weather, dict):
        raise ValueError("weather must be an object")
    required = (
        "tmin_c", "tmax_c", "relative_humidity_min_pct",
        "relative_humidity_max_pct", "wind_speed_2m_m_s",
        "net_radiation_mj_m2_day",
    )
    values = {}
    for name in required:
        if name not in weather:
            raise ValueError(f"Missing weather field: {name}")
        values[name] = _finite_number(weather[name], name)
    pressure = _finite_number(weather.get("atmospheric_pressure_kpa", 101.3),
                              "atmospheric_pressure_kpa")
    heat_flux = _finite_number(weather.get("soil_heat_flux_mj_m2_day", 0.0),
                               "soil_heat_flux_mj_m2_day")
    tmin, tmax = values["tmin_c"], values["tmax_c"]
    rhmin = values["relative_humidity_min_pct"]
    rhmax = values["relative_humidity_max_pct"]
    wind = values["wind_speed_2m_m_s"]
    radiation = values["net_radiation_mj_m2_day"]
    if not (-80 <= tmin <= tmax <= 60):
        raise ValueError("Temperatures must satisfy -80 <= tmin_c <= tmax_c <= 60")
    if not (0 <= rhmin <= rhmax <= 100):
        raise ValueError("Humidity must satisfy 0 <= relative_humidity_min_pct "
                         "<= relative_humidity_max_pct <= 100")
    if wind < 0:
        raise ValueError("wind_speed_2m_m_s must be nonnegative")
    if pressure <= 0:
        raise ValueError("atmospheric_pressure_kpa must be strictly positive")

    def saturation(temperature):
        return 0.6108 * math.exp(17.27 * temperature / (temperature + 237.3))

    temperature = (tmin + tmax) / 2
    vapour_min, vapour_max = saturation(tmin), saturation(tmax)
    saturation_pressure = (vapour_min + vapour_max) / 2
    actual_pressure = (vapour_min * rhmax + vapour_max * rhmin) / 200
    slope = 4098 * saturation(temperature) / (temperature + 237.3) ** 2
    psychrometric = 0.000665 * pressure
    numerator = (0.408 * slope * (radiation - heat_flux)
                 + psychrometric * 900 / (temperature + 273) * wind
                 * (saturation_pressure - actual_pressure))
    denominator = slope + psychrometric * (1 + 0.34 * wind)
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        raise ValueError("Weather magnitudes overflowed the FAO56 calculation")
    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError("Weather magnitudes overflowed the FAO56 calculation")
    return max(0.0, result)


@dataclass(frozen=True)
class SoilRetentionCurve:
    """User-calibrated tension magnitude (kPa) versus available water (mm).

    Available water is the scheduling model's root-zone storage above the
    permanent wilting point: zero at wilting and TAW at field capacity. It is
    not volumetric moisture. The increasing positive tension magnitude uses
    the dry-soil direction, rather than signed matric potential.
    """

    tension_kpa: tuple
    available_water_mm: tuple
    taw_mm: float

    @classmethod
    def from_dict(cls, data: dict, taw_mm: float):
        """Validate a monotone piecewise-linear calibration spanning 0..TAW."""
        if not isinstance(data, dict):
            raise ValueError("soil retention curve must be an object")
        taw = _finite_number(taw_mm, "taw_mm")
        if taw <= 0:
            raise ValueError("taw_mm must be strictly positive")
        series = {}
        for name in ("tension_kpa", "available_water_mm"):
            if name not in data:
                raise ValueError(f"Missing soil retention curve field: {name}")
            values = data[name]
            if not isinstance(values, (list, tuple)) or len(values) < 2:
                raise ValueError(f"{name} must contain at least two calibration points")
            series[name] = tuple(_finite_number(v, name) for v in values)
        tension = series["tension_kpa"]
        water = series["available_water_mm"]
        if len(tension) != len(water):
            raise ValueError("tension_kpa and available_water_mm must have equal lengths")
        if tension[0] < 0 or any(b <= a for a, b in zip(tension, tension[1:])):
            raise ValueError("tension_kpa must be nonnegative and strictly increasing")
        if (any(v < 0 or v > taw for v in water)
                or any(b >= a for a, b in zip(water, water[1:]))):
            raise ValueError("available_water_mm must be within [0, TAW] and strictly decreasing")
        if water[0] != taw or water[-1] != 0:
            raise ValueError("available_water_mm endpoints must be TAW and 0")
        return cls(tension, water, taw)

    def water_from_tension(self, tension) -> float:
        """Interpolate water; reject tension outside the calibrated domain."""
        tension = _finite_number(tension, "tension_kpa")
        if tension < self.tension_kpa[0] or tension > self.tension_kpa[-1]:
            raise ValueError("tension_kpa is outside the calibrated soil retention domain")
        return float(np.interp(tension, self.tension_kpa, self.available_water_mm))

    def tension_from_water(self, water) -> float:
        """Invert the calibration; reject available water outside [0, TAW]."""
        water = _finite_number(water, "available_water_mm")
        if water < self.available_water_mm[-1] or water > self.available_water_mm[0]:
            raise ValueError("available_water_mm is outside the calibrated soil retention domain")
        return float(np.interp(water, self.available_water_mm[::-1], self.tension_kpa[::-1]))
