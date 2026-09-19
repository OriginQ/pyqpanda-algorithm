"""Daily, weather-conditioned forecasting followed by fixed-baseline scheduling.

Weather and lagged observations drive daily ET0 and soil-water-tension
predictions. Calibrated tension forecasts define a no-new-irrigation water
baseline for the scheduling model. See docs/forecasting.md for the workflow.
"""

from copy import deepcopy
from datetime import date, timedelta

import numpy as np

from .forecast_physics import SoilRetentionCurve, daily_fao56
from .forecast_regression import make_regressor, regression_metrics
from .model import IrrigationInstance, _integer, _number


WEATHER_FEATURES = (
    "net_radiation_mj_m2_day", "tmin_c", "tmax_c",
    "relative_humidity_min_pct", "relative_humidity_max_pct",
    "wind_speed_2m_m_s", "atmospheric_pressure_kpa",
    "soil_heat_flux_mj_m2_day", "rainfall_mm",
)


def forecast_template(input_data):
    """Validate a planning template while allowing the three predicted fields to be omitted.

    Zero placeholders are used only for structural validation, never as
    observations or as returned predictions. Plain scheduling still requires
    all three fields.
    """
    prepared = deepcopy(input_data)
    if isinstance(prepared, dict) and isinstance(prepared.get("zones"), list):
        for zone in prepared["zones"]:
            if isinstance(zone, dict):
                for field in ("et0_mm", "rainfall_mm", "initial_moisture_mm"):
                    zone.setdefault(field, 0.0)
    return prepared, IrrigationInstance.from_dict(prepared)


def _records(rows, name):
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"{name} must be a nonempty list of daily objects")
    dates = []
    for row in rows:
        value = row.get("date")
        try:
            parsed = date.fromisoformat(value)
            if parsed.isoformat() != value:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: date must be YYYY-MM-DD") from exc
        if dates and parsed - dates[-1] != timedelta(days=1):
            raise ValueError(f"{name} must have consecutive, strictly increasing daily dates")
        dates.append(parsed)
    return dates


def _weather(rows):
    features, reference = [], []
    for original in rows:
        row = dict(original)
        row.setdefault("atmospheric_pressure_kpa", 101.3)
        row.setdefault("soil_heat_flux_mj_m2_day", 0.0)
        reference.append(daily_fao56(row))
        try:
            _number(row["rainfall_mm"], "rainfall_mm")
            features.append([row[key] for key in WEATHER_FEATURES])
        except KeyError as exc:
            raise ValueError(f"Missing weather field: {exc.args[0]}") from exc
    return np.asarray(features, dtype=float), np.asarray(reference)


def _lagged(target, index, lookback):
    return [target[index - offset] for offset in range(1, lookback + 1)]


def _et0_predict(model, weather, history, lookback):
    values = list(history)
    raw = []
    for row in weather:
        value = float(model.predict([list(row) + _lagged(values, len(values), lookback)])[0])
        if not np.isfinite(value):
            raise ValueError("ET0 prediction is nonfinite")
        raw.append(value)
        values.append(max(0.0, value))
    return np.asarray(values[-len(weather):]), np.asarray(raw)


def _soil_predict(model, weather, et0, irrigation, history, lookback, bounds):
    values, raw = list(history), []
    for row, demand, water in zip(weather, et0, irrigation):
        x = list(row) + [demand, water] + _lagged(values, len(values), lookback)
        value = float(model.predict([x])[0])
        if not np.isfinite(value):
            raise ValueError("Soil water tension prediction is nonfinite")
        raw.append(value)
        # Conversion is defined only on the supplied retention table. Preserve
        # raw outputs and flags; clipping here never clips a scheduled bucket.
        values.append(float(np.clip(value, *bounds)))
    return np.asarray(values[-len(weather):]), np.asarray(raw)


def prepare_forecast(input_data, forecast_input, method="ridge", seed=7, maxiter=30):
    """Return (resolved scheduling dict, JSON-safe forecast audit report).

    Holdout soil forecasts recurse from the training boundary, using observed
    holdout weather/irrigation as exogenous inputs. No holdout tension is fed
    back into prediction. Final models refit all pre-planning observations.
    """
    input_data, instance = forecast_template(input_data)
    if instance.start_date is None:
        raise ValueError("Forecasting requires a scheduling start_date")
    if not isinstance(forecast_input, dict):
        raise ValueError("Forecast input must be an object")
    if forecast_input.get("frequency") != "daily":
        raise ValueError("Forecast frequency must be 'daily'; hourly data must be explicitly aggregated first")
    note = forecast_input.get("source_note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("Forecast source_note must describe data provenance")
    lookback = _integer(forecast_input.get("lookback_days", 3), "lookback_days", 1)
    fraction = _number(forecast_input.get("holdout_fraction", 0.2), "holdout_fraction", strictly_positive=True)
    if fraction > 0.5:
        raise ValueError("holdout_fraction must be within (0, 0.5]")
    past = forecast_input.get("weather_history")
    future = forecast_input.get("weather_forecast")
    past_dates = _records(past, "weather_history")
    future_dates = _records(future, "weather_forecast")
    start = date.fromisoformat(instance.start_date)
    if start == date.min:
        raise ValueError("start_date must allow a preceding history day")
    if past_dates[-1] != start - timedelta(days=1):
        raise ValueError("weather_history must end the day before start_date (no future observations)")
    if future_dates[0] != start or len(future) != instance.days:
        raise ValueError("weather_forecast dates must match the complete scheduling horizon")
    n = len(past)
    if lookback >= n:
        raise ValueError("lookback_days must be shorter than weather_history")
    holdout_count = max(2, int(np.ceil((n - lookback) * fraction)))
    split = n - holdout_count
    if split - lookback < 8:
        raise ValueError("Insufficient history: need at least 8 training targets and 2 holdout days after lookback")
    weather, et0_observed = _weather(past)
    future_weather, future_fao56 = _weather(future)
    zone_inputs = forecast_input.get("zones")
    if (not isinstance(zone_inputs, list) or any(not isinstance(z, dict) for z in zone_inputs)
            or any(not isinstance(z.get("name"), str) for z in zone_inputs)):
        raise ValueError("Forecast zones must be a list of named objects")
    names = [z["name"] for z in zone_inputs]
    if len(set(names)) != len(names) or set(names) != {z.name for z in instance.zones}:
        raise ValueError("Forecast zones must match the scheduling zones exactly, without duplicates")
    zone_lookup = {z["name"]: z for z in zone_inputs}
    # Validate all observations and calibration tables before fitting a model.
    validated = []
    for zone in instance.zones:
        source = zone_lookup[zone.name]
        history = source.get("history")
        if _records(history, f"{zone.name}.history") != past_dates:
            raise ValueError(f"{zone.name}: soil history dates must match weather_history")
        curve = SoilRetentionCurve.from_dict(source.get("retention_curve"), zone.taw_mm)
        try:
            tension = np.array([_number(r["tension_kpa"], "tension_kpa") for r in history])
            irrigation = np.array([_number(r["net_irrigation_mm"], "net_irrigation_mm") for r in history])
        except KeyError as exc:
            raise ValueError(f"Missing soil history field: {exc.args[0]}") from exc
        for value in tension:
            curve.water_from_tension(float(value))
        validated.append((zone, curve, tension, irrigation))

    et0_x = np.array([list(weather[i]) + _lagged(et0_observed, i, lookback) for i in range(lookback, n)])
    et0_y = et0_observed[lookback:]
    et0_validation_model = make_regressor(method, seed=seed, maxiter=maxiter).fit(
        et0_x[:split - lookback], et0_y[:split - lookback])
    et0_holdout, et0_holdout_raw = _et0_predict(
        et0_validation_model, weather[split:], et0_observed[:split], lookback)
    et0_model = make_regressor(method, seed=seed, maxiter=maxiter).fit(et0_x, et0_y)
    et0_future, et0_future_raw = _et0_predict(et0_model, future_weather, et0_observed, lookback)
    resolved = deepcopy(input_data)
    report = {
        "reference": "Wheat irrigation forecasting workflow: docs/forecasting.md",
        "adaptation": "Daily weather-conditioned ET0 and autoregressive soil-tension forecasts; ridge or trainable 4-qubit VQC, coupled to fixed-baseline scheduling",
        "source_note": note, "frequency": "daily", "method": method, "seed": seed,
        "lookback_days": lookback,
        "units": {"et0": "mm/day", "soil_water_tension": "kPa (positive suction)", "available_water": "mm above wilting point"},
        "evaluation": {
            "mode": "chronological recursive holdout, conditioned on observed weather and recorded irrigation; not an archived-weather-forecast skill test",
            "training_start": past[lookback]["date"], "training_end": past[split - 1]["date"],
            "holdout_start": past[split]["date"], "holdout_end": past[-1]["date"],
            "training_samples": split - lookback, "holdout_samples": holdout_count,
            "final_refit_end": past[-1]["date"], "final_refit_samples": n - lookback,
        },
        "et0": {
            "target_source": "FAO56 computed reference, not measured evapotranspiration",
            "feature_names": list(WEATHER_FEATURES) + [f"et0_lag_{i}_mm" for i in range(1, lookback + 1)],
            "holdout_metrics": regression_metrics(et0_observed[split:], et0_holdout),
            "persistence_holdout_metrics": regression_metrics(et0_observed[split:], np.full(holdout_count, et0_observed[split - 1])),
            "holdout_observed_mm": et0_observed[split:].tolist(),
            "holdout_predicted_mm": et0_holdout.tolist(), "holdout_raw_predicted_mm": et0_holdout_raw.tolist(),
            "predicted_mm": et0_future.tolist(), "raw_predicted_mm": et0_future_raw.tolist(),
            "fao56_weather_forecast_mm": future_fao56.tolist(),
            "validation_model": et0_validation_model.describe(), "final_model": et0_model.describe(),
        },
        "soil": {}, "daily": [],
        "scheduling_assumption": "Forecast future net irrigation is zero. Convert tension to a fixed no-irrigation baseline B; schedule water = B + cumulative net irrigation. This additive approximation does not model irrigation-dependent drainage or learned causal treatment effects.",
    }
    for zi, (zone, curve, tension, irrigation) in enumerate(validated):
        soil_x = np.array([list(weather[i]) + [et0_observed[i], irrigation[i]]
                           + _lagged(tension, i, lookback) for i in range(lookback, n)])
        soil_y = tension[lookback:]
        soil_validation_model = make_regressor(method, seed=seed + zi + 1, maxiter=maxiter).fit(
            soil_x[:split - lookback], soil_y[:split - lookback])
        bounds = (curve.tension_kpa[0], curve.tension_kpa[-1])
        holdout, holdout_raw = _soil_predict(soil_validation_model, weather[split:], et0_holdout,
                                            irrigation[split:], tension[:split], lookback, bounds)
        soil_model = make_regressor(method, seed=seed + zi + 1, maxiter=maxiter).fit(soil_x, soil_y)
        predicted, raw = _soil_predict(soil_model, future_weather, et0_future,
                                       np.zeros(instance.days), tension, lookback, bounds)
        initial = curve.water_from_tension(float(tension[-1]))
        baseline = np.array([curve.water_from_tension(float(v)) for v in predicted])
        physical_forcing = (zone.effective_rain_fraction * future_weather[:, -1]
                            + zone.capillary_rise_mm - zone.crop_coefficient * et0_future)
        adjustment = np.diff(np.r_[initial, baseline]) - physical_forcing
        output_zone = resolved["zones"][zi]
        output_zone.update(initial_moisture_mm=initial, et0_mm=et0_future.tolist(),
                           rainfall_mm=future_weather[:, -1].tolist(), forecast_adjustment_mm=adjustment.tolist())
        report["soil"][zone.name] = {
            "target_source": "provided daily end-of-day soil water tension observations",
            "feature_names": list(WEATHER_FEATURES) + ["et0_mm", "net_irrigation_mm"]
                             + [f"tension_lag_{i}_kpa" for i in range(1, lookback + 1)],
            "holdout_metrics": regression_metrics(tension[split:], holdout),
            "persistence_holdout_metrics": regression_metrics(tension[split:], np.full(holdout_count, tension[split - 1])),
            "holdout_observed_kpa": tension[split:].tolist(), "holdout_predicted_kpa": holdout.tolist(),
            "holdout_raw_predicted_kpa": holdout_raw.tolist(),
            "retention_curve": deepcopy(zone_lookup[zone.name]["retention_curve"]),
            "initial_tension_kpa": float(tension[-1]), "initial_available_water_mm": initial,
            "predicted_tension_kpa": predicted.tolist(), "raw_predicted_tension_kpa": raw.tolist(),
            "no_irrigation_moisture_mm": baseline.tolist(), "forecast_adjustment_mm": adjustment.tolist(),
            "clipped_days": (np.flatnonzero(raw != predicted) + 1).tolist(),
            "historical_zero_irrigation_days": int(np.count_nonzero(irrigation == 0)),
            "validation_model": soil_validation_model.describe(), "final_model": soil_model.describe(),
        }
        for d in range(instance.days):
            report["daily"].append({
                "zone": zone.name, "day": d + 1, "date": future[d]["date"], "growth_stage": zone.growth_stage[d],
                "et0_mm": float(et0_future[d]), "et0_raw_mm": float(et0_future_raw[d]),
                "et0_fao56_mm": float(future_fao56[d]), "etc_mm": float(zone.crop_coefficient[d] * et0_future[d]),
                "rainfall_mm": float(future_weather[d, -1]), "forecast_net_irrigation_mm": 0.0,
                "soil_water_tension_kpa": float(predicted[d]), "soil_water_tension_raw_kpa": float(raw[d]),
                "tension_clipped": bool(predicted[d] != raw[d]),
                "no_irrigation_moisture_mm": float(baseline[d]), "forecast_adjustment_mm": float(adjustment[d]),
                "moisture_threshold_mm": float(zone.moisture_threshold_mm[d]),
                "below_threshold": bool(baseline[d] < zone.moisture_threshold_mm[d]),
            })
    resolved["source_note"] = input_data.get("source_note", "") + " | Forecast: " + note
    IrrigationInstance.from_dict(resolved)
    return resolved, report
