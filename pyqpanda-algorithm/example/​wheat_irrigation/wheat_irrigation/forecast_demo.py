"""Deterministic synthetic observations and forecast weather, never field data."""

from datetime import date, timedelta

import numpy as np

from .forecast_physics import daily_fao56
from .forecasting import forecast_template


def make_forecast_demo(input_data, seed=7):
    _, instance = forecast_template(input_data)
    if not instance.start_date:
        raise ValueError("Forecast demo requires start_date")
    rng = np.random.default_rng(seed)
    start = date.fromisoformat(instance.start_date)
    history_days = 84
    if start.toordinal() <= history_days:
        raise ValueError("start_date must allow 84 preceding days for the synthetic demo")
    stage_offset = {"jointing": 0, "heading": 4, "grain_filling": 8}.get(instance.zones[0].growth_stage[0], 0)
    weather = []
    for t in range(history_days + instance.days):
        phase = 2 * np.pi * t / 17
        rain = float(rng.uniform(2, 8)) if t % 11 in (2, 3) else 0.0
        low = float(7 + stage_offset + 3 * np.sin(phase) + rng.normal(0, 0.4))
        row = {
            "date": (start + timedelta(days=t - history_days)).isoformat(),
            "tmin_c": round(low, 4), "tmax_c": round(low + 11 + rng.uniform(0, 2), 4),
            "relative_humidity_min_pct": round(35 + 10 * np.cos(phase) + (10 if rain else 0), 4),
            "relative_humidity_max_pct": round(77 + 7 * np.cos(phase) + (5 if rain else 0), 4),
            "wind_speed_2m_m_s": round(float(1.7 + 0.6 * np.sin(phase + 1)), 4),
            "net_radiation_mj_m2_day": round(float(10 + stage_offset * 0.2 + 3 * np.sin(phase) - (2 if rain else 0)), 4),
            "atmospheric_pressure_kpa": 96.0, "soil_heat_flux_mj_m2_day": 0.0,
            "rainfall_mm": round(rain, 4),
        }
        weather.append(row)
    zones = []
    for zi, zone in enumerate(instance.zones):
        tension = 28.0 + 6 * zi
        history = []
        for t, row in enumerate(weather[:history_days]):
            irrigation = 15.0 if (t + 3 * zi) % 19 == 8 else 0.0
            tension = float(np.clip(0.96 * tension + 0.60 * daily_fao56(row)
                                    - 0.5 * row["rainfall_mm"] - 0.4 * irrigation
                                    + 0.65 + 0.10 * zi + rng.normal(0, 0.25), 2, 110))
            history.append({"date": row["date"], "tension_kpa": round(tension, 5),
                            "net_irrigation_mm": irrigation})
        zones.append({
            "name": zone.name,
            "retention_curve": {"tension_kpa": [0, 10, 30, 60, 1500],
                                "available_water_mm": [zone.taw_mm * f for f in (1, 0.85, 0.6, 0.35, 0)]},
            "history": history,
        })
    return {
        "frequency": "daily", "lookback_days": 3, "holdout_fraction": 0.2,
        "source_note": "完全合成：84天历史气象、土壤水势、灌溉记录及未来天气和水势-根区可利用水量标定表；固定种子生成，用于流程与算法演示，非实测数据、实际天气预报或当地土壤标定。后期根据特定区域的实际情况和条件再替换成实际数据。",
        "weather_history": weather[:history_days], "weather_forecast": weather[history_days:],
        "zones": zones,
    }
