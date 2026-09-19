"""Auditable forecast exports and optional figures."""

import csv
import json

import numpy as np


def export_forecast(report, output):
    for filename, value in (("forecast.json", report["forecast"]),
                            ("forecast_input.json", report["forecast_input"])):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    rows = report["forecast"]["daily"]
    with (output / "forecast.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_forecast(forecast, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 1 + len(forecast["soil"])
    figure, axes = plt.subplots((panels + 1) // 2, 2, squeeze=False,
                                figsize=(12, 3.6 * ((panels + 1) // 2)), layout="constrained")
    axes = axes.ravel()
    count = forecast["evaluation"]["holdout_samples"]
    hdays = np.arange(1 - count, 1)
    fdays = np.arange(1, len(forecast["et0"]["predicted_mm"]) + 1)
    et0 = forecast["et0"]
    axis = axes[0]
    axis.plot(hdays, et0["holdout_observed_mm"], label="Historical FAO56 reference", color="#687785")
    axis.plot(hdays, et0["holdout_predicted_mm"], label="Recursive holdout", color="#cb8436")
    axis.plot(fdays, et0["predicted_mm"], marker="o", markersize=3, label="Forecast after refit", color="#256b52")
    axis.plot(fdays, et0["fao56_weather_forecast_mm"], linestyle=":", label="FAO56 of forecast weather", color="#687785")
    axis.set_title("ET0 | weather-conditioned daily prediction", loc="left")
    axis.set_ylabel("Reference evapotranspiration (mm/day)")
    for axis, (name, soil) in zip(axes[1:], forecast["soil"].items()):
        axis.plot(hdays, soil["holdout_observed_kpa"], label="Provided observations", color="#687785")
        axis.plot(hdays, soil["holdout_predicted_kpa"], label="Recursive holdout", color="#cb8436")
        axis.plot(fdays, soil["predicted_tension_kpa"], marker="o", markersize=3,
                  label="Zero-irrigation forecast after refit", color="#256b52")
        if soil["clipped_days"]:
            axis.plot(fdays, soil["raw_predicted_tension_kpa"], linestyle=":", color="#b44848", label="Raw prediction")
        axis.set_title(f"{name} | soil water tension", loc="left")
        axis.set_ylabel("Positive suction (kPa)")
    for axis in axes[:panels]:
        axis.axvline(0.5, color="#46515a", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Day relative to planning start (1); dashed line = refit boundary")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=7)
    for axis in axes[panels:]:
        axis.set_visible(False)
    figure.suptitle(f"Prediction to scheduling | {forecast['method']} | daily adaptation", weight="bold")
    figure.savefig(output / "forecast.png", dpi=160)
    figure.savefig(output / "forecast.svg")
    plt.close(figure)
