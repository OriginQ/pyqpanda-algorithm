"""Forecast leakage boundaries, physical integration and portable CLI replay."""

import contextlib
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wheat_irrigation.cli import load_scenario, main
from wheat_irrigation.forecast_demo import make_forecast_demo
from wheat_irrigation.forecasting import prepare_forecast
from wheat_irrigation.model import IrrigationInstance, build_qubo
from wheat_irrigation.solvers import enumerate_spectrum


class ForecastPipelineTests(unittest.TestCase):
    def assert_nested_close(self, first, second):
        """BLAS may vary in its last few bits; structure and labels remain exact."""
        if isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_close(first[key], second[key])
        elif isinstance(first, list):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_close(a, b)
        elif isinstance(first, float):
            self.assertAlmostEqual(first, second, delta=1e-9 + abs(first) * 1e-10)
        else:
            self.assertEqual(first, second)

    @classmethod
    def setUpClass(cls):
        cls.source = load_scenario("jointing")
        cls.observations = make_forecast_demo(cls.source)
        cls.resolved, cls.forecast = prepare_forecast(cls.source, cls.observations)

    def test_deterministic_forecast_does_not_mutate_sources(self):
        source, observations = deepcopy(self.source), deepcopy(self.observations)
        resolved, forecast = prepare_forecast(source, observations)
        self.assertEqual(source, self.source)
        self.assertEqual(observations, self.observations)
        self.assert_nested_close(resolved, self.resolved)
        self.assert_nested_close(forecast, self.forecast)
        self.assertEqual(len(forecast["daily"]), 21)
        json.dumps(forecast, allow_nan=False)
        self.assertLess(forecast["evaluation"]["final_refit_end"], source["start_date"])

    def test_all_states_match_independent_fixed_baseline_and_budget(self):
        model = build_qubo(IrrigationInstance.from_dict(self.resolved))
        bits, energies = enumerate_spectrum(model)
        baseline = np.array([self.forecast["soil"][z.name]["no_irrigation_moisture_mm"]
                             for z in model.instance.zones])
        np.testing.assert_allclose(model.moisture_trajectory(np.zeros(model.n_decisions)), baseline, atol=1e-12)
        h, coupling, offset = model.to_ising()
        for full, energy in zip(bits, energies):
            schedule = np.zeros((3, 7))
            for bit, (zone, day) in zip(full, model.decisions):
                schedule[zone, day] = bit
            water = baseline + model.instance.dose_mm * np.cumsum(schedule, axis=1)
            cost = sum(np.sum(z.stress_weights * (water[i] - z.target_mm) ** 2)
                       for i, z in enumerate(model.instance.zones))
            cost += model.instance.water_cost * model.instance.dose_mm * schedule.sum()
            cost += model.instance.timing_weight * np.sum(schedule[:, :-1] * schedule[:, 1:])
            cost += model.instance.spatial_weight * sum(np.sum(schedule[i] * schedule[j]) for i, j in model.instance.adjacency)
            residual = schedule.sum() + full[model.n_decisions:] @ model.slack_weights - model.instance.budget_events
            self.assertAlmostEqual(energy, cost + model.budget_penalty * residual ** 2, places=7)
            spins = 1 - 2 * full.astype(float)
            self.assertAlmostEqual(energy, offset + spins @ h + spins @ coupling @ spins, places=7)
        for index, (zone, day) in enumerate(model.decisions):
            decisions = np.zeros(model.n_decisions)
            decisions[index] = 1
            expected = baseline.copy()
            expected[zone, day:] += model.instance.dose_mm
            np.testing.assert_allclose(model.moisture_trajectory(decisions), expected, atol=1e-12)

    def test_holdout_tension_is_never_fed_back_into_validation(self):
        changed = deepcopy(self.observations)
        split = len(changed["weather_history"]) - self.forecast["evaluation"]["holdout_samples"]
        for row in changed["zones"][0]["history"][split:]:
            row["tension_kpa"] += 10
        _, report = prepare_forecast(self.source, changed)
        old, new = self.forecast["soil"]["Wheat-A"], report["soil"]["Wheat-A"]
        self.assert_nested_close(old["validation_model"], new["validation_model"])
        self.assert_nested_close(old["holdout_predicted_kpa"], new["holdout_predicted_kpa"])
        self.assertNotEqual(old["holdout_metrics"], new["holdout_metrics"])
        self.assertNotEqual(old["final_model"], new["final_model"])

    def test_future_weather_changes_forecast_but_never_training_or_validation(self):
        changed = deepcopy(self.observations)
        for row in changed["weather_forecast"]:
            row["net_radiation_mj_m2_day"] += 5
        resolved, report = prepare_forecast(self.source, changed)
        for section in ("validation_model", "final_model", "holdout_metrics"):
            self.assert_nested_close(self.forecast["et0"][section], report["et0"][section])
        self.assertNotEqual(self.forecast["et0"]["predicted_mm"], report["et0"]["predicted_mm"])
        self.assertNotEqual(self.forecast["soil"]["Wheat-A"]["no_irrigation_moisture_mm"],
                            report["soil"]["Wheat-A"]["no_irrigation_moisture_mm"])
        self.assertNotEqual(self.resolved["zones"][0]["forecast_adjustment_mm"], resolved["zones"][0]["forecast_adjustment_mm"])

    def test_strict_dates_units_calibration_and_required_observations(self):
        def missing_irrigation(data):
            data["zones"][0]["history"][0].pop("net_irrigation_mm")
        mutations = [
            lambda d: d.update(frequency="hourly"),
            lambda d: d.update(lookback_days=True),
            lambda d: d.update(lookback_days=10 ** 1000),
            lambda d: d.update(holdout_fraction=0),
            lambda d: d.update(holdout_fraction=10 ** 1000),
            lambda d: d.update(weather_history=d["weather_history"][:5]),
            lambda d: d["weather_history"][-1].update(date=self.source["start_date"]),
            lambda d: d["weather_history"][1].update(date=d["weather_history"][0]["date"]),
            lambda d: d["weather_forecast"].pop(),
            lambda d: d["zones"][0]["history"][0].update(date="2020-01-01"),
            lambda d: d["zones"][0].update(name="unknown"),
            lambda d: d["zones"][0].update(retention_curve=None),
            lambda d: d["zones"][0]["history"][0].update(tension_kpa=-1),
            lambda d: d["zones"][0]["history"][0].update(tension_kpa=1501),
            lambda d: d["weather_forecast"][0].update(rainfall_mm=float("nan")),
            lambda d: d["weather_forecast"][0].update(rainfall_mm=10 ** 1000),
            lambda d: d["weather_forecast"][0].update(tmax_c=True),
            missing_irrigation,
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.observations)
                mutation(changed)
                with self.assertRaises(ValueError):
                    prepare_forecast(self.source, changed)

    def test_signed_adjustment_validates_and_default_keeps_original_scenarios(self):
        expected_costs = {"jointing": 7103.300119, "heading": 40521.998025, "grain_filling": 9008.408777}
        for scenario, cost in expected_costs.items():
            raw = load_scenario(scenario)
            original = build_qubo(IrrigationInstance.from_dict(raw))
            self.assertAlmostEqual(float(enumerate_spectrum(original)[1].min()), cost, places=5)
            for z in raw["zones"]:
                z["forecast_adjustment_mm"] = [0] * raw["days"]
            explicit = build_qubo(IrrigationInstance.from_dict(raw))
            np.testing.assert_array_equal(original.linear, explicit.linear)
            self.assertEqual(original.constant, explicit.constant)
        for bad in ([0] * 6, True, float("nan"), float("inf")):
            changed = deepcopy(self.resolved)
            changed["zones"][0]["forecast_adjustment_mm"] = bad
            with self.assertRaises(ValueError):
                IrrigationInstance.from_dict(changed)

    def test_cli_exports_raw_and_resolved_inputs_and_replays_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            output, replay, custom = [Path(directory) / name for name in ("forecast", "replay", "custom")]
            snapshot = Path(directory) / "saved.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--forecast-demo", "--backend", "classical", "--no-plot", "--shots", "16",
                                       "--export-input", str(snapshot), "--output", str(output)]), 0)
                self.assertEqual(main(["--input", str(snapshot), "--backend", "classical", "--no-plot",
                                       "--shots", "16", "--output", str(replay)]), 0)
                self.assertEqual(main(["--forecast-input", str(output / "forecast_input.json"),
                                       "--backend", "classical", "--no-plot", "--shots", "16", "--output", str(custom)]), 0)
            report = json.loads((output / "summary.json").read_text())
            repeated = json.loads((replay / "summary.json").read_text())
            self.assertEqual(report["solutions"], repeated["solutions"])
            self.assertEqual(report["input_data"], json.loads(snapshot.read_text()))
            self.assertEqual(report["input_data"], json.loads((output / "resolved_input.json").read_text()))
            self.assertEqual((output / "qubo.json").read_text(), (replay / "qubo.json").read_text())
            self.assert_nested_close(json.loads((output / "forecast.json").read_text()),
                                     json.loads((custom / "forecast.json").read_text()))
            with (output / "forecast.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 21)
            for row, expected in zip(rows, report["forecast"]["daily"]):
                self.assertEqual(float(row["et0_mm"]), expected["et0_mm"])
                self.assertEqual(float(row["no_irrigation_moisture_mm"]), expected["no_irrigation_moisture_mm"])
            self.assertIn("预测—调度", (output / "report.md").read_text())
            self.assertFalse((output / "forecast.png").exists())

    def test_forecast_templates_can_omit_only_the_predicted_fields(self):
        template = deepcopy(self.source)
        for zone in template["zones"]:
            for field in ("et0_mm", "rainfall_mm", "initial_moisture_mm"):
                zone.pop(field)
        with self.assertRaises(ValueError):
            IrrigationInstance.from_dict(template)
        self.assertEqual(make_forecast_demo(template), self.observations)
        resolved, _ = prepare_forecast(template, self.observations)
        self.assert_nested_close(resolved, self.resolved)
        template["zones"][0].pop("taw_mm")
        with self.assertRaises(ValueError):
            prepare_forecast(template, self.observations)

    def test_history_date_underflow_fails_cleanly(self):
        for start in ("0001-01-01", "0001-03-20"):
            source = deepcopy(self.source)
            source["start_date"] = start
            with self.assertRaises(ValueError):
                make_forecast_demo(source)

    def test_forecasting_runs_without_parent_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(ROOT / "wheat_irrigation", target / "wheat_irrigation",
                            ignore=shutil.ignore_patterns("__pycache__"))
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-m", "wheat_irrigation", "--scenario", "heading",
                                     "--forecast-demo", "--backend", "classical", "--no-plot", "--shots", "16"],
                                    cwd=target, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((target / "outputs" / "heading_forecast_ridge" / "summary.json").read_text())
            self.assertEqual(report["forecast"]["method"], "ridge")
            self.assertTrue(report["solutions"]["Exact"]["budget_feasible"])


if __name__ == "__main__":
    unittest.main()
