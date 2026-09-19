"""Exercise wheat scenarios, exported water accounting and portable execution."""

import contextlib
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

from wheat_irrigation.cli import SCENARIOS, load_scenario, main
from wheat_irrigation.model import IrrigationInstance, build_qubo
from wheat_irrigation.solvers import enumerate_spectrum
from test_irrigation import binary_vectors, independent_physics


class WheatApplicationTests(unittest.TestCase):
    def test_all_scenarios_match_independent_daily_physics(self):
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                data = load_scenario(name)
                model = build_qubo(IrrigationInstance.from_dict(data))
                self.assertEqual(model.n_variables, 11)
                self.assertEqual(model.instance.budget_events, 2)
                self.assertEqual(model.instance.crop, "winter_wheat")
                for x in binary_vectors(model.n_decisions):
                    if x.sum() > model.instance.budget_events:
                        continue
                    objective, _, _ = independent_physics(data, x)
                    self.assertAlmostEqual(model.energy(model.encode_schedule(x)), objective, places=7)
                bits, energies = enumerate_spectrum(model)
                exact = model.evaluate(bits[int(energies.argmin()), :model.n_decisions])
                self.assertLessEqual(exact["gross_water_m3"], data["water_budget_m3"])
                self.assertEqual(exact["moisture_excursions"], [])

    def test_cli_exports_gross_and_net_supply_consistently(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            snapshot = Path(directory) / "custom.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--scenario", "heading", "--backend", "classical", "--no-plot",
                             "--shots", "32", "--water-budget-m3", "350", "--efficiency", "0.8",
                             "--export-input", str(snapshot), "--output", str(output)])
            self.assertEqual(code, 0)
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["model"]["budget_events"], 1)
            self.assertEqual(report["model"]["water_budget_m3"], 350)
            self.assertEqual(report["input_data"], json.loads(snapshot.read_text(encoding="utf-8")))
            with (output / "schedules.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            for method, solution in report["solutions"].items():
                selected = [row for row in rows if row["solver"] == method]
                self.assertAlmostEqual(sum(float(row["net_water_m3"]) for row in selected), solution["net_water_m3"])
                self.assertAlmostEqual(sum(float(row["gross_water_m3"]) for row in selected), solution["gross_water_m3"])
                self.assertTrue(all(row["growth_stage"] == "heading" and row["date"] for row in selected))
            text = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("350.00 m³", text)
            self.assertIn("经典精确对照", text)

    def test_zero_supply_returns_only_zero_irrigation(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--backend", "classical", "--no-plot", "--shots", "32",
                      "--water-budget-m3", "0", "--output", directory])
            report = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["model"]["slack_variables"], 0)
            for solution in report["solutions"].values():
                self.assertEqual(solution["events"], 0)
                self.assertEqual(solution["gross_water_m3"], 0)

    def test_package_runs_when_copied_without_parent_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            shutil.copytree(ROOT / "wheat_irrigation", destination / "wheat_irrigation",
                            ignore=shutil.ignore_patterns("__pycache__"))
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-m", "wheat_irrigation", "--scenario", "grain_filling",
                 "--backend", "classical", "--no-plot", "--shots", "16"],
                cwd=destination, env=env, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((destination / "outputs" / "grain_filling" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["configuration"]["scenario"], "grain_filling")
            self.assertTrue(report["solutions"]["Exact"]["budget_feasible"])

    def test_invalid_cli_supply_fails_cleanly(self):
        for value in ("-1", "nan", "inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(["--water-budget-m3", value, "--no-plot"])
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
