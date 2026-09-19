"""Exercise persisted QPU experiments without credentials, the SDK or cloud access."""

from __future__ import annotations

import builtins
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation import cli, hardware
from wheat_irrigation.solvers import enumerate_spectrum, qaoa_probabilities


class HardwareWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "result"
        self.input = self.root / "tiny.json"
        self.input.write_text(json.dumps({
            "name": "offline hardware workflow fixture", "source_note": "Synthetic test data",
            "days": 2, "budget_events": 1, "dose_mm": 9.0,
            "water_cost": 0.4, "spatial_weight": 0.0, "timing_weight": 1.0,
            "zone_area_ha": 1.0, "adjacency": [],
            "zones": [{
                "name": "A", "taw_mm": 40.0, "initial_moisture_mm": 17.0,
                "window_days": [1, 2], "target_mm": [24.0, 25.0],
                "stress_weights": [0.7, 1.2], "et0_mm": [3.0, 5.0],
                "crop_coefficient": 1.0, "rainfall_mm": [0.0, 0.0],
                "effective_rain_fraction": 0.6, "capillary_rise_mm": 0.0,
            }],
        }), encoding="utf-8")
        self.argv = [
            "--input", str(self.input), "--backend", "qcloud", "--qcloud-chip", "test-real-chip",
            "--output", str(self.output), "--depth", "1", "--maxiter", "3", "--restarts", "1",
            "--shots", "16", "--seed", "19", "--max-qubits", "8", "--no-plot",
        ]
        # If a test forgets to mock the cloud boundary, fail before loading the
        # SDK or reading a credential. Circuit export remains real and offline.
        sdk_guard = patch.object(hardware, "_load_sdk", side_effect=AssertionError("SDK access forbidden"))
        service_guard = patch.object(hardware, "_service", side_effect=AssertionError("Credential access forbidden"))
        sdk_guard.start()
        service_guard.start()
        self.addCleanup(sdk_guard.stop)
        self.addCleanup(service_guard.stop)

    @staticmethod
    def local_result(model, **settings):
        """Supply deterministic ideal optimizer output, with distinct local shots."""
        gammas, betas = [0.21], [0.32]
        _, energies = enumerate_spectrum(model)
        h, couplings, offset = model.to_ising()
        return {
            "probabilities": qaoa_probabilities(model, gammas, betas, backend="numpy"),
            "counts": {int(energies.argmin()): settings["shots"]},
            "backend": "numpy", "engine": "offline deterministic optimizer fixture",
            "circuit_evaluations": 1, "parameters": {"gammas": gammas, "betas": betas},
            "normalization": float(max(np.max(np.abs(h)), np.max(np.abs(couplings)))) or 1.0,
            "ising_offset": float(offset), "evaluations": 1, "optimizer_runs": [],
        }

    @staticmethod
    def cloud_result(circuit, settings, status="finished", counts=None):
        counts = ({0: settings["shots"]} if counts is None else counts) if status == "finished" else {}
        return {
            "job_id": settings.get("job_id") or "saved-job-123", "status": status,
            "counts": counts,
            "raw_counts": {format(i, f"0{circuit['n_qubits']}b"): n for i, n in counts.items()},
            "chip": settings["chip"], "shots": settings["shots"],
        }

    def run_cli(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main(argv)

    def read_json(self, name="summary.json", output=None):
        return json.loads(((output or self.output) / name).read_text(encoding="utf-8"))

    def start_pending(self):
        def pending(circuit, **settings):
            settings["on_submitted"]("saved-job-123")
            return self.cloud_result(circuit, settings, "pending")

        with patch.object(cli, "solve_qaoa", side_effect=self.local_result), \
                patch.object(hardware, "run_hardware", side_effect=pending):
            self.assertEqual(self.run_cli(self.argv), 0)
        return self.output / "hardware_job.json"

    def test_dry_run_works_without_any_quantum_sdk_and_has_no_hardware_samples(self):
        real_import = builtins.__import__

        def no_quantum_sdk(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"pyqpanda3", "pyqpanda_alg", "pyqpanda"}:
                raise AssertionError("Offline dry run must not import a quantum SDK")
            return real_import(name, *args, **kwargs)

        # Use the actual NumPy/SciPy local optimizer here to cover the default
        # dry-run path on machines with no quantum SDK installed.
        with patch("builtins.__import__", side_effect=no_quantum_sdk), \
                patch.object(hardware, "run_hardware") as cloud:
            self.assertEqual(self.run_cli(self.argv + ["--qcloud-dry-run"]), 0)
        cloud.assert_not_called()
        report = self.read_json()
        manifest = self.read_json("hardware_job.json")
        self.assertEqual(report["hardware"]["status"], "dry_run")
        self.assertIsNone(report["hardware"]["job_id"])
        self.assertIn("qaoa_reference", report)
        self.assertNotIn("qaoa", report)
        self.assertNotIn("QAOA", report["solutions"])
        self.assertNotIn("counts_by_state_index", report["qaoa_reference"])
        self.assertEqual(manifest["report"], report)
        self.assertEqual((self.output / "qaoa.originir").read_text(), manifest["circuit"]["originir"])
        self.assertEqual(manifest["circuit"]["gate_counts"]["MEASURE"], report["model"]["total_qubits"])

    def test_pending_id_is_persisted_before_poll_and_resume_reuses_original_settings(self):
        def pending(circuit, **settings):
            self.assertIsNone(settings["job_id"])
            settings["on_submitted"]("saved-job-123")
            checkpoint = self.read_json("hardware_job.json")
            self.assertEqual(checkpoint["report"]["hardware"]["job_id"], "saved-job-123")
            self.assertEqual(checkpoint["report"]["hardware"]["status"], "pending")
            return self.cloud_result(circuit, settings, "pending")

        with patch.object(cli, "solve_qaoa", side_effect=self.local_result), \
                patch.object(hardware, "run_hardware", side_effect=pending) as submit:
            self.assertEqual(self.run_cli(self.argv), 0)
        submit.assert_called_once()
        original = self.read_json()
        self.assertNotIn("qaoa", original)
        self.assertNotIn("QAOA", original["solutions"])
        self.input.unlink()  # Resume must use its saved resolved input.
        resumed_output = self.root / "resumed"

        def finished(circuit, **settings):
            self.assertEqual(settings["job_id"], "saved-job-123")
            self.assertEqual(settings["chip"], "test-real-chip")
            self.assertEqual(settings["shots"], 16)
            self.assertEqual(settings["timeout"], 2.0)
            self.assertEqual(settings["poll_interval"], 0.25)
            self.assertEqual(settings["token_env"], "TEST_QCLOUD_TOKEN")
            self.assertEqual(circuit, self.read_json("hardware_job.json")["circuit"])
            return self.cloud_result(circuit, settings, counts={0: 5, 1: 11})

        with patch.object(hardware, "run_hardware", side_effect=finished) as query, \
                patch.object(cli, "solve_qaoa", side_effect=AssertionError("Resume must not optimize")), \
                patch.object(cli, "load_scenario", side_effect=AssertionError("Resume must not reload inputs")), \
                patch.object(np.random, "default_rng", side_effect=AssertionError("Resume must not resample")):
            code = self.run_cli([
                "--qcloud-resume", str(self.output / "hardware_job.json"),
                "--output", str(resumed_output), "--no-plot", "--qcloud-timeout", "2",
                "--qcloud-poll-interval", "0.25", "--qcloud-token-env", "TEST_QCLOUD_TOKEN",
            ])
        self.assertEqual(code, 0)
        query.assert_called_once()
        report = self.read_json(output=resumed_output)
        self.assertEqual(report["configuration"], original["configuration"])
        self.assertEqual(report["uniform_sampling"], original["uniform_sampling"])
        self.assertEqual(report["qaoa"]["counts_by_state_index"], {"0": 5, "1": 11})
        self.assertEqual(self.read_json("hardware_job.json")["report"], report)
        self.assertEqual(self.read_json("hardware_job.json", resumed_output)["report"], report)

    def test_only_observed_qpu_counts_select_the_qaoa_candidate(self):
        def finished(circuit, **settings):
            settings["on_submitted"]("saved-job-123")
            return self.cloud_result(circuit, settings, counts={0: 16})

        with patch.object(cli, "solve_qaoa", side_effect=self.local_result), \
                patch.object(hardware, "run_hardware", side_effect=finished):
            self.assertEqual(self.run_cli(self.argv), 0)
        report = self.read_json()
        self.assertEqual(report["qaoa"]["counts_by_state_index"], {"0": 16})
        self.assertEqual(report["hardware"]["raw_counts"], {"000": 16})
        self.assertEqual(report["qaoa"]["backend"], "qcloud")
        self.assertEqual(report["solutions"]["QAOA"]["sampled_state_index"], 0)
        self.assertEqual(report["solutions"]["QAOA"]["sampled_count"], 16)
        self.assertEqual(report["solutions"]["QAOA"]["events"], 0)
        self.assertGreater(report["solutions"]["Exact"]["events"], 0)
        self.assertEqual(report["solutions"]["QAOA"]["sampled_slack_residual"], -1)
        self.assertEqual(report["qaoa"]["sampled"]["zero_slack_residual_probability"], 0)

    def test_infeasible_hardware_samples_never_fall_back_to_classical_solution(self):
        def finished(circuit, **settings):
            settings["on_submitted"]("saved-job-123")
            return self.cloud_result(circuit, settings, counts={(1 << circuit["n_qubits"]) - 1: 16})

        with patch.object(cli, "solve_qaoa", side_effect=self.local_result), \
                patch.object(hardware, "run_hardware", side_effect=finished):
            self.assertEqual(self.run_cli(self.argv), 0)
        report = self.read_json()
        self.assertIsNone(report["qaoa"]["best_budget_feasible_sample"])
        self.assertEqual(report["qaoa"]["counts_by_state_index"], {"7": 16})
        self.assertEqual(report["qaoa"]["sampled"]["budget_feasible_probability"], 0)
        self.assertNotIn("QAOA", report["solutions"])
        self.assertTrue(report["solutions"]["Exact"]["budget_feasible"])
        with (self.output / "schedules.csv").open(encoding="utf-8", newline="") as stream:
            self.assertNotIn("QAOA", {row["solver"] for row in csv.DictReader(stream)})

    def test_interrupted_query_keeps_submitted_job_id_for_resume(self):
        def interrupted(circuit, **settings):
            settings["on_submitted"]("saved-job-123")
            raise KeyboardInterrupt()

        with patch.object(cli, "solve_qaoa", side_effect=self.local_result), \
                patch.object(hardware, "run_hardware", side_effect=interrupted), \
                self.assertRaises(KeyboardInterrupt):
            self.run_cli(self.argv)
        report = self.read_json("hardware_job.json")["report"]
        self.assertEqual(report["hardware"]["job_id"], "saved-job-123")
        self.assertEqual(report["hardware"]["status"], "pending")
        self.assertTrue(report["hardware"]["query_interrupted"])

    def test_tampered_manifest_is_rejected_before_query(self):
        original_path = self.start_pending()
        original = self.read_json("hardware_job.json")
        mutations = {
            "originir": lambda m: m["circuit"].update(originir=m["circuit"]["originir"] + "H q[0]\n"),
            "angles": lambda m: m["report"]["qaoa_reference"]["parameters"].update(gammas=[0.9]),
            "model": lambda m: m["report"]["input_data"].update(dose_mm=10.0),
            "checksum": lambda m: m["report"]["hardware"].update(circuit_sha256="invalid"),
            "shot_count": lambda m: m["report"]["configuration"].update(shots=17),
            "version": lambda m: m.update(schema_version=999),
        }
        for name, mutate in mutations.items():
            with self.subTest(tampering=name):
                manifest = copy.deepcopy(original)
                mutate(manifest)
                original_path.write_text(json.dumps(manifest), encoding="utf-8")
                with patch.object(hardware, "run_hardware") as cloud, \
                        patch.object(cli, "solve_qaoa") as optimizer, self.assertRaises(SystemExit) as error:
                    self.run_cli(["--qcloud-resume", str(original_path), "--no-plot"])
                self.assertEqual(error.exception.code, 2)
                cloud.assert_not_called()
                optimizer.assert_not_called()

    def test_reusing_submission_output_is_blocked_before_optimization(self):
        self.start_pending()
        original = (self.output / "hardware_job.json").read_bytes()
        with patch.object(cli, "solve_qaoa") as optimizer, \
                patch.object(hardware, "run_hardware") as cloud, self.assertRaises(SystemExit) as error:
            self.run_cli(self.argv)
        self.assertEqual(error.exception.code, 2)
        optimizer.assert_not_called()
        cloud.assert_not_called()
        self.assertEqual((self.output / "hardware_job.json").read_bytes(), original)

    def test_resume_cannot_overwrite_another_experiment(self):
        path = self.start_pending()
        other = self.root / "other"
        other.mkdir()
        occupied = other / "hardware_job.json"
        occupied.write_text("unrelated experiment", encoding="utf-8")
        with patch.object(hardware, "run_hardware") as cloud, self.assertRaises(SystemExit) as error:
            self.run_cli(["--qcloud-resume", str(path), "--output", str(other), "--no-plot"])
        self.assertEqual(error.exception.code, 2)
        cloud.assert_not_called()
        self.assertEqual(occupied.read_text(), "unrelated experiment")

    def test_dry_run_manifest_cannot_be_resumed_as_a_submitted_job(self):
        with patch.object(cli, "solve_qaoa", side_effect=self.local_result):
            self.assertEqual(self.run_cli(self.argv + ["--qcloud-dry-run"]), 0)
        with patch.object(hardware, "run_hardware") as cloud, self.assertRaises(SystemExit) as error:
            self.run_cli(["--qcloud-resume", str(self.output / "hardware_job.json"), "--no-plot"])
        self.assertEqual(error.exception.code, 2)
        cloud.assert_not_called()

    def test_failed_cloud_job_reports_failure_without_synthetic_samples(self):
        def failed(circuit, **settings):
            settings["on_submitted"]("saved-job-123")
            return self.cloud_result(circuit, settings, "failed")

        with patch.object(cli, "solve_qaoa", side_effect=self.local_result), \
                patch.object(hardware, "run_hardware", side_effect=failed):
            self.assertEqual(self.run_cli(self.argv), 2)
        report = self.read_json()
        self.assertEqual(report["hardware"]["status"], "failed")
        self.assertNotIn("qaoa", report)
        self.assertNotIn("QAOA", report["solutions"])

    def test_invalid_cloud_flags_fail_before_optimization_or_cloud_access(self):
        cases = [
            ["--backend", "qcloud"],
            ["--backend", "numpy", "--qcloud-dry-run"],
            ["--backend", "classical", "--qcloud-chip", "test-real-chip"],
            ["--backend", "numpy", "--qcloud-local-backend", "numpy"],
            ["--backend", "qcloud", "--qcloud-chip", "test-real-chip", "--qcloud-timeout", "nan"],
            ["--backend", "qcloud", "--qcloud-chip", "test-real-chip", "--qcloud-timeout", "-1"],
            ["--backend", "qcloud", "--qcloud-chip", "test-real-chip", "--qcloud-poll-interval", "0"],
            ["--backend", "qcloud", "--qcloud-chip", "test-real-chip", "--shots", "0"],
            ["--qcloud-resume", "missing.json", "--shots", "32"],
            ["--qcloud-resume", "missing.json", "--scenario", "heading"],
            ["--qcloud-resume", "missing.json", "--backend", "numpy"],
            ["--qcloud-resume", "missing.json", "--qcloud-dry-run"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), patch.object(cli, "solve_qaoa") as optimizer, \
                    patch.object(hardware, "run_hardware") as cloud, self.assertRaises(SystemExit) as error:
                self.run_cli(argv + ["--no-plot", "--output", str(self.output)])
            self.assertEqual(error.exception.code, 2)
            optimizer.assert_not_called()
            cloud.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
