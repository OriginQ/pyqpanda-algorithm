"""Offline checks for one native QCloud job containing several circuits."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation import hardware_batch_job as native_batch
from wheat_irrigation.hardware import HardwareError


class NativeBatchSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.output = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.output))
        self.stack.enter_context(redirect_stderr(self.output))
        self.manifest = {
            "schema_version": 1,
            "state": "prepared",
            "job_id": None,
            "configuration": {"chip": "fixture-chip", "shots": 1024},
            "circuits": [
                {"index": index, "scenario": scenario, "circuit_sha256": str(index + 1) * 64}
                for index, scenario in enumerate(("jointing", "heading", "grain_filling"))
            ],
        }
        self.write_manifest()
        self.plans = [
            {"circuit": {"originir": f"fixture-originir-{index}", "n_qubits": 11,
                         "sha256": entry["circuit_sha256"]}}
            for index, entry in enumerate(self.manifest["circuits"])
        ]
        self.progs = [object(), object(), object()]
        self.validate = self.stack.enter_context(
            patch.object(native_batch, "validated_plans", return_value=self.plans)
        )
        self.parse = self.stack.enter_context(
            patch.object(native_batch.hardware, "_parse_originir", side_effect=self.progs)
        )
        self.options = Mock()
        self.job = Mock()
        self.job.job_id.return_value = "native-batch-job-001"
        # Access to any result/status API during submission is a regression.
        for method in ("query", "result", "status"):
            getattr(self.job, method).side_effect = AssertionError("Submission must not query")
        self.backend = Mock()
        self.backend.chip_info.return_value.available_qubits.return_value = list(range(20))
        self.backend.run.return_value = self.job
        self.service = Mock()
        self.service.backends.return_value = {"fixture-chip": True}
        self.service.backend.return_value = self.backend
        self.sdk = SimpleNamespace(QCloudOptions=Mock(return_value=self.options))
        self.load_sdk = self.stack.enter_context(
            patch.object(native_batch.hardware, "_load_sdk", return_value=self.sdk)
        )
        self.new_service = self.stack.enter_context(
            patch.object(native_batch.hardware, "_service", return_value=self.service)
        )

    def write_manifest(self):
        (self.root / "native_batch.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def saved(self):
        return json.loads((self.root / "native_batch.json").read_text(encoding="utf-8"))

    def assert_no_submission(self):
        self.backend.run.assert_not_called()
        self.assertEqual(self.saved()["state"], "prepared")
        self.assertIsNone(self.saved()["job_id"])

    def test_one_native_list_submission_persists_intent_then_single_id_without_query(self):
        def run(progs, shots, options):
            saved = self.saved()
            self.assertEqual(saved["state"], "submitting")
            self.assertIsNone(saved["job_id"])
            self.assertIsInstance(progs, list)
            self.assertEqual(progs, self.progs)
            self.assertEqual(shots, 1024)
            self.assertIs(options, self.options)
            self.validate.assert_called_once_with(self.root, self.manifest)
            self.assertEqual(self.parse.call_count, 3)
            self.service.backends.assert_called_once()
            self.backend.chip_info.assert_called_once()
            return self.job

        self.backend.run.side_effect = run
        native_batch.submit_native_batch(self.root, self.manifest)

        self.backend.run.assert_called_once()
        self.assertEqual([call.args[0] for call in self.parse.call_args_list],
                         [plan["circuit"]["originir"] for plan in self.plans])
        self.service.backend.assert_called_once_with("fixture-chip")
        self.options.set_mapping.assert_called_once_with(True)
        self.options.set_optimization.assert_called_once_with(True)
        self.options.set_amend.assert_called_once_with(False)
        self.options.set_is_prob_counts.assert_called_once_with(True)
        saved = self.saved()
        self.assertEqual((saved["state"], saved["job_id"]),
                         ("submitted", "native-batch-job-001"))
        self.assertEqual(len(saved["circuits"]), 3)
        self.job.query.assert_not_called()
        self.job.result.assert_not_called()
        self.job.status.assert_not_called()

    def test_saved_id_prevents_resubmission_in_every_state(self):
        for state in ("prepared", "submitting", "submission_unknown", "submitted",
                      "pending", "completed_probabilities", "failed"):
            with self.subTest(state=state):
                self.manifest.update(state=state, job_id="existing-native-batch")
                self.write_manifest()
                before = deepcopy(self.manifest)
                native_batch.submit_native_batch(self.root, self.manifest)
                self.assertEqual(self.manifest, before)
                self.assertEqual(self.saved(), before)
        self.validate.assert_not_called()
        self.load_sdk.assert_not_called()
        self.new_service.assert_not_called()
        self.backend.run.assert_not_called()

    def test_ambiguous_previous_attempt_cannot_be_automatically_retried(self):
        for state in ("submitting", "submission_unknown"):
            with self.subTest(state=state):
                self.manifest.update(state=state, job_id=None)
                self.write_manifest()
                with self.assertRaises((ValueError, HardwareError)):
                    native_batch.submit_native_batch(self.root, self.manifest)
                self.assertEqual(self.saved()["state"], state)
        self.validate.assert_not_called()
        self.load_sdk.assert_not_called()
        self.new_service.assert_not_called()
        self.backend.run.assert_not_called()

    def test_submission_exception_preserves_unknown_state_and_blocks_restart(self):
        self.backend.run.side_effect = RuntimeError("private native diagnostics must not escape")
        with self.assertRaises(HardwareError) as caught:
            native_batch.submit_native_batch(self.root, self.manifest)
        self.assertNotIn("private native diagnostics", str(caught.exception))
        saved = self.saved()
        self.assertEqual(saved["state"], "submission_unknown")
        self.assertIsNone(saved["job_id"])
        self.assertNotIn("private native diagnostics", json.dumps(saved))
        with self.assertRaises((ValueError, HardwareError)):
            native_batch.submit_native_batch(self.root, saved)
        self.backend.run.assert_called_once()

    def test_missing_job_id_after_submission_is_not_retried(self):
        self.job.job_id.return_value = ""
        with self.assertRaises(HardwareError):
            native_batch.submit_native_batch(self.root, self.manifest)
        saved = self.saved()
        self.assertEqual(saved["state"], "submission_unknown")
        self.assertFalse(saved.get("job_id"))
        with self.assertRaises((ValueError, HardwareError)):
            native_batch.submit_native_batch(self.root, saved)
        self.backend.run.assert_called_once()

    def test_intent_write_failure_prevents_cloud_submission(self):
        with patch.object(native_batch, "_save", side_effect=OSError("Disk unavailable")):
            with self.assertRaises(OSError):
                native_batch.submit_native_batch(self.root, self.manifest)
        self.assert_no_submission()

    def test_report_failure_after_submission_does_not_lose_job_id(self):
        with patch.object(native_batch, "write_report", side_effect=OSError("Report unavailable")):
            with self.assertRaises(OSError):
                native_batch.submit_native_batch(self.root, self.manifest)
        saved = self.saved()
        self.assertEqual((saved["state"], saved["job_id"]),
                         ("submitted", "native-batch-job-001"))
        native_batch.submit_native_batch(self.root, saved)
        self.backend.run.assert_called_once()

    def test_process_interruption_retains_durable_intent_and_blocks_restart(self):
        self.backend.run.side_effect = KeyboardInterrupt()
        with self.assertRaises((KeyboardInterrupt, HardwareError)):
            native_batch.submit_native_batch(self.root, self.manifest)
        saved = self.saved()
        self.assertIn(saved["state"], ("submitting", "submission_unknown"))
        self.assertIsNone(saved["job_id"])
        with self.assertRaises((ValueError, HardwareError)):
            native_batch.submit_native_batch(self.root, saved)
        self.backend.run.assert_called_once()

    def test_invalid_plan_fails_before_sdk_or_credentials_are_accessed(self):
        self.validate.side_effect = ValueError("Invalid second saved circuit")
        with self.assertRaises(ValueError):
            native_batch.submit_native_batch(self.root, self.manifest)
        self.assert_no_submission()
        self.load_sdk.assert_not_called()
        self.new_service.assert_not_called()

    def test_originir_parse_failure_leaves_batch_prepared(self):
        self.parse.side_effect = [self.progs[0], HardwareError("Invalid second OriginIR circuit")]
        with self.assertRaises(HardwareError):
            native_batch.submit_native_batch(self.root, self.manifest)
        self.assert_no_submission()

    def test_unavailable_chip_leaves_batch_prepared(self):
        self.service.backends.return_value = {"fixture-chip": False}
        with self.assertRaises(HardwareError):
            native_batch.submit_native_batch(self.root, self.manifest)
        self.assert_no_submission()

    def test_insufficient_physical_qubits_leaves_batch_prepared(self):
        self.backend.chip_info.return_value.available_qubits.return_value = list(range(10))
        with self.assertRaises(HardwareError):
            native_batch.submit_native_batch(self.root, self.manifest)
        self.assert_no_submission()

    def test_chip_metadata_failure_leaves_batch_prepared(self):
        self.backend.chip_info.side_effect = RuntimeError("Chip metadata unavailable")
        with self.assertRaises(HardwareError):
            native_batch.submit_native_batch(self.root, self.manifest)
        self.assert_no_submission()


if __name__ == "__main__":
    unittest.main()
