"""Offline batch recovery checks: no SDK, credential file, or cloud access."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation import hardware_batch as batch_module
from wheat_irrigation import hardware_batch_analysis, hardware_batch_query
from wheat_irrigation.hardware import HardwareError


class HardwareBatchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.batch = {
            "configuration": {"chip": "fixture-chip", "shots": 1024},
            "jobs": [self.entry("jointing_01"), self.entry("heading_01")],
        }
        batch_module._save(self.root / "batch.json", self.batch)
        self.plan = self.start_patch(batch_module, "_plan", return_value={"circuit": {"fixture": True}})
        self.summary = self.start_patch(batch_module, "write_summary")
        self.backends = self.start_patch(batch_module.hardware, "list_hardware_backends",
                                        return_value={"fixture-chip": True})
        self.run = self.start_patch(batch_module.hardware, "run_hardware",
                                   side_effect=AssertionError("Unmocked hardware submission"))
        self.query = self.start_patch(hardware_batch_query, "query_cloud_response",
                                     side_effect=AssertionError("Unmocked cloud query"))
        self.analyze = self.start_patch(hardware_batch_analysis, "analyze_probability_result",
                                       return_value={"counts_synthesized": False})
        self.start_patch(batch_module.hardware, "_load_sdk",
                         side_effect=AssertionError("SDK access forbidden"))
        self.start_patch(batch_module.hardware, "_service",
                         side_effect=AssertionError("Credential access forbidden"))
        self.output = io.StringIO()
        self.stdout = redirect_stdout(self.output)
        self.stderr = redirect_stderr(self.output)
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def start_patch(self, target, name, **kwargs):
        patcher = patch.object(target, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    @staticmethod
    def entry(name, state="prepared", job_id=None):
        return {"name": name, "scenario": name.rsplit("_", 1)[0], "repeat": 1,
                "state": state, "job_id": job_id, "circuit_sha256": "fixture-digest"}

    def saved(self):
        return json.loads((self.root / "batch.json").read_text(encoding="utf-8"))

    @staticmethod
    def response(job_id, state="3", chip="fixture-chip"):
        return {"success": True, "code": 10000,
                "obj": {"taskId": job_id, "taskState": state, "chipId": chip, "errCode": 0}}

    def test_submission_intent_and_callback_id_are_saved_before_next_sdk_step(self):
        submitted = []

        def run(circuit, **kwargs):
            index = len(submitted)
            saved = self.saved()["jobs"][index]
            self.assertEqual(saved["state"], "submitting")
            self.assertIsNone(saved["job_id"])
            self.assertIn("submission_attempted_at", saved)
            self.assertEqual(self.plan.call_count, 2, "All circuits must be checked before submission")
            self.assertEqual(kwargs["timeout"], 0)
            self.assertEqual(kwargs["shots"], 1024)
            job_id = f"job-{index}"
            kwargs["on_submitted"](job_id)
            # This is the SDK's first opportunity to query after the callback.
            saved = self.saved()["jobs"][index]
            self.assertEqual((saved["state"], saved["job_id"]), ("submitted", job_id))
            self.assertIn("submitted_at", saved)
            submitted.append(job_id)
            return {"status": "pending", "job_id": job_id}

        self.run.side_effect = run
        batch_module.submit_jobs(self.root, self.batch)
        self.assertEqual(submitted, ["job-0", "job-1"])
        for entry in self.saved()["jobs"]:
            initial = self.root / "jobs" / entry["name"] / "initial_sdk_result.json"
            self.assertEqual(json.loads(initial.read_text())["job_id"], entry["job_id"])

    def test_every_saved_id_prevents_resubmission_even_with_inconsistent_state(self):
        states = ["prepared", "submitting", "submission_unknown", "submitted", "pending",
                  "completed_probabilities", "failed"]
        self.batch["jobs"] = [self.entry(f"case_{i}", state, f"saved-{i}")
                              for i, state in enumerate(states)]
        before = deepcopy(self.batch)
        batch_module.submit_jobs(self.root, self.batch)
        self.assertEqual(self.batch, before)
        self.run.assert_not_called()
        self.backends.assert_not_called()
        self.plan.assert_not_called()

    def test_ambiguous_attempt_blocks_all_further_submissions_before_cloud_access(self):
        for state in ("submitting", "submission_unknown"):
            with self.subTest(state=state):
                self.batch["jobs"][0]["state"] = state
                with self.assertRaisesRegex(ValueError, "no saved ID"):
                    batch_module.submit_jobs(self.root, self.batch)
        self.run.assert_not_called()
        self.backends.assert_not_called()
        self.plan.assert_not_called()

    def test_submission_exception_without_id_is_durable_and_never_automatically_retried(self):
        self.run.side_effect = RuntimeError("SDK failure containing private diagnostics")
        with self.assertRaisesRegex(HardwareError, "outcome is unknown") as caught:
            batch_module.submit_jobs(self.root, self.batch)
        self.assertNotIn("private diagnostics", str(caught.exception))
        saved = self.saved()
        self.assertEqual(saved["jobs"][0]["state"], "submission_unknown")
        self.assertEqual(saved["jobs"][0]["last_submission_exception_type"], "RuntimeError")
        self.assertEqual(saved["jobs"][1]["state"], "prepared")
        with self.assertRaises(ValueError):
            batch_module.submit_jobs(self.root, saved)
        self.assertEqual(self.run.call_count, 1)

    def test_process_interruption_retains_submission_intent_and_blocks_retry(self):
        self.run.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            batch_module.submit_jobs(self.root, self.batch)
        saved = self.saved()
        self.assertEqual(saved["jobs"][0]["state"], "submitting")
        self.assertIsNone(saved["jobs"][0]["job_id"])
        with self.assertRaises(ValueError):
            batch_module.submit_jobs(self.root, saved)
        self.assertEqual(self.run.call_count, 1)

    def test_saved_id_survives_sdk_read_failure_and_restart_collects_same_task(self):
        self.batch["jobs"] = self.batch["jobs"][:1]

        def submitted_then_failed(circuit, **kwargs):
            kwargs["on_submitted"]("durable-id")
            raise HardwareError("Completed SDK task cannot expose counts")

        self.run.side_effect = submitted_then_failed
        batch_module.submit_jobs(self.root, self.batch)
        restarted = self.saved()
        self.assertEqual(restarted["jobs"][0]["job_id"], "durable-id")
        self.assertEqual(restarted["jobs"][0]["state"], "submitted")
        batch_module.submit_jobs(self.root, restarted)
        self.assertEqual(self.run.call_count, 1)
        self.query.side_effect = None
        self.query.return_value = self.response("durable-id")
        batch_module.collect_once(self.root, restarted)
        self.query.assert_called_once_with("durable-id")
        self.analyze.assert_called_once_with(self.plan.return_value, self.query.return_value, "durable-id")
        saved = self.saved()["jobs"][0]
        self.assertEqual((saved["state"], saved["job_id"]), ("completed_probabilities", "durable-id"))
        self.assertTrue((self.root / "jobs" / saved["name"] / "cloud_response.json").is_file())
        self.assertTrue((self.root / "jobs" / saved["name"] / "result_analysis.json").is_file())

    def test_preflight_rejects_unavailable_chip_and_bad_circuit_without_submission(self):
        self.backends.return_value = {"fixture-chip": False}
        with self.assertRaises(HardwareError):
            batch_module.submit_jobs(self.root, self.batch)
        self.plan.assert_not_called()
        self.backends.return_value = {"fixture-chip": True}
        self.plan.side_effect = [{"circuit": {}}, ValueError("Invalid second circuit")]
        with self.assertRaisesRegex(ValueError, "Invalid second circuit"):
            batch_module.submit_jobs(self.root, self.batch)
        self.run.assert_not_called()
        self.assertTrue(all(job["state"] == "prepared" for job in self.saved()["jobs"]))

    def test_terminal_and_unsubmitted_entries_are_never_queried(self):
        self.batch["jobs"] = [self.entry("complete_01", "completed_probabilities", "done"),
                              self.entry("failed_01", "failed", "failed"),
                              self.entry("prepared_01"),
                              self.entry("unknown_01", "submission_unknown")]
        before = deepcopy(self.batch)
        batch_module.collect_once(self.root, self.batch)
        self.assertEqual(self.batch, before)
        self.query.assert_not_called()
        self.plan.assert_not_called()
        self.analyze.assert_not_called()

    def test_collection_error_is_retryable_with_the_same_id(self):
        self.batch["jobs"] = [self.entry("jointing_01", "submitted", "retry-id")]
        self.query.side_effect = [HardwareError("Temporarily unavailable"), self.response("retry-id", "2"),
                                  self.response("retry-id")]
        batch_module.collect_once(self.root, self.batch)
        saved = self.saved()["jobs"][0]
        self.assertEqual(saved["state"], "submitted")
        self.assertEqual(saved["last_collection_exception_type"], "HardwareError")
        batch_module.collect_once(self.root, self.batch)
        self.assertEqual(self.saved()["jobs"][0]["state"], "pending")
        batch_module.collect_once(self.root, self.batch)
        saved = self.saved()["jobs"][0]
        self.assertEqual(saved["state"], "completed_probabilities")
        self.assertNotIn("last_collection_exception_type", saved)
        self.assertEqual([call.args for call in self.query.call_args_list], [("retry-id",)] * 3)
        self.run.assert_not_called()

    def test_failed_cloud_result_is_saved_and_never_queried_again(self):
        self.batch["jobs"] = [self.entry("jointing_01", "submitted", "failed-id")]
        self.query.side_effect = None
        self.query.return_value = self.response("failed-id", "4")
        batch_module.collect_once(self.root, self.batch)
        self.assertEqual(self.saved()["jobs"][0]["state"], "failed")
        batch_module.collect_once(self.root, self.batch)
        self.query.assert_called_once_with("failed-id")
        self.analyze.assert_not_called()

    def test_wrong_chip_cannot_be_saved_as_completed(self):
        self.batch["jobs"] = [self.entry("jointing_01", "submitted", "other-chip-id")]
        self.query.side_effect = None
        self.query.return_value = self.response("other-chip-id", chip="other-chip")
        batch_module.collect_once(self.root, self.batch)
        saved = self.saved()["jobs"][0]
        self.assertEqual(saved["state"], "submitted")
        self.assertEqual(saved["last_collection_exception_type"], "ValueError")
        self.analyze.assert_not_called()

    def test_lock_excludes_second_handle_and_releases_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted owner"):
            with batch_module._lock(self.root):
                with self.assertRaisesRegex(ValueError, "Another process"):
                    with batch_module._lock(self.root):
                        self.fail("Second owner entered locked batch")
                raise RuntimeError("interrupted owner")
        with batch_module._lock(self.root):
            pass

    def test_cli_checks_lock_before_loading_credentials_or_submitting(self):
        with batch_module._lock(self.root), patch.object(batch_module, "_credential") as credential:
            result = batch_module.main(["submit", "--output", str(self.root)])
        self.assertEqual(result, 2)
        credential.assert_not_called()
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
