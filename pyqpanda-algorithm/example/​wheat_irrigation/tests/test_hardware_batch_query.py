"""Offline regression checks for completed-task SDK response compatibility."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation import hardware_batch_query as query
from wheat_irrigation.hardware import HardwareError


JOB_ID = "B3941F177F1AE6D9CAA33D229050B211"
SECRET = "test-credential-never-forward"


def response():
    # Field names, types and truncated probability precision match the real
    # WK_C180 completed-task snapshot; retain only two outcomes for this fixture.
    return {
        "success": True, "code": 10000, "message": "success",
        "obj": {
            "taskId": JOB_ID, "chipId": "WK_C180", "errCode": 0,
            "startTime": 1789720951689, "taskState": "3",
            "measureQubitSize": [11],
            "taskResult": ['{"value":[0.0009765,0.0019531],"key":["0x58f","0x22a"]}'],
        },
    }


def log(value):
    return "[native SDK] qcurl perform recv : " + json.dumps(value)


class ResponseTests(unittest.TestCase):
    def test_completed_snapshot_preserves_probabilities_and_original_json(self):
        raw = response()
        decoded = query.decode_cloud_response(log(raw), JOB_ID)
        self.assertEqual(decoded, {key: raw[key] for key in ("success", "code", "obj")})
        self.assertNotIn("counts", decoded["obj"])
        self.assertEqual(json.loads(decoded["obj"]["taskResult"][0])["value"],
                         [0.0009765, 0.0019531])

    def test_logger_line_breaks_inside_json_and_marker_are_reassembled(self):
        native_log = log(response())
        for width in (17, 37, 1024):
            with self.subTest(width=width):
                broken = "\r\n".join(native_log[i:i + width]
                                       for i in range(0, len(native_log), width))
                self.assertEqual(query.decode_cloud_response(broken, JOB_ID)["obj"],
                                 response()["obj"])

    def test_multiple_responses_select_last_matching_task(self):
        old = response()
        old["obj"]["taskState"] = "2"
        old["obj"]["taskResult"] = []
        unrelated = response()
        unrelated["obj"]["taskId"] = "unrelated-job"
        capture = "\n".join((log({"success": True, "obj": []}), log(old),
                              "qcurl perform recv : malformed", log(response()),
                              log(unrelated), "RuntimeError: value is not array"))
        decoded = query.decode_cloud_response(capture, JOB_ID)
        self.assertEqual(decoded["obj"]["taskState"], "3")
        self.assertEqual(decoded["obj"]["taskId"], JOB_ID)

    def test_unrelated_task_and_authentication_failure_rejected_without_leaking(self):
        unrelated = response()
        unrelated["obj"]["taskId"] = "another-job"
        rejected = response()
        rejected.update(success=False, code=401, message=SECRET)
        for capture in (log(unrelated), log({"success": False, "code": 401,
                                           "message": SECRET, "obj": None}),
                        log(rejected), "Authorization: " + SECRET,
                        "qcurl perform recv : not json " + SECRET):
            with self.subTest(capture=capture), self.assertRaises(HardwareError) as caught:
                query.decode_cloud_response(capture, JOB_ID)
            self.assertNotIn(SECRET, str(caught.exception))

    def test_final_matching_failure_does_not_reuse_earlier_success(self):
        rejected = response()
        rejected["success"] = False
        with self.assertRaisesRegex(HardwareError, "rejected"):
            query.decode_cloud_response(log(response()) + "\n" + log(rejected), JOB_ID)

    def test_allowlist_excludes_credentials_in_messages_unknown_and_nested_fields(self):
        raw = response()
        raw["message"] = SECRET
        raw["obj"].update(apiKey=SECRET, userId=SECRET, authCount=SECRET,
                           account=SECRET, errMsg=SECRET)
        result = json.loads(raw["obj"]["taskResult"][0])
        result["authorization"] = SECRET
        raw["obj"]["taskResult"] = [json.dumps(result)]
        decoded = query.decode_cloud_response(log(raw), JOB_ID)
        self.assertNotIn(SECRET, json.dumps(decoded))
        self.assertEqual(set(json.loads(decoded["obj"]["taskResult"][0])), {"key", "value"})

    def test_integer_counts_are_retained_without_rounding_probabilities(self):
        raw = response()
        raw["obj"]["taskResultCounts"] = ['{"key":["000","111"],"value":[1,2]}']
        raw["obj"]["raw_counts"] = {"000": 1, "111": 2}
        decoded = query.decode_cloud_response(log(raw), JOB_ID)
        self.assertEqual(decoded["obj"]["raw_counts"], {"000": 1, "111": 2})
        self.assertIs(type(decoded["obj"]["raw_counts"]["000"]), int)

    def test_platform_task_enums_are_preserved_without_guessing_their_meaning(self):
        for task_type, method in ((7, 2), ("7", "2")):
            raw = response()
            raw["obj"].update(taskType=task_type, taskCalMethod=method,
                              taskName=SECRET, QProg=SECRET)
            decoded = query.decode_cloud_response(log(raw), JOB_ID)
            self.assertEqual(decoded["obj"]["taskType"], task_type)
            self.assertIs(type(decoded["obj"]["taskType"]), type(task_type))
            self.assertEqual(decoded["obj"]["taskCalMethod"], method)
            self.assertNotIn(SECRET, json.dumps(decoded))

    def test_platform_task_enum_fields_reject_non_numeric_metadata(self):
        for field in ("taskType", "taskCalMethod"):
            for value in (True, -1, 1.5, SECRET, "1000", [1], {"token": SECRET}):
                raw = response()
                raw["obj"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(HardwareError) as caught:
                    query.decode_cloud_response(log(raw), JOB_ID)
                self.assertNotIn(SECRET, str(caught.exception))

    def test_pending_task_without_results_is_valid(self):
        raw = response()
        raw["obj"].update(taskState="1", taskResult=None, measureQubitSize=None)
        decoded = query.decode_cloud_response(log(raw), JOB_ID)
        self.assertEqual(decoded["obj"]["taskState"], "1")
        self.assertNotIn("taskResult", decoded["obj"])

    def test_malformed_allowlisted_values_are_rejected(self):
        invalid = ({"taskState": SECRET}, {"chipId": "Bearer " + SECRET},
                   {"measureQubitSize": [True]}, {"shots": True},
                   {"taskResult": [SECRET]},
                   {"taskResult": ['{"key":["0x1"],"value":[NaN]}']},
                   {"taskResult": ['{"key":["0x1"],"value":[-1]}']},
                   {"taskResult": ['{"key":["0x1"],"value":[]}']},
                   {"raw_counts": {"authorization": SECRET}})
        for fields in invalid:
            raw = response()
            raw["obj"].update(fields)
            with self.subTest(fields=fields), self.assertRaises(HardwareError) as caught:
                query.decode_cloud_response(log(raw), JOB_ID)
            self.assertNotIn(SECRET, str(caught.exception))


class ProcessTests(unittest.TestCase):
    def test_capture_is_used_for_normal_and_exceptional_sdk_exits(self):
        for exit_code in (0, 2):
            result = SimpleNamespace(returncode=exit_code,
                                     stdout="Authorization: " + SECRET + "\n" + log(response()),
                                     stderr="SDK exception: " + SECRET)
            with self.subTest(exit_code=exit_code), \
                    patch.dict(query.os.environ, {"TEST_QCLOUD_KEY": SECRET}, clear=True), \
                    patch.object(query.subprocess, "run", return_value=result) as run, \
                    patch.object(query.sys, "path", ["/project", "/workspace/sdk"]):
                decoded = query.query_cloud_response(JOB_ID, "TEST_QCLOUD_KEY", 12)
            self.assertEqual(decoded["obj"]["taskId"], JOB_ID)
            args, kwargs = run.call_args
            self.assertNotIn(SECRET, " ".join(args[0]))
            self.assertEqual(kwargs["env"]["TEST_QCLOUD_KEY"], SECRET)
            self.assertEqual(kwargs["env"]["PYTHONPATH"],
                             os.pathsep.join(["/project", "/workspace/sdk"]))
            self.assertTrue(kwargs["capture_output"])
            self.assertEqual(kwargs["timeout"], 12.0)
            self.assertNotIn(SECRET, json.dumps(decoded))

    def test_timeout_and_process_errors_never_forward_exception_logs(self):
        failures = [subprocess.TimeoutExpired("secret command " + SECRET, 1,
                                              output=SECRET, stderr=SECRET),
                    OSError(SECRET)]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), \
                    patch.dict(query.os.environ, {"QCLOUD_API_KEY": SECRET}, clear=True), \
                    patch.object(query.subprocess, "run", side_effect=failure), \
                    self.assertRaises(HardwareError) as caught:
                query.query_cloud_response(JOB_ID)
            self.assertNotIn(SECRET, str(caught.exception))
            self.assertIsNone(caught.exception.__cause__)
            self.assertTrue(caught.exception.__suppress_context__)

    def test_missing_key_and_invalid_arguments_do_not_start_process(self):
        with patch.dict(query.os.environ, {}, clear=True), \
                patch.object(query.subprocess, "run") as run:
            with self.assertRaises(HardwareError):
                query.query_cloud_response(JOB_ID)
            for kwargs in ({"job_id": ""}, {"token_env": "INVALID NAME"},
                           {"timeout": True}, {"timeout": 0}, {"timeout": -1},
                           {"timeout": float("nan")}, {"timeout": float("inf")}):
                args = {"job_id": JOB_ID, **kwargs}
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    query.query_cloud_response(**args)
            run.assert_not_called()

    def test_child_performs_only_query_and_suppresses_sdk_exception(self):
        for raises in (False, True):
            service, job = Mock(), Mock()
            if raises:
                job.query.side_effect = RuntimeError(SECRET)
            sdk = SimpleNamespace(QCloudService=Mock(return_value=service),
                                  QCloudJob=Mock(return_value=job),
                                  LogOutput=SimpleNamespace(CONSOLE="console"))
            output, error = io.StringIO(), io.StringIO()
            with self.subTest(raises=raises), \
                    patch.dict(sys.modules, {"pyqpanda3": SimpleNamespace(qcloud=sdk)}), \
                    patch.dict(os.environ, {"TEST_QCLOUD_KEY": SECRET}, clear=True), \
                    patch.object(sys, "argv", ["-c", JOB_ID, "TEST_QCLOUD_KEY"]), \
                    redirect_stdout(output), redirect_stderr(error):
                if raises:
                    with self.assertRaises(SystemExit) as caught:
                        exec(query._QUERY_CHILD, {})
                    self.assertEqual(caught.exception.code, 2)
                else:
                    exec(query._QUERY_CHILD, {})
            sdk.QCloudService.assert_called_once_with(api_key=SECRET)
            sdk.QCloudJob.assert_called_once_with(JOB_ID)
            self.assertEqual(job.mock_calls, [unittest.mock.call.query()])
            self.assertEqual(service.mock_calls,
                             [unittest.mock.call.setup_logging("console")])
            self.assertEqual(output.getvalue() + error.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
