"""Read existing QCloud jobs while containing native SDK request logs in memory.

Some pyqpanda3 releases reject completed task responses before returning a result
object. The authenticated response still appears in the native console log. A
short-lived child process captures that log, and only validated, allowlisted
task fields leave this module. This module never submits a circuit.
"""

from __future__ import annotations

import json
import math
from numbers import Real
import os
import re
import subprocess
import sys

from .hardware import HardwareError


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_OUTCOME = re.compile(r"(?:[0-9]+|0[xX][0-9A-Fa-f]+)\Z")
_RESPONSE_MARKER = re.compile(r"qcurl perform recv\s*:\s*")
_INTEGER_FIELDS = frozenset({
    "errCode", "startTime", "endTime", "createTime", "submitTime",
    "finishTime", "shots", "shot",
})
_ENUM_FIELDS = frozenset({"taskType", "taskCalMethod"})
_RESULT_FIELDS = frozenset({
    "taskResult", "taskResultCount", "taskResultCounts",
    "taskResultProbability", "counts", "raw_counts",
})
_QUERY_CHILD = r'''
import os
import sys
try:
    from pyqpanda3 import qcloud
    service = qcloud.QCloudService(api_key=os.environ[sys.argv[2]])
    service.setup_logging(qcloud.LogOutput.CONSOLE)
    job = qcloud.QCloudJob(sys.argv[1])
    job.query()
except Exception:
    # Even exception text can contain credentials. The parent needs only the
    # captured authenticated response; print neither errors nor SDK objects.
    raise SystemExit(2)
'''


def _job_identifier(job_id: object) -> str:
    if not isinstance(job_id, str) or _IDENTIFIER.fullmatch(job_id) is None:
        raise ValueError("job_id must be a nonempty QCloud task identifier")
    return job_id


def _number(value: object) -> bool:
    return (type(value) in (int, float) and value >= 0
            and (type(value) is int or math.isfinite(value)))


def _result_data(value: object) -> object:
    """Validate measurement data recursively without forwarding free text.

    Original JSON strings remain byte-for-byte intact when their entire parsed
    object is allowed. Unknown nested fields are removed before re-encoding.
    Numerical probabilities and integer counts retain their original types.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            raise HardwareError("QCloud returned invalid measurement data") from None
        if not isinstance(parsed, (dict, list)):
            raise HardwareError("QCloud returned invalid measurement data")
        cleaned = _result_data(parsed)
        return value if parsed == cleaned else json.dumps(cleaned, separators=(",", ":"))
    if isinstance(value, list):
        return [_result_data(item) for item in value]
    if isinstance(value, dict):
        if "key" in value or "value" in value:
            keys, values = value.get("key"), value.get("value")
            if (not isinstance(keys, list) or not isinstance(values, list)
                    or len(keys) != len(values)
                    or any(not isinstance(key, str) or _OUTCOME.fullmatch(key) is None
                           for key in keys)
                    or any(not _number(item) for item in values)):
                raise HardwareError("QCloud returned invalid measurement data")
            return {"key": keys, "value": values}
        if any(not isinstance(key, str) or _OUTCOME.fullmatch(key) is None
               or not _number(item) for key, item in value.items()):
            raise HardwareError("QCloud returned invalid measurement data")
        return dict(value)
    if _number(value):
        return value
    raise HardwareError("QCloud returned invalid measurement data")


def _safe_task(obj: dict) -> dict:
    safe = {"taskId": obj["taskId"]}
    if "chipId" in obj:
        chip = obj["chipId"]
        if not isinstance(chip, str) or _IDENTIFIER.fullmatch(chip) is None:
            raise HardwareError("QCloud returned invalid task metadata")
        safe["chipId"] = chip
    state = obj.get("taskState")
    if type(state) is int and state >= 0:
        safe["taskState"] = state
    elif isinstance(state, str) and re.fullmatch(r"[0-9]{1,3}", state):
        safe["taskState"] = state
    else:
        raise HardwareError("QCloud returned invalid task metadata")
    for key in _ENUM_FIELDS:
        if key in obj and obj[key] is not None:
            value = obj[key]
            if not ((type(value) is int and value >= 0)
                    or (isinstance(value, str) and re.fullmatch(r"[0-9]{1,3}", value))):
                raise HardwareError("QCloud returned invalid task metadata")
            # Retain the platform's numeric evidence without assigning an
            # undocumented meaning to an enum or coercing its original type.
            safe[key] = value
    for key in _INTEGER_FIELDS:
        if key in obj and obj[key] is not None:
            if type(obj[key]) is not int or obj[key] < 0:
                raise HardwareError("QCloud returned invalid task metadata")
            safe[key] = obj[key]
    if "measureQubitSize" in obj and obj["measureQubitSize"] is not None:
        sizes = obj["measureQubitSize"]
        if (not isinstance(sizes, list)
                or any(type(size) is not int or size < 0 for size in sizes)):
            raise HardwareError("QCloud returned invalid task metadata")
        safe["measureQubitSize"] = sizes
    for key in _RESULT_FIELDS:
        if key in obj and obj[key] is not None:
            safe[key] = _result_data(obj[key])
    return safe


def decode_cloud_response(log_text: str, job_id: str) -> dict:
    """Extract only the requested authenticated task response from SDK logs.

    Native logging may insert physical CR/LF characters inside JSON strings at
    fixed line widths. Escaped JSON newlines (``\\n``) remain untouched. Multiple
    response lines are supported, and the final matching response is used.
    """
    job_id = _job_identifier(job_id)
    if not isinstance(log_text, str):
        raise ValueError("log_text must be captured SDK text")
    flattened = log_text.replace("\r", "").replace("\n", "")
    decoder = json.JSONDecoder()
    matched = None
    for marker in _RESPONSE_MARKER.finditer(flattened):
        try:
            response, _ = decoder.raw_decode(flattened, marker.end())
        except (ValueError, RecursionError):
            continue
        if not isinstance(response, dict):
            continue
        obj = response.get("obj")
        if (isinstance(obj, dict) and obj.get("taskId") == job_id):
            matched = response
    if matched is None:
        raise HardwareError("QCloud did not return the requested authenticated task response")
    if matched.get("success") is not True or matched.get("code") != 10000:
        raise HardwareError("QCloud rejected the task query")
    try:
        safe = _safe_task(matched["obj"])
    except (RecursionError, OverflowError):
        raise HardwareError("QCloud returned invalid task data") from None
    return {"success": True, "code": 10000, "obj": safe}


def query_cloud_response(
    job_id: str, token_env: str = "QCLOUD_API_KEY", timeout: float = 45,
) -> dict:
    """Read an existing job once, keeping all native SDK logs process-local.

    ``timeout`` bounds the complete child process, including network calls.
    Credentials are inherited through the environment, never command arguments.
    Parent import paths are forwarded for workspace-local SDK installations.
    No raw SDK text or exception is printed, persisted, or chained to an error.
    """
    job_id = _job_identifier(job_id)
    if not isinstance(token_env, str) or _ENV_NAME.fullmatch(token_env) is None:
        raise ValueError("token_env must be an environment variable name")
    if (isinstance(timeout, bool) or not isinstance(timeout, Real)
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be finite and positive")
    if not os.environ.get(token_env, "").strip():
        raise HardwareError("QCloud API key is not configured")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in sys.path)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _QUERY_CHILD, job_id, token_env],
            env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=float(timeout), check=False,
        )
    except subprocess.TimeoutExpired:
        raise HardwareError("QCloud task query timed out; the saved task remains resumable") from None
    except Exception:
        raise HardwareError("QCloud task query could not run") from None
    # A nonzero exit after query() is expected for affected SDK versions. Only
    # the server's response with the exact requested task id determines success.
    return decode_cloud_response(result.stdout + "\n" + result.stderr, job_id)
