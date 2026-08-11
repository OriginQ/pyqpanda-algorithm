"""Shared fixtures for the runtime release-qualification suites.

Provides a schema-conforming ``valid_manifest`` pinned to the current
repository commit, the ``repo_commit`` string the release policy
compares against, and the credential-free qpanda3-runtime stand-ins the
preflight/QPU runner tests consume.  ``fake_service_with_bad_result``
is a plain helper, not a fixture, because the QPU runner test
constructs it directly.
"""

import subprocess
from pathlib import Path

import pytest

from test.execution.fakes import FakeQTaskManager, FakeRuntimeService

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Counts that pass the fixed bell threshold (correlated pairs dominate).
_BELL_PASS_COUNTS = {"00": 490, "11": 480, "01": 15, "10": 15}
#: Deterministic counts that fail the fixed bell threshold (uniform).
_BELL_FAIL_COUNTS = {"00": 250, "11": 250, "01": 250, "10": 250}


@pytest.fixture
def repo_commit():
    """Full 40-character SHA-1 of the current repository HEAD."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
    ).strip()


@pytest.fixture
def valid_manifest(repo_commit):
    """A full, schema-conforming qualification manifest for HEAD.

    Every field required by ``tools/release_qualification/schema.json``
    is present; the git commit is the real HEAD so policy tests compare
    against ``repo_commit``.
    """
    return {
        "version": "1",
        "timestamp": "2026-08-12T08:00:00Z",
        "git_commit": repo_commit,
        "wheel": "pyqpanda_alg-2.1.0-py3-none-any.whl",
        "wheel_sha256": "0" * 64,
        "python_version": "3.13.11",
        "pyqpanda3_version": "0.4.1",
        "qpanda3_runtime_version": "1.0.1",
        "device": {
            "chip_id": "WK_C180",
            "summary": "origin quantum superconducting chip, 128 qubits",
            "calibration_timestamp": "2026-08-12T07:00:00Z",
        },
        "cases": [
            {
                "algorithm": "bell",
                "execution_mode": "qpu",
                "task_ids": ["task-bell-001"],
                "shots": 1000,
                "threshold": 0.9,
                "raw_result_digest": "1" * 64,
                "parsed_result": dict(_BELL_PASS_COUNTS),
                "verdict": "passed",
                "transpiled": True,
            },
            {
                "algorithm": "grover",
                "execution_mode": "qpu",
                "task_ids": ["task-grover-001"],
                "shots": 2000,
                "threshold": 0.9,
                "raw_result_digest": "2" * 64,
                "parsed_result": {"success_probability": 0.95},
                "verdict": "passed",
                "transpiled": True,
            },
        ],
    }


@pytest.fixture
def fake_runtime_service():
    """Credential-free qpanda3-runtime stand-in shared by release tests."""
    return FakeRuntimeService()


@pytest.fixture
def good_task_result():
    """Fake task manager whose counts pass the fixed bell threshold."""
    return FakeQTaskManager([dict(_BELL_PASS_COUNTS)], kind="sample")


@pytest.fixture
def bad_task_result():
    """Fake task manager whose counts fail the fixed bell threshold."""
    return FakeQTaskManager([dict(_BELL_FAIL_COUNTS)], kind="sample")


def fake_service_with_bad_result():
    """Plain helper: fake service whose deterministic results fail.

    Not a fixture because the QPU runner test constructs it directly
    (``QPURunner(service=fake_service_with_bad_result())``).
    """
    service = FakeRuntimeService()
    service.sample_results = [dict(_BELL_FAIL_COUNTS)]
    return service
