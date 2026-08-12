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

from test.execution.fakes import FakeDevice, FakeQTaskManager, FakeRuntimeService
from tools.release_qualification.cases import (
    QUALIFICATION_CASES,
    SMOKE_CASES,
    TRANSPILATION_CASES,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Counts that pass the fixed bell threshold (correlated pairs dominate).
_BELL_PASS_COUNTS = {"00": 490, "11": 480, "01": 15, "10": 15}
#: Deterministic counts that fail the fixed bell threshold (uniform).
_BELL_FAIL_COUNTS = {"00": 250, "11": 250, "01": 250, "10": 250}

#: Gate set and topology the fake device must advertise so the fixed
#: cases pass the QPandaRuntimeBackend preflight validation (a real
#: device reports its own capabilities through the same surface).
_QPU_GATES = [
    "H", "X", "Y", "Z", "RX", "RY", "RZ", "S", "T", "SDG", "TDG", "SX",
    "CNOT", "CX", "CZ", "SWAP", "CP", "CSWAP", "CCX", "CCU1", "CCCP",
    "CCCU1", "CCH", "CCSWAP", "U1", "CU1", "P", "U2", "U3",
    "CORACLE", "ORACLE",
]


def qpu_fake_device() -> FakeDevice:
    """Fake device advertising the gates/topology the fixed cases use.

    Plain helper (not a fixture) so both the QPU runner tests and the
    per-case end-to-end tests construct it directly.
    """
    device = FakeDevice()
    device.basic_gates.return_value = list(_QPU_GATES)
    device.chip_topo_edges.return_value = [
        [i, j] for i in range(20) for j in range(i + 1, 20)
    ]
    return device


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
                "algorithm": case.algorithm,
                "execution_mode": "qpu",
                "task_ids": [f"task-{case.algorithm}-001"],
                "shots": case.shots,
                "threshold": case.threshold,
                "raw_result_digest": f"{index % 10}" * 64,
                "parsed_result": {"qualified": True},
                "verdict": "passed",
                "transpiled": False,
            }
            for index, case in enumerate((*QUALIFICATION_CASES, *SMOKE_CASES), start=1)
        ],
    }


@pytest.fixture
def valid_preflight_manifest(valid_manifest):
    payload = {
        key: value for key, value in valid_manifest.items() if key != "cases"
    }
    payload["cases"] = [
        {
            "algorithm": case.algorithm,
            "execution_mode": (
                "transpile" if case.mode == "transpile" else "preflight"
            ),
            "task_ids": [],
            "shots": case.shots,
            "threshold": case.threshold,
            "raw_result_digest": f"{index % 10}" * 64,
            "parsed_result": {"qualified": True},
            "verdict": "passed",
            "transpiled": True,
        }
        for index, case in enumerate(
            (*QUALIFICATION_CASES, *TRANSPILATION_CASES), start=1
        )
    ]
    return payload


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
