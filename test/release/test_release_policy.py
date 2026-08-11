"""Release policy gate: commit binding and all-passed verdicts.

The release-policy checker is the deterministic gate that release
creation runs: a manifest attesting any other commit is rejected, and
a single failed case verdict blocks the release.  Both tests below are
verbatim from the task brief.  The wheel-binding tests are the smoke
regression for the runtime-rc.yml deadlock: the RC workflow passes a
directory-prefixed wheel path into the qualification runner, which must
record only the basename so the gate's basename binding accepts the
artifact.
"""

import hashlib
from importlib.metadata import version

import pytest

from tools.release_qualification.check_release import (
    ReleasePolicyError,
    check_release,
)
from tools.release_qualification.run_preflight import (
    _DEFAULT_WHEEL,
    _wheel_basename,
    run_preflight,
)

#: The exact wheel filename the 2.1.0 candidate builds.
_WHEEL_NAME = "pyqpanda_alg-2.1.0-py3-none-any.whl"


def test_release_rejects_manifest_for_other_commit(valid_manifest, repo_commit):
    valid_manifest["git_commit"] = "0" * 40
    with pytest.raises(ReleasePolicyError, match="commit"):
        check_release(valid_manifest, expected_commit=repo_commit)


def test_release_requires_every_case_to_pass(valid_manifest):
    valid_manifest["cases"][0]["verdict"] = "failed"
    with pytest.raises(ReleasePolicyError, match="failed"):
        check_release(valid_manifest, expected_commit=valid_manifest["git_commit"])


def test_wheel_basename_keeps_default_for_none_or_dash():
    """None/``""``/``"-"`` keep the committed default wheel name."""
    assert _wheel_basename(None) == _DEFAULT_WHEEL
    assert _wheel_basename("") == _DEFAULT_WHEEL
    assert _wheel_basename("-") == _DEFAULT_WHEEL
    assert _wheel_basename(_WHEEL_NAME) == _WHEEL_NAME


def test_default_wheel_tracks_the_installed_package_version():
    """The default wheel name is derived from the installed pyqpanda_alg
    version, so the committed artifact name cannot drift from the
    release it qualifies."""
    installed = version("pyqpanda_alg")
    assert _DEFAULT_WHEEL == f"pyqpanda_alg-{installed}-py3-none-any.whl"
    assert _wheel_basename(f".artifacts/{installed}/wheel/{_WHEEL_NAME}") == _WHEEL_NAME


def test_qualification_runner_records_directory_prefixed_wheel_as_basename(
    fake_runtime_service,
):
    """Producer: a workflow-style directory-prefixed wheel path is
    recorded as its basename, exactly what the gate's basename binding
    and the schema's "Wheel filename" contract expect."""
    manifest = run_preflight(
        fake_runtime_service,
        "WK_C180",
        wheel=f".artifacts/2.1.0/wheel/{_WHEEL_NAME}",
    )
    assert manifest.wheel == _WHEEL_NAME


def test_release_gate_accepts_manifest_assembled_from_directory_prefixed_wheel(
    fake_runtime_service, tmp_path
):
    """Smoke: the runtime-rc.yml validate step now passes.

    The workflow hands the runner a directory-prefixed wheel path; the
    runner records the basename and digest, and the gate accepts the
    artifact and digest-verifies it against the same wheel path.
    Fake-backend case verdicts are pinned to ``passed`` -- the verdict
    binding itself is covered by the tests above -- so this test pins
    the wheel binding that previously deadlocked the RC pipeline.
    """
    wheel_path = tmp_path / ".artifacts" / "2.1.0" / "wheel" / _WHEEL_NAME
    wheel_path.parent.mkdir(parents=True)
    wheel_path.write_bytes(b"candidate wheel payload")
    digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()

    manifest = run_preflight(
        fake_runtime_service,
        "WK_C180",
        wheel=str(wheel_path),  # the workflow passes a path, not a basename
        wheel_sha256=digest,
    )
    payload = manifest.to_dict()
    for case in payload["cases"]:
        case["verdict"] = "passed"
    assert (
        check_release(payload, expected_commit=manifest.git_commit, wheel_path=wheel_path)
        is payload
    )
