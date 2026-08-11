"""Qualification manifest schema tests: closed structure and validation.

The manifest is the release gate's only evidence artifact, so the
schema is closed: every field is enumerated, unknown keys are rejected
at every level, and identifiers (git commit, wheel digest, raw-result
digest) are shape-checked.
"""

import pytest

from tools.release_qualification.manifest import (
    AlgorithmQualification,
    ManifestValidationError,
    QualificationManifest,
    validate_manifest,
)


def test_manifest_requires_commit_wheel_and_device(valid_manifest):
    valid_manifest["git_commit"] = "a" * 40
    valid_manifest["wheel_sha256"] = "0" * 64
    valid_manifest["device"]["chip_id"] = "WK_C180"
    validate_manifest(valid_manifest)


def test_valid_manifest_conforms_to_schema(valid_manifest):
    validate_manifest(valid_manifest)


@pytest.mark.parametrize(
    "field",
    [
        "version",
        "timestamp",
        "git_commit",
        "wheel",
        "wheel_sha256",
        "python_version",
        "pyqpanda3_version",
        "qpanda3_runtime_version",
        "device",
        "cases",
    ],
)
def test_manifest_requires_every_top_level_field(valid_manifest, field):
    del valid_manifest[field]
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


def test_manifest_rejects_non_hex_git_commit(valid_manifest):
    valid_manifest["git_commit"] = "z" * 40
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


def test_manifest_rejects_short_wheel_sha256(valid_manifest):
    valid_manifest["wheel_sha256"] = "0" * 63
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


def test_manifest_rejects_unknown_top_level_keys(valid_manifest):
    valid_manifest["surprise"] = 1
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


def test_manifest_rejects_unknown_device_keys(valid_manifest):
    valid_manifest["device"]["latency"] = "42ms"
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


def test_manifest_rejects_case_missing_fields(valid_manifest):
    del valid_manifest["cases"][0]["verdict"]
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


def test_manifest_rejects_unknown_case_keys(valid_manifest):
    valid_manifest["cases"][0]["callback_url"] = "https://example.invalid/cb"
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


def test_manifest_rejects_non_utc_timestamp(valid_manifest):
    valid_manifest["timestamp"] = "2026-08-12 08:00:00"
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-12T08:00:00+08:00",  # non-zero UTC offset
        "2026-08-12T08:00:00-05:30",
    ],
)
def test_manifest_rejects_non_zero_utc_offsets(valid_manifest, timestamp):
    """Timestamps must be UTC: only 'Z' or '+00:00' are accepted."""
    valid_manifest["timestamp"] = timestamp
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


def test_manifest_accepts_explicit_utc_zero_offset(valid_manifest):
    valid_manifest["timestamp"] = "2026-08-12T08:00:00+00:00"
    validate_manifest(valid_manifest)


def test_manifest_rejects_unpassed_verdict(valid_manifest):
    valid_manifest["cases"][0]["verdict"] = "skipped"
    with pytest.raises(ManifestValidationError):
        validate_manifest(valid_manifest)


def test_manifest_dataclass_round_trips_through_schema(valid_manifest):
    manifest = QualificationManifest(
        version=valid_manifest["version"],
        timestamp=valid_manifest["timestamp"],
        git_commit=valid_manifest["git_commit"],
        wheel=valid_manifest["wheel"],
        wheel_sha256=valid_manifest["wheel_sha256"],
        python_version=valid_manifest["python_version"],
        pyqpanda3_version=valid_manifest["pyqpanda3_version"],
        qpanda3_runtime_version=valid_manifest["qpanda3_runtime_version"],
        device=valid_manifest["device"],
        cases=tuple(
            AlgorithmQualification(
                algorithm=case["algorithm"],
                execution_mode=case["execution_mode"],
                task_ids=tuple(case["task_ids"]),
                shots=case["shots"],
                threshold=case["threshold"],
                raw_result_digest=case["raw_result_digest"],
                parsed_result=case["parsed_result"],
                verdict=case["verdict"],
                transpiled=case["transpiled"],
            )
            for case in valid_manifest["cases"]
        ),
    )
    payload = manifest.to_dict()
    assert payload == valid_manifest
    validate_manifest(payload)
