"""Sanitizer security tests: credential-shaped keys never survive.

The release artifact must never carry an API key or token, so the
sanitizer removes credential-shaped keys recursively (dicts, lists,
tuples) with case-insensitive substring matching, and
``contains_credentials`` lets runner tests assert a payload is clean.
"""

from tools.release_qualification.manifest import (
    AlgorithmQualification,
    QualificationManifest,
    contains_credentials,
    sanitize_payload,
)


def test_sanitizer_removes_nested_credentials():
    payload = {"api_key": "abc", "nested": {"token": "def", "task_id": "task-1"}}
    clean = sanitize_payload(payload)
    assert "api_key" not in clean
    assert "token" not in clean["nested"]
    assert clean["nested"]["task_id"] == "task-1"


def test_sanitizer_removes_credentials_from_lists():
    clean = sanitize_payload(
        {"results": [{"api_key": "a"}, {"token": "b"}, {"password": "c"}, 1, "x"]}
    )
    assert clean == {"results": [{}, {}, {}, 1, "x"]}


def test_sanitizer_matches_keys_case_insensitively():
    clean = sanitize_payload(
        {
            "API_Key": "a",
            "Token": "b",
            "Authorization": "c",
            "Bearer": "d",
            "access_key": "e",
        }
    )
    assert clean == {}


def test_sanitizer_redacts_credentials_inside_tuples():
    clean = sanitize_payload(({"secret": "x"}, 1, ("api_key", "y")))
    assert clean == ({}, 1, ("api_key", "y"))


def test_sanitizer_keeps_ordinary_fields():
    payload = {"task_id": "task-1", "counts": {"00": 490}, "verdict": "passed"}
    assert sanitize_payload(payload) == payload


def test_sanitizer_does_not_mutate_the_input():
    payload = {"api_key": "abc", "nested": {"token": "def", "task_id": "task-1"}}
    sanitize_payload(payload)
    assert payload == {"api_key": "abc", "nested": {"token": "def", "task_id": "task-1"}}


def test_contains_credentials_detects_nested_credentials():
    assert contains_credentials({"nested": {"api_key": "abc"}})
    assert not contains_credentials({"nested": {"task_id": "task-1"}})


def test_contains_credentials_is_case_insensitive():
    assert contains_credentials({"TOKEN": "abc"})
    assert not contains_credentials({"task_id": "task-1"})


def test_contains_credentials_scans_lists():
    assert contains_credentials({"results": [{"ok": 1}, {"token": "t"}]})
    assert not contains_credentials({"results": [{"ok": 1}, {"task_id": "t"}]})


def test_sanitizer_redacts_hyphenated_credential_keys():
    payload = {
        "api-key": "x",
        "X-API-Key": "y",
        "nested": {"Bearer-Token": "z"},
    }
    clean = sanitize_payload(payload)
    assert not contains_credentials(clean)
    assert "x" not in str(clean)
    assert "y" not in str(clean)
    assert "z" not in str(clean)
    assert clean == {"nested": {}}


def test_sanitizer_keeps_hyphenated_ordinary_keys():
    payload = {"task-id": "task-1", "task_id": "task-2"}
    assert sanitize_payload(payload) == payload


def test_contains_credentials_detects_hyphenated_keys():
    assert contains_credentials({"X-API-Key": "abc"})
    assert contains_credentials({"nested": {"Bearer-Token": "z"}})
    assert not contains_credentials({"task-id": "task-1"})


def test_manifest_to_dict_redacts_credentials(valid_manifest):
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
        cases=(
            AlgorithmQualification(
                algorithm="bell",
                execution_mode="qpu",
                task_ids=("task-1",),
                shots=1000,
                threshold=0.9,
                raw_result_digest="1" * 64,
                parsed_result={"api_key": "abc", "counts": {"00": 490}},
                verdict="passed",
                transpiled=True,
            ),
        ),
    )
    payload = manifest.to_dict()
    assert not contains_credentials(payload)
    assert "abc" not in str(payload)
    assert payload["cases"][0]["parsed_result"] == {"counts": {"00": 490}}
