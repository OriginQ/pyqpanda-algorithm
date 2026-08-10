"""Checkpoint security tests: credentials and live objects never serialize.

The checkpoint guarantee is enforced recursively: credential-named keys
are filtered from nested dicts and lists, and non-serializable values
(such as RuntimeService/QDevice handles) are rejected before anything
is written to disk.
"""

import json

import pytest

from pyqpanda_alg.execution import (
    AlgorithmInputError,
    AlgorithmTask,
    CompletedBackendTask,
    TaskRecoveryError,
)
from pyqpanda_alg.execution.checkpoint import redact_credentials


def test_checkpoint_does_not_serialize_credentials(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={"total": 0},
        metadata={"api_key": "secret", "nested": {"token": "secret-token"}},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = path.read_text(encoding="utf-8")
    assert "secret" not in payload
    assert "token" not in payload


def test_redaction_recurses_through_nested_structures():
    payload = redact_credentials(
        {
            "api_key": "top-level-key",
            "nested": {"access_token": "a", "fine": {"password": "b", "keep": 1}},
            "entries": [{"private_key": "c"}, {"keep": 2}],
            "keep": {"list": [1, {"token": "d"}]},
        }
    )
    assert payload == {
        "nested": {"fine": {"keep": 1}},
        "entries": [{}, {"keep": 2}],
        "keep": {"list": [1, {}]},
    }


def test_checkpoint_preserves_redacted_metadata_structure(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={"total": 0},
        metadata={"api_key": "secret", "nested": {"token": "x", "keep": "y"}},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"] == {"nested": {"keep": "y"}}


def test_checkpoint_redacts_credentials_inside_state(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={"api_key": "state-secret", "total": 0},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "api_key" not in payload["state"]
    assert "state-secret" not in path.read_text(encoding="utf-8")


def test_checkpoint_rejects_live_objects_in_state(tmp_path):
    class LiveObject:
        """Stand-in for RuntimeService/QDevice objects."""

    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={"session": LiveObject()},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    with pytest.raises(AlgorithmInputError, match="serializable"):
        task.checkpoint(tmp_path / "task.json")


def test_redaction_preserves_tuples():
    payload = redact_credentials(({"token": "x"}, 1, ("api_key", "y")))
    assert payload == ({}, 1, ("api_key", "y"))


def test_redaction_handles_non_string_keys():
    assert redact_credentials({1: "kept", "api_key": "dropped"}) == {1: "kept"}


def test_resume_of_missing_checkpoint_raises(tmp_path):
    with pytest.raises(TaskRecoveryError, match="could not read"):
        AlgorithmTask.resume(tmp_path / "missing.json")


def test_resume_rejects_non_object_checkpoint_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(TaskRecoveryError, match="object"):
        AlgorithmTask.resume(path)


def test_resume_rejects_invalid_checkpoint_status(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "exploded"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskRecoveryError, match="status"):
        AlgorithmTask.resume(path)


def test_resume_rejects_checkpoint_with_missing_fields(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["metadata"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskRecoveryError, match="metadata"):
        AlgorithmTask.resume(path)


def test_checkpoint_serializes_list_and_tuple_state(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={"params": [0.1, 0.2], "pair": (1, 2)},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == {"params": [0.1, 0.2], "pair": [1, 2]}


def test_resume_rejects_checkpoint_with_empty_algorithm_name(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["algorithm"] = ""
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskRecoveryError, match="algorithm name"):
        AlgorithmTask.resume(path)


def test_resume_rejects_non_string_task_id_lists(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["backend_task_ids"] = "oops"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskRecoveryError, match="backend_task_ids"):
        AlgorithmTask.resume(path)


def test_resume_rejects_non_object_metadata(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"] = ["not", "a", "dict"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskRecoveryError, match="metadata"):
        AlgorithmTask.resume(path)


def test_checkpoint_rejects_live_objects_in_metadata(tmp_path):
    class LiveObject:
        """Stand-in for RuntimeService/QDevice objects."""

    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={},
        metadata={"service": LiveObject()},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    with pytest.raises(AlgorithmInputError, match="serializable"):
        task.checkpoint(tmp_path / "task.json")
