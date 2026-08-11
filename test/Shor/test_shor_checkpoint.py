"""Checkpoint and recovery tests for the Shor attempt state machine.

A checkpointed run carries the modulus, config, RNG position, and the
full attempt history — attempted bases, completed task IDs, per-attempt
histograms, candidate orders, and rejection reasons — as JSON
primitives.  Resuming rebuilds the solver's attempt machine through the
registered ``shor`` advance factory, continues the exact base sequence,
and never resubmits a completed task.  All runtime-contract tests run
credential-free against scripted recording backends.
"""

import json
import random

import pytest

from pyqpanda_alg.Shor import Shor, ShorConfig
from pyqpanda_alg.execution import AlgorithmTask, TaskStatus

from test.execution.fakes import RecordingBackend
from test.Shor.test_shor_runtime import FixedBaseRng


@pytest.mark.runtime_contract
def test_shor_checkpoint_records_attempt_state(tmp_path):
    checkpoint = tmp_path / "shor.json"
    backend = RecordingBackend(sample_counts=[{"01000000": 600, "11000000": 400}])
    task = Shor(15, rng=FixedBaseRng(2), config=ShorConfig(max_attempts=2)).submit(
        backend=backend
    )
    assert task.poll() is TaskStatus.SUCCEEDED  # base 2 factors 15 on attempt one
    task.checkpoint(checkpoint)

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    state = payload["state"]
    assert payload["algorithm"] == "shor"
    assert payload["backend"] == {"type": "RecordingBackend"}
    assert payload["backend_task_ids"] == ["recorded-sample-1"]
    assert state["modulus"] == 15
    assert state["config"] == {"max_attempts": 2, "phase_qubits": None}
    assert state["classical_done"] is True
    # A constant double exposes no position; its value is the attempted
    # base recorded below, which is exactly what a resumed run re-draws.
    assert state["rng_state"] is None
    (record,) = state["history"]
    assert record["base"] == 2
    assert record["task_id"] == "recorded-sample-1"
    assert record["outcome"] == "factored"
    (sample,) = record["samples"]
    assert sample["sample"] == 64
    assert sample["count"] == 600
    assert sample["candidate_order"] == 4
    assert sample["rejection"] is None


@pytest.mark.runtime_contract
def test_shor_resume_continues_base_sequence_without_resubmission(tmp_path):
    checkpoint = tmp_path / "shor.json"
    # Every attempt is rejected (zero phase samples), so the run only
    # ends through exhaustion after the second attempt.
    backend = RecordingBackend(
        sample_counts=[{"00000000": 1000}, {"00000000": 1000}]
    )
    task = Shor(15, rng=random.Random(7), config=ShorConfig(max_attempts=2)).submit(
        backend=backend
    )
    assert task.poll() is TaskStatus.RUNNING  # attempt 1 rejected
    task.checkpoint(checkpoint)

    resumed = AlgorithmTask.resume(checkpoint, backend=backend)
    assert resumed.status() is TaskStatus.RUNNING
    assert resumed.backend_task_ids == ["recorded-sample-1"]
    calls_before = backend.sample_call_count
    assert resumed.poll() is TaskStatus.SUCCEEDED  # attempt 2, then exhaustion
    assert backend.sample_call_count == calls_before + 1  # exactly one new task
    assert resumed.backend_task_ids == ["recorded-sample-1", "recorded-sample-2"]
    assert resumed.result()["task_ids"] == ["recorded-sample-1", "recorded-sample-2"]
    assert resumed.result()["metadata"]["exhausted"] is True

    # The resumed run drew the exact base sequence of the checkpointed
    # RNG: the RNG position survived the round trip.
    resumed.checkpoint(checkpoint)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))["state"]
    assert state["rng_state"][0] == 3  # random.Random state version
    assert len(state["rng_state"][1]) == 625
    fresh = random.Random(7)
    assert state["history"][0]["base"] == fresh.randrange(2, 14)
    assert state["history"][1]["base"] == fresh.randrange(2, 14)
    assert state["history"][0]["task_id"] == "recorded-sample-1"
    assert state["history"][1]["task_id"] == "recorded-sample-2"


@pytest.mark.runtime_contract
def test_shor_resume_of_completed_checkpoint_never_resubmits(tmp_path):
    checkpoint = tmp_path / "shor.json"
    backend = RecordingBackend(sample_counts=[{"01000000": 600, "11000000": 400}])
    task = Shor(15, rng=FixedBaseRng(2), config=ShorConfig(max_attempts=1)).submit(
        backend=backend
    )
    assert task.poll() is TaskStatus.SUCCEEDED
    task.checkpoint(checkpoint)

    resumed = AlgorithmTask.resume(checkpoint, backend=backend)
    assert resumed.status() is TaskStatus.SUCCEEDED
    calls = backend.sample_call_count
    assert resumed.poll() is TaskStatus.SUCCEEDED  # terminal polling is a no-op
    assert resumed.result()["factors"] == [3, 5]  # JSON round trip: tuple to list
    assert backend.sample_call_count == calls  # nothing was resubmitted


@pytest.mark.runtime_contract
def test_shor_checkpoint_never_serializes_the_backend(tmp_path):
    checkpoint = tmp_path / "shor.json"
    backend = RecordingBackend(sample_counts=[{"00000000": 1000}])
    task = Shor(15, rng=random.Random(7), config=ShorConfig(max_attempts=2)).submit(
        backend=backend
    )
    task.poll()
    task.checkpoint(checkpoint)

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["backend"] == {"type": "RecordingBackend"}
    assert set(payload["state"]) == {
        "modulus",
        "config",
        "rng_state",
        "classical_done",
        "history",
    }
