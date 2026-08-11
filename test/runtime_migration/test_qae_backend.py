"""Runtime-contract tests for QAE and IQAE sampling execution.

Plan 3 Task 3: amplitude estimation flows through
``backend.submit_sample`` with normalized counts and the projection
``int(key, 2) >> qnumber`` onto the ancilla register, and iterative
estimation is exposed as a checkpointable state machine: ``submit()``
returns before any round, each ``poll()`` advances exactly one
iteration, and ``resume()`` continues from the checkpointed round
without re-submitting completed rounds.

The scripted counts are tuned to the IQAE loop: each iteration submits
``n_round`` shots (about 11-14 for epsilon=0.1) so a scripted ``m=1``
count keeps ``res = m / n_round`` inside ``[0, 1]`` and the Chernoff
bounds never degenerate (m=3 would overflow ``n_round`` when the query
index k grows and break the arccos math with NaN).
"""

import math

import numpy as np
import pytest
from pyqpanda3.core import QCircuit, RY, X

from pyqpanda_alg.QAE import QAE
from pyqpanda_alg.QAE.QAE import IQAE
from pyqpanda_alg.execution import AlgorithmTask, TaskStatus
from test.execution.fakes import RecordingBackend


def _create_cir(qlist):
    """Operator whose target amplitude is sin(pi/3) on |11>."""
    cir = QCircuit()
    cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
    return cir


def test_qae_runtime_uses_sampling_without_legacy_result_api():
    backend = RecordingBackend(sample_counts=[{"001000000": 700, "010000000": 300}])
    qae = QAE(
        operator_in=_create_cir, qnumber=2, epsilon=0.01, res_index=[0, 1], target_state="11"
    )
    prob = qae.run(backend=backend)
    # Ancilla register value 16 ("0010000") wins with 0.7 probability:
    # amplitude = sin(16 * pi / 2**7) = sin(pi / 8).
    assert prob == pytest.approx(math.sin(math.pi / 8) ** 2)
    assert backend.sample_call_count == 1
    assert "get_prob_dict" not in backend.accessed_result_attributes


def test_qae_submit_is_a_single_round_task():
    backend = RecordingBackend(sample_counts=[{"001000000": 700, "010000000": 300}])
    qae = QAE(
        operator_in=_create_cir, qnumber=2, epsilon=0.01, res_index=[0, 1], target_state="11"
    )
    task = qae.submit(backend=backend)
    assert task.status() is TaskStatus.PENDING
    assert backend.sample_call_count == 0
    assert task.poll() is TaskStatus.SUCCEEDED
    assert backend.sample_call_count == 1
    assert task.backend_task_ids == ["recorded-sample-1"]
    assert task.result() == pytest.approx(math.sin(math.pi / 8) ** 2)


def test_iqae_submit_polls_one_round_at_a_time():
    backend = RecordingBackend(sample_counts=[{"1": 1, "0": 9}])
    iqae = IQAE(operator_in=_create_cir, qnumber=2, epsilon=0.1, res_index=-1)
    task = iqae.submit(backend=backend)
    assert task.status() is TaskStatus.PENDING
    assert backend.sample_call_count == 0
    assert task.poll() is TaskStatus.RUNNING
    assert backend.sample_call_count == 1
    assert task.backend_task_ids == ["recorded-sample-1"]
    assert task.poll() is TaskStatus.RUNNING
    assert backend.sample_call_count == 2
    assert task.backend_task_ids == ["recorded-sample-1", "recorded-sample-2"]


def test_iqae_run_returns_probability_in_unit_interval():
    backend = RecordingBackend(sample_counts=[{"1": 1, "0": 9}])
    iqae = IQAE(operator_in=_create_cir, qnumber=2, epsilon=0.1, res_index=-1)
    result = iqae.run(backend=backend)
    assert 0.0 <= result <= 1.0
    assert backend.sample_call_count > 1
    assert "get_prob_dict" not in backend.accessed_result_attributes


def test_iqae_resume_continues_without_replaying_completed_rounds(tmp_path):
    backend = RecordingBackend(sample_counts=[{"1": 1, "0": 9}])
    iqae = IQAE(operator_in=_create_cir, qnumber=2, epsilon=0.1, res_index=-1)
    task = iqae.submit(backend=backend)
    assert task.poll() is TaskStatus.RUNNING
    assert task.poll() is TaskStatus.RUNNING
    assert task.backend_task_ids == ["recorded-sample-1", "recorded-sample-2"]
    checkpoint = tmp_path / "iqae-checkpoint.json"
    task.checkpoint(checkpoint)

    resumed = AlgorithmTask.resume(checkpoint, backend=backend)
    # Completed rounds are carried over, not replayed.
    assert resumed.backend_task_ids == task.backend_task_ids
    assert backend.sample_call_count == 2
    resumed.poll()
    assert resumed.backend_task_ids == [
        "recorded-sample-1",
        "recorded-sample-2",
        "recorded-sample-3",
    ]
    assert len(resumed.backend_task_ids) == len(set(resumed.backend_task_ids))
    assert backend.sample_call_count == len(resumed.backend_task_ids)
