"""State-machine tests for resumable :class:`AlgorithmTask`.

Each ``poll()`` call runs exactly one deterministic step through the
process-local ``advance`` callback; completed step results accumulate
into the algorithm result.  The recovery tests cover checkpointing and
``resume()`` through a registered algorithm factory.
"""

import json

import pytest

from pyqpanda_alg.execution import (
    AlgorithmInputError,
    AlgorithmTask,
    CompletedBackendTask,
    LocalBackend,
    TaskRecoveryError,
    TaskStatus,
    TaskSubmissionError,
    TaskTimeoutError,
    register_algorithm,
)


class _PendingBackendTask:
    """Backend task that has not finished yet."""

    id = "pending-1"

    def status(self):
        return TaskStatus.RUNNING

    def try_result(self):
        return None

    def result(self, timeout=None):
        return 1

    def checkpoint(self, path=None):
        return None


def test_poll_advances_two_backend_steps():
    values = iter([2, 3])

    def advance(state):
        value = next(values)
        state["total"] += value
        return CompletedBackendTask(value), state["total"] >= 5

    task = AlgorithmTask(algorithm="test-counter", initial_state={"total": 0}, advance=advance)
    assert task.poll() is TaskStatus.RUNNING
    assert task.poll() is TaskStatus.SUCCEEDED
    assert task.result() == 5


def test_initial_status_is_pending_and_success_is_terminal():
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={"total": 0},
        advance=lambda state: (CompletedBackendTask(1), True),
    )
    assert task.status() is TaskStatus.PENDING
    assert task.poll() is TaskStatus.SUCCEEDED
    assert task.poll() is TaskStatus.SUCCEEDED  # terminal polling is a no-op


def test_try_result_is_none_until_succeeded():
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={"total": 0},
        advance=lambda state: (CompletedBackendTask(2), False),
    )
    assert task.try_result() is None
    task.poll()
    assert task.try_result() is None


def test_backend_task_ids_record_every_step():
    def advance(state):
        state["n"] = state.get("n", 0) + 1
        return CompletedBackendTask(1, task_id=f"step-{state['n']}"), False

    task = AlgorithmTask(algorithm="test-ids", initial_state={}, advance=advance)
    task.poll()
    task.poll()
    assert task.backend_task_ids == ["step-1", "step-2"]


def test_result_times_out_when_algorithm_never_finishes():
    task = AlgorithmTask(
        algorithm="test-forever",
        initial_state={"total": 0},
        advance=lambda state: (CompletedBackendTask(1, task_id="t-1"), False),
    )
    with pytest.raises(TaskTimeoutError):
        task.result(timeout=0.0)
    assert task.status() is TaskStatus.TIMED_OUT


def test_poll_marks_task_failed_when_advance_raises():
    def failing_advance(state):
        raise TaskSubmissionError("backend refused the submission")

    task = AlgorithmTask(algorithm="test-failing", initial_state={}, advance=failing_advance)
    with pytest.raises(TaskSubmissionError):
        task.poll()
    assert task.status() is TaskStatus.FAILED


def test_resume_continues_from_checkpoint(tmp_path):
    def make_advance(backend=None):
        def advance(state):
            step = state.get("steps", 0) + 1
            state["steps"] = step
            state["total"] += step
            return CompletedBackendTask(step, task_id=f"task-{step}"), state["total"] >= 5

        return advance

    register_algorithm("resumable-counter", make_advance)
    task = AlgorithmTask(
        algorithm="resumable-counter",
        initial_state={"total": 0},
        advance=make_advance(),
    )
    assert task.poll() is TaskStatus.RUNNING  # step 1: total 1
    path = task.checkpoint(tmp_path / "task.json")

    resumed = AlgorithmTask.resume(path)
    assert resumed.status() is TaskStatus.RUNNING
    assert resumed.backend_task_ids == ["task-1"]
    assert resumed.poll() is TaskStatus.RUNNING  # step 2: total 3
    assert resumed.poll() is TaskStatus.SUCCEEDED  # step 3: total 6
    assert resumed.backend_task_ids == ["task-1", "task-2", "task-3"]
    assert resumed.result() == 6


def test_resume_of_unknown_algorithm_raises_recovery_error(tmp_path):
    task = AlgorithmTask(
        algorithm="no-such-algorithm",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    with pytest.raises(TaskRecoveryError, match="no-such-algorithm"):
        AlgorithmTask.resume(path)


def test_inflight_steps_do_not_accumulate_until_finished():
    task = AlgorithmTask(
        algorithm="test-pending",
        initial_state={},
        advance=lambda state: (_PendingBackendTask(), False),
    )
    assert task.poll() is TaskStatus.RUNNING
    assert task.backend_task_ids == ["pending-1"]
    assert task.try_result() is None


def test_constructor_validates_algorithm_and_advance():
    with pytest.raises(AlgorithmInputError, match="algorithm name"):
        AlgorithmTask(
            algorithm="", initial_state={}, advance=lambda s: (CompletedBackendTask(0), True)
        )
    with pytest.raises(AlgorithmInputError, match="callable"):
        AlgorithmTask(algorithm="test-counter", initial_state={}, advance="not-callable")


def test_register_algorithm_validates_name():
    with pytest.raises(AlgorithmInputError, match="algorithm name"):
        register_algorithm("", lambda backend: lambda state: (CompletedBackendTask(0), True))


def test_poll_after_timeout_is_a_noop():
    calls = []

    def advance(state):
        calls.append(1)
        return CompletedBackendTask(0, task_id="t-0"), False

    task = AlgorithmTask(algorithm="test-forever", initial_state={}, advance=advance)
    assert task.poll() is TaskStatus.RUNNING
    with pytest.raises(TaskTimeoutError):
        task.result(timeout=0.0)
    assert task.poll() is TaskStatus.TIMED_OUT
    assert len(calls) == 1  # timeout and post-timeout polls never advance


def test_poll_and_result_reraise_stored_step_failure():
    def failing_advance(state):
        raise TaskSubmissionError("backend refused")

    task = AlgorithmTask(algorithm="test-failing", initial_state={}, advance=failing_advance)
    with pytest.raises(TaskSubmissionError):
        task.poll()
    with pytest.raises(TaskSubmissionError):
        task.poll()  # re-raises the stored failure
    with pytest.raises(TaskSubmissionError):
        task.result()  # result() surfaces the failure instead of hanging


def test_try_result_returns_result_after_success():
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={"total": 0},
        advance=lambda state: (CompletedBackendTask(2, task_id="s-1"), True),
    )
    task.poll()
    assert task.try_result() == 2
    assert task.result() == 2


def test_result_after_timeout_raises_timeout_again():
    task = AlgorithmTask(
        algorithm="test-forever",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0, task_id="t"), False),
    )
    with pytest.raises(TaskTimeoutError):
        task.result(timeout=0.0)
    with pytest.raises(TaskTimeoutError):
        task.result(timeout=0.0)


def test_checkpoint_requires_explicit_path():
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    with pytest.raises(AlgorithmInputError, match="path"):
        task.checkpoint()


def test_checkpoint_records_backend_identity(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0), True),
        backend=LocalBackend(),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["backend"] == {"type": "LocalBackend"}


def test_result_polls_until_succeeded():
    def advance(state):
        state["total"] = state.get("total", 0) + 1
        return CompletedBackendTask(1, task_id=f"s-{state['total']}"), state["total"] >= 2

    task = AlgorithmTask(algorithm="test-counter", initial_state={}, advance=advance)
    assert task.result() == 2
    assert task.status() is TaskStatus.SUCCEEDED
    assert task.backend_task_ids == ["s-1", "s-2"]


def test_resumed_failed_task_reports_failure_without_stored_error(tmp_path):
    def failing_advance(state):
        raise TaskSubmissionError("backend refused")

    register_algorithm("resumable-failing", lambda backend: failing_advance)
    task = AlgorithmTask(
        algorithm="resumable-failing", initial_state={}, advance=failing_advance
    )
    with pytest.raises(TaskSubmissionError):
        task.poll()
    path = task.checkpoint(tmp_path / "task.json")

    resumed = AlgorithmTask.resume(path)
    assert resumed.status() is TaskStatus.FAILED
    assert resumed.poll() is TaskStatus.FAILED  # no stored error after resume
    with pytest.raises(TaskRecoveryError, match="failed without a recorded error"):
        resumed.result()


def test_resume_rejects_unsupported_checkpoint_version(tmp_path):
    task = AlgorithmTask(
        algorithm="resumable-counter",
        initial_state={},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["format_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskRecoveryError, match="version"):
        AlgorithmTask.resume(path)
