"""Credential-free contract tests for the qpanda3-runtime adapter.

Every test runs against :class:`FakeRuntimeService` and
:class:`FakeQTaskManager` (see ``test/execution/fakes.py``), so the
suite passes without API credentials or network access.  The
``runtime_contract`` marker groups these tests for CI.
"""

import asyncio
import json

import pytest

from pyqpanda_alg.execution import (
    BackendTask,
    BackendUnavailableError,
    DeviceCapabilityError,
    ExecutionOptions,
    EstimateBatchResult,
    QPandaRuntimeBackend,
    ResultDecodingError,
    RuntimeBackendTask,
    SampleBatchResult,
    TaskRecoveryError,
    TaskStatus,
    TaskSubmissionError,
    TaskTimeoutError,
)

from test.execution.fakes import FakeQTaskManager


@pytest.mark.runtime_contract
def test_runtime_sample_forwards_exact_options(fake_runtime_service, fake_device, bell_program):
    backend = QPandaRuntimeBackend(fake_runtime_service, fake_device)
    options = ExecutionOptions(shots=321, specified_block=(0, 1))
    task = backend.submit_sample(bell_program, options=options)
    assert fake_runtime_service.sample_calls[0]["shots"] == 321
    assert fake_runtime_service.sample_calls[0]["specified_block"] == [0, 1]
    assert task.result().single_counts() == {"00": 160, "11": 161}


@pytest.mark.runtime_contract
def test_runtime_sample_forwards_default_options(runtime_backend, bell_program):
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    call = runtime_backend.service.sample_calls[0]
    assert call["specified_block"] is None
    assert call["shots"] == 1000
    assert call["is_amend"] is True
    assert call["is_mapping"] is True
    assert call["is_optimization"] is True
    assert call["circuits"] is bell_program
    assert call["device"] is runtime_backend.device


@pytest.mark.runtime_contract
def test_runtime_estimate_forwards_exact_options(runtime_backend, bell_program, observable):
    options = ExecutionOptions(
        shots=111,
        specified_block=(2, 3),
        is_mapping=False,
        is_amend=False,
        is_optimization=False,
    )
    task = runtime_backend.submit_estimate((bell_program, observable), options=options)
    call = runtime_backend.service.estimate_calls[0]
    assert call["shots"] == 111
    assert call["specified_block"] == [2, 3]
    assert call["is_mapping"] is False
    assert call["is_amend"] is False
    assert call["is_optimization"] is False
    assert task.result().single_value() == 0.5


@pytest.mark.runtime_contract
def test_runtime_statevector_is_rejected_before_submission(runtime_backend, bell_program):
    with pytest.raises(DeviceCapabilityError, match="state-vector"):
        runtime_backend.submit_statevector(bell_program, options=ExecutionOptions())
    assert runtime_backend.service.sample_calls == []
    assert runtime_backend.service.estimate_calls == []


@pytest.mark.runtime_contract
def test_runtime_task_conforms_to_backend_task_protocol(runtime_backend, bell_program):
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    assert isinstance(task, BackendTask)


@pytest.mark.runtime_contract
def test_runtime_task_lifecycle_via_try_result(runtime_backend, bell_program):
    runtime_backend.service.finished = False
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions(shots=100))
    assert task.status() is TaskStatus.RUNNING
    assert task.try_result() is None
    task.raw_task.finished = True
    assert task.status() is TaskStatus.SUCCEEDED
    result = task.try_result()
    assert isinstance(result, SampleBatchResult)
    assert result.single_counts() == {"00": 160, "11": 161}


@pytest.mark.runtime_contract
def test_runtime_task_result_blocks_and_caches(runtime_backend, bell_program):
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions(shots=321))
    first = task.result(timeout=60)
    second = task.result(timeout=60)
    assert first is second
    assert isinstance(first, SampleBatchResult)
    assert first.shots == 321
    assert task.status() is TaskStatus.SUCCEEDED
    assert task.try_result() is first


@pytest.mark.runtime_contract
def test_runtime_task_id_is_a_stable_string():
    qtask = FakeQTaskManager([{"00": 1}], kind="sample", task_id=["sub-1", "sub-2"])
    task = RuntimeBackendTask(qtask, kind="sample", shots=1000)
    assert task.id == "sub-1,sub-2"


@pytest.mark.runtime_contract
def test_runtime_task_checkpoint_delegates_to_qtask_manager(
    runtime_backend, bell_program, tmp_path
):
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    path = tmp_path / "runtime-task.json"
    assert task.checkpoint(str(path)) is None
    assert task.raw_task.checkpoint_calls == [
        {"filepath": str(path), "user_data": None}
    ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["data"]["metadata"]["qtask_type"] == "sample"


@pytest.mark.runtime_contract
def test_runtime_task_recovery_rebuilds_from_checkpoint(runtime_backend, bell_program, tmp_path):
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions(shots=321))
    path = tmp_path / "task.json"
    task.checkpoint(str(path))
    recovered = RuntimeBackendTask.recover(runtime_backend.service, str(path))
    assert isinstance(recovered, RuntimeBackendTask)
    assert recovered.kind == "sample"
    assert recovered.result().single_counts() == {"00": 160, "11": 161}
    assert runtime_backend.service.recovered_task_paths == [str(path)]


@pytest.mark.runtime_contract
def test_runtime_recovery_rejects_checkpoint_without_task_kind(runtime_backend, tmp_path):
    qtask = FakeQTaskManager([{"00": 1}], kind="sample")
    qtask.task_state["data"]["metadata"] = None
    runtime_backend.service.recovered_task = qtask
    with pytest.raises(TaskRecoveryError, match="kind"):
        RuntimeBackendTask.recover(runtime_backend.service, "unreadable.json")


@pytest.mark.runtime_contract
def test_runtime_submission_failure_raises_public_error(runtime_backend, bell_program):
    cause = RuntimeError("network unreachable")
    runtime_backend.service.submit_error = cause
    with pytest.raises(TaskSubmissionError, match="network unreachable") as excinfo:
        runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    assert excinfo.value.__cause__ is cause


@pytest.mark.runtime_contract
def test_runtime_query_failure_raises_public_error(runtime_backend, bell_program):
    cause = RuntimeError("query failed")
    runtime_backend.service.query_error = cause
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    with pytest.raises(BackendUnavailableError) as excinfo:
        task.result(timeout=5)
    assert excinfo.value.__cause__ is cause


@pytest.mark.runtime_contract
def test_runtime_query_failure_surfaces_from_status_and_try_result(
    runtime_backend, bell_program
):
    cause = RuntimeError("query failed")
    runtime_backend.service.query_error = cause
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    with pytest.raises(BackendUnavailableError):
        task.status()
    with pytest.raises(BackendUnavailableError):
        task.try_result()


@pytest.mark.runtime_contract
def test_runtime_result_timeout_raises_public_error(runtime_backend, bell_program):
    runtime_backend.service.query_error = RuntimeError(
        "The query result took more than 30 seconds to resolve"
    )
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    with pytest.raises(TaskTimeoutError, match="30 seconds") as excinfo:
        task.result(timeout=30)
    assert excinfo.value.__cause__ is not None


@pytest.mark.runtime_contract
def test_runtime_undecodable_result_raises_public_error():
    qtask = FakeQTaskManager(None, kind="sample", finished=True)
    task = RuntimeBackendTask(qtask, kind="sample", shots=100)
    with pytest.raises(ResultDecodingError):
        task.result()


@pytest.mark.runtime_contract
def test_runtime_task_supports_async_result(runtime_backend, bell_program):
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions(shots=321))

    async def collect():
        return await task.result_async(timeout=60)

    result = asyncio.run(collect())
    assert result.single_counts() == {"00": 160, "11": 161}
