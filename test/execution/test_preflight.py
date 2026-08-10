"""Credential-free tests for the runtime preflight modes.

``NONE`` submits without any preflight work, ``TRANSPILE_ONLY``
validates the submission and runs the fake-backend transpile, and
``FAKE_EXECUTE`` additionally runs the fake task and records its
task/result metadata before the real submission.  All preflight steps
run before submission: a failing step must leave the service untouched.
"""

import pytest

from pyqpanda_alg.execution import (
    AlgorithmInputError,
    BackendUnavailableError,
    DeviceCapabilityError,
    ExecutionOptions,
    PreflightMode,
    TranspilationError,
)


def test_none_mode_skips_capability_checks(runtime_backend, five_qubit_prog):
    runtime_backend.device.available_qubits.return_value = [0, 1]
    task = runtime_backend.submit_sample(
        five_qubit_prog, options=ExecutionOptions(preflight=PreflightMode.NONE)
    )
    assert runtime_backend.service.sample_calls
    assert task.result().single_counts() == {"00": 160, "11": 161}


def test_none_mode_does_not_touch_fake_backend(runtime_backend, bell_program):
    runtime_backend.submit_sample(
        bell_program, options=ExecutionOptions(preflight=PreflightMode.NONE)
    )
    runtime_backend.device.fake_backend.assert_not_called()


def test_transpile_only_runs_fake_transpile_before_submission(
    runtime_backend, bell_program
):
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    fake = runtime_backend.device.fake_backend.return_value
    call = fake.transpile_calls[0]
    assert call["progs"] == [bell_program]
    assert call["specified_block"] is None
    assert call["is_optimization"] is True
    assert runtime_backend.service.sample_calls
    assert not hasattr(task, "fake_execution")


def test_transpile_only_forwards_specified_block_to_fake_backend(
    runtime_backend, bell_program
):
    options = ExecutionOptions(specified_block=(2, 3), is_optimization=False)
    runtime_backend.submit_sample(bell_program, options=options)
    call = runtime_backend.device.fake_backend.return_value.transpile_calls[0]
    assert call["specified_block"] == [2, 3]
    assert call["is_optimization"] is False


def test_transpile_only_surfaces_transpile_failure(runtime_backend, bell_program):
    fake = runtime_backend.device.fake_backend.return_value
    fake.transpile_error = RuntimeError("failed to trainspile on the fake backend")
    with pytest.raises(TranspilationError, match="trainspile"):
        runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    assert runtime_backend.service.sample_calls == []


def test_transpile_only_rejects_failed_transpiled_circuits(
    runtime_backend, bell_program
):
    fake = runtime_backend.device.fake_backend.return_value
    fake.transpile_result = (["ok"], ["bad-originir"])
    with pytest.raises(TranspilationError):
        runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    assert runtime_backend.service.sample_calls == []


def test_fake_execute_runs_fake_task_before_real_submission(
    runtime_backend, bell_program
):
    task = runtime_backend.submit_sample(
        bell_program, options=ExecutionOptions(preflight=PreflightMode.FAKE_EXECUTE)
    )
    fake = runtime_backend.device.fake_backend.return_value
    assert fake.sample_calls[0]["shots"] == 1000
    # the fake runs the *transpiled* circuit, so the recorded fidelity
    # describes what the device will actually run
    assert fake.sample_calls[0]["prog"] == "TRANSPILED-ORIGINIR"
    assert fake.transpile_calls  # transpile runs before the fake task
    assert runtime_backend.service.sample_calls  # real submission still happens
    assert task.fake_execution["kind"] == "sample"
    assert task.fake_execution["result"] == {"00": 0.5, "11": 0.5}
    assert task.result().single_counts() == {"00": 160, "11": 161}


def test_fake_execute_estimate_records_fake_value(
    runtime_backend, bell_program, observable
):
    task = runtime_backend.submit_estimate(
        (bell_program, observable),
        options=ExecutionOptions(preflight=PreflightMode.FAKE_EXECUTE),
    )
    fake = runtime_backend.device.fake_backend.return_value
    assert fake.estimate_calls[0]["shots"] == 1000
    assert fake.estimate_calls[0]["prog"] == "TRANSPILED-ORIGINIR"
    assert task.fake_execution["kind"] == "estimate"
    assert task.fake_execution["result"] == 0.5
    assert task.result().single_value() == 0.5


def test_fake_execute_failure_prevents_real_submission(runtime_backend, bell_program):
    fake = runtime_backend.device.fake_backend.return_value
    fake.sample_error = RuntimeError("simulation crashed")
    with pytest.raises(BackendUnavailableError, match="fake_execute"):
        runtime_backend.submit_sample(
            bell_program, options=ExecutionOptions(preflight=PreflightMode.FAKE_EXECUTE)
        )
    assert runtime_backend.service.sample_calls == []


def test_fake_execute_surfaces_multiprocessing_entry_guidance(
    runtime_backend, bell_program
):
    fake = runtime_backend.device.fake_backend.return_value
    fake.sample_error = RuntimeError(
        "An attempt has been made to start a new process before the current "
        "process has finished its bootstrapping phase. This probably means "
        "that you are not using fork to start your child processes and you "
        "have forgotten to use the proper idiom in the main module:\n"
        "    if __name__ == '__main__':\n"
        "        freeze_support()\n"
        "        ..."
    )
    with pytest.raises(BackendUnavailableError, match="fake_execute") as excinfo:
        runtime_backend.submit_sample(
            bell_program, options=ExecutionOptions(preflight=PreflightMode.FAKE_EXECUTE)
        )
    assert "if __name__ == '__main__':" in str(excinfo.value)
    assert runtime_backend.service.sample_calls == []


def test_fake_execute_still_validates_before_fake_run(runtime_backend, five_qubit_prog):
    runtime_backend.device.available_qubits.return_value = [0, 1]
    with pytest.raises(DeviceCapabilityError, match=r"requires qubit"):
        runtime_backend.submit_sample(
            five_qubit_prog,
            options=ExecutionOptions(preflight=PreflightMode.FAKE_EXECUTE),
        )
    runtime_backend.device.fake_backend.assert_not_called()


def test_qubit_capacity_check_rejects_dead_qubits_by_membership(
    runtime_backend, bell_program
):
    # A device whose highest qubit index is 5 but that only exposes
    # [0, 4, 5] must reject a circuit using qubit 1 -- a span-versus-
    # count comparison (max+1 <= len) would wrongly accept it.
    runtime_backend.device.available_qubits.return_value = [0, 4, 5]
    prog = bell_program  # uses qubits 0 and 1
    with pytest.raises(DeviceCapabilityError, match=r"qubit\(s\) \[1\]"):
        runtime_backend.submit_sample(prog, options=ExecutionOptions())
    assert runtime_backend.service.sample_calls == []


def test_preflight_rejects_string_observable_with_input_error(
    runtime_backend, bell_program
):
    with pytest.raises(AlgorithmInputError, match="observable"):
        runtime_backend.submit_estimate(
            (bell_program, "Z0 Z1"), options=ExecutionOptions()
        )
    assert runtime_backend.service.estimate_calls == []


def test_preflight_surfaces_device_fetch_failure(runtime_backend, bell_program):
    runtime_backend.device.available_qubits.side_effect = RuntimeError(
        "device unreachable"
    )
    with pytest.raises(BackendUnavailableError, match="available_qubits") as excinfo:
        runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert runtime_backend.service.sample_calls == []
