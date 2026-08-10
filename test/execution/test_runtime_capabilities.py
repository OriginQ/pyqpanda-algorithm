"""Credential-free tests for runtime device capability validation.

Every test runs against :class:`FakeDevice` and
:class:`FakeRuntimeService` (see ``test/execution/fakes.py``), so the
suite passes without API credentials or network access.  Capability
failures must be raised as :class:`DeviceCapabilityError` before any
submission reaches the service.
"""

import pytest
from pyqpanda3.core import H, QProg

from pyqpanda_alg.execution import DeviceCapabilityError, ExecutionOptions


def test_preflight_rejects_circuit_larger_than_device(runtime_backend, five_qubit_prog):
    runtime_backend.device.available_qubits.return_value = [0, 1]
    with pytest.raises(DeviceCapabilityError, match="requires 5 qubits"):
        runtime_backend.submit_sample(five_qubit_prog, options=ExecutionOptions())
    assert runtime_backend.service.sample_calls == []


def test_preflight_rejects_device_without_available_qubits(runtime_backend, bell_program):
    runtime_backend.device.available_qubits.return_value = []
    with pytest.raises(DeviceCapabilityError, match="available qubits"):
        runtime_backend.submit_sample(bell_program, options=ExecutionOptions())


def test_preflight_rejects_observable_qubits_outside_device(
    runtime_backend, observable
):
    single_qubit_prog = QProg() << H(0)
    runtime_backend.device.available_qubits.return_value = [0]
    with pytest.raises(DeviceCapabilityError, match="observable"):
        runtime_backend.submit_estimate(
            (single_qubit_prog, observable), options=ExecutionOptions()
        )
    assert runtime_backend.service.estimate_calls == []


def test_preflight_rejects_unsupported_gate(runtime_backend, bell_program):
    runtime_backend.device.basic_gates.return_value = ["H", "X"]
    with pytest.raises(DeviceCapabilityError, match="CNOT"):
        runtime_backend.submit_sample(bell_program, options=ExecutionOptions())


def test_preflight_rejects_gate_pair_not_in_topology(runtime_backend, bell_program):
    runtime_backend.device.chip_topo_edges.return_value = [[1, 2], [2, 3]]
    with pytest.raises(DeviceCapabilityError, match="topolog"):
        runtime_backend.submit_sample(bell_program, options=ExecutionOptions())


def test_preflight_rejects_specified_block_outside_device(runtime_backend, bell_program):
    runtime_backend.device.available_qubits.return_value = [2, 3]
    options = ExecutionOptions(specified_block=(0, 1))
    with pytest.raises(DeviceCapabilityError, match="specified block"):
        runtime_backend.submit_sample(bell_program, options=options)


def test_preflight_accepts_capable_circuit_and_submits(runtime_backend, bell_program):
    task = runtime_backend.submit_sample(bell_program, options=ExecutionOptions())
    assert runtime_backend.service.sample_calls
    assert task.result().single_counts() == {"00": 160, "11": 161}
