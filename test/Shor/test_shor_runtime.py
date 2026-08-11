"""Runtime sampling contract tests for the Shor solver.

Covers the runtime provenance of a quantum order-finding run — the
``used_quantum`` flag and the recorded task IDs — and the
pre-submission capacity gate that rejects a device with fewer
available qubits than the order-finding circuit needs before any
sample task is submitted.  All runtime-contract tests run
credential-free against scripted recording backends and the fake
runtime service/device.
"""

import pytest

from pyqpanda_alg.Shor import Shor, ShorConfig
from pyqpanda_alg.Shor.resources import estimate_shor_resources
from pyqpanda_alg.execution import DeviceCapabilityError

from test.execution.fakes import RecordingBackend


class FixedBaseRng:
    """Test double whose ``randrange`` always returns its constructor value.

    The fixed base makes the drawn attempt deterministic: base 2 modulo
    15 has order 4, so the scripted phase samples below factor 15.
    """

    def __init__(self, value):
        self.value = value

    def randrange(self, a, b):
        return self.value


class ShorRecordingBackend(RecordingBackend):
    """RecordingBackend with scripted ``sample_results`` and a task-ID log.

    ``sample_results`` aliases the parent's per-round ``sample_counts``
    (the last dict repeats for overflow), and ``task_ids`` records every
    submitted task ID in submission order.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_ids = []

    @property
    def sample_results(self):
        """The scripted per-round count dicts (see ``sample_counts``)."""
        return self.sample_counts

    @sample_results.setter
    def sample_results(self, counts):
        self.sample_counts = [dict(c) for c in counts] if counts is not None else None

    def submit_sample(self, circuit, *, options):
        task = super().submit_sample(circuit, options=options)
        self.task_ids.append(task.id)
        return task


@pytest.fixture
def recording_backend():
    """Shor recording backend: scripted samples and recorded task IDs."""
    return ShorRecordingBackend()


@pytest.mark.runtime_contract
def test_runtime_shor_records_quantum_task_ids(recording_backend):
    recording_backend.sample_results = [{"01000000": 600, "11000000": 400}]
    result = Shor(15, rng=FixedBaseRng(2), config=ShorConfig(max_attempts=1)).run(
        backend=recording_backend
    )
    assert result.used_quantum is True
    assert result.task_ids == tuple(recording_backend.task_ids)


@pytest.mark.runtime_contract
def test_runtime_shor_rejects_device_with_too_few_qubits_before_submission(
    runtime_backend,
):
    estimate = estimate_shor_resources(15)
    runtime_backend.device.available_qubits.return_value = list(
        range(estimate.total_qubits - 1)
    )
    with pytest.raises(DeviceCapabilityError):
        Shor(15, rng=FixedBaseRng(2), config=ShorConfig(max_attempts=1)).run(
            backend=runtime_backend
        )
    assert runtime_backend.service.sample_calls == []
