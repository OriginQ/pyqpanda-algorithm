"""Repository-wide fakes for credential-free execution tests.

These stand-ins let execution-layer tests exercise real submission and
result flows without network access or API credentials.  They are
consumed by the shared fixtures in ``test/conftest.py`` and by later
algorithm plans, so their behavior is deterministic and their records
are inspectable.

- :class:`RecordingBackend` records every submission and returns fixed
  batch results.
- :class:`FakeRuntimeService` records ``sample``/``estimate`` calls and
  returns :class:`FakeQTaskManager` handles.
- :class:`FakeQTaskManager` simulates the qpanda3-runtime task surface
  with controllable results, completion, and failure.
- :class:`FakeDevice` stands in for a ``QDevice`` with mock-configurable
  capability data.
"""

import json
from unittest.mock import MagicMock

from pyqpanda_alg.execution import (
    BackendCapabilities,
    CompletedBackendTask,
    DeviceCapabilityError,
    EstimateBatchResult,
    SampleBatchResult,
    StatevectorBatchResult,
)


class RecordingBackend:
    """Backend that records every submission and returns fixed results."""

    capabilities = BackendCapabilities()

    def __init__(self) -> None:
        self.sample_calls: list = []
        self.estimate_calls: list = []
        self.statevector_calls: list = []
        self.sample_result = SampleBatchResult(counts=({"00": 1000},), shots=1000)
        self.estimate_result = EstimateBatchResult(values=(0.5,))
        self.statevector_result = StatevectorBatchResult(statevectors=([1.0, 0.0],))

    def submit_sample(self, circuit, *, options):
        """Record the sampling call and return the fixed sample task."""
        self.sample_calls.append((circuit, options))
        return CompletedBackendTask(
            self.sample_result, task_id=f"recorded-sample-{len(self.sample_calls)}"
        )

    def submit_estimate(self, circuit_and_observable, *, options):
        """Record the estimation call and return the fixed estimate task."""
        self.estimate_calls.append((circuit_and_observable, options))
        return CompletedBackendTask(
            self.estimate_result, task_id=f"recorded-estimate-{len(self.estimate_calls)}"
        )

    def submit_statevector(self, circuit, *, options):
        """Record the state-vector call and return the fixed state task."""
        self.statevector_calls.append((circuit, options))
        return CompletedBackendTask(
            self.statevector_result,
            task_id=f"recorded-statevector-{len(self.statevector_calls)}",
        )

    def create_variational_session(self, ansatz, observable, *, options):
        """Reject variational sessions: this backend does not provide them."""
        raise DeviceCapabilityError(
            "RecordingBackend does not support variational sessions"
        )


class FakeDevice:
    """Deterministic stand-in for a qpanda3-runtime ``QDevice``.

    Capability accessors are ``MagicMock`` instances so tests can pin
    exact device data with ``device.available_qubits.return_value = [...]``.
    """

    def __init__(self, chip_id: str = "fake-chip", channel: str = "qcloud") -> None:
        self.chip_id = MagicMock(return_value=chip_id)
        self.channel = MagicMock(return_value=channel)
        self.available_qubits = MagicMock(return_value=[0, 1, 2, 3, 4, 5])
        self.basic_gates = MagicMock(
            return_value=["H", "X", "Y", "Z", "RX", "RY", "RZ", "CNOT", "CZ", "SWAP"]
        )
        self.chip_topo_edges = MagicMock(
            return_value=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
        )


class FakeQTaskManager:
    """Deterministic stand-in for a qpanda3-runtime ``QTaskManager``.

    ``results`` holds the raw runtime-shaped result (a list of counts
    dicts for sampling, a list of floats for estimation) that
    ``try_get_result()`` and ``get_result_sync()`` return.  ``finished``
    is a mutable public flag, and ``error`` makes every result access
    raise, simulating a transport failure.
    """

    def __init__(
        self,
        results,
        *,
        kind: str = "sample",
        finished: bool = True,
        task_id=None,
        channel: str = "qcloud",
        error=None,
    ) -> None:
        self.results = results
        self.kind = kind
        self.finished = finished
        self.task_id = task_id if task_id is not None else "fake-task-id"
        self.channel = channel
        self.error = error
        self.checkpoint_calls: list = []
        state_task_id = (
            self.task_id
            if isinstance(self.task_id, str)
            else ",".join(str(part) for part in self.task_id)
        )
        self.task_state = {
            "qtask_manager_init_time": 0.0,
            "session_id": "",
            "data": {
                "task_id": [state_task_id],
                "channel": self.channel,
                "metadata": {"qtask_type": self.kind, "shots": 1000},
            },
            "used_remote_backend": True,
            "used_program_set": self.kind == "estimate",
            "sub_task_id__result_id": {state_task_id: 0},
            "result": [] if self.results is None else self.results,
            "is_transpile_successed": True,
            "processing_sub_task_ids": [] if self.finished else [self.task_id],
            "used_fake_backend": False,
        }

    def id(self):
        """Return the subtask identifier(s), like the real manager."""
        return self.task_id

    def channel(self):
        """Return the backend channel where the task resides."""
        return self.channel

    def used_remote_backend(self):
        """Remote tasks always use the remote backend in this fake."""
        return True

    def try_get_result(self, additional_info=None):
        """Non-blocking poll: ``(results, finished, progress)``."""
        if self.error is not None:
            raise self.error
        progress = "finished 100%" if self.finished else "finished 0%"
        return self.results, self.finished, progress

    def get_result_sync(self, additional_info=None, timeout=1000 * 1800, using_http=True):
        """Blocking result fetch; raises ``error`` when configured."""
        if self.error is not None:
            raise self.error
        return self.results

    async def get_result_async(self, additional_info=None, timeout=1800):
        """Coroutine result fetch; delegates to the sync path."""
        return self.get_result_sync(additional_info, timeout)

    def check_point(self, filepath=None, user_data=None):
        """Record the call and persist the serializable task state."""
        self.checkpoint_calls.append({"filepath": filepath, "user_data": user_data})
        if filepath is not None:
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(self.task_state, handle)
        return str(filepath) if filepath is not None else "checkpoint_FAKE.json"

    def get_task_state(self):
        """Return the serializable state used for checkpoint recovery."""
        return dict(self.task_state)


class FakeRuntimeService:
    """Deterministic stand-in for a qpanda3-runtime ``RuntimeService``.

    Records every ``sample``/``estimate`` call with the exact keyword
    arguments the adapter forwards, so tests can assert option mapping.
    ``submit_error`` makes submission raise and ``query_error`` makes
    the returned task's result access raise, both simulating transport
    failures without any network.
    """

    def __init__(self) -> None:
        self.sample_calls: list = []
        self.estimate_calls: list = []
        self.recovered_task_paths: list = []
        self.submit_error = None
        self.query_error = None
        self.sample_results = [{"00": 160, "11": 161}]
        self.estimate_results = [0.5]
        self.finished = True
        self.recovered_task = None

    def sample(self, circuits, device, **kwargs):
        """Record the sampling submission and return a fake task."""
        if self.submit_error is not None:
            raise self.submit_error
        self.sample_calls.append({"circuits": circuits, "device": device, **kwargs})
        return FakeQTaskManager(
            self.sample_results,
            kind="sample",
            finished=self.finished,
            error=self.query_error,
        )

    def estimate(self, circuit_with_observable, device, **kwargs):
        """Record the estimation submission and return a fake task."""
        if self.submit_error is not None:
            raise self.submit_error
        self.estimate_calls.append(
            {"circuit_with_observable": circuit_with_observable, "device": device, **kwargs}
        )
        return FakeQTaskManager(
            self.estimate_results,
            kind="estimate",
            finished=self.finished,
            error=self.query_error,
        )

    def recover_qtask_manager(self, checkpoint_file, recover_completely=True):
        """Rebuild a fake task from a checkpoint path."""
        self.recovered_task_paths.append(checkpoint_file)
        if self.recovered_task is not None:
            return self.recovered_task
        return FakeQTaskManager(self.sample_results, kind="sample", finished=True)
