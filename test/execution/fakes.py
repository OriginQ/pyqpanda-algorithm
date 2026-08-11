"""Repository-wide fakes for credential-free execution tests.

These stand-ins let execution-layer tests exercise real submission and
result flows without network access or API credentials.  They are
consumed by the shared fixtures in ``test/conftest.py`` and by later
algorithm plans, so their behavior is deterministic and their records
are inspectable.

- :class:`RecordingBackend` records every submission and returns fixed
  batch results.
- :class:`FakeRuntimeService` records ``sample``/``estimate``/``vqsession``
  calls and returns :class:`FakeQTaskManager` handles.
- :class:`FakeQTaskManager` simulates the qpanda3-runtime task surface
  with controllable results, completion, and failure.
- :class:`FakeVQSession` simulates the qpanda3-runtime ``VQSession``
  surface with a controllable run failure and a release counter.
- :class:`FakeDevice` stands in for a ``QDevice`` with mock-configurable
  capability data.
- :class:`FakeFakeBackend` stands in for the ``FakeBackend`` a real
  device exposes through ``fake_backend()``, recording the transpile,
  sample, and estimate calls of the preflight modes.
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
    """Backend that records every submission and returns fixed results.

    ``expectations`` and ``statevector`` optionally pin canned results
    that are replayed in submission order (the last value repeats for
    overflow), so optimization loops can be driven deterministically.
    Per-family call counts expose the submission mix an algorithm
    actually produces.
    """

    capabilities = BackendCapabilities()

    def __init__(self, expectations=None, statevector=None) -> None:
        self.sample_calls: list = []
        self.estimate_calls: list = []
        self.statevector_calls: list = []
        self.sample_call_count = 0
        self.estimate_call_count = 0
        self.statevector_call_count = 0
        self.expectations = list(expectations) if expectations is not None else None
        self.statevector_values = (
            [list(sv) for sv in statevector] if statevector is not None else None
        )
        self.sample_result = SampleBatchResult(counts=({"00": 1000},), shots=1000)
        self.estimate_result = EstimateBatchResult(values=(0.5,))
        self.statevector_result = StatevectorBatchResult(statevectors=([1.0, 0.0],))

    def submit_sample(self, circuit, *, options):
        """Record the sampling call and return the fixed sample task."""
        self.sample_call_count += 1
        self.sample_calls.append((circuit, options))
        return CompletedBackendTask(
            self.sample_result, task_id=f"recorded-sample-{self.sample_call_count}"
        )

    def submit_estimate(self, circuit_and_observable, *, options):
        """Record the estimation call and return the fixed estimate task."""
        self.estimate_call_count += 1
        self.estimate_calls.append((circuit_and_observable, options))
        if self.expectations is None:
            result = self.estimate_result
        else:
            index = min(self.estimate_call_count - 1, len(self.expectations) - 1)
            result = EstimateBatchResult(values=(self.expectations[index],))
        return CompletedBackendTask(
            result, task_id=f"recorded-estimate-{self.estimate_call_count}"
        )

    def submit_statevector(self, circuit, *, options):
        """Record the state-vector call and return the fixed state task."""
        self.statevector_call_count += 1
        self.statevector_calls.append((circuit, options))
        if self.statevector_values is None:
            result = self.statevector_result
        else:
            index = min(self.statevector_call_count - 1, len(self.statevector_values) - 1)
            result = StatevectorBatchResult(statevectors=(self.statevector_values[index],))
        return CompletedBackendTask(
            result, task_id=f"recorded-statevector-{self.statevector_call_count}"
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
        self.fake_backend = MagicMock(return_value=FakeFakeBackend())


class FakeFakeBackend:
    """Deterministic stand-in for the ``FakeBackend`` of a real device.

    Mirrors the qpanda3-runtime surface the preflight modes consume:
    ``transpile`` returns ``(transpiled, failed)``, ``sample`` returns a
    probability dict, and ``estimate`` returns a float.  Every call is
    recorded, and ``*_error`` makes the matching call raise.
    """

    def __init__(self) -> None:
        self.transpile_calls: list = []
        self.sample_calls: list = []
        self.estimate_calls: list = []
        self.transpile_error = None
        self.sample_error = None
        self.estimate_error = None
        self.transpile_result = (["TRANSPILED-ORIGINIR"], [])
        self.sample_result = {"00": 0.5, "11": 0.5}
        self.estimate_result = 0.5

    def transpile(self, progs, specified_block=None, is_optimization=True):
        """Record the transpile call and return the configured outcome."""
        self.transpile_calls.append(
            {
                "progs": progs,
                "specified_block": specified_block,
                "is_optimization": is_optimization,
            }
        )
        if self.transpile_error is not None:
            raise self.transpile_error
        return self.transpile_result

    def sample(self, prog, shots=1):
        """Record the sampling call and return the configured probabilities."""
        self.sample_calls.append({"prog": prog, "shots": shots})
        if self.sample_error is not None:
            raise self.sample_error
        return self.sample_result

    def estimate(self, prog, observable, shots=1):
        """Record the estimation call and return the configured value."""
        self.estimate_calls.append(
            {"prog": prog, "observable": observable, "shots": shots}
        )
        if self.estimate_error is not None:
            raise self.estimate_error
        return self.estimate_result


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
        self.result_calls: list = []
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
        self.result_calls.append({"timeout": timeout})
        if self.error is not None:
            raise self.error
        return self.results

    async def get_result_async(self, additional_info=None, timeout=1800):
        """Coroutine result fetch; delegates to the sync path."""
        return self.get_result_sync(additional_info, timeout)

    def check_point(self, filepath=None, user_data=None):
        """Record the call and persist the serializable task state.

        The file layout mirrors the real qpanda3-runtime checkpoint:
        ``{"meta": ..., "task_state": ..., "user_data": ...}``.
        """
        self.checkpoint_calls.append({"filepath": filepath, "user_data": user_data})
        if filepath is not None:
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "meta": {"version": 1, "created_at": ""},
                        "task_state": self.task_state,
                        "user_data": user_data if user_data is not None else {},
                    },
                    handle,
                )
        return str(filepath) if filepath is not None else "checkpoint_FAKE.json"

    def get_task_state(self):
        """Return the serializable state used for checkpoint recovery."""
        return dict(self.task_state)


class FakeVQSession:
    """Deterministic stand-in for a qpanda3-runtime ``VQSession``.

    Mirrors the session surface the adapter consumes: ``__enter__``,
    ``run_vqtask(gate_params, measure_list)``, and a release counter.
    ``raise_on_run`` makes the next run raise (simulating a session-side
    failure); the returned task carries ``results`` and the
    ``finished``/``error`` flags like the other fakes.
    """

    def __init__(self, results=None, *, finished: bool = True, error=None) -> None:
        self.release_calls = 0
        self.raise_on_run = None
        self.run_calls: list = []
        self.results = results if results is not None else [0.5]
        self.finished = finished
        self.error = error

    def __enter__(self):
        """Activate the session, like the real VQSession context."""
        return self

    def run_vqtask(self, gate_params, measure_list=None):
        """Record the run; optionally raise, else return a fake task."""
        self.run_calls.append(
            {"gate_params": list(gate_params), "measure_list": measure_list}
        )
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return FakeQTaskManager(
            self.results, kind="estimate", finished=self.finished, error=self.error
        )

    def release(self):
        """Record the release request (idempotent in the real session)."""
        self.release_calls += 1


class FakeRuntimeService:
    """Deterministic stand-in for a qpanda3-runtime ``RuntimeService``.

    Records every ``sample``/``estimate``/``vqsession`` call with the
    exact keyword arguments the adapter forwards, so tests can assert
    option mapping.  ``submit_error`` makes submission raise and
    ``query_error`` makes the returned task's result access raise, both
    simulating transport failures without any network.
    """

    def __init__(self) -> None:
        self.sample_calls: list = []
        self.estimate_calls: list = []
        self.vqsession_calls: list = []
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

    def vqsession(self, vqcircuit, device, shots=1000, life_time=360, observable=None):
        """Record the session request and return a fake VQ session."""
        if self.submit_error is not None:
            raise self.submit_error
        self.vqsession_calls.append(
            {
                "vqcircuit": vqcircuit,
                "device": device,
                "shots": shots,
                "life_time": life_time,
                "observable": observable,
            }
        )
        return FakeVQSession(
            results=self.estimate_results, finished=self.finished, error=self.query_error
        )

    def recover_qtask_manager(self, checkpoint_file, recover_completely=True):
        """Rebuild a fake task from a checkpoint path."""
        self.recovered_task_paths.append(checkpoint_file)
        if self.recovered_task is not None:
            return self.recovered_task
        return FakeQTaskManager(self.sample_results, kind="sample", finished=True)
