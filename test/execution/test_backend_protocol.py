"""Contract tests for the backend and backend-task protocols.

Backends always submit work and return :class:`BackendTask` objects;
these tests pin the shared task surface (``id``, ``status()``,
``try_result()``, ``result(timeout=...)``, ``checkpoint(path=...)``)
and the :class:`ExecutionBackend` submission surface with keyword-only
``ExecutionOptions``.
"""

import inspect

from pyqpanda_alg.execution import (
    BackendTask,
    BackendCapabilities,
    CompletedBackendTask,
    ExecutionBackend,
    TaskStatus,
)


def test_completed_task_has_stable_result():
    task = CompletedBackendTask({"0": 10}, task_id="local-1")
    assert task.status() is TaskStatus.SUCCEEDED
    assert task.result() == {"0": 10}
    assert task.id == "local-1"


def test_task_status_has_the_five_plan_states():
    assert {state.name for state in TaskStatus} == {
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "TIMED_OUT",
    }


def test_completed_task_implements_backend_task_contract():
    task = CompletedBackendTask([1, 2], task_id="local-2")
    assert isinstance(task, BackendTask)
    assert task.status() is TaskStatus.SUCCEEDED
    assert task.try_result() == [1, 2]
    assert task.result(timeout=0.001) == [1, 2]
    assert task.checkpoint() is None


def test_execution_backend_options_are_keyword_only():
    for name in (
        "submit_sample",
        "submit_estimate",
        "submit_statevector",
        "create_variational_session",
    ):
        signature = inspect.signature(getattr(ExecutionBackend, name))
        options_param = signature.parameters["options"]
        assert options_param.kind is inspect.Parameter.KEYWORD_ONLY


class _StubBackend:
    """Minimal structural stand-in for a concrete backend."""

    capabilities = BackendCapabilities(statevector=False)

    def submit_sample(self, circuit, *, options):
        return CompletedBackendTask(circuit, task_id="stub")

    def submit_estimate(self, circuit_and_observable, *, options):
        return CompletedBackendTask(circuit_and_observable, task_id="stub")

    def submit_statevector(self, circuit, *, options):
        return CompletedBackendTask(circuit, task_id="stub")

    def create_variational_session(self, ansatz, observable, *, options):
        return None


def test_stub_backend_conforms_to_execution_backend():
    backend = _StubBackend()
    assert isinstance(backend, ExecutionBackend)
    assert backend.capabilities.statevector is False
