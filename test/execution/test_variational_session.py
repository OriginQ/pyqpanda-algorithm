"""Variational session tests: context-manager release and float tasks.

Both backends expose the same :class:`VariationalSession` surface:
``run(parameters)`` returns a task whose result is the expectation value
(a float), and the session releases its underlying resource exactly once
on context exit -- including when the body raises.  These tests run
against the repository fakes, so no network credentials are needed.
"""

import json
import math

import pytest

from pyqpanda_alg.execution import (
    AlgorithmInputError,
    BackendTask,
    ExecutionOptions,
    LocalBackend,
    QPandaRuntimeBackend,
    TaskStatus,
    VariationalSession,
)


def test_runtime_variational_session_releases_on_error(runtime_backend, ansatz, observable):
    session = runtime_backend.create_variational_session(ansatz, observable, options=ExecutionOptions())
    with pytest.raises(RuntimeError):
        with session:
            session.raw_session.raise_on_run = RuntimeError("failure")
            session.run([0.1, 0.2])
    assert session.raw_session.release_calls == 1


def test_runtime_session_creation_binds_ansatz_device_and_observable(
    runtime_backend, ansatz, observable
):
    session = runtime_backend.create_variational_session(
        ansatz, observable, options=ExecutionOptions(shots=321)
    )
    call = runtime_backend.service.vqsession_calls[0]
    assert call["vqcircuit"] is ansatz
    assert call["device"] is runtime_backend.device
    assert call["shots"] == 321
    assert call["life_time"] == 360
    assert call["observable"] is observable
    assert isinstance(session, VariationalSession)


def test_runtime_session_run_returns_float_expectation_task(
    runtime_backend, ansatz, observable
):
    session = runtime_backend.create_variational_session(
        ansatz, observable, options=ExecutionOptions()
    )
    with session:
        task = session.run([0.1, 0.2])
    assert isinstance(task, BackendTask)
    assert isinstance(task.result(), float)
    assert task.result() == 0.5
    assert session.raw_session.run_calls[0]["gate_params"] == [0.1, 0.2]


def test_runtime_session_forwards_ansatz_qubits_as_measure_list(
    runtime_backend, ansatz, observable
):
    session = runtime_backend.create_variational_session(
        ansatz, observable, options=ExecutionOptions()
    )
    with session:
        session.run([0.1, 0.2])
    # the ansatz acts on qubits 0 and 1, and the session must forward
    # exactly those as the measure_list of every run
    assert session.raw_session.run_calls[0]["measure_list"] == [0, 1]


def test_runtime_session_release_is_idempotent(runtime_backend, ansatz, observable):
    session = runtime_backend.create_variational_session(
        ansatz, observable, options=ExecutionOptions()
    )
    with session:
        session.run([0.1, 0.2])
    session.release()
    assert session.raw_session.release_calls == 1


def test_runtime_session_reentry_after_release_raises(runtime_backend, ansatz, observable):
    session = runtime_backend.create_variational_session(
        ansatz, observable, options=ExecutionOptions()
    )
    with session:
        session.run([0.1, 0.2])
    assert session.raw_session.release_calls == 1
    with pytest.raises(AlgorithmInputError, match="re-enter"):
        with session:
            session.run([0.3, 0.4])
    # re-entry released nothing and ran nothing: no second network call
    assert session.raw_session.release_calls == 1
    assert len(session.raw_session.run_calls) == 1


def test_local_session_reentry_after_release_raises(ansatz, observable):
    session = LocalBackend().create_variational_session(
        ansatz, observable, options=ExecutionOptions()
    )
    with session:
        session.run([0.1, 0.2])
    with pytest.raises(AlgorithmInputError, match="re-enter"):
        with session:
            session.run([0.3, 0.4])
    assert session.history == [[0.1, 0.2]]  # the second run never happened


def test_local_session_run_returns_expectation_float(ansatz, observable):
    session = LocalBackend().create_variational_session(
        ansatz, observable, options=ExecutionOptions(shots=1000)
    )
    with session:
        task = session.run([0.1, 0.2])
    value = task.result()
    assert isinstance(value, float)
    assert abs(value - math.cos(0.1) * math.cos(0.2)) < 1e-9


def test_local_session_context_manager_and_idempotent_release(ansatz, observable):
    session = LocalBackend().create_variational_session(
        ansatz, observable, options=ExecutionOptions()
    )
    with session:
        task = session.run([0.1, 0.2])
    assert isinstance(session, VariationalSession)
    assert task.status() is TaskStatus.SUCCEEDED
    session.release()
    assert session.history == [[0.1, 0.2]]


def test_backends_advertise_variational_sessions(fake_runtime_service, fake_device):
    pytest.importorskip("qpanda3_runtime")
    assert LocalBackend().capabilities.variational_session is True
    assert (
        QPandaRuntimeBackend(
            fake_runtime_service, fake_device
        ).capabilities.variational_session
        is True
    )


def test_session_checkpoint_state_holds_parameters_only(runtime_backend, ansatz, observable):
    session = runtime_backend.create_variational_session(
        ansatz, observable, options=ExecutionOptions()
    )
    with session:
        session.run([0.1, 0.2])
        session.run([0.3, 0.4])
    assert json.dumps(session.history) == "[[0.1, 0.2], [0.3, 0.4]]"
