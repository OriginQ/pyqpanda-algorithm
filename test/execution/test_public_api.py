"""Public API surface tests: the approved execution-layer exports.

Task 8 publishes the execution package surface for all later algorithm
plans.  These tests pin that surface: the plan's import contract must
resolve, and ``__all__`` must contain exactly the approved names from
Tasks 1-7 (value objects, capabilities, tasks, backends, algorithm
state machines, variational sessions, and the exception hierarchy).
"""

import pyqpanda_alg.execution as execution

from pyqpanda_alg.execution import (
    AlgorithmTask,
    BackendCapabilities,
    ExecutionOptions,
    LocalBackend,
    QPandaRuntimeBackend,
    StatevectorBatchResult,
    resolve_backend,
)


def test_execution_public_types_are_importable():
    assert all((AlgorithmTask, BackendCapabilities, ExecutionOptions, LocalBackend, QPandaRuntimeBackend, StatevectorBatchResult, resolve_backend))


def test_approved_surface_is_exactly_the_exported_names():
    approved = {
        # value objects and capabilities
        "BackendCapabilities",
        "ExecutionOptions",
        "PreflightMode",
        "SampleBatchResult",
        "EstimateBatchResult",
        "StatevectorBatchResult",
        # tasks, protocols, and status
        "BackendTask",
        "CompletedBackendTask",
        "ExecutionBackend",
        "RuntimeBackendTask",
        "TaskStatus",
        # backends
        "LocalBackend",
        "QPandaRuntimeBackend",
        "resolve_backend",
        # resumable algorithm state machines
        "AlgorithmStep",
        "AlgorithmTask",
        "register_algorithm",
        # variational sessions
        "VariationalSession",
        # exception hierarchy
        "AlgorithmExecutionError",
        "AlgorithmInputError",
        "BackendUnavailableError",
        "DeviceCapabilityError",
        "MissingRuntimeDependencyError",
        "ResultDecodingError",
        "TaskRecoveryError",
        "TaskSubmissionError",
        "TaskTimeoutError",
        "TranspilationError",
    }
    assert set(execution.__all__) == approved


def test_every_exported_name_resolves_to_a_real_object():
    for name in execution.__all__:
        assert getattr(execution, name) is not None
