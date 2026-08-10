"""Capability-based execution layer.

Backends always submit work and return task objects; this package
provides the shared value objects (options, capabilities, result
wrappers) and the public exception hierarchy used by every backend.
"""

from .algorithm_task import AlgorithmStep, AlgorithmTask, register_algorithm
from .backend import ExecutionBackend
from .backend_task import BackendTask, CompletedBackendTask, TaskStatus
from .capabilities import BackendCapabilities
from .errors import (
    AlgorithmExecutionError,
    AlgorithmInputError,
    BackendUnavailableError,
    DeviceCapabilityError,
    MissingRuntimeDependencyError,
    ResultDecodingError,
    TaskRecoveryError,
    TaskSubmissionError,
    TaskTimeoutError,
    TranspilationError,
)
from .local import LocalBackend, resolve_backend
from .options import ExecutionOptions, PreflightMode
from .results import EstimateBatchResult, SampleBatchResult, StatevectorBatchResult
from .runtime import QPandaRuntimeBackend
from .runtime_task import RuntimeBackendTask
from .variational import VariationalSession

__all__ = [
    "AlgorithmExecutionError",
    "AlgorithmInputError",
    "AlgorithmStep",
    "AlgorithmTask",
    "BackendCapabilities",
    "BackendTask",
    "BackendUnavailableError",
    "CompletedBackendTask",
    "DeviceCapabilityError",
    "EstimateBatchResult",
    "ExecutionBackend",
    "ExecutionOptions",
    "LocalBackend",
    "MissingRuntimeDependencyError",
    "PreflightMode",
    "QPandaRuntimeBackend",
    "ResultDecodingError",
    "RuntimeBackendTask",
    "SampleBatchResult",
    "StatevectorBatchResult",
    "TaskRecoveryError",
    "TaskStatus",
    "TaskSubmissionError",
    "TaskTimeoutError",
    "TranspilationError",
    "VariationalSession",
    "register_algorithm",
    "resolve_backend",
]
