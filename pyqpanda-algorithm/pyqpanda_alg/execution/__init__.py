"""Capability-based execution layer.

Backends always submit work and return task objects; this package
provides the shared value objects (options, capabilities, result
wrappers) and the public exception hierarchy used by every backend.
"""

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
from .options import ExecutionOptions, PreflightMode
from .results import EstimateBatchResult, SampleBatchResult, StatevectorBatchResult

__all__ = [
    "AlgorithmExecutionError",
    "AlgorithmInputError",
    "BackendCapabilities",
    "BackendTask",
    "BackendUnavailableError",
    "CompletedBackendTask",
    "DeviceCapabilityError",
    "EstimateBatchResult",
    "ExecutionBackend",
    "ExecutionOptions",
    "MissingRuntimeDependencyError",
    "PreflightMode",
    "ResultDecodingError",
    "SampleBatchResult",
    "StatevectorBatchResult",
    "TaskRecoveryError",
    "TaskStatus",
    "TaskSubmissionError",
    "TaskTimeoutError",
    "TranspilationError",
]
