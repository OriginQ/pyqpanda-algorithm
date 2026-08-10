"""Capability-based execution layer.

Backends always submit work and return task objects; callers poll
:class:`BackendTask` handles and collect results through the batch
result wrappers.  This package publishes the shared value objects
(options, capabilities, result wrappers), the task protocols, the
resumable :class:`AlgorithmTask` state machine, the variational
session abstraction, and the public exception hierarchy used by every
backend.

The approved public surface is exactly ``__all__``; algorithm plans
must import from ``pyqpanda_alg.execution`` and nothing deeper.

Rules that bind every backend:

* ``resolve_backend(None)`` returns the CPU default
  :class:`LocalBackend`; existing CPU calls are unchanged.
* :class:`QPandaRuntimeBackend` is optional (install with
  ``pip install pyqpanda-algorithm[runtime]``) and must be constructed
  with an already logged-in service and an explicit device; it never
  authenticates or selects a device on its own.
* A runtime failure never silently falls back to the local backend or
  to a classical substitute result; it surfaces as a public
  execution-layer exception that retains the original failure as
  ``__cause__``.
* Checkpoints never contain credentials, live services, or device
  handles; :meth:`AlgorithmTask.resume` rebuilds the step callback
  only through a factory registered with :func:`register_algorithm`.
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
