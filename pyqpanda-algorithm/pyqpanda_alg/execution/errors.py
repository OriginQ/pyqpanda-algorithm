"""Public exception hierarchy for the execution layer.

Every error raised by the execution layer derives from
:class:`AlgorithmExecutionError`, so algorithms can catch one base
type while callers may still distinguish failure modes.
"""


class AlgorithmExecutionError(RuntimeError):
    """Base class for all execution-layer failures."""


class AlgorithmInputError(AlgorithmExecutionError):
    """The caller supplied an invalid program, option, or result request."""


class MissingRuntimeDependencyError(AlgorithmExecutionError):
    """The optional qpanda3-runtime package is not installed."""


class BackendUnavailableError(AlgorithmExecutionError):
    """The selected backend or device is not reachable."""


class DeviceCapabilityError(AlgorithmExecutionError):
    """The device does not support the requested operation."""


class TranspilationError(AlgorithmExecutionError):
    """A program could not be transpiled for the target device."""


class TaskSubmissionError(AlgorithmExecutionError):
    """Submitting a task to the backend failed."""


class TaskTimeoutError(AlgorithmExecutionError):
    """A backend task did not finish within the requested timeout."""


class TaskRecoveryError(AlgorithmExecutionError):
    """A task could not be recovered from a checkpoint or backend state."""


class ResultDecodingError(AlgorithmExecutionError):
    """A backend result could not be decoded into a batch result wrapper."""
