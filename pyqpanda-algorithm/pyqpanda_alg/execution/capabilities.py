"""Declarative capabilities of an execution backend.

Backends advertise which operations they support; algorithms and the
preflight layer use this to fail fast with
:class:`~pyqpanda_alg.execution.errors.DeviceCapabilityError` instead of
discovering a missing capability after submission.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    """Explicit support flags for every backend operation family."""

    sampling: bool = True
    estimation: bool = True
    variational_session: bool = True
    statevector: bool = True
    tomography: bool = True
