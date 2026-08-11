"""HHL data types: configuration, normalized system, and solution record.

:class:`HHLConfig` carries the classical knobs of a solve: the
ill-conditioning threshold and the phase-estimation register size.
:class:`NormalizedLinearSystem` is the validated, padded, and
normalized input produced by :func:`normalize_linear_system`; it
records everything needed to truncate and rescale a quantum-computed
solution back to the original system.  :class:`HHLSolution` is the
immutable snapshot of a solved system.  Arrays and containers are
copied on construction so these objects can never be mutated through
caller-owned objects.
"""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HHLConfig:
    """Immutable HHL run configuration.

    ``max_condition_number`` is the largest accepted ratio of the
    largest to the smallest singular value of the system matrix;
    systems with a higher condition number are rejected before any
    task submission.  ``phase_qubits`` is the size of the phase
    estimation register used by circuit synthesis.
    """

    max_condition_number: float = 1e6
    phase_qubits: int = 4

    def __post_init__(self) -> None:
        if not (self.max_condition_number > 0):
            # ``not (x > 0)`` rejects NaN, which ``x <= 0`` silently
            # accepts and would otherwise disable the rejection itself.
            raise ValueError(
                f"max_condition_number must be a positive number, got {self.max_condition_number}"
            )
        if self.phase_qubits < 1:
            raise ValueError(
                f"phase_qubits must be at least 1, got {self.phase_qubits}"
            )


@dataclass(frozen=True)
class NormalizedLinearSystem:
    """Validated, padded, and normalized form of a linear system.

    ``matrix`` is the ``padded_dimension``-sized Hermitian system
    matrix and ``vector`` the unit-norm right-hand vector of the same
    size; both are immutable copies.  ``original_dimension`` and
    ``original_vector_norm`` make the padding reversible: truncate a
    computed solution to its first ``original_dimension`` entries and
    scale by ``original_vector_norm`` to recover the solution of the
    original (unpadded, unnormalized) system.  ``metadata`` keeps
    free-form provenance without exposing any live object.
    """

    original_dimension: int
    padded_dimension: int
    matrix: np.ndarray
    vector: np.ndarray
    original_vector_norm: float
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, copy=True)
        vector = np.array(self.vector, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"matrix must be square, got shape {matrix.shape}"
            )
        if matrix.shape[0] != self.padded_dimension:
            raise ValueError(
                f"matrix size {matrix.shape[0]} does not match "
                f"padded_dimension {self.padded_dimension}"
            )
        if vector.ndim != 1 or vector.shape[0] != self.padded_dimension:
            raise ValueError(
                f"vector must have {self.padded_dimension} entries, "
                f"got shape {vector.shape}"
            )
        matrix.setflags(write=False)
        vector.setflags(write=False)
        object.__setattr__(self, "original_dimension", int(self.original_dimension))
        object.__setattr__(self, "padded_dimension", int(self.padded_dimension))
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "original_vector_norm", float(self.original_vector_norm))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))


@dataclass(frozen=True)
class HHLSolution:
    """Immutable outcome of an HHL solve.

    ``classical_vector`` is the reconstructed solution direction,
    always normalized to unit norm; the original vector norm and the
    unpadded dimension needed to restore the original scale are
    carried by :class:`NormalizedLinearSystem` (``original_vector_norm``
    and ``original_dimension``).  ``statevector`` is the full
    post-selected data-register state when a statevector backend
    produced one, and ``success_probability`` the ancilla success
    branch probability.  ``residual`` is the error against the
    original unpadded system when it was computed.  ``metadata`` keeps
    free-form provenance.  Fields that a solver did not produce remain
    None.
    """

    classical_vector: np.ndarray | None = None
    statevector: np.ndarray | None = None
    success_probability: float | None = None
    residual: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("classical_vector", "statevector"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.array(value, copy=True)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("success_probability", "residual"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "success_probability" and not 0.0 <= value <= 1.0:
                raise ValueError("success_probability must lie in [0, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
