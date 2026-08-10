"""Batch result wrappers with one canonical shape per backend.

Backends return batches of results (counts, expectation values, or
statevectors) plus transport diagnostics.  These wrappers normalize the
batch shape so algorithms do not have to special-case local versus
runtime results, while keeping the diagnostics available as sanitized
``raw_metadata``.  Callers must use the ``single_*`` accessor when
exactly one result was requested.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .errors import AlgorithmInputError


def _sanitize_metadata(raw_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a private deep copy of the diagnostics dict, or None.

    Metadata is small, so a deep copy is cheap; a shallow copy would
    keep caller-owned nested dicts aliased, letting the caller mutate
    the wrapped result's diagnostics after the fact.
    """
    if raw_metadata is None:
        return None
    return copy.deepcopy(raw_metadata)


def _readonly_complex_copy(array: Any) -> np.ndarray:
    """Return a fresh, complex, read-only copy of a statevector."""
    copied = np.array(array, dtype=complex, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class SampleBatchResult:
    """Sampling outcomes: one counts dict per submitted circuit."""

    counts: Tuple[Dict[str, int], ...]
    shots: int
    raw_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", tuple(dict(c) for c in self.counts))
        object.__setattr__(self, "raw_metadata", _sanitize_metadata(self.raw_metadata))

    def single_counts(self) -> Dict[str, int]:
        """Return the counts dict when exactly one result was requested."""
        if len(self.counts) != 1:
            raise AlgorithmInputError(
                f"expected exactly one sample result, got {len(self.counts)}"
            )
        return self.counts[0]


@dataclass(frozen=True)
class EstimateBatchResult:
    """Expectation values: one float per submitted circuit."""

    values: Tuple[float, ...]
    raw_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        object.__setattr__(self, "raw_metadata", _sanitize_metadata(self.raw_metadata))

    def single_value(self) -> float:
        """Return the expectation value when exactly one was requested."""
        if len(self.values) != 1:
            raise AlgorithmInputError(
                f"expected exactly one estimate result, got {len(self.values)}"
            )
        return self.values[0]


@dataclass(frozen=True)
class StatevectorBatchResult:
    """Statevectors: one read-only complex array per submitted circuit."""

    statevectors: Tuple[np.ndarray, ...]
    raw_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "statevectors", tuple(_readonly_complex_copy(s) for s in self.statevectors)
        )
        object.__setattr__(self, "raw_metadata", _sanitize_metadata(self.raw_metadata))

    def single_statevector(self) -> np.ndarray:
        """Return the statevector when exactly one was requested."""
        if len(self.statevectors) != 1:
            raise AlgorithmInputError(
                f"expected exactly one statevector result, got {len(self.statevectors)}"
            )
        return self.statevectors[0]
