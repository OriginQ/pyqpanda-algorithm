"""Pure input canonicalization for QSEncode-Insight.

This implementation is independent of the legacy QSpare_Code validator and has
no PyQPanda or Frozen-evidence runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import InputPolicy
from .exceptions import InputValidationError
from .models import InputSummary


ORIGINAL_INPUT_DOMAIN = b"qseencode-original-input-v1\0"
EFFECTIVE_PROBABILITY_DOMAIN = b"qseencode-effective-probability-v1\0"
NORMALIZED_CHANGE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ValidatedInput:
    """Private pure-core output; probabilities are a read-only float64 copy."""

    summary: InputSummary
    probabilities: NDArray[np.float64]


def canonical_probability_sha256(vector: ArrayLike, *, domain: bytes) -> str:
    """Hash a one-dimensional vector under the frozen binary v1 contract."""

    if not isinstance(domain, bytes) or not domain:
        raise ValueError("domain must be non-empty bytes")
    try:
        array = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputValidationError(code="not_numeric") from error
    if array.ndim != 1:
        raise InputValidationError(code="not_one_dimensional")

    little_endian = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
    payload = (
        domain
        + struct.pack("<Q", int(little_endian.size))
        + little_endian.tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def canonicalize_probabilities(
    probabilities: ArrayLike,
    *,
    policy: InputPolicy | None = None,
) -> ValidatedInput:
    """Validate, normalize, pad, and hash a probability vector."""

    active_policy = policy if policy is not None else InputPolicy()
    if not isinstance(active_policy, InputPolicy):
        raise InputValidationError(code="invalid_input_policy")

    try:
        source = np.asarray(probabilities)
        if np.iscomplexobj(source):
            raise InputValidationError(code="complex_probability")
        converted = np.asarray(source, dtype=np.float64)
    except InputValidationError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise InputValidationError(code="not_numeric") from error
    if converted.ndim != 1:
        raise InputValidationError(code="not_one_dimensional")
    if converted.size < 2:
        raise InputValidationError(code="insufficient_dimension")

    original = np.array(converted, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(original)):
        raise InputValidationError(code="nonfinite_probability")
    if np.any(original < 0.0):
        raise InputValidationError(code="negative_probability")

    original_sum = float(np.sum(original, dtype=np.float64))
    if not np.isfinite(original_sum):
        raise InputValidationError(code="nonfinite_probability")
    if original_sum <= 0.0:
        raise InputValidationError(code="zero_mass")

    original_hash = canonical_probability_sha256(
        original, domain=ORIGINAL_INPUT_DOMAIN
    )
    sum_difference = abs(original_sum - 1.0)
    if (
        active_policy.normalization == "strict"
        and sum_difference > active_policy.normalization_tolerance
    ):
        raise InputValidationError(code="not_normalized")

    normalized = original / original_sum
    original_length = int(normalized.size)
    if active_policy.padding == "reject":
        if not _is_power_of_two(original_length):
            raise InputValidationError(code="non_power_of_two")
        padded_length = original_length
    else:
        padded_length = _next_power_of_two(original_length)

    padding_count = padded_length - original_length
    if padding_count:
        effective = np.pad(
            normalized,
            (0, padding_count),
            mode="constant",
            constant_values=0.0,
        )
    else:
        effective = np.array(normalized, dtype=np.float64, order="C", copy=True)
    effective = np.ascontiguousarray(effective, dtype=np.float64)

    effective_hash = canonical_probability_sha256(
        effective, domain=EFFECTIVE_PROBABILITY_DOMAIN
    )
    effective.setflags(write=False)
    summary = InputSummary(
        original_length=original_length,
        padded_length=padded_length,
        original_sum=original_sum,
        normalized=sum_difference > NORMALIZED_CHANGE_TOLERANCE,
        padding_count=padding_count,
        original_input_sha256=original_hash,
        effective_probability_sha256=effective_hash,
    )
    return ValidatedInput(summary=summary, probabilities=effective)


__all__ = [
    "ValidatedInput",
    "ORIGINAL_INPUT_DOMAIN",
    "EFFECTIVE_PROBABILITY_DOMAIN",
    "canonical_probability_sha256",
    "canonicalize_probabilities",
]
