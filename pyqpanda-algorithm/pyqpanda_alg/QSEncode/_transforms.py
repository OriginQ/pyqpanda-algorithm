"""Pure orthonormal transform layer for QSEncode-Insight.

Walsh production uses an iterative O(N log N) FWHT.  The explicit Hadamard
matrix is restricted to small diagnostics and tests.  Fourier production uses
SciPy's orthonormal FFT without constructing a quantum circuit.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import fft
from scipy.linalg import hadamard

from .exceptions import InputValidationError, InternalInvariantError
from .models import TransformDiagnostics


TRANSFORM_ATOL = 1e-12
TRANSFORM_RTOL = 1e-10


def _transform_array(values: ArrayLike) -> NDArray[np.float64] | NDArray[np.complex128]:
    try:
        source = np.asarray(values)
        dtype = np.complex128 if np.iscomplexobj(source) else np.float64
        array = np.array(source, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputValidationError(code="transform_not_numeric") from error
    if array.ndim != 1:
        raise InputValidationError(code="transform_not_one_dimensional")
    if not np.all(np.isfinite(array)):
        raise InputValidationError(code="transform_nonfinite")
    return array


def _require_power_of_two_length(array: np.ndarray) -> None:
    size = int(array.size)
    if size <= 0 or size & (size - 1):
        raise InputValidationError(code="transform_non_power_of_two")


def _validated_transform_energy(array: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        energy = float(np.sum(np.abs(array) ** 2, dtype=np.float64))
    if not math.isfinite(energy):
        raise InternalInvariantError(code="nonfinite_transform_energy")
    if energy <= 0.0:
        raise InternalInvariantError(code="zero_transform_energy")
    return energy


def _validate_transform_output(
    coefficients: np.ndarray,
    *,
    input_energy: float,
) -> tuple[float, float]:
    if not np.all(np.isfinite(coefficients)):
        raise InternalInvariantError(code="nonfinite_transform_coefficients")
    output_energy = _validated_transform_energy(coefficients)
    parseval_error = abs(output_energy - input_energy)
    allowed_error = TRANSFORM_ATOL + TRANSFORM_RTOL * input_energy
    if parseval_error > allowed_error:
        raise InternalInvariantError(code="parseval_invariant_failed")
    return output_energy, parseval_error


def normalized_fwht(
    values: ArrayLike,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Return the orthonormal iterative Walsh-Hadamard transform."""

    result = _transform_array(values)
    _require_power_of_two_length(result)
    input_energy = _validated_transform_energy(result)
    size = int(result.size)
    width = 1
    with np.errstate(over="ignore", invalid="ignore"):
        while width < size:
            block = width * 2
            for start in range(0, size, block):
                left = result[start : start + width].copy()
                right = result[start + width : start + block].copy()
                result[start : start + width] = left + right
                result[start + width : start + block] = left - right
            width = block
        result /= math.sqrt(size)
    _validate_transform_output(result, input_energy=input_energy)
    return result


def explicit_walsh_oracle(
    values: ArrayLike,
    *,
    maximum_size: int = 64,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Return the O(N^2) explicit Walsh oracle for small diagnostics only."""

    array = _transform_array(values)
    _require_power_of_two_length(array)
    input_energy = _validated_transform_energy(array)
    if array.size > maximum_size:
        raise InputValidationError(code="walsh_oracle_size_exceeded")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(hadamard(array.size) @ array / math.sqrt(array.size))
    _validate_transform_output(result, input_energy=input_energy)
    return result


def normalized_fourier(values: ArrayLike) -> NDArray[np.complex128]:
    """Return SciPy FFT coefficients in orthonormal ordering/convention."""

    array = _transform_array(values)
    _require_power_of_two_length(array)
    input_energy = _validated_transform_energy(array)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(fft(array, norm="ortho"), dtype=np.complex128)
    _validate_transform_output(result, input_energy=input_energy)
    return result


def analyze_transform(
    amplitudes: ArrayLike,
    *,
    basis: str,
    oracle_check: bool = False,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], TransformDiagnostics]:
    """Run one explicit basis transform and return its pure diagnostics."""

    source = _transform_array(amplitudes)
    _require_power_of_two_length(source)
    input_energy = _validated_transform_energy(source)
    if basis == "walsh":
        coefficients = normalized_fwht(source)
        implementation = "iterative_fwht_v1"
        oracle_checked = bool(oracle_check)
        oracle_error: float | None = None
        if oracle_checked:
            oracle = explicit_walsh_oracle(source)
            oracle_error = float(np.max(np.abs(coefficients - oracle)))
    elif basis == "fourier":
        coefficients = normalized_fourier(source)
        implementation = "scipy_fft_ortho_v1"
        oracle_checked = False
        oracle_error = None
    else:
        raise InputValidationError(code="invalid_transform_basis")

    _, parseval_error = _validate_transform_output(
        coefficients,
        input_energy=input_energy,
    )
    diagnostics = TransformDiagnostics(
        basis=basis,
        coefficient_count=int(coefficients.size),
        parseval_error=parseval_error,
        implementation=implementation,
        normalization="orthonormal",
        oracle_checked=oracle_checked,
        oracle_max_abs_error=oracle_error,
    )
    return coefficients, diagnostics


__all__ = [
    "TRANSFORM_ATOL",
    "TRANSFORM_RTOL",
    "normalized_fwht",
    "explicit_walsh_oracle",
    "normalized_fourier",
    "analyze_transform",
]
