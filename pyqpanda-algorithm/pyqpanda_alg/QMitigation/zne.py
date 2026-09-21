"""Zero-noise extrapolation helpers and circuit-folding schedules."""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, TypeVar

import numpy as np


T = TypeVar("T")


@dataclass(frozen=True)
class PolynomialExtrapolation:
    """Polynomial zero-noise fit and diagnostics."""

    value: np.ndarray
    coefficients: np.ndarray
    residual_norm: float
    degree: int


def _normalize_scales(scales: Sequence[float]) -> np.ndarray:
    values = np.asarray(scales, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("noise scales must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("noise scales must be positive and finite")
    if len(np.unique(values)) != values.size:
        raise ValueError("noise scales must be distinct")
    return values


def global_fold_sequence(
    operations: Sequence[T],
    scale: int,
    inverse: Callable[[T], T],
) -> Tuple[T, ...]:
    """Return U (U_dagger U)^n for an odd integer noise scale 2n+1.

    The operation sequence is assumed to be in execution order. The inverse
    callback must return the adjoint of one operation.
    """

    scale = int(scale)
    if scale < 1 or scale % 2 != 1:
        raise ValueError("global folding scale must be a positive odd integer")

    original = tuple(operations)
    if not original:
        raise ValueError("operations may not be empty")

    folded = list(original)
    inverse_sequence = tuple(inverse(operation) for operation in reversed(original))
    for _ in range((scale - 1) // 2):
        folded.extend(inverse_sequence)
        folded.extend(original)
    return tuple(folded)


def richardson_coefficients(scales: Sequence[float]) -> np.ndarray:
    """Return coefficients that cancel the first m-1 noise powers."""

    values = _normalize_scales(scales)
    vandermonde = np.vstack(
        [values ** power for power in range(values.size)]
    )
    target = np.zeros(values.size, dtype=float)
    target[0] = 1.0
    return np.linalg.solve(vandermonde, target)


def richardson_extrapolate(scales: Sequence[float], noisy_values) -> np.ndarray:
    """Extrapolate an observable to zero noise with Richardson cancellation."""

    scale_values = _normalize_scales(scales)
    values = np.asarray(noisy_values)
    if values.ndim == 0 or values.shape[0] != scale_values.size:
        raise ValueError("noisy_values first axis must match noise scales")
    if not np.all(np.isfinite(values)):
        raise ValueError("noisy_values must be finite")

    coefficients = richardson_coefficients(scale_values)
    return np.tensordot(coefficients, values, axes=(0, 0))


def polynomial_extrapolate(
    scales: Sequence[float],
    noisy_values,
    *,
    degree: int = 2,
    weights=None,
) -> PolynomialExtrapolation:
    """Least-squares polynomial zero-noise extrapolation."""

    scale_values = _normalize_scales(scales)
    values = np.asarray(noisy_values, dtype=float)
    if values.ndim == 0 or values.shape[0] != scale_values.size:
        raise ValueError("noisy_values first axis must match noise scales")
    if not np.all(np.isfinite(values)):
        raise ValueError("noisy_values must be finite")

    degree = int(degree)
    if degree < 0 or degree >= scale_values.size:
        raise ValueError("degree must satisfy 0 <= degree < number of scales")

    design = np.column_stack(
        [scale_values ** power for power in range(degree + 1)]
    )
    flat_values = values.reshape(scale_values.size, -1)

    if weights is not None:
        weight_values = np.asarray(weights, dtype=float)
        if weight_values.shape != (scale_values.size,):
            raise ValueError("weights must provide one value per noise scale")
        if not np.all(np.isfinite(weight_values)) or np.any(weight_values <= 0):
            raise ValueError("weights must be positive and finite")
        root = np.sqrt(weight_values)[:, None]
        design_fit = design * root
        values_fit = flat_values * root
    else:
        design_fit = design
        values_fit = flat_values

    coefficients, _, _, _ = np.linalg.lstsq(
        design_fit,
        values_fit,
        rcond=None,
    )
    fitted = design @ coefficients
    residual_norm = float(np.linalg.norm(flat_values - fitted))
    output_shape = values.shape[1:]
    intercept = coefficients[0].reshape(output_shape)
    coefficient_shape = (degree + 1,) + output_shape

    return PolynomialExtrapolation(
        value=intercept,
        coefficients=coefficients.reshape(coefficient_shape),
        residual_norm=residual_norm,
        degree=degree,
    )


def evaluate_noise_scales(
    executor: Callable[[float], object],
    scales: Sequence[float],
) -> Tuple[object, ...]:
    """Evaluate a caller-provided noisy circuit executor at every scale."""

    normalized = _normalize_scales(scales)
    return tuple(executor(float(scale)) for scale in normalized)


def zero_noise_extrapolate(
    executor: Callable[[float], object],
    scales: Sequence[float],
) -> np.ndarray:
    """Execute all requested scales and apply Richardson extrapolation."""

    values = evaluate_noise_scales(executor, scales)
    return richardson_extrapolate(scales, values)
