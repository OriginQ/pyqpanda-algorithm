"""Pure ranking and error-budget core for QSEncode-Insight."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import InputValidationError
from .models import ErrorBudgetResult


RANKING_POLICY = "frozen_stable_v1"
MINIMALITY_TOLERANCE = 1e-12


def _coefficient_array(values: ArrayLike) -> NDArray[np.float64] | NDArray[np.complex128]:
    try:
        source = np.asarray(values)
        dtype = np.complex128 if np.iscomplexobj(source) else np.float64
        coefficients = np.array(source, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputValidationError(code="coefficients_not_numeric") from error
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise InputValidationError(code="invalid_coefficient_dimension")
    if not np.all(np.isfinite(coefficients)):
        raise InputValidationError(code="nonfinite_coefficients")
    return coefficients


def stable_magnitude_order(coefficients: ArrayLike) -> NDArray[np.intp]:
    """Return the exact Frozen ascending stable magnitude ordering."""

    values = _coefficient_array(coefficients)
    return np.argsort(np.abs(values), kind="stable")


def _validate_k(k: int, size: int) -> None:
    if type(k) is not int or not 1 <= k <= size:
        raise InputValidationError(code="invalid_k")


def top_k_indices(coefficients: ArrayLike, k: int) -> NDArray[np.intp]:
    values = _coefficient_array(coefficients)
    _validate_k(k, int(values.size))
    order = np.argsort(np.abs(values), kind="stable")
    return np.asarray(order[-k:], dtype=np.intp)


def top_k_coefficients(
    coefficients: ArrayLike,
    k: int,
    *,
    normalize: bool = False,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    values = _coefficient_array(coefficients)
    indices = top_k_indices(values, k)
    selected = np.zeros_like(values)
    selected[indices] = values[indices]
    if normalize:
        norm = float(np.linalg.norm(selected))
        if not math.isfinite(norm) or norm <= 0.0:
            raise InputValidationError(code="zero_selected_norm")
        selected = selected / norm
    return selected


def _energy_components(
    coefficients: ArrayLike,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], NDArray[np.float64], float, NDArray[np.intp]]:
    values = _coefficient_array(coefficients)
    energy = np.asarray(np.abs(values) ** 2, dtype=np.float64)
    total = float(np.sum(energy, dtype=np.float64))
    if not math.isfinite(total):
        raise InputValidationError(code="nonfinite_coefficient_energy")
    if total <= 0.0:
        raise InputValidationError(code="zero_coefficient_energy")
    order = np.argsort(np.abs(values), kind="stable")
    return values, energy, total, order


def retained_energy_ratio(coefficients: ArrayLike, k: int) -> float:
    values, energy, total, order = _energy_components(coefficients)
    _validate_k(k, int(values.size))
    return float(np.sum(energy[order[-k:]], dtype=np.float64) / total)


def candidate_neighborhood(k_star: int, size: int) -> tuple[int, ...]:
    if type(size) is not int or size < 2:
        raise InputValidationError(code="invalid_candidate_dimension")
    if type(k_star) is not int or not 1 <= k_star <= size:
        raise InputValidationError(code="invalid_k_star")
    return tuple(
        candidate
        for candidate in (k_star - 1, k_star, k_star + 1)
        if 1 <= candidate <= size - 1
    )


def find_k_star(
    coefficients: ArrayLike,
    fidelity_target: float,
) -> ErrorBudgetResult:
    """Return the minimal retained-energy k and its frozen neighborhood."""

    if (
        not isinstance(fidelity_target, (float, int))
        or isinstance(fidelity_target, bool)
        or not math.isfinite(float(fidelity_target))
        or not 0.0 < float(fidelity_target) <= 1.0
    ):
        raise InputValidationError(code="invalid_fidelity_target")
    target = float(fidelity_target)
    values, energy, total, order = _energy_components(coefficients)

    previous = 0.0
    retained = 0.0
    k_star = int(values.size)
    for k in range(1, int(values.size) + 1):
        retained = float(np.sum(energy[order[-k:]], dtype=np.float64) / total)
        if retained >= target - MINIMALITY_TOLERANCE:
            k_star = k
            break
        previous = retained

    minimality_pass = (
        previous < target + MINIMALITY_TOLERANCE
        and retained >= target - MINIMALITY_TOLERANCE
    )
    return ErrorBudgetResult(
        fidelity_target=target,
        k_star=k_star,
        retained_energy=retained,
        previous_retained_energy=previous,
        candidate_k=candidate_neighborhood(k_star, int(values.size)),
        minimality_pass=minimality_pass,
        ranking_policy=RANKING_POLICY,
    )


__all__ = [
    "RANKING_POLICY",
    "MINIMALITY_TOLERANCE",
    "stable_magnitude_order",
    "top_k_indices",
    "top_k_coefficients",
    "retained_energy_ratio",
    "candidate_neighborhood",
    "find_k_star",
]
