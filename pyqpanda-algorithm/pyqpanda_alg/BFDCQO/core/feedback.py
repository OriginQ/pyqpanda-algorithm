# Licensed under the Apache License, Version 2.0.
"""Feedback rules for the BF-DCQO family.

The basic rule estimates <Z_i> from every sample. The CVaR rule keeps only an
``alpha`` fraction of the lowest-energy sample mass before estimating <Z_i>.
Threshold-energy degeneracies are handled proportionally, avoiding any
bitstring-index bias when the CVaR quantile cuts through a degenerate level.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .measurement import z_expectation_from_probabilities


@dataclass(frozen=True)
class FeedbackInfo:
    """Bias estimate plus effective-sample diagnostics for one update."""

    bias: np.ndarray
    effective_samples: float
    energy_threshold: float
    alpha: float


def _validate(prob: np.ndarray, energies: np.ndarray, n: int):
    prob = np.asarray(prob, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if prob.shape != energies.shape or prob.shape != (1 << n,):
        raise ValueError("probabilities/energies have incompatible shape")
    if np.any(prob < -1e-15):
        raise ValueError("probabilities must be nonnegative")
    s = float(prob.sum())
    if s <= 0:
        raise ValueError("probability mass must be positive")
    return prob / s, energies


def mean_feedback_from_probabilities(prob: np.ndarray, n: int) -> FeedbackInfo:
    """Compute the standard BF update from a complete probability vector."""
    prob = np.asarray(prob, dtype=float)
    prob = prob / prob.sum()
    return FeedbackInfo(
        bias=z_expectation_from_probabilities(prob, n),
        effective_samples=float(prob.sum()),
        energy_threshold=float("inf"),
        alpha=1.0,
    )


def _select_low_energy_mass(weights: np.ndarray, energies: np.ndarray, target: float):
    """Return weights restricted to lowest energies with degeneracy-safe cutoff.

    If the target falls inside an energy-degenerate group, all members of that
    group are scaled by the same fractional factor. This is the discrete analogue
    of random subsampling at the quantile boundary and prevents arbitrary basis
    ordering from choosing a preferred state.
    """
    selected = np.zeros_like(weights, dtype=float)
    used = 0.0
    threshold = float("inf")
    for energy in np.unique(np.sort(energies)):
        idx = np.flatnonzero(np.isclose(energies, energy, atol=1e-12, rtol=1e-12) & (weights > 0))
        if idx.size == 0:
            continue
        group = float(weights[idx].sum())
        remaining = target - used
        if remaining <= 1e-15:
            break
        if group <= remaining + 1e-15:
            selected[idx] = weights[idx]
            used += group
        else:
            frac = remaining / group
            selected[idx] = weights[idx] * frac
            used += remaining
        threshold = float(energy)
        if used >= target - 1e-12:
            break
    if used <= 0:
        raise RuntimeError("CVaR selected no sample mass")
    return selected, used, threshold


def cvar_feedback_from_probabilities(
    prob: np.ndarray,
    energies: np.ndarray,
    n: int,
    alpha: float,
) -> FeedbackInfo:
    """CVaR feedback from an exact probability vector."""
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must lie in (0,1]")
    p, e = _validate(prob, energies, n)
    if alpha >= 1.0 - 1e-15:
        return FeedbackInfo(
            z_expectation_from_probabilities(p, n), 1.0,
            float(np.max(e[p > 0])), 1.0,
        )
    selected, used, threshold = _select_low_energy_mass(p, e, float(alpha))
    selected /= used
    return FeedbackInfo(
        bias=z_expectation_from_probabilities(selected, n),
        effective_samples=float(used),
        energy_threshold=threshold,
        alpha=float(alpha),
    )


def cvar_feedback_from_counts(
    counts: np.ndarray,
    energies: np.ndarray,
    n: int,
    alpha: float,
) -> FeedbackInfo:
    """CVaR feedback from finite-shot counts.

    The target is ``ceil(alpha * shots)`` samples.  If that target cuts through
    a degenerate energy level, counts at the threshold are scaled proportionally
    instead of taking a basis-order-dependent subset.
    """
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must lie in (0,1]")
    counts = np.asarray(counts, dtype=np.int64)
    energies = np.asarray(energies, dtype=float)
    if counts.shape != energies.shape or counts.shape != (1 << n,):
        raise ValueError("counts/energies have incompatible shape")
    if np.any(counts < 0):
        raise ValueError("counts must be nonnegative")
    shots = int(counts.sum())
    if shots <= 0:
        raise ValueError("counts must contain at least one shot")
    if alpha >= 1.0 - 1e-15:
        p = counts.astype(float) / float(shots)
        return FeedbackInfo(
            bias=z_expectation_from_probabilities(p, n),
            effective_samples=float(shots),
            energy_threshold=float(np.max(energies[counts > 0])),
            alpha=1.0,
        )
    target = max(1, int(np.ceil(alpha * shots)))
    selected, used, threshold = _select_low_energy_mass(counts.astype(float), energies, float(target))
    selected /= used
    return FeedbackInfo(
        bias=z_expectation_from_probabilities(selected, n),
        effective_samples=float(used),
        energy_threshold=threshold,
        alpha=float(alpha),
    )
