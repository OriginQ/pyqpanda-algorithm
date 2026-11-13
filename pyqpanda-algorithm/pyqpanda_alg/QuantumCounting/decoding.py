# Licensed under the Apache License, Version 2.0.
"""Runtime-independent readout for the corrected Grover phase convention."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def phase_to_probability(phase: float) -> float:
    """Return marked probability for Q=-A S0 A† Sf and phase in [0, 1)."""
    if isinstance(phase, bool) or not isinstance(phase, Real):
        raise ValueError("phase must be finite and in [0, 1)")
    phase = float(phase)
    if not math.isfinite(phase) or not 0.0 <= phase < 1.0:
        raise ValueError("phase must be finite and in [0, 1)")
    # Fold first to avoid asymmetric rounding for conjugate eigenphases.
    return math.sin(math.pi * min(phase, 1.0 - phase)) ** 2


@dataclass(frozen=True)
class PhaseEstimate:
    probability: float
    phase: float
    phase_state: str
    phase_probability: float
    pair_probability: float


def decode_phase_distribution(weights: Mapping[str, Real]) -> PhaseEstimate:
    """Decode exact probabilities or a measured phase-register histogram.

    Mirror bins y and 2**t-y estimate the same marked probability. Pool their
    weight before selecting a peak, rather than splitting their evidence.
    Ties select the smaller folded phase. Return the selected original bin
    and its own weight as well as the complete pair weight. Weights need not
    sum to one; they must be finite, nonnegative and have positive total.
    This is a peak estimator, not an unbiased estimate or confidence interval.
    """
    if not weights:
        raise ValueError("phase distribution must not be empty")
    width = None
    values = {}
    for key, value in weights.items():
        if not isinstance(key, str) or not key or set(key) - {"0", "1"}:
            raise ValueError("phase keys must be non-empty binary strings")
        if width is None:
            width = len(key)
        if len(key) != width:
            raise ValueError("phase keys must have equal width")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("phase weights must be finite and nonnegative")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("phase weights must be finite and nonnegative")
        values[int(key, 2)] = value
    scale = max(values.values())
    if scale <= 0:
        raise ValueError("phase distribution must have positive total weight")
    # Scaling avoids overflow when callers provide large finite counts.
    values = {index: value / scale for index, value in values.items()}
    total = math.fsum(values.values())
    size = 1 << width
    pairs = {}
    for index, value in values.items():
        folded = min(index, (-index) % size)
        pairs.setdefault(folded, []).append(value)
    pairs = {index: math.fsum(value) for index, value in pairs.items()}
    chosen = min(pairs, key=lambda index: (-pairs[index], index))
    mirror = (-chosen) % size
    original = min({chosen, mirror}, key=lambda index: (-values.get(index, 0), index))
    phase = chosen / size
    return PhaseEstimate(
        probability=phase_to_probability(phase), phase=original / size,
        phase_state=format(original, f"0{width}b"),
        phase_probability=values.get(original, 0) / total,
        pair_probability=pairs[chosen] / total,
    )
