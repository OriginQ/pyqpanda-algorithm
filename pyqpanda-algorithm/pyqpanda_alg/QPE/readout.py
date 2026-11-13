# Licensed under the Apache License, Version 2.0.
# Structured result and eigenvalue helper preserved from OriginQ PR #81;
# validation and normalization for both public interfaces.
"""Generic eigenphase readout: distinct phases are never folded together."""

from dataclasses import dataclass
from math import frexp, fsum, isfinite, ldexp, pi
from numbers import Integral, Real
from typing import Dict, Optional


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class PhaseEstimationResult:
    phase: float
    bitstring: str
    probability: float
    resolution: float
    probabilities: Dict[str, float]
    # None means exact statevector probabilities; a positive value is the
    # actual number of measurement shots. Original five-argument construction
    # remains valid.
    sampling_shots: Optional[int] = None


def normalize_distribution(weights, precision_qubits):
    """Validate every entry and normalize finite nonnegative readout weights."""
    precision_qubits = positive_integer(precision_qubits, "precision_qubits")
    if not weights:
        raise ValueError("probabilities cannot be empty")
    values = {}
    for key, weight in weights.items():
        if (not isinstance(key, str) or len(key) != precision_qubits
                or any(bit not in "01" for bit in key)):
            raise ValueError(f"expected {precision_qubits}-bit binary keys, got {key!r}")
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise ValueError("readout weights must be finite nonnegative real numbers")
        try:
            value = float(weight)
        except OverflowError as exc:
            raise ValueError("readout weights must be finite") from exc
        if not isfinite(value) or value < 0:
            raise ValueError("readout weights must be finite nonnegative real numbers")
        values[key] = value
    largest = max(values.values())
    if largest == 0:
        raise ValueError("readout weights must have positive total")
    scaled = {key: value / largest for key, value in values.items()}
    total = fsum(scaled.values())
    return {key: value / total for key, value in scaled.items()}


def decode_phase(probabilities, precision_qubits, *, sampling_shots=None):
    """Decode the most likely generic eigenphase, preserving its own bin.

    Weights can be exact probabilities or observed counts. Equal-weight ties
    retain input insertion order, as in the original structured-result API.
    Generic QPE does not combine phase phi with 1-phi.
    """
    precision_qubits = positive_integer(precision_qubits, "precision_qubits")
    if sampling_shots is not None:
        sampling_shots = positive_integer(sampling_shots, "sampling_shots")
    normalized = normalize_distribution(probabilities, precision_qubits)
    bitstring = max(normalized, key=normalized.get)
    denominator = 1 << precision_qubits
    return PhaseEstimationResult(
        phase=int(bitstring, 2) / denominator,
        bitstring=bitstring,
        probability=normalized[bitstring],
        resolution=1.0 / denominator,
        probabilities=normalized,
        sampling_shots=sampling_shots,
    )


def phase_to_eigenvalue(phase, evolution_time=1.0, centered=False):
    """Convert phase to an eigenvalue branch for U = exp(i * H * t)."""
    if not isfinite(phase) or not 0.0 <= phase < 1.0:
        raise ValueError("phase must lie in [0, 1)")
    if not isfinite(evolution_time) or evolution_time <= 0:
        raise ValueError("evolution_time must be finite and positive")
    wrapped = phase - 1.0 if centered and phase >= 0.5 else phase
    # Form the ratio through bounded mantissas. Multiplying a subnormal phase
    # by 2*pi first loses precision; dividing phase/time first can underflow
    # even when the final result remains representable.
    phase_mantissa, phase_exponent = frexp(wrapped)
    time_mantissa, time_exponent = frexp(evolution_time)
    try:
        value = ldexp(2.0 * pi * (phase_mantissa / time_mantissa),
                      phase_exponent - time_exponent)
    except OverflowError as exc:
        raise ValueError("eigenvalue is not representable as a finite float") from exc
    if not isfinite(value):
        raise ValueError("eigenvalue is not representable as a finite float")
    return value
