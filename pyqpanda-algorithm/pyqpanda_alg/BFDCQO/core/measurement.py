# Licensed under the Apache License, Version 2.0.
"""State-preparation and measurement post-processing utilities."""
from __future__ import annotations

import numpy as np


def initial_ground_angles(
    bias: np.ndarray,
    hx: float | np.ndarray = -1.0,
) -> np.ndarray:
    """Return RY angles for the product ground state of ``hx*X - bias*Z``."""
    bias = np.asarray(bias, dtype=float)
    hx_array = np.broadcast_to(np.asarray(hx, dtype=float), bias.shape)
    minimum = -np.sqrt(bias**2 + hx_array**2)
    theta = 2.0 * np.arctan2(bias + minimum, hx_array)
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def z_expectation_from_probabilities(prob: np.ndarray, n: int) -> np.ndarray:
    """Compute every ``<Z_i>`` from a computational-basis distribution."""
    prob = np.asarray(prob, dtype=float)
    if prob.shape != (1 << n,):
        raise ValueError("probability vector has wrong length")
    expectation = np.zeros(n)
    for index, probability in enumerate(prob):
        if probability == 0:
            continue
        for qubit in range(n):
            sign = 1.0 if ((index >> qubit) & 1) == 0 else -1.0
            expectation[qubit] += probability * sign
    return expectation


def counts_to_probability(counts: dict[str, int], n: int) -> np.ndarray:
    """Convert a bitstring-count dictionary into an integer-indexed distribution."""
    probability = np.zeros(1 << n, dtype=float)
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("empty counts")
    for bits, count in counts.items():
        key = str(bits).replace(" ", "").zfill(n)
        probability[int(key, 2)] += int(count) / total
    return probability
