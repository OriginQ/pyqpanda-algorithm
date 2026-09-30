# Licensed under the Apache License, Version 2.0.
"""Hamiltonian construction utilities for quadratic and polynomial Ising models."""
from __future__ import annotations

import numpy as np

from ..problems.ising import IsingProblem
from ..problems.polynomial import PolynomialIsingProblem
from .pauli import PauliOp, add, pair, scale, single, z_string


def build_initial_hamiltonian(
    n: int,
    bias: np.ndarray,
    hx: float | np.ndarray = -1.0,
) -> PauliOp:
    """Build ``sum_i hx_i X_i - bias_i Z_i`` for one BF-DCQO iteration."""
    bias = np.asarray(bias, dtype=float)
    hx_array = np.broadcast_to(np.asarray(hx, dtype=float), (n,))
    if bias.shape != (n,):
        raise ValueError("bias has wrong shape")
    operator: PauliOp = {}
    for i in range(n):
        operator = add(operator, {single(n, i, "X"): hx_array[i]})
        operator = add(operator, {single(n, i, "Z"): -bias[i]})
    return operator


def build_problem_hamiltonian(problem) -> PauliOp:
    """Build the non-constant Pauli Hamiltonian for quadratic Ising or HUBO."""
    n = problem.n
    operator: PauliOp = {}
    if isinstance(problem, PolynomialIsingProblem):
        for term, coefficient in problem.terms.items():
            if abs(coefficient) > 1e-15:
                operator = add(operator, {z_string(n, term): float(coefficient)})
        return operator

    if isinstance(problem, IsingProblem) or (
        hasattr(problem, "h") and hasattr(problem, "J")
    ):
        for i, field in enumerate(problem.h):
            if field != 0:
                operator = add(operator, {single(n, i, "Z"): field})
        for i in range(n):
            for j in range(i + 1, n):
                coupling = problem.J[i, j]
                if coupling != 0:
                    operator = add(
                        operator,
                        {pair(n, i, "Z", j, "Z"): coupling},
                    )
        return operator
    raise TypeError(f"unsupported problem type: {type(problem)!r}")


def interpolate(hi: PauliOp, hf: PauliOp, lam: float) -> PauliOp:
    """Return ``(1-lam) * Hi + lam * Hf``."""
    return add(scale(hi, 1.0 - lam), scale(hf, lam))
