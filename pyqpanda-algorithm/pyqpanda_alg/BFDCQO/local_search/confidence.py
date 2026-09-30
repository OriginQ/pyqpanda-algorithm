# Licensed under the Apache License, Version 2.0.
"""Confidence-guided and random block-local-search implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np


@dataclass(frozen=True)
class LocalSearchStep:
    """Diagnostics for one exact block-neighborhood refinement step."""

    round_index: int
    variables: tuple[int, ...]
    start_energy: float
    end_energy: float
    evaluations: int
    improved: bool


@dataclass(frozen=True)
class LocalSearchResult:
    """Result and evaluation count of bounded block local search."""

    start_solution: str
    best_solution: str
    start_energy: float
    best_energy: float
    evaluations: int
    method: str
    variable_order: tuple[int, ...]
    steps: tuple[LocalSearchStep, ...]

    @property
    def improvement(self) -> float:
        return float(self.start_energy - self.best_energy)


def bitstring_to_bits(bitstring: str) -> np.ndarray:
    """Convert displayed q_(n-1)...q_0 bitstring to q0-first bits."""
    return np.asarray([int(c) for c in bitstring[::-1]], dtype=int)


def bits_to_bitstring(bits: Iterable[int]) -> str:
    """Convert q0-first bits to displayed q_(n-1)...q_0 bitstring."""
    b = np.asarray(list(bits), dtype=int)
    return "".join(str(int(x)) for x in b[::-1])


def uncertain_variable_order(bias: np.ndarray) -> tuple[int, ...]:
    """Return variables from least to most confident, stable under ties."""
    b = np.asarray(bias, dtype=float)
    return tuple(int(i) for i in np.argsort(np.abs(b), kind="stable"))


def random_variable_order(n: int, seed: int) -> tuple[int, ...]:
    """Return a reproducible random permutation for ablation experiments."""
    rng = np.random.default_rng(int(seed))
    return tuple(int(i) for i in rng.permutation(int(n)))


def _optimize_block(problem, bits: np.ndarray, variables: tuple[int, ...]):
    """Exact minimization over one small coordinate block with others fixed."""
    start = bits.copy()
    best_bits = start.copy()
    best_e = float(problem.energy_bits(best_bits))
    evals = 0
    k = len(variables)
    for mask in range(1 << k):
        candidate = start.copy()
        for pos, q in enumerate(variables):
            candidate[q] = (mask >> pos) & 1
        e = float(problem.energy_bits(candidate))
        evals += 1
        # Stable tie-break: prefer lexicographically smaller displayed bitstring.
        if e < best_e - 1e-12 or (
            np.isclose(e, best_e, atol=1e-12)
            and bits_to_bitstring(candidate) < bits_to_bitstring(best_bits)
        ):
            best_e = e
            best_bits = candidate.copy()
    return best_bits, best_e, evals


def block_local_search(
    problem,
    start_solution: str,
    *,
    variable_order: Iterable[int],
    neighborhood_size: int = 4,
    rounds: int = 1,
    method: str = "block",
) -> LocalSearchResult:
    """Run equal-cost exact block neighborhoods following a supplied variable order.

    Each round optimizes at most ``neighborhood_size`` variables exactly, costing
    ``2**k`` classical energy evaluations for a k-variable block.  Consecutive
    rounds consume consecutive blocks of the supplied order (wrapping if needed),
    so confidence-guided and random ablations have exactly the same evaluation
    budget.
    """
    bits = bitstring_to_bits(start_solution)
    if bits.shape != (problem.n,):
        raise ValueError("start solution length does not match problem")
    order = tuple(int(q) for q in variable_order)
    if sorted(order) != list(range(problem.n)):
        raise ValueError("variable_order must be a permutation of all variables")
    k = min(int(neighborhood_size), problem.n)
    if k < 1 or int(rounds) < 1:
        raise ValueError("neighborhood_size and rounds must be >=1")
    start_e = float(problem.energy_bits(bits))
    current_e = start_e
    steps = []
    total_evals = 0
    cursor = 0
    for r in range(int(rounds)):
        if k == problem.n:
            block = order
        else:
            block = tuple(order[(cursor + j) % problem.n] for j in range(k))
            cursor = (cursor + k) % problem.n
        before = current_e
        bits, current_e, evals = _optimize_block(problem, bits, block)
        total_evals += evals
        steps.append(LocalSearchStep(
            round_index=r,
            variables=tuple(block),
            start_energy=float(before),
            end_energy=float(current_e),
            evaluations=int(evals),
            improved=bool(current_e < before - 1e-12),
        ))
    return LocalSearchResult(
        start_solution=start_solution,
        best_solution=bits_to_bitstring(bits),
        start_energy=float(start_e),
        best_energy=float(current_e),
        evaluations=int(total_evals),
        method=str(method),
        variable_order=order,
        steps=tuple(steps),
    )


def confidence_guided_local_search(
    problem,
    start_solution: str,
    bias: np.ndarray,
    *,
    neighborhood_size: int = 4,
    rounds: int = 1,
) -> LocalSearchResult:
    """Refine the least-confident BF variables under a bounded block budget."""
    return block_local_search(
        problem,
        start_solution,
        variable_order=uncertain_variable_order(bias),
        neighborhood_size=neighborhood_size,
        rounds=rounds,
        method="confidence",
    )


def random_block_local_search(
    problem,
    start_solution: str,
    *,
    neighborhood_size: int = 4,
    rounds: int = 1,
    seed: int = 0,
) -> LocalSearchResult:
    """Run the equal-cost random-block ablation baseline."""
    return block_local_search(
        problem,
        start_solution,
        variable_order=random_variable_order(problem.n, seed),
        neighborhood_size=neighborhood_size,
        rounds=rounds,
        method="random",
    )
