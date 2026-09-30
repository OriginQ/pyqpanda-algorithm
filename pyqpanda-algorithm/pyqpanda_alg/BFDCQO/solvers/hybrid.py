# Licensed under the Apache License, Version 2.0.
"""Hybrid BF-DCQO with bounded confidence-guided local refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from .bfdcqo import BFDCQO, CVaRBFDCQO, AdaptiveShotBFDCQO, BFDCQOResult
from ..local_search import confidence_guided_local_search, random_block_local_search, LocalSearchResult


@dataclass
class HybridBFDCQOResult:
    """Combined quantum-stage and bounded local-refinement result."""

    quantum_result: BFDCQOResult
    local_result: LocalSearchResult
    best_solution: str
    best_energy: float
    exact_ground_energy: float
    exact_ground_states: list[str]
    variant: str

    @property
    def total_shots(self) -> int:
        return int(self.quantum_result.total_shots)

    @property
    def circuit_evaluations(self) -> int:
        return int(self.quantum_result.circuit_evaluations)

    @property
    def local_evaluations(self) -> int:
        return int(self.local_result.evaluations)

    @property
    def success(self) -> bool:
        return bool(np.isclose(self.best_energy, self.exact_ground_energy, atol=1e-9))

    @property
    def quantum_best_energy(self) -> float:
        return float(self.quantum_result.best_energy)

    @property
    def local_improvement(self) -> float:
        return float(self.quantum_result.best_energy - self.best_energy)


class HybridBFDCQO:
    """BF-DCQO followed by a tightly-budgeted confidence-guided local refinement.

    The quantum stage is any existing BF-DCQO family member (basic, CVaR, adaptive,
    adaptive+CVaR).  The default local stage exactly optimizes only a small block
    of the least-confident variables ranked by ``|<Z_i>|`` from the final BF
    feedback.  This preserves the quantum shot budget and makes the classical
    contribution auditable through an explicit ``local_evaluations`` count.
    """

    def __init__(
        self,
        problem,
        *,
        base_variant: Literal["basic", "cvar", "adaptive", "adaptive_cvar"] = "basic",
        local_method: Literal["confidence", "random"] = "confidence",
        neighborhood_size: int = 4,
        local_rounds: int = 1,
        local_seed: int | None = None,
        cvar_alpha: float = 0.10,
        total_shot_budget: int | None = None,
        max_iterations: int = 4,
        shots: int = 128,
        min_shots: int | None = None,
        max_shots: int | None = None,
        initial_shots: int | None = None,
        n_trotter: int = 3,
        dt: float = 0.1,
        backend: str = "numpy",
        seed: int = 7,
        **solver_kwargs,
    ):
        self.problem = problem
        self.base_variant = str(base_variant)
        self.local_method = str(local_method)
        self.neighborhood_size = int(neighborhood_size)
        self.local_rounds = int(local_rounds)
        self.local_seed = int(seed + 100003 if local_seed is None else local_seed)
        common = dict(problem=problem, n_trotter=n_trotter, dt=dt, backend=backend, seed=seed)
        common.update(solver_kwargs)
        if self.base_variant == "basic":
            self.base_solver = BFDCQO(iterations=max_iterations, shots=shots, **common)
        elif self.base_variant == "cvar":
            self.base_solver = CVaRBFDCQO(
                iterations=max_iterations, shots=shots, cvar_alpha=cvar_alpha, **common
            )
        elif self.base_variant in ("adaptive", "adaptive_cvar"):
            if total_shot_budget is None:
                total_shot_budget = int(max_iterations * shots)
            if min_shots is None:
                min_shots = max(16, int(total_shot_budget) // (4 * max_iterations))
            if max_shots is None:
                max_shots = max(shots, 2 * int(total_shot_budget) // max_iterations)
            if initial_shots is None:
                initial_shots = shots
            self.base_solver = AdaptiveShotBFDCQO(
                total_shot_budget=int(total_shot_budget),
                max_iterations=max_iterations,
                min_shots=int(min_shots),
                max_shots=int(max_shots),
                initial_shots=int(initial_shots),
                cvar_alpha=cvar_alpha if self.base_variant == "adaptive_cvar" else None,
                **common,
            )
        else:
            raise ValueError(f"unknown base_variant {base_variant}")

    def solve(self) -> HybridBFDCQOResult:
        q = self.base_solver.solve()
        # Confidence should be estimated from all shots, even when the quantum
        # feedback rule itself uses a filtered CVaR subset. This avoids making
        # confidence artificially binary/saturated at small CVaR alpha and costs
        # no extra quantum samples.
        confidence_bias = q.history[-1].raw_bias
        if confidence_bias is None:
            confidence_bias = q.history[-1].bias
        confidence_bias = np.asarray(confidence_bias, dtype=float)
        if self.local_method == "confidence":
            local = confidence_guided_local_search(
                self.problem, q.best_solution, confidence_bias,
                neighborhood_size=self.neighborhood_size, rounds=self.local_rounds,
            )
        elif self.local_method == "random":
            local = random_block_local_search(
                self.problem, q.best_solution,
                neighborhood_size=self.neighborhood_size, rounds=self.local_rounds,
                seed=self.local_seed,
            )
        else:
            raise ValueError(f"unknown local_method {self.local_method}")
        # The local optimizer is monotone, so this is normally local.best_solution;
        # retain the explicit min for defensive robustness.
        if local.best_energy <= q.best_energy + 1e-12:
            best_solution, best_energy = local.best_solution, float(local.best_energy)
        else:
            best_solution, best_energy = q.best_solution, float(q.best_energy)
        return HybridBFDCQOResult(
            quantum_result=q,
            local_result=local,
            best_solution=best_solution,
            best_energy=best_energy,
            exact_ground_energy=float(q.exact_ground_energy),
            exact_ground_states=list(q.exact_ground_states),
            variant=f"hybrid:{self.base_variant}+{self.local_method}",
        )
