# Licensed under the Apache License, Version 2.0.
"""Unified user-facing API for the BF-DCQO solver family.

The low-level solver classes remain public for advanced use.  Most users can
construct a :class:`BFDCQOSolver` with a :class:`BFDCQOConfig` and switch among
basic, CVaR, adaptive-shot, adaptive-CVaR, and hybrid variants without changing
application code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .problems import IsingProblem, PolynomialIsingProblem
from .solvers import (
    AdaptiveShotBFDCQO,
    BFDCQO,
    CVaRBFDCQO,
    HybridBFDCQO,
)

Variant = Literal["basic", "cvar", "adaptive", "adaptive_cvar", "hybrid"]
Backend = Literal["numpy", "pyqpanda"]


@dataclass(frozen=True)
class BFDCQOConfig:
    """Configuration shared by the BF-DCQO family.

    Parameters
    ----------
    variant:
        Solver variant. ``"basic"`` reproduces the baseline feedback rule;
        ``"cvar"`` filters bias updates to a low-energy sample fraction;
        ``"adaptive"`` reallocates a fixed shot budget across iterations;
        ``"adaptive_cvar"`` combines both; ``"hybrid"`` adds a small classical
        block refinement after a selected quantum variant.
    n_trotter, dt:
        Number of digitized counterdiabatic steps and time step size.
    iterations, shots:
        Fixed-iteration/fixed-shot settings used by basic and CVaR variants.
    total_shot_budget:
        Total quantum-sampling budget used by adaptive variants.  If omitted,
        ``iterations * shots`` is used.
    cvar_alpha:
        Fraction of lowest-energy measurements used by the CVaR feedback rule.
    base_variant:
        Quantum stage used by ``variant="hybrid"``.
    neighborhood_size, local_rounds, local_method:
        Controls the bounded classical refinement stage of Hybrid BF-DCQO.
    """

    variant: Variant = "basic"
    n_trotter: int = 3
    dt: float = 0.1
    iterations: int = 6
    shots: int = 1024
    total_shot_budget: int | None = None
    cvar_alpha: float = 0.1
    min_shots: int | None = None
    max_shots: int | None = None
    initial_shots: int | None = None
    early_stop: bool = False
    min_iterations: int = 3
    uncertainty_tol: float = 0.03
    bias_delta_tol: float = 0.02
    impulse_only: bool = False
    base_variant: Literal["basic", "cvar", "adaptive", "adaptive_cvar"] = "adaptive_cvar"
    local_method: Literal["confidence", "random"] = "confidence"
    neighborhood_size: int = 4
    local_rounds: int = 1


class BFDCQOSolver:
    """Facade that constructs and runs one member of the BF-DCQO family.

    Examples
    --------
    Basic solver::

        solver = BFDCQOSolver(problem, BFDCQOConfig(variant="basic"))
        result = solver.solve()

    Adaptive-CVaR solver::

        cfg = BFDCQOConfig(
            variant="adaptive_cvar",
            total_shot_budget=4096,
            cvar_alpha=0.1,
        )
        result = BFDCQOSolver(problem, cfg).solve()
    """

    def __init__(
        self,
        problem: IsingProblem | PolynomialIsingProblem,
        config: BFDCQOConfig | None = None,
        *,
        backend: Backend = "pyqpanda",
        seed: int = 7,
    ) -> None:
        self.problem = problem
        self.config = config or BFDCQOConfig()
        self.backend = backend
        self.seed = int(seed)
        self._solver = self._build_solver()

    @property
    def solver(self):
        """Return the concrete low-level solver selected by the configuration."""
        return self._solver

    def _adaptive_kwargs(self) -> dict:
        cfg = self.config
        budget = cfg.total_shot_budget
        if budget is None:
            budget = int(cfg.iterations * cfg.shots)
        min_shots = cfg.min_shots
        if min_shots is None:
            min_shots = max(16, budget // (4 * cfg.iterations))
        max_shots = cfg.max_shots
        if max_shots is None:
            max_shots = max(cfg.shots, 2 * budget // cfg.iterations)
        initial_shots = cfg.initial_shots if cfg.initial_shots is not None else cfg.shots
        return {
            "total_shot_budget": budget,
            "max_iterations": cfg.iterations,
            "min_shots": min_shots,
            "max_shots": max_shots,
            "initial_shots": initial_shots,
            "early_stop": cfg.early_stop,
            "min_iterations": cfg.min_iterations,
            "uncertainty_tol": cfg.uncertainty_tol,
            "bias_delta_tol": cfg.bias_delta_tol,
        }

    def _build_solver(self):
        cfg = self.config
        common = {
            "problem": self.problem,
            "n_trotter": cfg.n_trotter,
            "dt": cfg.dt,
            "backend": self.backend,
            "seed": self.seed,
            "impulse_only": cfg.impulse_only,
        }
        if cfg.variant == "basic":
            return BFDCQO(iterations=cfg.iterations, shots=cfg.shots, **common)
        if cfg.variant == "cvar":
            return CVaRBFDCQO(
                iterations=cfg.iterations,
                shots=cfg.shots,
                cvar_alpha=cfg.cvar_alpha,
                **common,
            )
        if cfg.variant in {"adaptive", "adaptive_cvar"}:
            adaptive = self._adaptive_kwargs()
            return AdaptiveShotBFDCQO(
                cvar_alpha=cfg.cvar_alpha if cfg.variant == "adaptive_cvar" else None,
                **adaptive,
                **common,
            )
        if cfg.variant == "hybrid":
            adaptive = self._adaptive_kwargs()
            extra = {"impulse_only": cfg.impulse_only}
            if cfg.base_variant in {"adaptive", "adaptive_cvar"}:
                extra.update(
                    early_stop=cfg.early_stop,
                    min_iterations=cfg.min_iterations,
                    uncertainty_tol=cfg.uncertainty_tol,
                    bias_delta_tol=cfg.bias_delta_tol,
                )
            return HybridBFDCQO(
                problem=self.problem,
                base_variant=cfg.base_variant,
                local_method=cfg.local_method,
                neighborhood_size=cfg.neighborhood_size,
                local_rounds=cfg.local_rounds,
                max_iterations=cfg.iterations,
                shots=cfg.shots,
                cvar_alpha=cfg.cvar_alpha,
                total_shot_budget=adaptive["total_shot_budget"],
                min_shots=adaptive["min_shots"],
                max_shots=adaptive["max_shots"],
                initial_shots=adaptive["initial_shots"],
                n_trotter=cfg.n_trotter,
                dt=cfg.dt,
                backend=self.backend,
                seed=self.seed,
                **extra,
            )
        raise ValueError(f"unknown BF-DCQO variant: {cfg.variant}")

    def solve(self):
        """Run the configured solver and return its result object."""
        return self._solver.solve()
