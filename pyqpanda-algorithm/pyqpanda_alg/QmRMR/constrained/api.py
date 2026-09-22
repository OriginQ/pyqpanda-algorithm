"""High-level QmRMR feature-selection API."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .problem import QmRMRProblem
from .solvers import (
    BitOrder,
    ConstrainedQAOAQmRMR,
    ExchangeAnsatzSelector,
    OptimizationResult,
    OptimizerName,
)


AnsatzName = Literal["constrained_qaoa", "exchange"]
ObjectiveMode = Literal["canonical_mrmr", "plus_relevance"]
OptimizationObjective = Literal["expectation", "cvar"]


class QmRMRFeatureSelection:
    """Feature-selection facade with constrained and exchange ansatz options.

    Parameters
    ----------
    quadratic:
        Symmetric redundancy coefficient matrix.  The symmetric part is used.
    relevance:
        Positive relevance vector.
    select_num:
        Exact number of features to select.
    ansatz:
        ``constrained_qaoa`` uses the Dicke plus XY construction; ``exchange``
        selects the exchange-ansatz baseline.
    objective_convention:
        ``canonical_mrmr`` minimizes redundancy minus relevance.
        ``plus_relevance`` uses a plus sign on the supplied relevance vector.

    The solver returns :class:`OptimizationResult` in ``optimize``/``run``.
    ``get_his_res`` returns a compact tuple for callers that prefer the
    example-style interface; its state keys are in feature-index order.
    """

    def __init__(
        self,
        quadratic,
        relevance,
        select_num: int,
        *,
        ansatz: AnsatzName = "constrained_qaoa",
        objective_convention: ObjectiveMode = "canonical_mrmr",
        layers: int = 1,
        initializer: str = "auto",
        initializer_strength: float = 1.0,
        ring: bool = False,
        optimization_objective: OptimizationObjective = "expectation",
        cvar_alpha: float = 0.25,
        bit_order: BitOrder = "qubit",
        optimizer: OptimizerName = "slsqp",
        maxiter: int = 60,
        seed: int = 0,
        restarts: int = 1,
        initial_scale: float = 0.5,
    ) -> None:
        if ansatz not in {"constrained_qaoa", "exchange"}:
            raise ValueError(
                "ansatz must be 'constrained_qaoa' or 'exchange'"
            )
        if objective_convention not in {"canonical_mrmr", "plus_relevance"}:
            raise ValueError(
                "objective_convention must be 'canonical_mrmr' or 'plus_relevance'"
            )
        if optimization_objective not in {"expectation", "cvar"}:
            raise ValueError(
                "optimization_objective must be 'expectation' or 'cvar'"
            )
        if ansatz == "exchange" and optimization_objective != "expectation":
            raise ValueError(
                "optimization_objective='cvar' requires ansatz='constrained_qaoa'"
            )
        if bit_order not in {"qubit", "feature"}:
            raise ValueError("bit_order must be 'qubit' or 'feature'")
        self.problem = QmRMRProblem.from_mrmr(
            quadratic,
            relevance,
            select_num,
            convention=objective_convention,
        )
        self.ansatz = ansatz
        self.objective_convention = objective_convention
        self.optimizer = optimizer
        self.maxiter = int(maxiter)
        self.seed = int(seed)
        self.restarts = int(restarts)
        self.initial_scale = float(initial_scale)
        self.optimization_objective = optimization_objective
        self.cvar_alpha = float(cvar_alpha)
        if ansatz == "constrained_qaoa":
            self.solver = ConstrainedQAOAQmRMR(
                self.problem,
                layers=layers,
                initializer=initializer,
                initializer_strength=initializer_strength,
                ring=ring,
                objective_mode=optimization_objective,
                cvar_alpha=cvar_alpha,
            )
        else:
            self.solver = ExchangeAnsatzSelector(
                self.problem,
                bit_order=bit_order,
            )
        self.last_result: OptimizationResult | None = None

    def optimize(
        self,
        initial_parameters: Sequence[float] | None = None,
        *,
        optimizer: OptimizerName | None = None,
        maxiter: int | None = None,
        seed: int | None = None,
        restarts: int | None = None,
        initial_scale: float | None = None,
    ) -> OptimizationResult:
        """Run the selected ansatz and retain its structured result."""

        options = {
            "optimizer": self.optimizer if optimizer is None else optimizer,
            "maxiter": self.maxiter if maxiter is None else int(maxiter),
            "seed": self.seed if seed is None else int(seed),
            "restarts": self.restarts if restarts is None else int(restarts),
        }
        if self.ansatz == "constrained_qaoa":
            options["initial_scale"] = (
                self.initial_scale if initial_scale is None else float(initial_scale)
            )
        result = self.solver.optimize(initial_parameters, **options)
        self.last_result = result
        return result

    def run(self, *args, **kwargs) -> OptimizationResult:
        """Alias for :meth:`optimize` for algorithm-library style callers."""

        return self.optimize(*args, **kwargs)

    def progressive_optimize(
        self,
        *,
        optimizer: OptimizerName | None = None,
        maxiter: int | None = None,
        seed: int | None = None,
        restarts: int | None = None,
        initial_scale: float | None = None,
    ) -> list[OptimizationResult]:
        """Run all QAOA depths up to ``layers`` using warm starts."""

        if self.ansatz != "constrained_qaoa":
            raise ValueError("progressive_optimize requires constrained_qaoa")
        options = {
            "optimizer": self.optimizer if optimizer is None else optimizer,
            "maxiter": self.maxiter if maxiter is None else int(maxiter),
            "seed": self.seed if seed is None else int(seed),
            "restarts": self.restarts if restarts is None else int(restarts),
            "initial_scale": self.initial_scale
            if initial_scale is None
            else float(initial_scale),
        }
        stages = self.solver.optimize_progressive(**options)
        if stages:
            self.last_result = stages[-1]
        return stages

    def get_his_res(
        self,
        initial_parameters: Sequence[float] | None = None,
        **kwargs,
    ) -> tuple[list[float], list[int], dict[str, float]]:
        """Return history, selected state, and top states in one tuple."""

        result = self.optimize(initial_parameters, **kwargs)
        if self.ansatz == "exchange":
            reverse = self.solver.bit_order == "qubit"
        else:
            reverse = True
        probabilities = {
            (key[::-1] if reverse else key): float(value)
            for key, value in result.probabilities.items()
            if (key[::-1] if reverse else key).count("1")
            == self.problem.select_num
        }
        top = dict(
            sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:10]
        )
        return result.history, list(result.selected_state), top


__all__ = ["QmRMRFeatureSelection"]

