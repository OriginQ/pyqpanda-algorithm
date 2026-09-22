"""Differentiable quantum-circuit solvers for small ODE benchmarks.

The public API keeps the state-vector solver, benchmark definitions, and
diagnostic helpers together while leaving cloud and hardware execution outside
the module.  NumPy is the dependency-light default; the optional ``pyqpanda3``
backend uses the local CPU state-vector simulator.
"""

from .differentiable_ode_solver import (
    ExperimentConfig,
    ODECase,
    StatevectorDQC,
    TorchStatevectorDQC,
    make_cases,
    plot_results,
    residual_loss,
    run_experiments,
    train_case,
)

__all__ = [
    "ExperimentConfig",
    "ODECase",
    "StatevectorDQC",
    "TorchStatevectorDQC",
    "make_cases",
    "plot_results",
    "residual_loss",
    "run_experiments",
    "train_case",
]

