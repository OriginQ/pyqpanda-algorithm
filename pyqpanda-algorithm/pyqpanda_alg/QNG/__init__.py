"""Quantum Natural Gradient and Fubini-Study metric utilities."""

from .qng import (
    MetricDiagnostics,
    NaturalGradientResult,
    NaturalGradientStep,
    block_diagonal_metric,
    fubini_study_metric,
    natural_gradient_step,
    quantum_natural_gradient,
    solve_natural_gradient,
)

__all__ = [
    "MetricDiagnostics",
    "NaturalGradientResult",
    "NaturalGradientStep",
    "block_diagonal_metric",
    "fubini_study_metric",
    "natural_gradient_step",
    "quantum_natural_gradient",
    "solve_natural_gradient",
]
