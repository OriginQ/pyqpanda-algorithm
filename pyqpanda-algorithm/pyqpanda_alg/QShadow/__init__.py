"""QShadow: observable-aware local-Pauli classical-shadow estimation.

QShadow combines independently implemented locally-biased measurement planning,
backend-neutral records, unbiased inverse-propensity estimation, simultaneous
empirical-Bernstein intervals, and explicit shot-budget control.  See README.md
for references, originality scope, and limitations.
"""

from .dataset import ShadowDataset
from .estimator import (
    BudgetReport,
    ShadowBudgetController,
    ShadowEstimate,
    empirical_bernstein_radius,
    estimate_observable,
    estimate_observables,
    observable_shadow_samples,
    term_shadow_samples,
)
from .observable import PauliObservable, PauliTerm, pauli_expectation
from .planner import (
    AXES,
    BasisOptimizationResult,
    MeasurementPlan,
    optimize_basis_probabilities,
    planning_loss,
    sample_measurement_plan,
    uniform_probabilities,
)
from .sampler import (
    PyQPandaRunner,
    build_pyqpanda_program,
    run_measurement_plan,
    simulate_pauli_measurements,
)

__all__ = [
    "AXES",
    "BasisOptimizationResult",
    "BudgetReport",
    "MeasurementPlan",
    "PauliObservable",
    "PauliTerm",
    "PyQPandaRunner",
    "ShadowBudgetController",
    "ShadowDataset",
    "ShadowEstimate",
    "build_pyqpanda_program",
    "empirical_bernstein_radius",
    "estimate_observable",
    "estimate_observables",
    "observable_shadow_samples",
    "optimize_basis_probabilities",
    "pauli_expectation",
    "planning_loss",
    "run_measurement_plan",
    "sample_measurement_plan",
    "simulate_pauli_measurements",
    "term_shadow_samples",
    "uniform_probabilities",
]
