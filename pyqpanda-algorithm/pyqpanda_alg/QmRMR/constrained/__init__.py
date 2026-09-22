"""Constraint-preserving QAOA extensions for QmRMR.

The subpackage adds a fixed-cardinality Dicke-state initializer and a
number-preserving XY mixer while leaving the original ``Feature_Selection``
interface untouched.
"""

from .api import QmRMRFeatureSelection
from .baselines import GreedySolution, greedy_forward_selection
from .problem import ExactSolution, ObjectiveConvention, QmRMRProblem, cvar_from_probabilities
from .solvers import ConstrainedQAOAQmRMR, ExchangeAnsatzSelector, OptimizationResult

__all__ = [
    "ConstrainedQAOAQmRMR",
    "ExactSolution",
    "GreedySolution",
    "ExchangeAnsatzSelector",
    "OptimizationResult",
    "ObjectiveConvention",
    "QmRMRFeatureSelection",
    "QmRMRProblem",
    "cvar_from_probabilities",
    "greedy_forward_selection",
]

