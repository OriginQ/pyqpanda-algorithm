"""PyBF-DCQO: QUBO/HUBO BF-DCQO family, applications, and benchmarks."""
from .problems import (
    IsingProblem, qubo_to_ising, PolynomialIsingProblem,
    binary_hubo_to_ising, binary_hubo_energy, MaxCutProblem,
)
from .solvers import (
    ExactSolver, ExactResult, BFDCQO, BFDCQOResult,
    CVaRBFDCQO, AdaptiveShotBFDCQO, AdaptiveBFDCQO,
    QAOASolver, QAOAResult, HybridBFDCQO, HybridBFDCQOResult,
)
from .applications import (
    WeightedClause, Max3SATProblem, QuadratizedQUBO, quadratize_cubic_hubo,
    max3sat_to_qubo, MWISProblem,
)
from .local_search import (
    LocalSearchStep, LocalSearchResult, uncertain_variable_order,
    block_local_search, confidence_guided_local_search, random_block_local_search,
)

__all__ = [
    "IsingProblem", "qubo_to_ising", "PolynomialIsingProblem", "binary_hubo_to_ising",
    "binary_hubo_energy", "MaxCutProblem", "ExactSolver", "ExactResult", "BFDCQO",
    "BFDCQOResult", "CVaRBFDCQO", "AdaptiveShotBFDCQO", "AdaptiveBFDCQO",
    "QAOASolver", "QAOAResult", "HybridBFDCQO", "HybridBFDCQOResult",
    "WeightedClause", "Max3SATProblem", "QuadratizedQUBO", "quadratize_cubic_hubo",
    "max3sat_to_qubo", "MWISProblem", "LocalSearchStep", "LocalSearchResult",
    "uncertain_variable_order", "block_local_search", "confidence_guided_local_search",
    "random_block_local_search",
    "BFDCQOConfig", "BFDCQOSolver",
]

from .api import BFDCQOConfig, BFDCQOSolver
