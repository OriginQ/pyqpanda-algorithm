"""Problem representations supported by PyBF-DCQO."""
from .ising import IsingProblem, qubo_to_ising
from .polynomial import PolynomialIsingProblem, binary_hubo_to_ising, binary_hubo_energy
from .maxcut import MaxCutProblem

__all__ = [
    "IsingProblem",
    "qubo_to_ising",
    "PolynomialIsingProblem",
    "binary_hubo_to_ising",
    "binary_hubo_energy",
    "MaxCutProblem",
]
