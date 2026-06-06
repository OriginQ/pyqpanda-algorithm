'''
Variational Quantum Eigensolver (VQE) Module.

Provides tools for finding the ground state energy of a quantum system
using hybrid quantum-classical optimization.

Core components:
    - VQESolver: main solver class
    - PauliHamiltonian: Hamiltonian representation as sum of Pauli terms
    - hardware_efficient_ansatz: commonly used parameterized circuit
    - measure_expectation: Pauli expectation value measurement

Author: Bai
'''

from .hamiltonian import PauliHamiltonian
from .ansatz import hardware_efficient_ansatz, ry_linear_ansatz
from .measurement import measure_expectation, compute_energy
from .vqe_solver import VQESolver

__all__ = [
    'VQESolver',
    'PauliHamiltonian',
    'hardware_efficient_ansatz',
    'ry_linear_ansatz',
    'measure_expectation',
    'compute_energy',
]
