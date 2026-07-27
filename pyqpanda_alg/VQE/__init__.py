'''
Variational Quantum Eigensolver (VQE)

VQE is a hybrid quantum-classical algorithm designed to find the
ground-state energy of a given Hamiltonian. It uses a parameterized
quantum circuit (ansatz) to prepare trial states, measures the
expectation value of the Hamiltonian, and iteratively improves the
parameters via a classical optimizer.

The VQE module provides:

    - VQE : Main class for running the VQE algorithm.
'''

from .vqe import VQE

__all__ = ['VQE']
