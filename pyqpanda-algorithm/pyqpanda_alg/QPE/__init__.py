'''
Quantum Phase Estimation (QPE)

QPE is a fundamental quantum algorithm that estimates the phase φ
of an eigenvalue e^(2πiφ) of a unitary operator U.  Given an
eigenstate |ψ⟩ such that U|ψ⟩ = e^(2πiφ)|ψ⟩, QPE outputs an
n-bit estimate of φ.

This module provides:

    - QPE : Standard Quantum Phase Estimation.
'''

from .qpe import QPE

__all__ = ['QPE']
