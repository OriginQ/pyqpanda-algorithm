'''
Quantum Kernel Algorithm
reference: `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_
'''

from .quantum_kernel_svm import QuantumKernel_vqnet

__all__ = ['QuantumKernel_vqnet']
from .non_hermitian_quantum_kernel import (
    NonHermitianQuantumKernel,
    PTPhaseAnalyzer,
    SupervisedGammaScheduler,
    GammaScheduler,
    symmetrize_and_psd,
)
__all__ += [
    'NonHermitianQuantumKernel',
    'PTPhaseAnalyzer',
    'SupervisedGammaScheduler',
    'GammaScheduler',
    'symmetrize_and_psd',
]