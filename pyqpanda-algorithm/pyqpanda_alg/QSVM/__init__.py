'''
Quantum Kernel Algorithm
reference: `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_
'''

from .quantum_kernel_svm import QuantumKernel_vqnet
from .factorized_kernel import FactorizedQuantumKernel

__all__ = ['QuantumKernel_vqnet', 'FactorizedQuantumKernel']
