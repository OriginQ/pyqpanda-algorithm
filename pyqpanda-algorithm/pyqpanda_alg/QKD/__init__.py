'''
Quantum key distribution (QKD) is a secure communication method that implements
a cryptographic protocol based on the laws of quantum mechanics, specifically
quantum entanglement, measurement-disturbance principle, and no-cloning theorem.
~ref: https://en.wikipedia.org/wiki/Quantum_key_distribution

Five classic protocols are provided, all built on real ``pyqpanda3`` circuits and
sharing the same sift / QBER-estimation pipeline:

    BB84    prepare-and-measure, four states in two bases
    B92     prepare-and-measure, two non-orthogonal states
    E91     entanglement-based, secured by the CHSH Bell inequality
    BBM92   entanglement-based twin of BB84 (EPR pairs)
    SARG04  four BB84 states, basis-encoded key, exclusion-based sifting

Example:
    >>> from pyqpanda_alg.QKD import BB84
    >>> key = BB84(seed=42).keygen(16)
    >>> len(key)
    16
'''

from .QKD import QKD, QKDResult, QKDInsecureError
from .B92 import B92
from .BB84 import BB84
from .BBM92 import BBM92
from .E91 import E91
from .SARG04 import SARG04

__all__ = [
  'QKD',
  'QKDResult',
  'QKDInsecureError',
  'B92',
  'BB84',
  'BBM92',
  'E91',
  'SARG04',
]
