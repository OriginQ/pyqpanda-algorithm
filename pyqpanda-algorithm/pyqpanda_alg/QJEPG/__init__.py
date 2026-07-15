'''
Adapted implementation of the essay "Quantum JPEG" (arXiv:2306.09323) from the official repo,
and the inversed form (i.e. unQJPEG) as we originally proposed :)

The QJPEG algorithm uses the quantum Fourier transform to discard the high spatial-frequency
qubits of an image, downsampling it to a lower resolution. This allows one to capture,
compress, and send images even with limited quantum resources for storage and communication.
~ref: https://arxiv.org/abs/2306.09323
~ref: https://github.com/simoneroncallo/quantum-jpeg
'''

from .QJPEG import QJPEG
from .unQJPEG import unQJPEG

__all__ = [
  'QJPEG',
  'unQJPEG',
]
