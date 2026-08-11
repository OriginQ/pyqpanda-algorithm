"""Hamiltonian-level variational quantum eigensolver.

Publishes the stable VQE surface: the :class:`VQE` solver (local
energy evaluation and classical optimization over the built-in
hardware-efficient ansatz), the immutable :class:`VQEConfig` carrying
the classical optimization and sampling knobs, and the immutable
:class:`VQEResult` snapshot of a completed or interrupted run.  The
approved public surface is exactly ``__all__``; callers import from
``pyqpanda_alg.VQE`` and nothing deeper.
"""

from .config import VQEConfig
from .result import VQEResult
from .vqe import VQE

__all__ = ["VQEConfig", "VQEResult", "VQE"]
