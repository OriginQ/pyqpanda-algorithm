"""Shor factorization package: stable small-scale Shor with classical preprocessing.

Publishes the stable Shor surface: the immutable :class:`ShorConfig`
and :class:`ShorResult`, :func:`classical_preprocess` which resolves
even, prime, and perfect-power moduli classically and marks the rest
as needing quantum order finding, :func:`recover_order` which turns a
measured phase sample into a candidate multiplicative order via
continued fractions, and the :class:`Shor` solver facade.  The
approved public surface is exactly ``__all__``; callers import from
``pyqpanda_alg.Shor`` and nothing deeper.
"""

from .classical import classical_preprocess, recover_order
from .model import NEEDS_QUANTUM, RESOLVED, PreprocessOutcome, ShorConfig, ShorResult
from .shor import Shor

__all__ = [
    "NEEDS_QUANTUM",
    "RESOLVED",
    "PreprocessOutcome",
    "Shor",
    "ShorConfig",
    "ShorResult",
    "classical_preprocess",
    "recover_order",
]
