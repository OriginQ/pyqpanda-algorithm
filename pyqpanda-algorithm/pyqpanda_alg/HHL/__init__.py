"""HHL linear-system solver package.

Publishes the stable HHL surface: the immutable :class:`HHLConfig`
carrying the ill-conditioning threshold and phase-estimation knobs,
:func:`normalize_linear_system` which validates and reversibly pads an
input system into an immutable :class:`NormalizedLinearSystem`, and
the immutable :class:`HHLSolution` snapshot of a solved system.  The
approved public surface is exactly ``__all__``; callers import from
``pyqpanda_alg.HHL`` and nothing deeper.
"""

from .model import HHLConfig, HHLSolution, NormalizedLinearSystem
from .validation import normalize_linear_system

__all__ = [
    "HHLConfig",
    "HHLSolution",
    "NormalizedLinearSystem",
    "normalize_linear_system",
]
