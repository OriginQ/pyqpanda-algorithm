"""HHL linear-system solver package.

Publishes the stable HHL surface: the immutable :class:`HHLConfig`
carrying the ill-conditioning threshold and phase-estimation knobs,
:func:`normalize_linear_system` which validates and reversibly pads an
input system into an immutable :class:`NormalizedLinearSystem`, the
immutable :class:`HHLSolution` snapshot of a solved system, and the
:class:`HHL` solver facade which validates a system, synthesizes its
circuit, and solves it on an execution backend.  The approved public
surface is exactly ``__all__``; callers import from
``pyqpanda_alg.HHL`` and nothing deeper.
"""

from .hhl import HHL
from .model import HHLConfig, HHLSolution, NormalizedLinearSystem
from .validation import normalize_linear_system

__all__ = [
    "HHL",
    "HHLConfig",
    "HHLSolution",
    "NormalizedLinearSystem",
    "normalize_linear_system",
]
