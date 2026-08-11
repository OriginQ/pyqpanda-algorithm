"""HHL linear-system solver package.

Publishes the stable HHL surface: the immutable :class:`HHLConfig`
carrying the ill-conditioning threshold and phase-estimation knobs,
:func:`normalize_linear_system` which validates and reversibly pads an
input system into an immutable :class:`NormalizedLinearSystem`, the
immutable :class:`HHLSolution` snapshot of a solved system, the
:class:`HHL` solver facade which validates a system, synthesizes its
circuit, and solves it on an execution backend, the low-level
:func:`build_hhl_circuit` and :func:`estimate_hhl_resources` entry
points, and the legacy notebook compatibility wrappers
:func:`build_HHL_circuit`, :func:`expand_linear_equations`, and
:func:`HHL_solve_linear_equations` which delegate to the stable
surface.  The approved public surface is exactly ``__all__``; callers
import from ``pyqpanda_alg.HHL`` and nothing deeper.
"""

from .circuit import build_hhl_circuit
from .hhl import HHL
from .model import HHLConfig, HHLSolution, NormalizedLinearSystem
from .resources import estimate_hhl_resources
from .validation import normalize_linear_system
from .wrappers import HHL_solve_linear_equations, build_HHL_circuit, expand_linear_equations

__all__ = [
    "HHL",
    "HHLConfig",
    "HHLSolution",
    "NormalizedLinearSystem",
    "build_HHL_circuit",
    "build_hhl_circuit",
    "estimate_hhl_resources",
    "expand_linear_equations",
    "HHL_solve_linear_equations",
    "normalize_linear_system",
]
