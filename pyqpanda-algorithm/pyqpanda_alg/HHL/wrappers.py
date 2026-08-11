"""Compatibility wrappers for the legacy notebook HHL entry points.

The legacy notebooks under ``pyqpanda-algorithm/test/00-HHL`` called
the three names :func:`build_HHL_circuit`, :func:`expand_linear_equations`,
and :func:`HHL_solve_linear_equations` with the system matrix given as a
flat list and an integer as the third positional argument.  The old
package documented that integer as ``precision_cnt`` -- the number of
decimal digits the solution is accurate to -- and the wrappers keep
that single meaning: the count maps to the QPE phase-resolution target
``precision = 10 ** (-precision_cnt)`` of the stable solver, which
yields the phase register sizes the legacy package's documentation
reported (4 qubits for one decimal digit, 7 for two, ...).  A
``precision_cnt`` of 0 asks for integer-accuracy only and leaves the
solver at its one-qubit minimum; the phase register must be able to
resolve the system spectrum for the reciprocal rotation to invert it
faithfully, so coarse counts can return inaccurate directions on
ill-conditioned spectra -- the same caveat the legacy entry points
carried.

The wrappers are thin by design: they delegate to the stable
:class:`~pyqpanda_alg.HHL.HHL` surface (and to
:func:`~pyqpanda_alg.HHL.validation.normalize_linear_system`) and
implement no solving logic of their own.
"""

import numpy as np
from pyqpanda3.core import QProg

from .hhl import HHL
from .model import HHLConfig
from .validation import normalize_linear_system


def _legacy_precision(precision_cnt: int) -> float:
    """Phase-resolution target implied by the legacy decimal-digit count.

    ``precision_cnt`` digits after the decimal point mean a phase
    resolution target of ``10 ** (-precision_cnt)``; the stable solver
    turns that into a phase register of ``ceil(log2(10 ** precision_cnt))``
    qubits (4 for one digit, 7 for two, ...).  Negative counts are
    rejected before any task submission.
    """
    precision_cnt = int(precision_cnt)
    if precision_cnt < 0:
        raise ValueError(
            f"precision_cnt must be non-negative, got {precision_cnt}"
        )
    return 10.0 ** (-precision_cnt)


def build_HHL_circuit(matrix, vector, precision_cnt: int = 1) -> QProg:
    """Build the HHL circuit for a linear system (legacy entry point).

    ``matrix`` may be a 2-D array or a flat list whose length is a
    perfect square (the legacy flattened form); ``vector`` is the
    matching right-hand vector.  ``precision_cnt`` is the legacy
    decimal-digit precision count described in the module docstring.
    The circuit is synthesized by the stable
    :class:`~pyqpanda_alg.HHL.HHL` solver, so validation and padding
    rules are identical; the returned value is the measurement-free
    :class:`~pyqpanda3.core.QProg` the legacy notebooks printed.
    """
    solver = HHL(matrix, vector, precision=_legacy_precision(precision_cnt))
    return solver.build_circuit().program


def expand_linear_equations(matrix, vector):
    """Expand a linear system to the padded form the HHL circuit consumes.

    ``matrix`` and ``vector`` are validated and padded exactly as in
    :func:`~pyqpanda_alg.HHL.validation.normalize_linear_system`: the
    matrix is block-diagonally augmented ``[[A, 0], [0, I]]`` to the
    next power of two and the vector is zero-padded and normalized to
    unit norm, so the returned pair is what the circuit amplitude-encodes.
    The returned matrix and vector are read-only copies.
    """
    system = normalize_linear_system(matrix, vector, HHLConfig())
    return system.matrix, system.vector


def HHL_solve_linear_equations(matrix, vector, precision_cnt: int = 1) -> np.ndarray:
    """Solve a linear system on the local CPU backend (legacy entry point).

    ``matrix`` may be a 2-D array or a flat list whose length is a
    perfect square (the legacy flattened form); ``vector`` is the
    matching right-hand vector.  ``precision_cnt`` is the legacy
    decimal-digit precision count described in the module docstring.
    The solve is delegated to the stable
    :class:`~pyqpanda_alg.HHL.HHL` solver with
    ``precision = 10 ** (-precision_cnt)``, and the returned value is
    the reconstructed classical solution direction -- unit norm,
    truncated to the original unpadded dimension -- so the legacy
    callers' ``for key in result`` iteration keeps working over the
    solution entries.
    """
    solver = HHL(matrix, vector, precision=_legacy_precision(precision_cnt))
    return solver.run(reconstruct=True).classical_vector
