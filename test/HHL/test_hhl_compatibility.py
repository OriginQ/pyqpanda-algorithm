"""Compatibility tests for the legacy HHL notebook entry points.

The legacy notebooks under ``pyqpanda-algorithm/test/00-HHL`` called
the three names ``build_HHL_circuit``, ``expand_linear_equations``, and
``HHL_solve_linear_equations`` with the system matrix given as a flat
list and an integer as the third positional argument -- the number of
decimal digits the solution is accurate to.  The compatibility wrappers
keep those calls working on the stable HHL surface, delegating to it
without maintaining a second implementation.
"""

import numpy as np
from pyqpanda3.core import QProg

from pyqpanda_alg.HHL import (
    HHL_solve_linear_equations,
    build_HHL_circuit,
    expand_linear_equations,
)


def normalized(vector):
    return vector / np.linalg.norm(vector)


def test_legacy_solver_accepts_flat_matrix():
    result = HHL_solve_linear_equations([1.0, 0.0, 0.0, 2.0], [1.0, 1.0], 1)
    assert np.allclose(normalized(result), normalized([1.0, 0.5]), atol=0.1)


def test_legacy_solver_identity_two_by_two():
    # demo03 cell 2: A = I, b = [0.6, 0.8] -> x = [0.6, 0.8]
    result = HHL_solve_linear_equations([1.0, 0, 0, 1.0], [0.6, 0.8], 1)
    assert np.allclose(normalized(result), normalized([0.6, 0.8]), atol=0.05)


def test_legacy_solver_diagonal_four_by_four():
    # demo03 cell 4: diag([2, 3, 4, 5]) x = [2, 3, 4, 5] -> x = [1, 1, 1, 1]
    matrix = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5]
    result = HHL_solve_linear_equations(matrix, [2, 3, 4, 5], 1)
    assert np.allclose(normalized(result), normalized([1, 1, 1, 1]), atol=0.1)


def test_legacy_circuit_builder_returns_program():
    prog = build_HHL_circuit([1.0, 0, 0, 1.0], [0.6, 0.8], 1)
    assert isinstance(prog, QProg)


def test_legacy_expand_pads_three_by_three_to_four_by_four():
    matrix = [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0]
    padded_matrix, padded_vector = expand_linear_equations(matrix, [1.0, 1.0, 1.0])
    assert padded_matrix.shape == (4, 4)
    assert padded_vector.shape == (4,)
    assert np.allclose(padded_matrix[:3, :3], np.array(matrix).reshape(3, 3))
    assert np.allclose(padded_matrix[3, :3], 0.0)
    assert padded_matrix[3, 3] == 1.0  # identity padding block
    assert np.allclose(normalized(padded_vector[:3]), normalized([1.0, 1.0, 1.0]))
