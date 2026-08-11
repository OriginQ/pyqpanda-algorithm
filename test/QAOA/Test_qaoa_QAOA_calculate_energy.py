"""Restored from ``test/legacy_disabled/QAOA/qaoa_QAOA_calculate_energy.py``.

Pure Python assertion of ``QAOA.calculate_energy`` against the symbolic
objective; needs no quantum backend.
"""

import pytest
import sympy as sp

from pyqpanda_alg.QAOA.qaoa import QAOA


class TestCalculateEnergy:

    def test_basic_functionality_with_symbolic_problem(self):
        # f = 2*x0*x1 + 3*x2 - 1
        vars = sp.symbols('x0:3')
        f = 2 * vars[0] * vars[1] + 3 * vars[2] - 1

        qaoa_f = QAOA(f)

        test_cases = [
            ([1, 0, 0], 2 * 1 * 0 + 3 * 0 - 1),  # f(1,0,0) = -1
            ([0, 1, 1], 2 * 0 * 1 + 3 * 1 - 1),  # f(0,1,1) = 2
            ([1, 1, 0], 2 * 1 * 1 + 3 * 0 - 1),  # f(1,1,0) = 1
            ([0, 0, 0], 2 * 0 * 0 + 3 * 0 - 1),  # f(0,0,0) = -1
            ([1, 1, 1], 2 * 1 * 1 + 3 * 1 - 1),  # f(1,1,1) = 4
        ]

        for solution, expected_energy in test_cases:
            calculated_energy = qaoa_f.calculate_energy(solution)
            assert abs(calculated_energy - expected_energy) < 1e-10, \
                f"solve {solution} error : expect {expected_energy}, " \
                f"get {calculated_energy}"
