"""Restored from ``test/legacy_disabled/QAOA/qaoa_problem_to_z_operator.py``.

Pure Python assertion of the symbolic-problem-to-Pauli-operator mapping.
"""

import pytest
import sympy as sp

from pyqpanda3.hamiltonian import PauliOperator

from pyqpanda_alg.QAOA import qaoa


class TestProblemToZOperator:

    def test_basic_functionality(self):
        vars = sp.symbols('x0:3')
        f = 2 * vars[0] * vars[1] + 3 * vars[2] - 1

        hamiltonian = qaoa.problem_to_z_operator(f)

        assert isinstance(hamiltonian, PauliOperator)
        assert str(hamiltonian) is not None
        assert len(str(hamiltonian)) > 0
