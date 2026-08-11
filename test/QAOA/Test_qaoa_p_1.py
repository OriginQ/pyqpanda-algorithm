"""Restored from ``test/legacy_disabled/QAOA/qaoa_p_1.py``.

Pure Python assertion of the ``p_1`` mixing-operator builder: each qubit
index yields a distinct operator.
"""

import pytest

from pyqpanda3.hamiltonian import PauliOperator

from pyqpanda_alg.QAOA import qaoa


class TestP1Interface:

    def test_p1_basic_functionality(self):
        operator_0 = qaoa.p_1(0)
        operator_1 = qaoa.p_1(1)
        operator_2 = qaoa.p_1(2)

        assert isinstance(operator_0, PauliOperator)
        assert isinstance(operator_1, PauliOperator)
        assert isinstance(operator_2, PauliOperator)
        assert str(operator_0) != str(operator_1)
        assert str(operator_1) != str(operator_2)
