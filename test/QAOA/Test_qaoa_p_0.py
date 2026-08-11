"""Restored from ``test/legacy_disabled/QAOA/qaoa_p_0.py``.

Pure Python assertion of the ``p_0`` phase-separation operator builder.
"""

import pytest

from pyqpanda3.hamiltonian import PauliOperator

from pyqpanda_alg.QAOA import qaoa


class TestP0Interface:

    def test_p0_basic_functionality(self):
        operator_0 = qaoa.p_0(0)

        assert isinstance(operator_0, PauliOperator)
        assert str(operator_0) is not None
        assert len(str(operator_0)) > 0
