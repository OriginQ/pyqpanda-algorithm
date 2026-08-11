"""Restored from ``test/legacy_disabled/QAlgBase/QAE_IQAE.py``.

Runs on the execution-layer default backend (``LocalBackend``).  The
[0.2, 0.3] assertion window matches the value documented by ``QAE`` for
the RY(pi/3) + CX operator (``sin(theta)**2`` with theta estimated from
the ancilla register, approximately 0.24-0.26).
"""

import numpy as np
import pytest

from pyqpanda3.core import QCircuit, RY, X

from pyqpanda_alg.QAE.QAE import IQAE


class Test_QAE_IQAE:

    def create_cir_basic(self, qlist):
        cir = QCircuit()
        cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
        return cir

    def test_iqae_basic_functionality(self):
        W = IQAE(
            operator_in=self.create_cir_basic,
            qnumber=2,
            epsilon=0.01,
            res_index=-1
        ).run()

        assert W is not None, "amplitude estimation result should not be None"
        assert isinstance(W, (float, np.floating)), \
            f"amplitude should be numeric, got {type(W)}"
        assert 0.2 <= W <= 0.3, f"amplitude should be in [0.2, 0.3], got {W}"
