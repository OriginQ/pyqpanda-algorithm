"""Restored from ``test/legacy_disabled/QAlgBase/QAE_QAE.py``.

Runs on the execution-layer default backend (``LocalBackend``).  The
[0.2, 0.3] assertion window matches the value documented by ``QAE`` for
the RY(pi/3) + CX operator: ``_prob_from_counts`` measures theta (not
2*theta), so the standard example returns approximately 0.24.
"""

import math

import numpy as np
import pytest

from pyqpanda3.core import QCircuit, RY, X

from pyqpanda_alg.QAE import QAE


def create_cir_basic(qlist):
    """basic test circuit"""
    cir = QCircuit()
    cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
    return cir


class Test_QAE_QAE:

    def test_qae_basic_functionality(self):
        W = QAE(
            operator_in=create_cir_basic,
            qnumber=2,
            epsilon=0.01,
            res_index=[0, 1],
            target_state='11'
        ).run()

        assert W is not None, "amplitude estimation result should not be None"
        assert isinstance(W, (float, np.floating)), \
            f"amplitude should be numeric, got {type(W)}"
        assert 0.2 <= W <= 0.3, f"amplitude should be in [0.2, 0.3], got {W}"
