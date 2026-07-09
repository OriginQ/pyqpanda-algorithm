import numpy as np
from pyqpanda3.core import QCircuit, RY

from pyqpanda_alg.QAE import IQAE


def asymmetric_two_qubit_state(qlist):
    cir = QCircuit()
    cir << RY(qlist[0], np.pi / 5)
    cir << RY(qlist[1], np.pi / 3)
    return cir


def test_iqae_estimates_non_last_result_qubit():
    prob = IQAE(
        operator_in=asymmetric_two_qubit_state,
        qnumber=2,
        epsilon=0.005,
        res_index=0,
    ).run()

    assert np.isclose(prob, np.sin(np.pi / 10) ** 2, atol=0.08)


def test_iqae_estimates_last_result_qubit():
    prob = IQAE(
        operator_in=asymmetric_two_qubit_state,
        qnumber=2,
        epsilon=0.005,
        res_index=1,
    ).run()

    assert np.isclose(prob, np.sin(np.pi / 6) ** 2, atol=0.08)


def test_iqae_negative_index_keeps_last_qubit_default():
    prob = IQAE(
        operator_in=asymmetric_two_qubit_state,
        qnumber=2,
        epsilon=0.005,
        res_index=-1,
    ).run()

    assert np.isclose(prob, np.sin(np.pi / 6) ** 2, atol=0.08)
