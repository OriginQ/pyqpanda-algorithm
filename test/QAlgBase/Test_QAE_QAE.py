import numpy as np
from pyqpanda3.core import QCircuit, RY

from pyqpanda_alg.QAE import QAE


def single_qubit_state(qlist):
    cir = QCircuit()
    cir << RY(qlist[0], np.pi / 3)
    return cir


def asymmetric_two_qubit_state(qlist):
    cir = QCircuit()
    cir << RY(qlist[0], np.pi / 3)
    cir << RY(qlist[1], np.pi / 5)
    return cir


def test_qae_single_qubit_target_states():
    prob_one = QAE(
        operator_in=single_qubit_state,
        qnumber=1,
        epsilon=0.005,
        res_index=0,
        target_state="1",
    ).run()
    prob_zero = QAE(
        operator_in=single_qubit_state,
        qnumber=1,
        epsilon=0.005,
        res_index=0,
        target_state="0",
    ).run()

    assert np.isclose(prob_one, 0.25, atol=0.02)
    assert np.isclose(prob_zero, 0.75, atol=0.02)


def test_qae_target_state_follows_res_index_order():
    p0 = np.sin(np.pi / 6) ** 2
    p1 = np.sin(np.pi / 10) ** 2
    expected = {
        "00": (1 - p0) * (1 - p1),
        "01": (1 - p0) * p1,
        "10": p0 * (1 - p1),
        "11": p0 * p1,
    }

    for target_state, target_prob in expected.items():
        prob = QAE(
            operator_in=asymmetric_two_qubit_state,
            qnumber=2,
            epsilon=0.005,
            res_index=[0, 1],
            target_state=target_state,
        ).run()

        assert np.isclose(prob, target_prob, atol=0.02)


def test_qae_non_natural_res_index_order():
    p0 = np.sin(np.pi / 6) ** 2
    p1 = np.sin(np.pi / 10) ** 2

    prob = QAE(
        operator_in=asymmetric_two_qubit_state,
        qnumber=2,
        epsilon=0.005,
        res_index=[1, 0],
        target_state="01",
    ).run()

    assert np.isclose(prob, (1 - p1) * p0, atol=0.02)
