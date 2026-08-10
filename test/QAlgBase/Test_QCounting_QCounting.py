import pytest
import numpy as np
from pyqpanda_alg.QCounting import QCounting
from pyqpanda3.core import QCircuit, TOFFOLI, Z


def _states(n):
    return [format(i, '0{}b'.format(n)) for i in range(2 ** n)]


def _subset(n, m, seed):
    states = _states(n)
    index = list(range(2 ** n))
    for i in range(len(index) - 1, 0, -1):
        seed = (1103515245 * seed + 12345) % (2 ** 31)
        j = seed % (i + 1)
        index[i], index[j] = index[j], index[i]
    return [states[i] for i in index[:m]]


def _and_oracle(qubits):
    cir = QCircuit()
    cir << TOFFOLI(qubits[0], qubits[1], qubits[3])
    cir << Z(qubits[3])
    cir << TOFFOLI(qubits[0], qubits[1], qubits[3])
    return cir


class Test_QCounting_QCounting:

    def test_two_qubit_search_space(self):
        for m in range(5):
            count = QCounting(qnumber=2, mark_data=_states(2)[:m]).run()
            assert int(round(count)) == m, 'count {} for {} marked items'.format(count, m)

    def test_three_qubit_search_space(self):
        for m in range(9):
            count = QCounting(qnumber=3, mark_data=_states(3)[:m]).run()
            assert int(round(count)) == m, 'count {} for {} marked items'.format(count, m)

    def test_scattered_marked_states(self):
        for m in range(1, 8):
            marked = _subset(3, m, seed=m)
            count = QCounting(qnumber=3, mark_data=marked).run()
            assert int(round(count)) == m, 'count {} for {}'.format(count, marked)

    def test_four_qubit_search_space(self):
        for m in (1, 5, 11, 16):
            marked = _subset(4, m, seed=m)
            count = QCounting(qnumber=4, mark_data=marked, counting_qubits=6).run()
            assert int(round(count)) == m, 'count {} for {} marked items'.format(count, m)

    def test_no_marked_item_is_exact(self):
        count = QCounting(qnumber=3, mark_data=[]).run()
        assert count == pytest.approx(0.0, abs=1e-9)

    def test_all_items_marked_is_exact(self):
        count = QCounting(qnumber=3, mark_data=_states(3)).run()
        assert count == pytest.approx(8.0, abs=1e-9)

    def test_single_mark_data_string(self):
        count = QCounting(qnumber=3, mark_data='101').run()
        assert int(round(count)) == 1

    def test_flip_operator_with_ancilla(self):
        counter = QCounting(qnumber=3, flip_operator=_and_oracle, ancilla_qubits=1)
        count = counter.run()
        assert int(round(count)) == 2

    def test_theta_matches_count(self):
        counter = QCounting(qnumber=3, mark_data=_states(3)[:3])
        count = counter.run()
        assert 8 * np.sin(counter.theta / 2) ** 2 == pytest.approx(count, abs=1e-9)

    def test_larger_counting_register_is_closer(self):
        marked = _states(3)[:3]
        coarse = QCounting(qnumber=3, mark_data=marked, counting_qubits=5).run()
        fine = QCounting(qnumber=3, mark_data=marked, counting_qubits=7).run()
        assert abs(fine - 3) < abs(coarse - 3)

    def test_default_counting_qubits(self):
        assert QCounting(qnumber=3, mark_data='000').counting_qubits == 5

    def test_missing_oracle_raises(self):
        with pytest.raises(ValueError):
            QCounting(qnumber=3)

    def test_bad_qnumber_raises(self):
        with pytest.raises(ValueError):
            QCounting(qnumber=0, mark_data=[])

    def test_bad_mark_data_raises(self):
        with pytest.raises(ValueError):
            QCounting(qnumber=3, mark_data=['0101'])
        with pytest.raises(ValueError):
            QCounting(qnumber=3, mark_data=['012'])

    def test_bad_counting_qubits_raises(self):
        with pytest.raises(ValueError):
            QCounting(qnumber=3, mark_data='000', counting_qubits=0)
        with pytest.raises(ValueError):
            QCounting(qnumber=3, mark_data='000', counting_qubits=20)

    def test_bad_ancilla_qubits_raises(self):
        with pytest.raises(ValueError):
            QCounting(qnumber=3, mark_data='000', ancilla_qubits=-1)

    def test_cir_register_size_raises(self):
        counter = QCounting(qnumber=2, mark_data='11', counting_qubits=3)
        with pytest.raises(ValueError):
            counter.cir(q_search=[0], q_count=[1, 2, 3])
        with pytest.raises(ValueError):
            counter.cir(q_search=[0, 1], q_count=[2, 3])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
