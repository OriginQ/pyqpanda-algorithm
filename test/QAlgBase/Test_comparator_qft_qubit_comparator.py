import pytest
from pyqpanda_alg.QCmp import qft_qubit_comparator
from pyqpanda3.core import *


PREDICATES = {
    'g': lambda a, b: a > b,
    'geq': lambda a, b: a >= b,
    's': lambda a, b: a < b,
    'seq': lambda a, b: a <= b,
}


class Test_comparator_qft_qubit_comparator:

    def setup_method(self):
        self.machine = CPUQVM()

    def _prob_high(self, prog, q_cmp):
        self.machine.run(prog, 1000)
        return self.machine.result().get_prob_dict([q_cmp]).get('1', 0.0)

    def _basis_prob(self, n, a, b, function):
        # q_state_1 = 0..n-1, q_state_2 = n..2n-1, q_cmp = 2n, lowest index is LSB
        q_state_1 = list(range(n))
        q_state_2 = list(range(n, 2 * n))
        q_cmp = 2 * n

        prog = QProg()
        for j in range(n):
            if (a >> j) & 1:
                prog << X(q_state_1[j])
            if (b >> j) & 1:
                prog << X(q_state_2[j])
        prog << qft_qubit_comparator(q_state_1, q_state_2, [q_cmp], function=function)
        return self._prob_high(prog, q_cmp)

    def test_qft_qubit_comparator_example_from_doc(self):
        prog = QProg()
        prog << H(0) << H(1)
        prog << X(3)

        cir = qft_qubit_comparator([0, 1], [2, 3], [4], function='g')
        prog << cir

        prob_high = self._prob_high(prog, 4)
        # uniform state over 0..3 compared with 2, only 3 is greater
        assert abs(prob_high - 0.25) < 1e-6, (
            f"expected 0.2500, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("function, expected", [
        ('g', 0.25),
        ('geq', 0.50),
        ('s', 0.50),
        ('seq', 0.75),
    ])
    def test_superposition_all_functions(self, function, expected):
        prog = QProg()
        prog << H(0) << H(1)
        prog << X(3)

        prog << qft_qubit_comparator([0, 1], [2, 3], [4], function=function)

        prob_high = self._prob_high(prog, 4)
        assert abs(prob_high - expected) < 1e-6, (
            f"function={function}: expected {expected:.4f}, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("function", ['g', 'geq', 's', 'seq'])
    def test_equality_boundary(self, function):
        # a == b separates g from geq and s from seq
        expected = 1.0 if PREDICATES[function](2, 2) else 0.0
        prob_high = self._basis_prob(2, 2, 2, function)
        assert abs(prob_high - expected) < 1e-6, (
            f"function={function}, a=b=2: expected {expected:.4f}, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("function", ['g', 'geq', 's', 'seq'])
    def test_all_basis_pairs_two_qubits(self, function):
        for a in range(4):
            for b in range(4):
                expected = 1.0 if PREDICATES[function](a, b) else 0.0
                prob_high = self._basis_prob(2, a, b, function)
                assert abs(prob_high - expected) < 1e-6, (
                    f"function={function}, a={a}, b={b}: "
                    f"expected {expected:.4f}, got {prob_high:.4f}"
                )

    @pytest.mark.parametrize("function", ['g', 'geq', 's', 'seq'])
    def test_all_basis_pairs_three_qubits(self, function):
        for a in range(8):
            for b in range(8):
                expected = 1.0 if PREDICATES[function](a, b) else 0.0
                prob_high = self._basis_prob(3, a, b, function)
                assert abs(prob_high - expected) < 1e-6, (
                    f"function={function}, a={a}, b={b}: "
                    f"expected {expected:.4f}, got {prob_high:.4f}"
                )

    def test_state_registers_restored(self):
        prog = QProg()
        prog << H(0) << H(1) << H(2) << X(3)
        prog << qft_qubit_comparator([0, 1], [2, 3], [4], function='geq')

        self.machine.run(prog, 1000)
        prob_dict = self.machine.result().get_prob_dict([0, 1, 2, 3])
        for state, prob in prob_dict.items():
            expected = 0.125 if state[0] == '1' else 0.0
            assert abs(prob - expected) < 1e-6, (
                f"state {state}: expected {expected:.4f}, got {prob:.4f}"
            )

    def test_unknown_function_raises(self):
        with pytest.raises(NameError):
            qft_qubit_comparator([0, 1], [2, 3], [4], function='eq')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
