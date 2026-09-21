import pytest
from pyqpanda_alg.QCmp import qubit_comparator
from pyqpanda3.core import *


PREDICATES = {
    'g': lambda a, b: a > b,
    'geq': lambda a, b: a >= b,
    's': lambda a, b: a < b,
    'seq': lambda a, b: a <= b,
    'eq': lambda a, b: a == b,
    'neq': lambda a, b: a != b,
}

FUNCTIONS = ['g', 'geq', 's', 'seq', 'eq', 'neq']


class Test_comparator_qubit_comparator:

    def setup_method(self):
        self.machine = CPUQVM()

    def _prob_high(self, prog, qlist, q_cmp):
        for q in qlist:
            prog << I(q)
        self.machine.run(prog, 1000)
        return self.machine.result().get_prob_dict([q_cmp]).get('1', 0.0)

    def _basis_prob(self, n1, n2, a, b, function):
        # q_state_1 = 0..n1-1, q_state_2 = n1..n1+n2-1, ancilla after that,
        # lowest index is the LSB and the result qubit is the last ancilla
        q_state_1 = list(range(n1))
        q_state_2 = list(range(n1, n1 + n2))
        q_anc_cmp = list(range(n1 + n2, n1 + n2 + min(n1, n2)))
        q_cmp = q_anc_cmp[-1]

        prog = QProg()
        for j in range(n1):
            if (a >> j) & 1:
                prog << X(q_state_1[j])
        for j in range(n2):
            if (b >> j) & 1:
                prog << X(q_state_2[j])
        prog << qubit_comparator(q_state_1, q_state_2, q_anc_cmp, function=function)
        return self._prob_high(prog, q_state_1 + q_state_2 + q_anc_cmp, q_cmp)

    def _sweep(self, n1, n2, function):
        for a in range(2 ** n1):
            for b in range(2 ** n2):
                expected = 1.0 if PREDICATES[function](a, b) else 0.0
                prob_high = self._basis_prob(n1, n2, a, b, function)
                assert abs(prob_high - expected) < 1e-6, (
                    f"function={function}, n1={n1}, n2={n2}, a={a}, b={b}: "
                    f"expected {expected:.4f}, got {prob_high:.4f}"
                )

    def test_qubit_comparator_example_from_doc(self):
        prog = QProg()
        prog << H(0) << H(1)
        prog << X(3)

        cir = qubit_comparator([0, 1], [2, 3], [4, 5], function='g')
        prog << cir

        prob_high = self._prob_high(prog, [0, 1, 2, 3, 4, 5], 5)
        # uniform state over 0..3 compared with 2, only 3 is greater
        assert abs(prob_high - 0.25) < 1e-6, (
            f"expected 0.2500, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("function, expected", [
        ('g', 0.25),
        ('geq', 0.50),
        ('s', 0.50),
        ('seq', 0.75),
        ('eq', 0.25),
        ('neq', 0.75),
    ])
    def test_superposition_all_functions(self, function, expected):
        prog = QProg()
        prog << H(0) << H(1)
        prog << X(3)

        prog << qubit_comparator([0, 1], [2, 3], [4, 5], function=function)

        prob_high = self._prob_high(prog, [0, 1, 2, 3, 4, 5], 5)
        assert abs(prob_high - expected) < 1e-6, (
            f"function={function}: expected {expected:.4f}, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("function", FUNCTIONS)
    def test_equality_boundary(self, function):
        # a == b separates g from geq, s from seq and eq from neq
        expected = 1.0 if PREDICATES[function](2, 2) else 0.0
        prob_high = self._basis_prob(2, 2, 2, 2, function)
        assert abs(prob_high - expected) < 1e-6, (
            f"function={function}, a=b=2: expected {expected:.4f}, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("function", FUNCTIONS)
    def test_unequal_width_equality_boundary(self, function):
        # eq and neq are symmetric, a wider first register must not invert them
        expected = 1.0 if PREDICATES[function](2, 2) else 0.0
        prob_high = self._basis_prob(3, 2, 2, 2, function)
        assert abs(prob_high - expected) < 1e-6, (
            f"function={function}, n1=3, n2=2, a=b=2: "
            f"expected {expected:.4f}, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("function", FUNCTIONS)
    def test_all_basis_pairs_two_qubits(self, function):
        self._sweep(2, 2, function)

    @pytest.mark.parametrize("function", FUNCTIONS)
    def test_all_basis_pairs_three_qubits(self, function):
        self._sweep(3, 3, function)

    @pytest.mark.parametrize("function", FUNCTIONS)
    def test_all_basis_pairs_first_register_wider(self, function):
        self._sweep(3, 2, function)

    @pytest.mark.parametrize("function", FUNCTIONS)
    def test_all_basis_pairs_second_register_wider(self, function):
        self._sweep(2, 3, function)

    @pytest.mark.parametrize("function", FUNCTIONS)
    def test_all_basis_pairs_width_gap_two(self, function):
        self._sweep(4, 2, function)

    def test_unknown_function_raises(self):
        with pytest.raises(NameError):
            qubit_comparator([0, 1], [2, 3], [4, 5], function='lt')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
