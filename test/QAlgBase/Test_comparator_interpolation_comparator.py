import pytest
from pyqpanda_alg.QCmp import interpolation_comparator
from pyqpanda3.core import *


def interval_prob(state, value, function):
    # state a is treated as the unit interval [a - 0.5, a + 0.5]
    lo, hi = state - 0.5, state + 0.5
    if function == 'g':
        return hi - max(lo, min(hi, value))
    return min(hi, max(lo, value)) - lo


class Test_comparator_interpolation_comparator:

    def setup_method(self):
        self.machine = CPUQVM()

    def _basis_prob(self, n, state, value, function, reuse=True):
        # q_state = 0..n-1 with the lowest index as LSB, ancilla + flag = n..2n-1
        q_state = list(range(n))
        q_anc_cmp = list(range(n, 2 * n))

        prog = QProg()
        for j in range(n):
            prog << (X(q_state[j]) if (state >> j) & 1 else I(q_state[j]))
        for q in q_anc_cmp:
            prog << I(q)
        prog << interpolation_comparator(value, q_state, q_anc_cmp,
                                         function=function, reuse=reuse)

        self.machine.run(prog, 1000)
        return self.machine.result().get_prob_dict([q_anc_cmp[-1]]).get('1', 0.0)

    def test_interpolation_comparator_example_from_doc(self):
        value = 3.3
        prog = QProg()
        prog << X(0) << X(1) << I(2)
        prog << I(3) << I(4) << I(5)

        cir = interpolation_comparator(value, [0, 1, 2], [3, 4, 5], function='g', reuse=True)
        prog << cir

        self.machine.run(prog, 1000)
        prob_dict_result = self.machine.result().get_prob_dict([5])

        prob_high = prob_dict_result.get('1', 0.0)
        assert abs(prob_high - 0.2) < 1e-6, (
            f"expected 0.2000, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("value, state, function, expected", [
        (1.3, 1, 'g', 0.2),
        (1.3, 1, 's', 0.8),
        (2.7, 3, 'g', 0.8),
        (2.7, 3, 's', 0.2),
        (3.3, 3, 'g', 0.2),
        (3.3, 3, 's', 0.8),
        (5.25, 5, 'g', 0.25),
        (5.25, 5, 's', 0.75),
    ])
    def test_interpolated_probability(self, value, state, function, expected):
        # inside the interval of the nearest integer the flag is a partial probability
        prob_high = self._basis_prob(3, state, value, function)
        assert abs(prob_high - expected) < 1e-6, (
            f"value={value}, state={state}, function={function}: "
            f"expected {expected:.4f}, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("value, state, function", [
        (1.3, 0, 'g'), (1.3, 2, 'g'), (1.3, 7, 'g'),
        (1.3, 0, 's'), (1.3, 2, 's'), (1.3, 7, 's'),
        (2.7, 2, 'g'), (2.7, 4, 'g'), (2.7, 2, 's'), (2.7, 4, 's'),
    ])
    def test_non_integer_value_is_sharp_outside_its_interval(self, value, state, function):
        expected = 1.0 if (state > value if function == 'g' else state < value) else 0.0
        prob_high = self._basis_prob(3, state, value, function)
        assert abs(prob_high - expected) < 1e-6, (
            f"value={value}, state={state}, function={function}: "
            f"expected {expected:.4f}, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("function", ['g', 's'])
    def test_integer_value_matches_classical_predicate(self, n, function):
        for value in range(2 ** n):
            for state in range(2 ** n):
                if state == value:
                    expected = 0.5
                elif function == 'g':
                    expected = float(state > value)
                else:
                    expected = float(state < value)
                prob_high = self._basis_prob(n, state, float(value), function)
                assert abs(prob_high - expected) < 1e-6, (
                    f"n={n}, value={value}, state={state}, function={function}: "
                    f"expected {expected:.4f}, got {prob_high:.4f}"
                )

    @pytest.mark.parametrize("function", ['g', 's'])
    def test_state_equal_to_integer_value_is_half(self, function):
        prob_high = self._basis_prob(3, 5, 5.0, function)
        assert abs(prob_high - 0.5) < 1e-6, (
            f"function={function}: expected 0.5000, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("value", [4.0, 5.0, 8.0, 100.0])
    @pytest.mark.parametrize("function", ['g', 's'])
    def test_value_above_register_range(self, value, function):
        # every state of a 2-qubit register is below the compared value
        expected = 0.0 if function == 'g' else 1.0
        for state in range(4):
            prob_high = self._basis_prob(2, state, value, function)
            assert abs(prob_high - expected) < 1e-6, (
                f"value={value}, state={state}, function={function}: "
                f"expected {expected:.4f}, got {prob_high:.4f}"
            )

    @pytest.mark.parametrize("function", ['g', 's'])
    def test_value_at_upper_edge_of_register_range(self, function):
        # 3.5 is the top edge of the interval of the largest 2-qubit state
        expected = 0.0 if function == 'g' else 1.0
        for state in range(4):
            prob_high = self._basis_prob(2, state, 3.5, function)
            assert abs(prob_high - expected) < 1e-6, (
                f"state={state}, function={function}: "
                f"expected {expected:.4f}, got {prob_high:.4f}"
            )

    @pytest.mark.parametrize("function", ['g', 's'])
    def test_value_at_lower_edge_of_register_range(self, function):
        for state in range(4):
            expected = interval_prob(state, 0.0, function)
            prob_high = self._basis_prob(2, state, 0.0, function)
            assert abs(prob_high - expected) < 1e-6, (
                f"state={state}, function={function}: "
                f"expected {expected:.4f}, got {prob_high:.4f}"
            )

    @pytest.mark.parametrize("function", ['g', 's'])
    def test_reuse_does_not_change_flag(self, function):
        for state in range(4):
            without = self._basis_prob(2, state, 1.4, function, reuse=False)
            with_reuse = self._basis_prob(2, state, 1.4, function, reuse=True)
            assert abs(without - with_reuse) < 1e-6, (
                f"state={state}, function={function}: "
                f"reuse=False gave {without:.4f}, reuse=True gave {with_reuse:.4f}"
            )

    def test_superposition_state_register_is_undisturbed(self):
        prog = QProg()
        for q in range(3):
            prog << H(q)
        for q in range(3, 6):
            prog << I(q)
        prog << interpolation_comparator(2.4, [0, 1, 2], [3, 4, 5], function='g', reuse=True)

        self.machine.run(prog, 1000)
        prob_dict_result = self.machine.result().get_prob_dict([0, 1, 2])

        assert len(prob_dict_result) == 8
        for key, prob in prob_dict_result.items():
            assert abs(prob - 0.125) < 1e-6, (
                f"state {key}: expected 0.1250, got {prob:.4f}"
            )

    def test_superposition_flag_is_the_mean_of_the_basis_results(self):
        prog = QProg()
        for q in range(2):
            prog << H(q)
        for q in range(2, 4):
            prog << I(q)
        prog << interpolation_comparator(1.4, [0, 1], [2, 3], function='g', reuse=True)

        self.machine.run(prog, 1000)
        prob_high = self.machine.result().get_prob_dict([3]).get('1', 0.0)

        expected = sum(interval_prob(s, 1.4, 'g') for s in range(4)) / 4
        assert abs(prob_high - expected) < 1e-6, (
            f"expected {expected:.4f}, got {prob_high:.4f}"
        )

    @pytest.mark.parametrize("value", [-0.1, -1.0, -4.0])
    def test_negative_value_raises(self, value):
        with pytest.raises(ValueError):
            interpolation_comparator(value, [0, 1], [2, 3], function='g')

    @pytest.mark.parametrize("function", ['geq', 'seq', 'eq', 'neq', ''])
    def test_unknown_function_raises(self, function):
        with pytest.raises(NameError):
            interpolation_comparator(1.5, [0, 1], [2, 3], function=function)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
