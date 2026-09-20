import pytest
import sympy as sp
import numpy as np
from pyqpanda_alg.QUBO import QUBO


class Test_QUBO_function_value:

    def test_function_value_basic(self):
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        test0 = QUBO.QuadraticBinary(function)

        value = test0.function_value([0, 1, 0])
        value_rel = -0.5 * 0 * 1 - 0.7 * 0 * 1 + 0.9 * 1 * 0 + 1.3 * 0 - 1 - 0.5 * 0
        assert value_rel == value
        assert value is not None, "function_value应该返回非None结果"
        assert isinstance(value, (int, float, np.number)), f"函数值应该是数值类型，实际是{type(value)}"

    def test_binary_power_normalization(self):
        x0, x1 = sp.symbols('x0 x1')
        function = x0 ** 3 + 2 * x0 ** 2 * x1 - 4 * x1
        qubo = QUBO.QuadraticBinary(function)

        assert qubo.function_value([1, 1]) == -1.0
        assert qubo.function_value([0, 1]) == -4.0
        assert qubo.function_value([1, 0]) == 1.0

    def test_binary_normalization_accumulates_equivalent_terms(self):
        x0, x1 = sp.symbols('x0 x1')
        function = x0 ** 2 * x1 + 2 * x0 * x1 ** 3
        qubo = QUBO.QuadraticBinary(function)

        assert qubo.function_value([1, 1]) == 3.0
        assert qubo.function_value([1, 0]) == 0.0

    def test_rejects_non_quadratic_binary_support(self):
        x0, x1, x2 = sp.symbols('x0 x1 x2')

        with pytest.raises(ValueError, match="at most two distinct variables"):
            QUBO.QuadraticBinary(x0 * x1 * x2)

    def test_constant_only_dict_keeps_quadratic_matrix_shape(self):
        qubo = QUBO.QuadraticBinary({
            'quadratic': None,
            'linear': None,
            'constant': 3,
        })

        assert qubo.quadratic == [[0]]
        assert qubo.linear == [0]
        assert qubo.function_value([0]) == 3.0


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])