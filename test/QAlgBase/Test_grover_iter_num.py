import pytest
import math
import numpy as np
from pyqpanda_alg.Grover import iter_num


def _success_probability(q_num, sol_num, iternum):
    """给定迭代次数下 Grover 搜索的成功概率。"""
    theta = 2 * math.asin(math.sqrt(sol_num / 2 ** q_num))
    return math.sin((2 * iternum + 1) * theta / 2) ** 2


class Test_grover_iter_num:

    def test_iter_num_basic_case(self):
        q_num = 3
        sol_num = 2

        # 执行计算
        result = iter_num(q_num=q_num, sol_num=sol_num)

        assert isinstance(result, int), "迭代次数应该是整数类型"
        assert result >= 0, "迭代次数应该大于等于0"
        N = 2 ** q_num
        M = sol_num
        theoretical_r = round((math.pi / 4) * math.sqrt(N / M))
        assert abs(result - theoretical_r) <= 1, f"迭代次数 {result} 与理论值 {theoretical_r} 差异过大"

    def test_iter_num_is_the_first_maximum(self):
        """返回值应当正好是成功概率第一个极大值处的迭代次数。"""
        for q_num in range(1, 11):
            for sol_num in range(1, 2 ** q_num + 1):
                result = iter_num(q_num=q_num, sol_num=sol_num)
                theta = 2 * math.asin(math.sqrt(sol_num / 2 ** q_num))
                expected = max(0, int(round(math.pi / (2 * theta) - 0.5)))
                assert result == expected, \
                    f"q_num={q_num}, sol_num={sol_num}: 期望 {expected}，实际 {result}"

    def test_iter_num_no_better_neighbour(self):
        """相邻迭代次数的成功概率都不应优于返回值处的成功概率。"""
        for q_num in range(1, 11):
            for sol_num in range(1, 2 ** q_num):
                best = iter_num(q_num=q_num, sol_num=sol_num)
                p_best = _success_probability(q_num, sol_num, best)
                for neighbour in (best - 1, best + 1):
                    if neighbour < 0:
                        continue
                    p_neighbour = _success_probability(q_num, sol_num, neighbour)
                    assert p_best >= p_neighbour - 1e-12, \
                        (f"q_num={q_num}, sol_num={sol_num}: 迭代 {neighbour} 次的成功概率 "
                         f"{p_neighbour:.6f} 高于返回值 {best} 次的 {p_best:.6f}")

    def test_iter_num_dense_solution_set(self):
        """解集较大时不应使用小角度近似，否则会多迭代一次并显著降低成功概率。"""
        # N = 8192, M = 5053 时，小角度近似给出 1，而最优值为 0。
        q_num, sol_num = 13, 5053
        result = iter_num(q_num=q_num, sol_num=sol_num)
        assert result == 0, f"稠密解集下最优迭代次数应为 0，实际为 {result}"
        assert _success_probability(q_num, sol_num, 0) > \
            _success_probability(q_num, sol_num, 1), "0 次迭代的成功概率应高于 1 次"

    def test_iter_num_single_solution_matches_classic_formula(self):
        """单解且解集稀疏时，应与经典公式 floor(pi/4 * sqrt(N)) 一致。"""
        for q_num in range(4, 15):
            expected = int(np.floor(np.pi / 4 * np.sqrt(2 ** q_num)))
            assert iter_num(q_num=q_num, sol_num=1) == expected, \
                f"q_num={q_num} 时单解情形应与经典公式一致"

    @pytest.mark.parametrize("q_num, sol_num", [(3, 0), (3, -1), (3, 9), (0, 2)])
    def test_iter_num_rejects_invalid_solution_count(self, q_num, sol_num):
        """解的数量超出 [1, 2 ** q_num] 时应抛出 ValueError 而不是返回错误结果。"""
        with pytest.raises(ValueError):
            iter_num(q_num=q_num, sol_num=sol_num)

    def test_iter_num_rejects_negative_qubit_number(self):
        with pytest.raises(ValueError):
            iter_num(q_num=-1, sol_num=1)


if __name__ == "__main__":
    # 直接运行所有测试
    pytest.main([__file__, "-v"])
