# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the quantum risk analysis application.

Every quantum estimate is checked against the exact value computed classically
from the same distribution, so the tests verify the workflow rather than merely
exercising it.
"""

import itertools

import numpy as np
import pytest

from quantum_risk import (
    LossDistribution,
    QuantumRiskAnalyzer,
    classical_monte_carlo_samples,
    recommended_epsilon,
)

SIMPLE_PROBABILITIES = [0.30, 0.25, 0.18, 0.12, 0.08, 0.04, 0.02, 0.01]
PORTFOLIO_DEFAULTS = [0.10, 0.20, 0.05, 0.30]
PORTFOLIO_EXPOSURES = [1, 2, 1, 3]


class Test_LossDistribution:

    def test_normalises_and_pads_to_power_of_two(self):
        dist = LossDistribution([1.0, 1.0, 1.0])
        assert dist.probabilities.size == 4, "概率向量应补齐到 2 的幂"
        assert dist.num_qubits == 2, "3 个取值需要 2 个量子比特"
        assert abs(dist.probabilities.sum() - 1.0) < 1e-12, "概率应归一化"
        assert dist.probabilities[3] == 0.0, "补齐位置的概率应为 0"

    def test_tail_probability_matches_direct_sum(self):
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        probabilities = np.asarray(SIMPLE_PROBABILITIES) / sum(SIMPLE_PROBABILITIES)
        for threshold in range(len(SIMPLE_PROBABILITIES) + 1):
            expected = probabilities[threshold:].sum()
            assert abs(dist.tail_probability(threshold) - expected) < 1e-12, \
                f"阈值 {threshold} 的尾部概率不正确"

    def test_expected_shortfall_matches_conditional_mean(self):
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        for confidence in (0.5, 0.8, 0.9, 0.95):
            var = dist.value_at_risk(confidence)
            mask = dist.levels >= var
            expected = (dist.levels[mask] * dist.probabilities[mask]).sum() / \
                dist.tail_probability(var)
            assert abs(dist.expected_shortfall(confidence) - expected) < 1e-12, \
                f"置信度 {confidence} 的期望损失不等于条件均值"

    def test_credit_portfolio_matches_brute_force(self):
        """卷积得到的组合损失分布应与穷举所有违约组合一致。"""
        dist = LossDistribution.from_credit_portfolio(PORTFOLIO_DEFAULTS, PORTFOLIO_EXPOSURES)
        brute = np.zeros(sum(PORTFOLIO_EXPOSURES) + 1)
        for indicators in itertools.product([0, 1], repeat=len(PORTFOLIO_DEFAULTS)):
            probability, loss = 1.0, 0
            for indicator, default, exposure in zip(indicators, PORTFOLIO_DEFAULTS,
                                                    PORTFOLIO_EXPOSURES):
                probability *= default if indicator else (1.0 - default)
                loss += exposure * indicator
            brute[loss] += probability
        padded = np.zeros(dist.probabilities.size)
        padded[:brute.size] = brute
        assert np.max(np.abs(dist.probabilities - padded)) < 1e-12, "组合损失分布与穷举结果不一致"

    def test_state_preparation_reproduces_the_distribution(self):
        """态制备电路产生的测量概率应等于目标分布。"""
        from pyqpanda3.core import CPUQVM, QProg

        dist = LossDistribution(SIMPLE_PROBABILITIES)
        qubits = list(range(dist.num_qubits))
        prog = QProg()
        prog << dist.state_preparation(qubits)
        machine = CPUQVM()
        machine.run(prog, 1)
        amplitudes = np.asarray(machine.result().get_state_vector())
        measured = np.abs(amplitudes) ** 2
        assert np.max(np.abs(measured - dist.probabilities)) < 1e-9, "态制备结果与目标分布不符"

    @pytest.mark.parametrize("probabilities", [[], [-0.1, 0.5], [0.0, 0.0]])
    def test_rejects_invalid_probabilities(self, probabilities):
        with pytest.raises(ValueError):
            LossDistribution(probabilities)

    def test_rejects_mismatched_portfolio_lengths(self):
        with pytest.raises(ValueError):
            LossDistribution.from_credit_portfolio([0.1, 0.2], [1])

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
    def test_rejects_invalid_confidence(self, confidence):
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        with pytest.raises(ValueError):
            dist.value_at_risk(confidence)


class Test_QuantumRiskAnalyzer:

    def test_tail_probability_matches_exact_value(self):
        """振幅估计得到的尾部概率应落在给定精度内。"""
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        epsilon = 0.01
        analyzer = QuantumRiskAnalyzer(dist, epsilon=epsilon)
        for threshold in range(1, dist.probabilities.size):
            exact = dist.tail_probability(threshold)
            estimate = analyzer.tail_probability(threshold)
            assert abs(estimate - exact) < 10 * epsilon, \
                f"阈值 {threshold}: 估计值 {estimate:.5f} 与精确值 {exact:.5f} 偏差过大"

    def test_tail_probability_boundaries_need_no_estimation(self):
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        analyzer = QuantumRiskAnalyzer(dist, epsilon=0.05)
        assert analyzer.tail_probability(0) == 1.0, "阈值为 0 时尾部概率必为 1"
        assert analyzer.tail_probability(dist.probabilities.size) == 0.0, \
            "阈值超出取值范围时尾部概率必为 0"
        assert analyzer.estimate_calls == 0, "边界情形不应调用振幅估计"

    def test_value_at_risk_matches_exact_value(self):
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        for confidence in (0.80, 0.90):
            analyzer = QuantumRiskAnalyzer(dist, epsilon=recommended_epsilon(confidence))
            assert analyzer.value_at_risk(confidence) == dist.value_at_risk(confidence), \
                f"置信度 {confidence} 下的 VaR 与精确值不一致"

    def test_value_at_risk_uses_logarithmic_number_of_estimations(self):
        """二分搜索的振幅估计次数应为 O(log2(取值数))，而非线性扫描。"""
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        analyzer = QuantumRiskAnalyzer(dist, epsilon=0.02)
        analyzer.estimate_calls = 0
        analyzer.value_at_risk(0.80)
        levels = dist.probabilities.size
        assert analyzer.estimate_calls <= int(np.ceil(np.log2(levels))) + 1, \
            f"{levels} 个取值上的二分搜索调用了 {analyzer.estimate_calls} 次振幅估计"

    def test_expected_shortfall_matches_exact_value(self):
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        confidence = 0.80
        analyzer = QuantumRiskAnalyzer(dist, epsilon=recommended_epsilon(confidence))
        exact = dist.expected_shortfall(confidence)
        estimate = analyzer.expected_shortfall(confidence)
        assert abs(estimate - exact) < 0.5, \
            f"期望损失估计 {estimate:.4f} 与精确值 {exact:.4f} 偏差过大"

    def test_report_contains_quantum_and_exact_values(self):
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        confidence = 0.80
        analyzer = QuantumRiskAnalyzer(dist, epsilon=recommended_epsilon(confidence))
        summary = analyzer.report(confidence)
        for key in ('var_quantum', 'var_exact', 'expected_shortfall_quantum',
                    'expected_shortfall_exact', 'amplitude_estimations', 'num_qubits'):
            assert key in summary, f"报告缺少字段 {key}"
        assert summary['num_qubits'] == 2 * dist.num_qubits, "寄存器宽度应为损失寄存器的两倍"
        assert summary['amplitude_estimations'] > 0, "报告应记录振幅估计的调用次数"

    def test_warns_when_epsilon_cannot_resolve_the_confidence_level(self):
        """精度不足以分辨 1 - alpha 时应给出警告。"""
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        analyzer = QuantumRiskAnalyzer(dist, epsilon=0.05)
        with pytest.warns(RuntimeWarning):
            analyzer.value_at_risk(0.99)

    def test_rejects_invalid_arguments(self):
        dist = LossDistribution(SIMPLE_PROBABILITIES)
        with pytest.raises(TypeError):
            QuantumRiskAnalyzer(SIMPLE_PROBABILITIES)
        with pytest.raises(ValueError):
            QuantumRiskAnalyzer(dist, epsilon=0.0)
        with pytest.raises(ValueError):
            QuantumRiskAnalyzer(dist, epsilon=1.5)


class Test_query_complexity:

    def test_recommended_epsilon_scales_with_the_tail(self):
        assert recommended_epsilon(0.90) > recommended_epsilon(0.99), \
            "置信度越高，所需精度应越小"
        assert abs(recommended_epsilon(0.99) - 0.2 * 0.01) < 1e-12

    def test_classical_sample_count_grows_quadratically(self):
        coarse = classical_monte_carlo_samples(0.01)
        fine = classical_monte_carlo_samples(0.005)
        ratio = fine / coarse
        assert 3.5 < ratio < 4.5, \
            f"精度提高一倍时经典采样量应约变为 4 倍，实际比值为 {ratio:.2f}"

    @pytest.mark.parametrize("epsilon", [0.0, 1.0, -0.1])
    def test_classical_sample_count_rejects_invalid_accuracy(self, epsilon):
        with pytest.raises(ValueError):
            classical_monte_carlo_samples(epsilon)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
