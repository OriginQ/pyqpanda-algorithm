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

import itertools

import numpy as np
import pytest
from pyqpanda3.core import QCircuit, RY
from pyqpanda_alg.QAE import IQAE


class EndOfSamples(Exception):
    """Stop a scripted run after all prescribed batches have been processed."""


class ScriptedIQAE(IQAE):
    def __init__(self, powers):
        super().__init__(operator_in=lambda q: QCircuit() << RY(q[0], 0.7),
                         qnumber=1, epsilon=0.01)
        self.powers = powers
        self.batches = []
        self.intervals = []

    def _findnextk(self, k_i, theta_l, theta_u, up):
        if len(self.batches) == len(self.powers):
            raise EndOfSamples
        return self.powers[len(self.batches)], True

    def _measure(self, k, n_round):
        # Distinct batch proportions detect losing the first batch after a switch.
        fraction = (0.2, 0.7, 0.1, 0.9)[len(self.batches) % 4]
        successes = int(fraction * n_round)
        self.batches.append((k, successes, n_round))
        return successes

    def _chernoff_confint(self, value, shots, max_rounds):
        self.intervals.append((value, shots))
        # Keep the loop open so aggregation can be tested independently of
        # stopping-time fluctuations and the power-selection heuristic.
        return 0.0, 1.0

    def replay(self):
        self.batches.clear()
        self.intervals.clear()
        with pytest.raises(EndOfSamples):
            self.run()


def grouped_prefixes(batches):
    """Compute the sufficient statistics of consecutive equal-power batches."""
    for _, group in itertools.groupby(batches, key=lambda row: row[0]):
        successes = shots = 0
        for _, count, size in group:
            successes += count
            shots += size
            yield successes / shots, shots


@pytest.mark.parametrize('powers', [
    [0, 0, 1, 1, 3, 3],
    [0, 1, 3, 7],
    [0, 0, 0, 1, 1, 1],
])
def test_confidence_interval_uses_only_current_power_shots(powers):
    estimator = ScriptedIQAE(powers)
    estimator.replay()
    expected = list(grouped_prefixes(estimator.batches))
    assert [shots for _, shots in estimator.intervals] == [n for _, n in expected]
    assert estimator.n_sum == sum(n for _, _, n in estimator.batches)


def test_repeated_power_includes_its_first_batch():
    estimator = ScriptedIQAE([0, 0, 1, 1, 1])
    estimator.replay()
    expected = list(grouped_prefixes(estimator.batches))
    np.testing.assert_allclose([p for p, _ in estimator.intervals],
                               [p for p, _ in expected], atol=1e-15, rtol=0)


def test_n_sum_is_reset_for_each_run():
    estimator = ScriptedIQAE([0, 0, 1, 1])
    estimator.replay()
    first = list(estimator.intervals)
    estimator.replay()
    assert estimator.intervals == first
    assert estimator.n_sum == sum(n for _, _, n in estimator.batches)


def test_same_power_accumulation_can_exceed_per_batch_cap():
    estimator = ScriptedIQAE([0] * 8)
    estimator.N_max = 60
    estimator.replay()
    total_shots = sum(n for _, _, n in estimator.batches)
    assert total_shots > estimator.N_max
    assert estimator.intervals[-1][1] == total_shots


def test_lower_half_circle_does_not_shift_at_period_endpoint(monkeypatch):
    estimator = IQAE(qnumber=1, epsilon=0.01)
    bounds = []
    # First narrow theta/(2*pi) to [1/12, 1/6]. K=6 then maps this
    # interval to the lower half-circle [1/2, 1], ending at a full period.
    confidence_intervals = iter([(0.25, 0.75), (0.0, 1.0)])

    def next_power(k, lower, upper, up):
        bounds.append((lower, upper))
        if len(bounds) == 1:
            return 0, True
        if len(bounds) == 2:
            return 1, False
        raise EndOfSamples

    monkeypatch.setattr(estimator, '_findnextk', next_power)
    monkeypatch.setattr(estimator, '_measure', lambda k, n: n // 2)
    monkeypatch.setattr(estimator, '_chernoff_confint',
                        lambda *args: next(confidence_intervals))
    with pytest.raises(EndOfSamples):
        estimator.run()
    np.testing.assert_allclose(bounds[1], [1 / 12, 1 / 6], atol=1e-15, rtol=0)
    np.testing.assert_allclose(bounds[2], bounds[1], atol=1e-15, rtol=0)


def test_upper_half_circle_keeps_period_after_roundoff_at_lower_bound(monkeypatch):
    estimator = IQAE(qnumber=1, epsilon=0.01)
    period, bigk = 13, 90
    lower, upper = period / bigk, (period + 0.5) / bigk
    # This exact phase boundary rounds just below the integer when rescaled.
    assert bigk * lower < period
    initial = np.sin(2 * np.pi * np.array([
        (period + 0.05) / bigk, (period + 0.45) / bigk,
    ])) ** 2
    confidence_intervals = iter([initial, (0.0, 1.0), (0.0, 1.0)])
    bounds = []

    def next_power(k, theta_l, theta_u, up):
        bounds.append((theta_l, theta_u))
        if len(bounds) == 1:
            return 0, True
        if len(bounds) <= 3:
            return (bigk - 2) // 4, True
        raise EndOfSamples

    monkeypatch.setattr(estimator, '_findnextk', next_power)
    monkeypatch.setattr(estimator, '_measure', lambda k, n: n // 2)
    monkeypatch.setattr(estimator, '_chernoff_confint',
                        lambda *args: next(confidence_intervals))
    with pytest.raises(EndOfSamples):
        estimator.run()
    np.testing.assert_allclose(bounds[2], [lower, upper], atol=1e-15, rtol=0)
    np.testing.assert_allclose(bounds[3], [lower, upper], atol=1e-15, rtol=0)


class RoundedProbabilityIQAE(IQAE):
    """Deterministic Bernoulli-count oracle, retaining the real adaptive loop."""
    def __init__(self, probability):
        super().__init__(qnumber=1, epsilon=0.01)
        self.probability = probability
        self.shots = []

    def _measure(self, k, n_round):
        angle = np.arcsin(np.sqrt(self.probability))
        probability = np.sin((2 * k + 1) * angle) ** 2
        self.shots.append(n_round)
        return int(round(n_round * probability))


@pytest.mark.parametrize('probability', [0.0, 0.02, 0.2, 0.5, 0.8, 0.98, 1.0])
def test_adaptive_estimate_with_deterministic_counts(probability):
    estimator = RoundedProbabilityIQAE(probability)
    estimate = estimator.run()
    assert estimator.a_l <= probability <= estimator.a_u
    assert abs(estimate - probability) <= estimator.epsilon
    assert estimator.n_sum == sum(estimator.shots)


def test_completed_runs_do_not_reuse_previous_statistics():
    estimator = RoundedProbabilityIQAE(0.2)
    first = (estimator.run(), estimator.a_l, estimator.a_u, estimator.n_sum)
    estimator.shots.clear()
    second = (estimator.run(), estimator.a_l, estimator.a_u, estimator.n_sum)
    assert second == first
    estimator.probability = 0.8
    estimator.shots.clear()
    estimate = estimator.run()
    assert estimator.a_l <= 0.8 <= estimator.a_u
    assert abs(estimate - 0.8) <= estimator.epsilon
    assert estimator.n_sum == sum(estimator.shots)


@pytest.mark.parametrize('probability', [0.0, 1.0])
def test_cpuqvm_endpoint_states(probability):
    angle = 2 * np.arcsin(np.sqrt(probability))
    estimator = IQAE(operator_in=lambda q: QCircuit() << RY(q[0], angle),
                     qnumber=1, epsilon=0.01)
    estimate = estimator.run()
    assert estimator.a_l <= probability <= estimator.a_u
    assert abs(estimate - probability) <= estimator.epsilon
