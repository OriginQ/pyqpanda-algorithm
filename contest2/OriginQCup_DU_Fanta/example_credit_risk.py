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

"""Credit portfolio risk analysis with quantum amplitude estimation.

Runs the full workflow on a small loan book and prints every quantum estimate
next to the exact value, so the accuracy of the method is visible rather than
asserted.

Usage
    python example_credit_risk.py
"""

import numpy as np

from quantum_risk import (
    LossDistribution,
    QuantumRiskAnalyzer,
    classical_monte_carlo_samples,
    recommended_epsilon,
)

# A small loan book: five obligors with individual default probabilities and
# integer exposures in units of 100k CNY.
DEFAULT_PROBABILITIES = [0.08, 0.15, 0.04, 0.22, 0.10]
EXPOSURES = [2, 1, 3, 1, 2]
UNIT = 1e5
EPSILON = 0.01
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]


def print_portfolio():
    print('=' * 72)
    print('Credit portfolio')
    print('=' * 72)
    print(f"{'obligor':>8} {'default prob':>14} {'exposure':>10} {'exposure (CNY)':>16}")
    for index, (probability, exposure) in enumerate(zip(DEFAULT_PROBABILITIES, EXPOSURES), 1):
        print(f'{index:>8} {probability:>14.2%} {exposure:>10} {exposure * UNIT:>16,.0f}')
    expected = sum(p * e for p, e in zip(DEFAULT_PROBABILITIES, EXPOSURES))
    print(f"\ntotal exposure : {sum(EXPOSURES) * UNIT:,.0f} CNY")
    print(f'expected loss  : {expected * UNIT:,.0f} CNY')


def print_distribution(distribution):
    print()
    print('=' * 72)
    print(f'Loss distribution on {distribution.num_qubits} qubits '
          f'({distribution.probabilities.size} levels)')
    print('=' * 72)
    print(f"{'loss':>6} {'CNY':>12} {'P(L = l)':>12} {'P(L >= l)':>12}")
    for level, probability in enumerate(distribution.probabilities):
        if probability < 1e-6:
            continue
        print(f'{level:>6} {level * UNIT:>12,.0f} {probability:>12.5f} '
              f'{distribution.tail_probability(level):>12.5f}')


def print_tail_probabilities(distribution, analyzer):
    print()
    print('=' * 72)
    print('Tail probabilities: quantum amplitude estimation vs exact')
    print('=' * 72)
    print(f"{'threshold':>10} {'exact':>10} {'quantum':>10} {'abs error':>11}")
    errors = []
    for threshold in range(1, distribution.probabilities.size):
        exact = distribution.tail_probability(threshold)
        if exact < 1e-6:
            continue
        estimate = analyzer.tail_probability(threshold)
        errors.append(abs(estimate - exact))
        print(f'{threshold:>10} {exact:>10.5f} {estimate:>10.5f} {abs(estimate - exact):>11.5f}')
    print(f'\nmax absolute error {max(errors):.5f}, target accuracy {EPSILON}')


def print_risk_measures(analyzer):
    print()
    print('=' * 72)
    print('Risk measures')
    print('=' * 72)
    # Value at Risk is decided by comparing tail probabilities against 1 - alpha,
    # so each confidence level needs its own accuracy; a coarse epsilon cannot
    # resolve a 1% tail. Each report is also estimated once and reused, because
    # amplitude estimation is randomised.
    summaries = []
    for confidence in CONFIDENCE_LEVELS:
        epsilon = min(EPSILON, recommended_epsilon(confidence))
        summaries.append(QuantumRiskAnalyzer(analyzer.distribution, epsilon=epsilon)
                         .report(confidence))

    print(f"{'alpha':>7} {'epsilon':>9} {'VaR (q)':>9} {'VaR (exact)':>12} {'ES (q)':>9} "
          f"{'ES (exact)':>11} {'ES error':>10} {'AE calls':>9}")
    for summary in summaries:
        print(f"{summary['confidence']:>7.2f} {summary['epsilon']:>9.4f} "
              f"{summary['var_quantum']:>9} "
              f"{summary['var_exact']:>12} "
              f"{summary['expected_shortfall_quantum']:>9.4f} "
              f"{summary['expected_shortfall_exact']:>11.4f} "
              f"{summary['expected_shortfall_abs_error']:>10.4f} "
              f"{summary['amplitude_estimations']:>9}")

    print()
    for summary in summaries:
        print(f"alpha = {summary['confidence']:.0%}: "
              f"VaR = {summary['var_monetary']:,.0f} CNY, "
              f"ES = {summary['expected_shortfall_monetary']:,.0f} CNY")

    boundary = [s for s in summaries if s['var_quantum'] != s['var_exact']]
    if boundary:
        print('\nNote: Value at Risk is an integer level, so a confidence level that')
        print('falls very close to a jump of the cumulative distribution can round to')
        print('either side. The levels affected here are: '
              + ', '.join(f"{s['confidence']:.0%}" for s in boundary) + '.')


def print_query_comparison(distribution):
    print()
    print('=' * 72)
    print('Oracle queries vs classical Monte Carlo samples')
    print('=' * 72)
    print(f"{'accuracy':>10} {'classical samples':>19} {'ratio to 1/eps':>16}")
    for epsilon in (0.05, 0.01, 0.005, 0.001):
        samples = classical_monte_carlo_samples(epsilon)
        print(f'{epsilon:>10} {samples:>19,} {samples * epsilon:>16,.1f}')
    print('\nClassical Monte Carlo needs O(1/eps^2) samples; iterative amplitude')
    print('estimation needs O(1/eps) oracle queries. The advantage is in the query')
    print('count of the tail-probability primitive, not in the whole workflow, and')
    print('it is asymptotic: on a simulator the classical computation is faster.')

    levels = distribution.probabilities.size
    print(f'\nValue at Risk uses bisection over {levels} loss levels, so it needs')
    print(f'O(log2({levels})) = {int(np.ceil(np.log2(levels)))} amplitude estimations')
    print(f'rather than the {levels} a linear scan would take.')


def main():
    print_portfolio()

    distribution = LossDistribution.from_credit_portfolio(
        DEFAULT_PROBABILITIES, EXPOSURES, unit=UNIT)
    print_distribution(distribution)

    analyzer = QuantumRiskAnalyzer(distribution, epsilon=EPSILON)
    print(f'\nquantum register: {analyzer.num_qubits} qubits '
          f'({distribution.num_qubits} loss + {distribution.num_qubits} comparator)')

    print_tail_probabilities(distribution, analyzer)
    print_risk_measures(analyzer)
    print_query_comparison(distribution)


if __name__ == '__main__':
    main()
