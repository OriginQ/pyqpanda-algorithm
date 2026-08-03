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

"""Quantum risk analysis: Value at Risk and Expected Shortfall by amplitude estimation.

This module composes two existing ``pyqpanda_alg`` components into a complete
risk-measurement workflow:

* ``QCmp.int_comparator`` builds the reversible predicate  :math:`L \\geq t` ;
* ``QAE.IQAE`` estimates the probability that the predicate holds.

Together they evaluate the tail probability  :math:`P(L \\geq t)`  with
:math:`O(1 / \\epsilon)`  oracle queries, against the  :math:`O(1 / \\epsilon ^ 2)`
samples a classical Monte Carlo simulation needs for the same accuracy.

Every risk figure produced here reduces to tail probabilities, which is what
makes a single quantum primitive sufficient for the whole workflow. For an
integer-valued loss  :math:`L` :

.. math::

    E[(L - t) ^ +] = \\sum_{k > t} P(L \\geq k)

    \\mathrm{ES}_\\alpha = \\mathrm{VaR}_\\alpha
        + \\frac{E[(L - \\mathrm{VaR}_\\alpha) ^ +]}{P(L \\geq \\mathrm{VaR}_\\alpha)}

so Value at Risk follows from a bisection over tail probabilities and Expected
Shortfall from a short sum of them. No separate estimation primitive is needed.

References
    [1] S. Woerner, D. J. Egger, Quantum risk analysis. npj Quantum Information 5, 15 (2019).
        https://doi.org/10.1038/s41534-019-0130-6
    [2] D. Grinko, J. Gacon, C. Zoufal, S. Woerner, Iterative quantum amplitude estimation.
        npj Quantum Information 7, 52 (2021). https://doi.org/10.1038/s41534-021-00379-1
"""

import warnings

import numpy as np
from pyqpanda3.core import QCircuit, Encode

from pyqpanda_alg import QAE
from pyqpanda_alg import QCmp

__all__ = ['LossDistribution', 'QuantumRiskAnalyzer', 'recommended_epsilon',
           'classical_monte_carlo_samples']

# Value at Risk is decided by comparing tail probabilities against 1 - alpha, so
# the estimation accuracy has to be a small fraction of that gap or the
# comparison is dominated by estimation error. A fifth of the gap keeps the
# probability of choosing the wrong level low without making the estimate
# needlessly expensive.
EPSILON_SAFETY_FACTOR = 0.2


def recommended_epsilon(confidence: float, safety: float = EPSILON_SAFETY_FACTOR) -> float:
    """Accuracy needed to resolve Value at Risk at a given confidence level.

    The bisection in :meth:`QuantumRiskAnalyzer.value_at_risk` tests whether
    :math:`P(L \\geq t + 1) \\leq 1 - \\alpha` . If the estimation accuracy
    :math:`\\epsilon`  is comparable to  :math:`1 - \\alpha` , that comparison is
    decided by noise and the returned level can be off by one. This helper
    returns  :math:`\\text{safety} \\times (1 - \\alpha)` .

    Parameters
        confidence : ``float``\n
            Confidence level  :math:`\\alpha`  in  :math:`(0, 1)` .
        safety : ``float``\n
            Fraction of the gap  :math:`1 - \\alpha`  to use. Default 0.2.

    Returns
        epsilon : ``float``\n
            The recommended estimation accuracy.

    Examples
    >>> from quantum_risk import recommended_epsilon
    >>> round(recommended_epsilon(0.99), 4)
    0.002

    """
    _check_confidence(confidence)
    if not 0 < safety < 1:
        raise ValueError(f'safety must lie in (0, 1), got {safety}')
    return safety * (1.0 - confidence)


class LossDistribution:
    """A discrete loss distribution held on  :math:`n`  qubits.

    The loss takes the integer values  :math:`0, 1, \\ldots, 2 ^ n - 1` , which is
    the representation the quantum comparator works on directly. Monetary losses
    are recovered by scaling with :attr:`unit`.

    Parameters
        probabilities : ``list[float]``, ``np.ndarray``\n
            Probability of each loss level. The length is padded up to the next
            power of two and the result is normalised.
        unit : ``float``\n
            Monetary value of one loss level. Default 1.0.

    Examples
    >>> from quantum_risk import LossDistribution
    >>> dist = LossDistribution([0.5, 0.3, 0.15, 0.05])
    >>> dist.num_qubits
    2
    >>> round(dist.tail_probability(2), 4)
    0.2

    """

    def __init__(self, probabilities, unit: float = 1.0):
        probabilities = np.asarray(probabilities, dtype=float).flatten()
        if probabilities.size == 0:
            raise ValueError('probabilities must contain at least one entry')
        if np.any(probabilities < 0):
            raise ValueError('probabilities must be non-negative')
        total = probabilities.sum()
        if total <= 0:
            raise ValueError('probabilities must contain at least one positive entry')
        if unit <= 0:
            raise ValueError(f'unit must be positive, got {unit}')

        size = 1 << max(int(np.ceil(np.log2(probabilities.size))), 1)
        padded = np.zeros(size)
        padded[:probabilities.size] = probabilities
        self.probabilities = padded / padded.sum()
        self.unit = float(unit)

    @property
    def num_qubits(self) -> int:
        """Number of qubits holding the loss value."""
        return int(np.log2(self.probabilities.size))

    @property
    def levels(self) -> np.ndarray:
        """The integer loss levels  :math:`0, \\ldots, 2 ^ n - 1` ."""
        return np.arange(self.probabilities.size)

    @property
    def amplitudes(self) -> np.ndarray:
        """Amplitudes  :math:`\\sqrt{p_i}`  loaded by the state-preparation circuit."""
        return np.sqrt(self.probabilities)

    def state_preparation(self, qubits) -> QCircuit:
        """Circuit mapping  :math:`|0\\rangle ^ {\\otimes n}`  to  :math:`\\sum_i \\sqrt{p_i} |i\\rangle` .

        Parameters
            qubits : ``list[int]``\n
                The  :math:`n`  qubits holding the loss value.

        Returns
            circuit : ``QCircuit``\n
                The state-preparation circuit.

        """
        if len(qubits) != self.num_qubits:
            raise ValueError(
                f'expected {self.num_qubits} qubits for the loss register, got {len(qubits)}')
        encoder = Encode()
        encoder.amplitude_encode(list(qubits), list(self.amplitudes))
        circuit = QCircuit()
        circuit << encoder.get_circuit()
        return circuit

    def tail_probability(self, threshold: int) -> float:
        """Exact  :math:`P(L \\geq \\text{threshold})` , used as the reference value."""
        threshold = int(np.clip(threshold, 0, self.probabilities.size))
        return float(self.probabilities[threshold:].sum())

    def value_at_risk(self, confidence: float) -> int:
        """Exact Value at Risk: the smallest level  :math:`t`  with  :math:`P(L \\leq t) \\geq \\alpha` ."""
        _check_confidence(confidence)
        cumulative = np.cumsum(self.probabilities)
        return int(np.searchsorted(cumulative, confidence))

    def expected_shortfall(self, confidence: float) -> float:
        """Exact Expected Shortfall  :math:`E[L \\mid L \\geq \\mathrm{VaR}_\\alpha]` ."""
        var = self.value_at_risk(confidence)
        tail = self.tail_probability(var)
        if tail <= 0:
            return float(var)
        mask = self.levels >= var
        return float((self.levels[mask] * self.probabilities[mask]).sum() / tail)

    @classmethod
    def from_credit_portfolio(cls, default_probabilities, exposures, unit: float = 1.0):
        """Build the loss distribution of a portfolio of independent obligors.

        Each obligor  :math:`i`  defaults independently with probability
        :math:`p_i`  and contributes an integer exposure  :math:`e_i`  when it
        does, so the portfolio loss is  :math:`L = \\sum_i e_i X_i`  with
        :math:`X_i \\sim \\mathrm{Bernoulli}(p_i)` . The exact distribution is
        obtained by convolving the per-obligor distributions.

        Parameters
            default_probabilities : ``list[float]``\n
                Default probability of each obligor, each in  :math:`[0, 1]` .
            exposures : ``list[int]``\n
                Integer loss contributed by each obligor on default.
            unit : ``float``\n
                Monetary value of one exposure unit. Default 1.0.

        Returns
            distribution : ``LossDistribution``\n
                The exact portfolio loss distribution.

        Examples
        >>> from quantum_risk import LossDistribution
        >>> dist = LossDistribution.from_credit_portfolio([0.1, 0.2], [1, 2])
        >>> round(dist.tail_probability(3), 4)
        0.02

        """
        default_probabilities = np.asarray(default_probabilities, dtype=float).flatten()
        exposures = np.asarray(exposures, dtype=int).flatten()
        if default_probabilities.size != exposures.size:
            raise ValueError('default_probabilities and exposures must have the same length')
        if default_probabilities.size == 0:
            raise ValueError('the portfolio must contain at least one obligor')
        if np.any(default_probabilities < 0) or np.any(default_probabilities > 1):
            raise ValueError('every default probability must lie in [0, 1]')
        if np.any(exposures < 0):
            raise ValueError('exposures must be non-negative')

        distribution = np.zeros(int(exposures.sum()) + 1)
        distribution[0] = 1.0
        for probability, exposure in zip(default_probabilities, exposures):
            updated = distribution * (1.0 - probability)
            if exposure > 0:
                updated[exposure:] += distribution[:distribution.size - exposure] * probability
            else:
                updated += distribution * probability
            distribution = updated
        return cls(distribution, unit=unit)


class QuantumRiskAnalyzer:
    """Estimate tail risk measures with iterative quantum amplitude estimation.

    The circuit acting on the loss register and the comparator ancillas is

    .. parsed-literal::

        loss register  ---- state preparation ----*----
                                                  |
        comparator     ------------------- int_comparator ----> objective qubit

    and ``QAE.IQAE`` estimates the probability that the objective qubit is
    :math:`|1\\rangle` , which is exactly  :math:`P(L \\geq t)` .

    Parameters
        distribution : ``LossDistribution``\n
            The loss distribution to analyse.
        epsilon : ``float``\n
            Target accuracy of each amplitude estimate. Default 0.005.

    Examples
    >>> from quantum_risk import LossDistribution, QuantumRiskAnalyzer
    >>> dist = LossDistribution([0.5, 0.3, 0.15, 0.05])
    >>> analyzer = QuantumRiskAnalyzer(dist, epsilon=0.01)
    >>> estimate = analyzer.tail_probability(2)
    >>> abs(estimate - dist.tail_probability(2)) < 0.05
    True

    """

    def __init__(self, distribution: LossDistribution, epsilon: float = 5e-3):
        if not isinstance(distribution, LossDistribution):
            raise TypeError('distribution must be a LossDistribution instance')
        if not 0 < epsilon < 1:
            raise ValueError(f'epsilon must lie in (0, 1), got {epsilon}')

        self.distribution = distribution
        self.epsilon = float(epsilon)

        n = distribution.num_qubits
        # int_comparator needs one ancilla register the same width as the loss
        # register, and reports the outcome on its last qubit.
        self.loss_qubits = list(range(n))
        self.comparator_qubits = list(range(n, 2 * n))
        self.objective_qubit = self.comparator_qubits[-1]
        self.num_qubits = 2 * n
        self.estimate_calls = 0

    def _state_operator(self, threshold: int):
        """The operator  :math:`A`  whose  :math:`|1\\rangle`  amplitude on the objective qubit is the tail probability."""
        distribution = self.distribution
        loss_qubits = self.loss_qubits
        comparator_qubits = self.comparator_qubits

        def operator(qlist):
            circuit = QCircuit()
            circuit << distribution.state_preparation([qlist[i] for i in loss_qubits])
            circuit << QCmp.int_comparator(int(threshold),
                                           [qlist[i] for i in loss_qubits],
                                           [qlist[i] for i in comparator_qubits],
                                           function='geq')
            return circuit

        return operator

    def tail_probability(self, threshold: int) -> float:
        """Estimate  :math:`P(L \\geq \\text{threshold})`  by amplitude estimation.

        Parameters
            threshold : ``int``\n
                The loss level to compare against.

        Returns
            probability : ``float``\n
                The estimated tail probability.

        """
        size = self.distribution.probabilities.size
        # The comparator is only defined on representable levels; outside that
        # range the answer is known exactly and no quantum work is needed.
        if threshold <= 0:
            return 1.0
        if threshold >= size:
            return 0.0

        self.estimate_calls += 1
        return float(QAE.IQAE(operator_in=self._state_operator(threshold),
                              qnumber=self.num_qubits,
                              epsilon=self.epsilon,
                              res_index=self.objective_qubit).run())

    def value_at_risk(self, confidence: float) -> int:
        """Estimate Value at Risk by bisection on quantum tail probabilities.

        :math:`\\mathrm{VaR}_\\alpha`  is the smallest level  :math:`t`  with
        :math:`P(L \\geq t + 1) \\leq 1 - \\alpha` . Because  :math:`P(L \\geq t)`
        is monotone in  :math:`t` , bisection finds it in
        :math:`O(\\log 2 ^ n) = O(n)`  amplitude estimations rather than the
        :math:`O(2 ^ n)`  a linear scan would need.

        The accuracy :attr:`epsilon` must be small compared with  :math:`1 - \\alpha` ;
        see :func:`recommended_epsilon`. A warning is issued otherwise, because
        the comparison would then be decided by estimation noise.

        Parameters
            confidence : ``float``\n
                Confidence level  :math:`\\alpha`  in  :math:`(0, 1)` .

        Returns
            var : ``int``\n
                The estimated Value at Risk, as a loss level.

        """
        _check_confidence(confidence)
        advised = recommended_epsilon(confidence)
        if self.epsilon > advised:
            warnings.warn(
                f'epsilon={self.epsilon:g} is too coarse to resolve Value at Risk at '
                f'confidence {confidence:g}: the bisection compares tail probabilities '
                f'against {1.0 - confidence:g}. Use epsilon <= {advised:g}.',
                RuntimeWarning, stacklevel=2)
        low, high = 0, self.distribution.probabilities.size - 1
        target = 1.0 - confidence
        while low < high:
            middle = (low + high) // 2
            if self.tail_probability(middle + 1) <= target:
                high = middle
            else:
                low = middle + 1
        return int(low)

    def expected_shortfall(self, confidence: float, var: int = None) -> float:
        """Estimate Expected Shortfall  :math:`E[L \\mid L \\geq \\mathrm{VaR}_\\alpha]` .

        Uses  :math:`E[(L - t) ^ +] = \\sum_{k > t} P(L \\geq k)` , so the whole
        quantity is a sum of tail probabilities and needs no second estimation
        primitive.

        Parameters
            confidence : ``float``\n
                Confidence level  :math:`\\alpha`  in  :math:`(0, 1)` .
            var : ``int``, optional\n
                A previously estimated Value at Risk. Recomputed when omitted.

        Returns
            shortfall : ``float``\n
                The estimated Expected Shortfall, as a loss level.

        """
        _check_confidence(confidence)
        if var is None:
            var = self.value_at_risk(confidence)

        tail = self.tail_probability(var)
        if tail <= 0:
            return float(var)

        size = self.distribution.probabilities.size
        excess = sum(self.tail_probability(k) for k in range(var + 1, size))
        return float(var + excess / tail)

    def report(self, confidence: float) -> dict:
        """Estimate every risk measure at one confidence level and compare with the exact values.

        Parameters
            confidence : ``float``\n
                Confidence level  :math:`\\alpha`  in  :math:`(0, 1)` .

        Returns
            summary : ``dict``\n
                Quantum estimates, exact references, absolute errors, the number
                of amplitude estimations used, and the monetary values obtained
                by scaling with the distribution's unit.

        """
        _check_confidence(confidence)
        self.estimate_calls = 0

        quantum_var = self.value_at_risk(confidence)
        quantum_tail = self.tail_probability(quantum_var)
        quantum_shortfall = self.expected_shortfall(confidence, var=quantum_var)

        exact_var = self.distribution.value_at_risk(confidence)
        exact_tail = self.distribution.tail_probability(exact_var)
        exact_shortfall = self.distribution.expected_shortfall(confidence)
        unit = self.distribution.unit

        return {
            'confidence': confidence,
            'epsilon': self.epsilon,
            'num_qubits': self.num_qubits,
            'amplitude_estimations': self.estimate_calls,
            'var_quantum': quantum_var,
            'var_exact': exact_var,
            'var_abs_error': abs(quantum_var - exact_var),
            'var_monetary': quantum_var * unit,
            'tail_probability_quantum': quantum_tail,
            'tail_probability_exact': exact_tail,
            'tail_probability_abs_error': abs(quantum_tail - exact_tail),
            'expected_shortfall_quantum': quantum_shortfall,
            'expected_shortfall_exact': exact_shortfall,
            'expected_shortfall_abs_error': abs(quantum_shortfall - exact_shortfall),
            'expected_shortfall_monetary': quantum_shortfall * unit,
        }


def classical_monte_carlo_samples(epsilon: float, alpha: float = 0.05) -> int:
    """Samples a classical Monte Carlo estimate needs for accuracy ``epsilon``.

    From the Chernoff-Hoeffding bound, estimating a probability to within
    :math:`\\epsilon`  with confidence  :math:`1 - \\alpha`  needs
    :math:`\\lceil \\log(2 / \\alpha) / (2 \\epsilon ^ 2) \\rceil`  samples. It is
    quoted here so the  :math:`O(1 / \\epsilon)`  versus  :math:`O(1 / \\epsilon ^ 2)`
    scaling can be stated with a concrete number rather than asymptotically.

    Parameters
        epsilon : ``float``\n
            Target accuracy.
        alpha : ``float``\n
            Failure probability. Default 0.05.

    Returns
        samples : ``int``\n
            The required number of classical samples.

    """
    if not 0 < epsilon < 1:
        raise ValueError(f'epsilon must lie in (0, 1), got {epsilon}')
    return int(np.ceil(np.log(2.0 / alpha) / (2.0 * epsilon ** 2)))


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError(f'confidence must lie in (0, 1), got {confidence}')
