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

import numpy as np
import pytest

from pyqpanda3.core import CPUQVM, QCircuit, QProg, H, RY, RZ, X
from pyqpanda_alg.QAE import QAE
from pyqpanda_alg.plugin import QFT


def state_preparation(qubits):
    return (QCircuit() << RY(qubits[0], 0.71)
            << RY(qubits[1], 1.13) << RZ(qubits[0], 0.37)
            << X(qubits[1]).control(qubits[0]))


def legacy_program(estimator):
    """Build the pre-optimization circuit to compare all output probabilities."""
    qubits = QProg(estimator.qnumber + estimator.n_anc).qubits()
    targets = [qubits[i] for i in range(estimator.qnumber)
               if i in [list(range(estimator.qnumber))[j]
                        for j in estimator.res_index]]
    program = QProg() << estimator.operator(qubits[:estimator.qnumber])
    for k, bit in enumerate(estimator.target_state):
        if bit == '0':
            program << X(targets[-k - 1])
    for qubit in qubits[estimator.qnumber:]:
        program << H(qubit)
    for i in range(estimator.n_anc):
        for _ in range(2 ** i):
            program << estimator._Q_cir(qubits[:estimator.qnumber]).control(
                [qubits[estimator.qnumber + i]])
    return program << QFT(qubits[estimator.qnumber:]).dagger()


@pytest.mark.parametrize("ancillas", [1, 2, 4, 6])
@pytest.mark.parametrize("indices,target", [
    ([0], '1'), ([1], '0'), ([-1], '1'),
    ([0, 1], '00'), ([0, 1], '01'), ([0, 1], '10'), ([0, 1], '11'),
])
def test_qae_reuse_preserves_full_output_distribution(ancillas, indices, target):
    estimator = QAE(operator_in=state_preparation, qnumber=2,
                    res_index=indices, target_state=target,
                    epsilon=2.0 ** -ancillas)
    reference = CPUQVM()
    reference.run(legacy_program(estimator), 1000)
    expected = reference.result().get_prob_list(list(range(2 + ancillas)))
    estimator.run()
    actual = estimator.machine.result().get_prob_list(list(range(2 + ancillas)))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("ancillas", [1, 2, 4, 6])
def test_qae_state_preparation_build_count_is_constant(ancillas):
    calls = []

    def counted_preparation(qubits):
        calls.append(tuple(qubits))
        return state_preparation(qubits)

    estimator = QAE(operator_in=counted_preparation, qnumber=2,
                    epsilon=2.0 ** -ancillas)
    estimator.run()
    # Initial A plus A-dagger and A in the one reused Grover iterate.
    assert calls == [(0, 1)] * 3


def test_qae_rebuilds_after_preparation_changes_between_runs():
    probability = [0.25]

    def preparation(qubits):
        angle = 2 * np.arcsin(np.sqrt(probability[0]))
        return QCircuit() << RY(qubits[0], angle)

    estimator = QAE(operator_in=preparation, qnumber=1, epsilon=1 / 64)
    first = estimator.run()
    probability[0] = 0.75
    second = estimator.run()
    assert abs(first - 0.25) < 0.025
    assert abs(second - 0.75) < 0.025


@pytest.mark.parametrize("probability", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_qae_estimate_matches_prepared_probability(probability):
    def preparation(qubits):
        return QCircuit() << RY(qubits[0], 2 * np.arcsin(np.sqrt(probability)))

    estimate = QAE(operator_in=preparation, qnumber=1, epsilon=1 / 64).run()
    assert abs(estimate - probability) < 0.025
