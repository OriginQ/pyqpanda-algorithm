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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, TOFFOLI, SWAP, CNOT, U1, U3, CR, I, H, X, Y, measure
import numpy as np
import math

from .. plugin import *



class _QMachine:
    def __init__(self, q_bit_count, c_bit_count):
        self.machine = CPUQVM()
        self.q_list = QProg(q_bit_count).qubits()

    def __del__(self):
        pass


def _init_cir(qm, state_vector, n=2):
    prog = QProg()
    prog << I(qm.q_list[0])
    return qm.machine.run(prog, 1000)


def _phase_estimation_cir(q_list, m, tao, n=2):
    qc = QCircuit()
    qc << H(q_list[1])
    qc << H(q_list[2])

    if n == 2:
        qc << U3(q_list[3], -math.pi / 2, -math.pi / 2, math.pi / 2).control(q_list[2])
        if bool(tao < min(m)):
            qc << U1(q_list[2], 5 * math.pi / 4)
            qc << CNOT(q_list[1], q_list[3])
            qc << U1(q_list[1], math.pi)
        if bool(tao >= min(m)) and bool(tao < max(m)):
            qc << U1(q_list[2], 3 * math.pi / 4)
            qc << CNOT(q_list[1], q_list[3])

    if n == 4:
        qc << X(q_list[3])
        qc << CR(q_list[3], q_list[4], math.pi / 4)
        qc << CNOT(q_list[2], q_list[3])
        qc << CR(q_list[3], q_list[4], -math.pi / 4)
        qc << CNOT(q_list[2], q_list[3])
        qc << CR(q_list[2], q_list[4], math.pi / 4)
        qc << X(q_list[3])

        qc << Y(q_list[4]).control(q_list[3])
        qc << CNOT(q_list[3], q_list[4])
        qc << CNOT(q_list[2], q_list[3])
        qc << CNOT(q_list[3], q_list[4])
        qc << Y(q_list[4]).control(q_list[3])
        qc << CNOT(q_list[2], q_list[3])
        qc << Y(q_list[4]).control(q_list[2])
        qc << CNOT(q_list[2], q_list[4])
        qc << CR(q_list[3], q_list[4], math.pi / 4)
        qc << CNOT(q_list[2], q_list[3])
        qc << CR(q_list[3], q_list[4], -math.pi / 4)
        qc << CNOT(q_list[2], q_list[3])
        qc << CR(q_list[2], q_list[4], math.pi / 4)

        qc << Y(q_list[3]).control(q_list[1])
        qc << CNOT(q_list[1], q_list[3])
        qc << CNOT(q_list[4], q_list[1])
        qc << CNOT(q_list[1], q_list[3])
        qc << Y(q_list[3]).control(q_list[1])
        qc << CNOT(q_list[4], q_list[1])
        qc << Y(q_list[3]).control(q_list[4])
        qc << CNOT(q_list[4], q_list[3])

    qc << SWAP(q_list[1], q_list[2])
    qc << H(q_list[2])
    qc << CR(q_list[1], q_list[2], -math.pi / 2)
    qc << H(q_list[1])
    return qc


def _transition_cir(q_list, n=2):
    qc = QCircuit()
    qc << X(q_list[2])

    if n == 2:
        qc << CNOT(q_list[2], q_list[1])

    if n == 4:
        qc << TOFFOLI(q_list[1], q_list[2], q_list[7])
        qc << X(q_list[2])
        qc << CNOT(q_list[7], q_list[1])

    return qc


def _cnot_cir(q_list):
    qc = QCircuit()
    qc << X(q_list[1])
    qc << X(q_list[2])
    qc << TOFFOLI(q_list[1], q_list[2], q_list[0])
    qc << X(q_list[1])
    qc << X(q_list[2])
    qc << X(q_list[0])
    return qc


def _transition_reverse_cir(q_list, n=2):
    qc = QCircuit()

    if n == 2:
        qc << CNOT(q_list[2], q_list[1])

    if n == 4:
        qc << CNOT(q_list[7], q_list[1])
        qc << X(q_list[2])
        qc << TOFFOLI(q_list[1], q_list[2], q_list[7])

    qc << X(q_list[2])
    return qc


def _phase_estimation_reverse_cir(q_list, m, tao, n=2):
    qc = QCircuit()
    qc << H(q_list[1])
    qc << U1(q_list[2], math.pi / 2).control(q_list[1])
    qc << H(q_list[2])
    qc << SWAP(q_list[1], q_list[2])

    if n == 2:
        if bool(tao < min(m)):
            qc << U1(q_list[1], -math.pi)
            qc << CNOT(q_list[1], q_list[3])
            qc << U1(q_list[2], -5 * math.pi / 4)
        if bool(tao >= min(m)) and bool(tao < max(m)):
            qc << CNOT(q_list[1], q_list[3])
            qc << U1(q_list[2], -3 * math.pi / 4)
        qc << U3(q_list[3], -math.pi / 2, math.pi / 2, -math.pi / 2).control(q_list[2])

    if n == 4:
        qc << CNOT(q_list[4], q_list[3])
        qc << Y(q_list[3]).control(q_list[4])
        qc << CNOT(q_list[4], q_list[1])
        qc << Y(q_list[3]).control(q_list[1])
        qc << CNOT(q_list[1], q_list[3])
        qc << CNOT(q_list[4], q_list[1])
        qc << CNOT(q_list[1], q_list[3])
        qc << Y(q_list[3]).control(q_list[1])

        qc << CR(q_list[2], q_list[4], -math.pi / 4)
        qc << CNOT(q_list[2], q_list[3])
        qc << CR(q_list[3], q_list[4], math.pi / 4)
        qc << CNOT(q_list[2], q_list[3])
        qc << CR(q_list[3], q_list[4], -math.pi / 4)
        qc << CNOT(q_list[2], q_list[4])
        qc << Y(q_list[4]).control(q_list[2])
        qc << CNOT(q_list[2], q_list[3])
        qc << Y(q_list[4]).control(q_list[3])
        qc << CNOT(q_list[3], q_list[4])
        qc << CNOT(q_list[2], q_list[3])
        qc << CNOT(q_list[3], q_list[4])
        qc << Y(q_list[4]).control(q_list[3])

        qc << X(q_list[3])
        qc << CR(q_list[2], q_list[4], -math.pi / 4)
        qc << CNOT(q_list[2], q_list[3])
        qc << CR(q_list[3], q_list[4], math.pi / 4)
        qc << CNOT(q_list[2], q_list[3])
        qc << CR(q_list[3], q_list[4], -math.pi / 4)
        qc << X(q_list[3])

    qc << H(q_list[1])
    qc << H(q_list[2])
    return qc


def _measure_cir(prog, q_list, c_list):
    qc = QProg()
    qc << measure(q_list, c_list)
    return qc


def _preprocessing(x):
    n_samples, n_features = x.shape
    mean = np.array([np.mean(x[:, i]) for i in range(n_features)])
    norm_x = x - mean
    covariance_matrix = np.dot(np.transpose(norm_x), norm_x)
    return norm_x, covariance_matrix


def qpca(sample_A, k):
    """
    QPCA is a quantum version of the classical PCA algorithm, which is widely used in data analysis and machine learning.

    Only 2-feature input is supported. The n=4 branches of the circuit helpers
    are incomplete, so a larger input raises NotImplementedError.

    Parameters:
        sample_A: ``ndarray``\n
            the input matrix for analysis, shape (n_samples, 2)
        k: ``int``\n
            the dimension to reduce, 1 or 2

    Returns:
        out: ``ndarray``\n
            the output matrix after reducing dimension, shape (n_samples, k)

    Raises:
        ValueError: k is not 1 or 2.\n
        NotImplementedError: sample_A has other than 2 features.

    Examples:
        .. code-block:: python

            import numpy as np
            from pyqpanda_alg.QPCA import qpca

            A = np.array([[-1, 2], [-2, -1], [-1, -2], [1, 3], [2, 1], [3, 2]])
            data_q = qpca(A, 1)
            print(data_q)
    """
    norm_x, A = _preprocessing(sample_A)
    n = A.shape[0]
    if n != 2:
        raise NotImplementedError(
            'qpca supports 2-feature input only, got %d features' % n)
    if k not in (1, 2):
        raise ValueError('k must be 1 or 2, got %r' % (k,))

    # the covariance matrix is symmetric, so eigh gives real eigenvalues and an
    # orthonormal basis. sorted descending, principal component first.
    lambda_A, vector_A = np.linalg.eigh(A)
    order = np.argsort(lambda_A)[::-1]
    lambda_A = lambda_A[order]
    vector_A = vector_A[:, order]
    A1 = A.reshape(1, 2 ** n)[0]
    state_vector = np.sqrt(1 / np.sum(A1 * A1)) * A1

    # tao is the eigenvalue threshold. k=1 keeps the top eigenvalue only,
    # k=2 sits below both and keeps the whole space.
    if k == 2:
        tao = min(lambda_A) - 1
    else:
        tao = np.sum(lambda_A) / 2
    qm = _QMachine(5, 5)

    prog = QProg()
    cir = QCircuit()
    _init_cir(qm, state_vector, n)
    cir << _phase_estimation_cir(qm.q_list, lambda_A, tao, n)
    cir << _transition_cir(qm.q_list, n)
    cir << _cnot_cir(qm.q_list)
    cir << _transition_reverse_cir(qm.q_list, n)
    cir << _phase_estimation_reverse_cir(qm.q_list, lambda_A, tao, n)
    prog << cir
    prog <<_measure_cir(prog, qm.q_list, qm.q_list)
    qm.machine.run(prog, 8192)
    result = qm.machine.result().get_prob_dict(qm.q_list)

    if k == 2:
        # tao is below every eigenvalue, so post-selection succeeds with
        # probability 1 and nothing is filtered out. the retained subspace is
        # the whole feature space and its basis comes from vector_A.
        return np.dot(norm_x, vector_A)

    # post-select the ancilla, qubit 0, which is the last character of the key.
    # sorted() so the branch order does not depend on dict insertion order.
    a = [result[v] for v in sorted(result) if int(v[-1]) == 1]
    sum_A = np.sum(np.array(a))
    # probabilities back to amplitudes. unit norm when post-selection leaves
    # two outcomes, which is the non-degenerate case.
    result_circuitA = [np.sqrt(x / sum_A) for x in a]
    vector_qpca = np.array([result_circuitA[0], result_circuitA[-1]]).reshape(1, 2)
    return np.dot(norm_x, np.transpose(vector_qpca))
