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

from pyqpanda3.core import QCircuit, QProg, TOFFOLI, SWAP, CNOT, U1, U3, CR, H, X, Y, measure
import numpy as np
import math
import sys

from ..execution import ExecutionOptions, resolve_backend
from .. plugin import *



class _QMachine:
    def __init__(self, q_bit_count, c_bit_count):
        self.q_list = QProg(q_bit_count).qubits()

    def __del__(self):
        pass


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


def qpca(sample_A, k, *, backend=None, execution_options=None):
    """
    QPCA is a quantum version of the classical PCA algorithm, which is widely used in data analysis and machine learning.

    Parameters:
        sample_A: ``ndarray``
            the input matrix for analysis
        k: ``int``
            the dimension to reduce
        backend: execution backend, keyword-only
            The sampling backend running the phase-estimation circuit.
            Defaults to the local simulator.
        execution_options: ``ExecutionOptions``, keyword-only
            Sampling options (shots, timeout, ...) applied to the
            phase-estimation submission.

    Returns:
        out: ``ndarray``
            the output matrix after reducing dimension

    Examples:
        .. code-block:: python

            import numpy as np
            from pyqpanda_alg.QPCA import qpca

            A = np.array([[-1, 2], [-2, -1], [-1, -2], [1, 3], [2, 1], [3, 2]])
            data_q = qpca(A, 1)
            print(data_q)
    """
    norm_x, A = _preprocessing(sample_A)
    lambda_A, vector_A = np.linalg.eig(A)
    A1 = A.reshape(1, 2 ** A.shape[0])[0]
    state_vector = np.sqrt(1 / np.sum(A1 * A1)) * A1

    if A.shape[0] == 2:
        if k == 2:
            tao = min(lambda_A) - 1
        elif k == 1:
            tao = np.sum(lambda_A)/2
        else:
            print('The K Error!')
            sys.exit()
        qm = _QMachine(5, 5)
    if A.shape[0] == 4:
        qm = _QMachine(8, 8)

    prog = QProg()
    cir = QCircuit()
    cir << _phase_estimation_cir(qm.q_list, lambda_A, tao, A.shape[0])
    cir << _transition_cir(qm.q_list, A.shape[0])
    cir << _cnot_cir(qm.q_list)
    cir << _transition_reverse_cir(qm.q_list, A.shape[0])
    cir << _phase_estimation_reverse_cir(qm.q_list, lambda_A, tao, A.shape[0])
    prog << cir
    prog << _measure_cir(prog, qm.q_list, qm.q_list)

    exec_backend = resolve_backend(backend)
    options = execution_options if execution_options is not None else ExecutionOptions()
    sample_task = exec_backend.submit_sample(prog, options=options)
    counts = sample_task.result().single_counts()
    shots = sum(counts.values())
    result = {key: value / shots if shots else 0 for key, value in counts.items()}
    a = []
    data = 0
    if A.shape[0] == 2:
        for i, v in enumerate(result.keys()):
            if int(v[-1]) == 1:
                # a.append(float("%.4f" % (result[v] / 8192)))
                a.append(float("%.4f" % (result[v])))
        if k == 2:
            result_idealA = state_vector
        if k == 1:
            i = np.argmax(lambda_A)
            result_idealA = (np.kron(vector_A[:, i], vector_A[:, i]) * lambda_A[i]) / (
                np.sqrt(lambda_A[i] * lambda_A[i]))
        result_idealA1 = []
        for i in range(len(result_idealA)):
            if result_idealA[i] != 0:
                result_idealA1.append(result_idealA[i])
        result_idealA = result_idealA1
        sum_A = np.sum(np.array(a))
        result_circuitA = []
        for i in range(len(a)):
            result_circuitA.append(np.sqrt(a[i] / sum_A))
        if k == 1:
            vector_qpca = []
            vector_qpca.append(np.sqrt(result_circuitA[0]))
            vector_qpca.append(np.sqrt(result_circuitA[-1]))
            vector_qpca = np.array(vector_qpca).reshape(1, 2)
            data = np.dot(norm_x, np.transpose(vector_qpca))
    return data
