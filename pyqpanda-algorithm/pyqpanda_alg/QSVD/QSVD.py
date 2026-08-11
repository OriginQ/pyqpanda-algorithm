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
import pandas as pd
from pyqpanda3.core import QCircuit, QProg, RY, X, Encode
from pyqpanda3.hamiltonian import Hamiltonian
from scipy.optimize import minimize

from ..execution import DeviceCapabilityError, ExecutionOptions, resolve_backend

class SVD:
    """
    This class provides a framework for the Quantum Singular Value Decomposition (QSVD) algorithm,
    which estimates the singular values of an input matrix via parameterized quantum circuits.

    Parameters
        matrix_in : ``array_like``, optional
            The input real-valued matrix to be decomposed. The matrix will be zero-padded to the nearest power-of-two shape.
        depth : ``int``, optional (default=8)
            The number of variational layers used in the parameterized quantum circuits. Must be in the range [2, 10].

    Attributes
        matrix : ``ndarray``
            The input matrix converted into a 2D NumPy array.
        q0 : ``int``
            Number of qubits required to encode the number of matrix rows.
        q1 : ``int``
            Number of qubits required to encode the number of matrix columns.
        q_matrix : ``ndarray``
            The zero-padded version of the input matrix to fit a 2^n × 2^m dimension.
        normal_value : ``float``
            Normalization factor (Frobenius norm of the input matrix).
        iter_depth : ``int``
            Depth of the variational quantum circuit layers.
        parameter : ``ndarray``
            Initial random parameters used for variational optimization.

    Methods
        cir(qlist, para)
            Builds a parameterized variational quantum circuit based on the input qubit list and parameter vector.

        loss(ls, return_type=True)
            Computes the loss function between the diagonal of the original matrix and that of the decomposed one.
            If `return_type` is False, returns the reconstructed matrix and its dominant singular vector index.

        QSVD_min()
            Optimizes the variational parameters using classical optimization (SLSQP) to minimize the loss function.

        return_diag(par)
            Returns the diagonal-approximated matrix reconstructed from the optimized parameters.

        max_eig(return_mat='0', par=None, max_index=0)
            Extracts the principal eigenvector of the reconstructed left or right unitary matrix from QSVD based on
            the given `return_mat` flag ('0' for left unitary, '1' for right unitary), using the inverse quantum circuit.

    Note
    
        This implementation follows a hybrid quantum-classical approach using amplitude encoding, variational circuits,
        and classical optimization to extract approximated singular values of the input matrix.

    Requirements
        This class depends on pyQPanda as the quantum programming interface and NumPy for numerical operations.

    Examples:
        .. code-block:: python

            from pyqpanda_alg import QSVD
            import numpy as np

            matrix = np.random.random(64).reshape([8, 8])
            para = QSVD.SVD(matrix_in=matrix).QSVD_min()
            qeig = QSVD.SVD(matrix_in=matrix).return_diag(para)
            l1 = QSVD.SVD(matrix_in=matrix).max_eig('0', para, 0)
            l2 = QSVD.SVD(matrix_in=matrix).max_eig('1', para, 0)

            print(l1)
            print(l2)
            print(np.sort(np.diag(qeig)))

            u, s, v = np.linalg.svd(matrix)
            print(u[:, 0])
            print(v[0])
            print(np.sort(s))
    """
    def __init__(self,
                 matrix_in=None,
                 depth=8,
                 ):
        self.matrix = np.array(matrix_in)
        if len(self.matrix.shape) != 2:
            raise TypeError('please input data with 2 dimension !')
        if depth not in [i for i in range(2, 11)]:
            raise ValueError('please input integer number in 2~10 !')
        self.q0 = int(np.ceil(np.log2(self.matrix.shape[0])))
        self.q1 = int(np.ceil(np.log2(self.matrix.shape[1])))
        self.q_matrix = np.pad(self.matrix, ((0, 2**self.q0-self.matrix.shape[0]), (0, 2**self.q1-self.matrix.shape[1])))
        self.normal_value = np.sum(self.matrix.copy().flatten()**2)**0.5
        self.iter_depth = depth
        self.parameter = 0.5 * np.pi * np.random.random((self.q0 + self.q1) * self.iter_depth)
        self._backend = None
        self._execution_options = None
        self.loss_value = None
        self.loss_uncertainty = None

    def cir(self, qlist, para):
        Qcir = QCircuit()
        for i in range(self.iter_depth):
            for j in range(len(qlist)):
                Qcir << RY(qlist[j], para[j+len(qlist)*i])
            for j in range(len(qlist)):
                if j <= len(qlist) - 2:
                    Qcir << X(qlist[j+1]).control(qlist[j])
        return Qcir

    def _overlap_observable(self):
        """
        Diagonal-overlap Hamiltonian whose expectation is the QSVD cost.

        The cost counts the probability that the row and column registers
        read the same index: ``sum_i P(row=i, col=i)``.  For
        ``n = min(q0, q1)`` shared index bits this projector sum expands
        into ``2**n`` Pauli terms ``2**-n * sum_A Z_A(row) Z_A(col)`` over
        every bit subset ``A`` of the diagonal, which is exactly what
        ``submit_estimate`` can evaluate.
        """
        n = min(self.q0, self.q1)
        hamiltonian = {}
        for mask in range(1 << n):
            coefficient = 2.0 ** (-n)
            if mask == 0:
                hamiltonian[""] = coefficient
                continue
            terms = []
            for a in range(n):
                if mask & (1 << a):
                    terms.append("Z%d" % (self.q0 + a))
                    terms.append("Z%d" % a)
            hamiltonian[" ".join(terms)] = coefficient
        return Hamiltonian(hamiltonian)

    def loss(self, ls, return_type=True):
        prog = QProg(self.q0 + self.q1)
        qvec0 = prog.qubits()[:self.q0]
        qvec1 = prog.qubits()[self.q0:]
        cir = QCircuit()
        encode = Encode()
        encode.amplitude_encode(qvec1+qvec0, self.q_matrix.flatten()/self.normal_value)
        cir << encode.get_circuit()
        cir << self.cir(qlist=qvec0, para=ls[0:self.q0 * self.iter_depth])
        cir << self.cir(qlist=qvec1, para=ls[self.q0 * self.iter_depth:(self.q0 + self.q1) * self.iter_depth])
        prog << cir
        if return_type:
            backend = self._backend if self._backend is not None else resolve_backend(None)
            options = self._execution_options if self._execution_options is not None else ExecutionOptions()
            estimate = backend.submit_estimate((prog, self._overlap_observable()), options=options)
            same_p = estimate.result().single_value()
            self.loss_value = 1 - same_p
            # Statistical uncertainty of the estimate: the overlap operator
            # is a projector sum (0 <= O <= I), so its variance is bounded
            # by p(1-p) with p = E[O]; the Bernoulli standard error is
            # therefore a conservative shot-count estimate.
            p = min(max(same_p, 0.0), 1.0)
            self.loss_uncertainty = (p * (1 - p) / options.shots) ** 0.5
            return 1 - same_p
        return self._singular_vector_phase(prog)

    def _singular_vector_phase(self, prog):
        """Return the real phase matrix of ``prog`` and its peak index.

        Exact singular vectors need full state access, so backends
        without state-vector (or tomography) capability are rejected
        before any submission.
        """
        backend = self._backend if self._backend is not None else resolve_backend(None)
        options = self._execution_options if self._execution_options is not None else ExecutionOptions()
        if not (backend.capabilities.statevector or backend.capabilities.tomography):
            raise DeviceCapabilityError(
                "exact singular vectors require a backend with state-vector "
                "or tomography capability"
            )
        statevector_task = backend.submit_statevector(prog, options=options)
        phase = np.asarray(statevector_task.result().single_statevector()).real
        phase = phase.reshape(2**self.q1, 2**self.q0)
        return phase, np.argmax(abs(phase))

    def QSVD_min(self, *, backend=None, execution_options=None, maxiter=100):
        """
        Optimize the variational parameters using classical optimization (SLSQP).

        Parameters
            backend : ``ExecutionBackend``, ``optional``
                The execution backend evaluating the overlap cost as an
                observable expectation. Keyword-only. If not given, the
                CPU LocalBackend is used.

            execution_options : ``ExecutionOptions``, ``optional``
                Options controlling the backend submissions. Keyword-only.

            maxiter : ``int``, ``optional``
                Maximum number of SLSQP iterations. Keyword-only.
        """
        self._backend = resolve_backend(backend)
        self._execution_options = (
            execution_options if execution_options is not None else ExecutionOptions()
        )
        final_x = minimize(
            self.loss, x0=self.parameter, method='SLSQP', tol=1e-10,
            options={'maxiter': maxiter},
        ).x
        return final_x

    def return_diag(self, par):
        res = self.normal_value * self.loss(par, return_type=False)[0]
        return abs(res)

    def max_eig(self, return_mat='0', par=None, max_index=0):
        backend = self._backend if self._backend is not None else resolve_backend(None)
        options = self._execution_options if self._execution_options is not None else ExecutionOptions()
        if not (backend.capabilities.statevector or backend.capabilities.tomography):
            raise DeviceCapabilityError(
                "exact singular vectors require a backend with state-vector "
                "or tomography capability"
            )
        cir = QCircuit()
        ss = max_index % 2**self.q0
        bi0 = '{:b}'.format(ss).rjust(self.q0, '0')
        bi1 = '{:b}'.format(ss).rjust(self.q1, '0')
        if return_mat == '0':
            prog = QProg(self.q0)
            qvec = prog.qubits()
            for j in range(len(qvec)):
                if bi0[-j-1] == '1':
                    cir << X(qvec[j])
            cir << self.cir(qlist=qvec, para=par[0:self.q0 * self.iter_depth]).dagger()
            prog << cir
            statevector_task = backend.submit_statevector(prog, options=options)
            return np.asarray(statevector_task.result().single_statevector()).real

        elif return_mat == '1':
            prog = QProg(self.q1)
            qvec = prog.qubits()
            for j in range(len(qvec)):
                if bi1[-j-1] == '1':
                    cir << X(qvec[j])
            cir << self.cir(qlist=qvec, para=par[self.q0 * self.iter_depth:(self.q0 + self.q1) * self.iter_depth]).dagger()
            prog << cir
            statevector_task = backend.submit_statevector(prog, options=options)
            return np.asarray(statevector_task.result().single_statevector()).real
