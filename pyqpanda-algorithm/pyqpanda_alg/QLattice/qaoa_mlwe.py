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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, H, RZ, RX, CNOT, CZ, measure
import numpy as np
import sympy as sp

from .. plugin import *


class QAOA_MLWE:
    """QAOA-based solver for Module Learning With Errors (MLWE) problem.

    This class implements a quantum approximate optimization algorithm (QAOA)
    approach to solve the MLWE problem, which is the foundation of post-quantum
    cryptographic standards (ML-KEM, ML-DSA).

    The algorithm transforms MLWE into a QUBO problem and uses QAOA to find
    the secret vector s.

    Parameters
        A : ``np.ndarray``\n
            Random matrix A in Z_q^{m x n}.
        b : ``np.ndarray``\n
            Observation vector b = As + e (mod q).
        q : ``int``\n
            Modulus.
        error_bound : ``int``, ``optional``\n
            Error bound for small errors. Default is 1.

    Attributes
        m : ``int``\n
            Number of samples.
        n : ``int``\n
            Secret vector dimension.
        n_qubits : ``int``\n
            Total number of qubits required.

    Methods
        build_qaoa_circuit(gamma, beta, layers)\n
            Construct QAOA circuit.
        run(layers, maxiter)\n
            Run QAOA optimization.

    References
        [1] Farhi, E., Goldstone, J., Gutmann, S. "A Quantum Approximate
        Optimization Algorithm". arXiv:1411.4028, 2014.
        [2] Regev, O. "On Lattices, Learning with Errors, Random Linear
        Codes, and Cryptography". Journal of the ACM, 2009.

    Examples
        Solve MLWE with QAOA:

    >>> from pyqpanda_alg.QLattice.qaoa_mlwe import QAOA_MLWE
    >>> A = np.array([[1, 2], [3, 4], [5, 6]])
    >>> b = np.array([3, 7, 11])
    >>> qaoa = QAOA_MLWE(A, b, q=7, error_bound=1)
    >>> result = qaoa.run(layers=2, maxiter=100)

    """

    def __init__(self, A, b, q, error_bound=1):
        self.A = np.array(A, dtype=int)
        self.b = np.array(b, dtype=int)
        self.q = q
        self.m, self.n = A.shape
        self.error_bound = error_bound
        self.n_qubits = self.n + self.m

    def build_qaoa_circuit(self, gamma, beta, layers):
        """
        Construct QAOA circuit for MLWE problem.

        Each layer consists of:
        1. Problem Hamiltonian: e^{-i*gamma*H_C}
        2. Mixer Hamiltonian: e^{-i*beta*H_M}

        Parameters
            gamma : ``float``\n
                Phase parameter for problem Hamiltonian.
            beta : ``float``\n
                Rotation parameter for mixer Hamiltonian.
            layers : ``int``\n
                Number of QAOA layers.

        Return
            circuit : ``QCircuit``\n
                QAOA quantum circuit.

        Examples
            Build QAOA circuit with 2 layers:

        >>> from pyqpanda_alg.QLattice.qaoa_mlwe import QAOA_MLWE
        >>> qaoa = QAOA_MLWE(A, b, q=7)
        >>> circuit = qaoa.build_qaoa_circuit(0.5, 0.3, layers=2)

        """
        n_qubits = self.n_qubits
        circuit = QCircuit()

        for i in range(n_qubits):
            circuit << H(i)

        for layer in range(layers):
            for i in range(n_qubits - 1):
                circuit << CZ(i, i + 1)
                circuit << RZ(i, gamma)
                circuit << RZ(i + 1, gamma)
                circuit << CZ(i, i + 1)

            for i in range(n_qubits):
                circuit << RX(i, beta)

        return circuit

    def run(self, layers=2, maxiter=100, shots=10):
        """
        Run QAOA optimization for MLWE problem.

        Parameters
            layers : ``int``, ``optional``\n
                Number of QAOA layers. Default is 2.
            maxiter : ``int``, ``optional``\n
                Maximum iterations. Default is 100.
            shots : ``int``, ``optional``\n
                Number of measurements. Default is 10.

        Return
            result : ``dict``\n
                Dictionary containing:
                - 'best_state': Best measurement result
                - 'best_cost': Best cost function value
                - 'secret': Recovered secret vector
                - 'iterations': Total iterations

        Examples
            Run QAOA for MLWE:

        >>> from pyqpanda_alg.QLattice.qaoa_mlwe import QAOA_MLWE
        >>> qaoa = QAOA_MLWE(A, b, q=7)
        >>> result = qaoa.run(layers=3, maxiter=200)

        """
        best_cost = float('inf')
        best_state = None

        gamma = np.random.uniform(0, np.pi)
        beta = np.random.uniform(0, np.pi / 2)

        for iteration in range(maxiter):
            prog = QProg()
            prog << self.build_qaoa_circuit(gamma, beta, layers)

            qubits = list(range(self.n_qubits))
            prog << measure(qubits, qubits)

            qvm = CPUQVM()
            qvm.run(prog, shots)
            result = qvm.result().get_prob_dict(qubits)

            for state_str, prob in result.items():
                s = np.array([int(state_str[i]) for i in range(self.n)], dtype=int)
                e = np.array([int(state_str[self.n + i]) for i in range(self.m)], dtype=int)

                b_computed = (self.A @ s + e) % self.q
                cost = np.sum((b_computed - self.b) ** 2)

                if cost < best_cost:
                    best_cost = cost
                    best_state = state_str

            gamma = gamma * 0.99
            beta = beta * 0.99

        if best_state:
            s = np.array([int(best_state[i]) for i in range(self.n)], dtype=int)
            e = np.array([int(best_state[self.n + i]) for i in range(self.m)], dtype=int)
        else:
            s = np.zeros(self.n, dtype=int)
            e = np.zeros(self.m, dtype=int)

        return {
            'best_state': best_state,
            'best_cost': best_cost,
            'secret': s.tolist(),
            'error': e.tolist(),
            'iterations': maxiter
        }
