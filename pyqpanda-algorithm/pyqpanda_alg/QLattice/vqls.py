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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, H, RX, RY, RZ, CNOT, measure
import numpy as np

from .. plugin import *


class VQLS:
    """Variational Quantum Lattice Solver for Shortest Vector Problem.

    This class implements a variational quantum eigensolver (VQE) approach
    to solve the Shortest Vector Problem (SVP) on lattices.

    The algorithm encodes ||v||^2 as a problem Hamiltonian and uses a
    hardware-efficient ansatz to find the ground state, which corresponds
    to the shortest lattice vector.

    Parameters
        lattice_basis : ``list[list[int]]``\n
            The lattice basis matrix.
        coefficient_bits : ``int``, ``optional``\n
            Number of bits per coefficient. Default is 3.
        ansatz_depth : ``int``, ``optional``\n
            Depth of the variational ansatz. Default is 3.

    Attributes
        n_qubits : ``int``\n
            Total number of qubits required.
        n_params : ``int``\n
            Number of variational parameters.

    Methods
        build_ansatz(params)\n
            Construct variational ansatz circuit.
        compute_expectation(params, shots)\n
            Compute cost function expectation value.
        run(maxiter, learning_rate, shots)\n
            Run VQLS optimization.

    References
        [1] Peruzzo, A., et al. "A variational eigenvalue solver on a photonic
        quantum processor". Nature Communications, 2014.

    Examples
        Run VQLS on a 2D lattice:

    >>> from pyqpanda_alg.QLattice.vqls import VQLS
    >>> basis = [[2, 1], [1, 3]]
    >>> vqls = VQLS(basis, coefficient_bits=3, ansatz_depth=3)
    >>> result = vqls.run(maxiter=50)
    >>> print(f"Best cost: {result['best_cost']:.4f}")

    """

    def __init__(self, lattice_basis, coefficient_bits=3, ansatz_depth=3):
        self.basis = np.array(lattice_basis, dtype=int)
        self.dimension = len(lattice_basis)
        self.coefficient_bits = coefficient_bits
        self.ansatz_depth = ansatz_depth
        self.n_qubits = self.dimension * coefficient_bits
        self.n_params = self.n_qubits * ansatz_depth * 3

    def build_ansatz(self, params):
        """
        Construct hardware-efficient variational ansatz circuit.

        Each layer consists of:
        1. Single-qubit rotation gates: RX(theta), RY(theta), RZ(theta)
        2. Two-qubit entangling gates: CNOT

        Parameters
            params : ``np.ndarray``\n
                Variational parameters of length n_qubits * depth * 3.

        Return
            circuit : ``QCircuit``\n
                Variational ansatz circuit.

        Examples
            Build ansatz for 6 qubits with depth 3:

        >>> from pyqpanda_alg.QLattice.vqls import VQLS
        >>> vqls = VQLS([[2, 1], [1, 3]], coefficient_bits=3)
        >>> params = np.random.uniform(0, 2*np.pi, vqls.n_params)
        >>> circuit = vqls.build_ansatz(params)

        """
        n = self.n_qubits
        depth = self.ansatz_depth
        circuit = QCircuit()

        for i in range(n):
            circuit << H(i)

        param_idx = 0
        for d in range(depth):
            for i in range(n):
                circuit << RX(i, params[param_idx])
                param_idx += 1
                circuit << RY(i, params[param_idx])
                param_idx += 1
                circuit << RZ(i, params[param_idx])
                param_idx += 1

            for i in range(n - 1):
                circuit << CNOT(i, i + 1)

        return circuit

    def compute_expectation(self, params, shots=1000):
        """
        Compute cost function expectation value.

        Measures <psi(theta)|H_C|psi(theta)> where H_C encodes ||v||^2.

        Parameters
            params : ``np.ndarray``\n
                Variational parameters.
            shots : ``int``, ``optional``\n
                Number of measurements. Default is 1000.

        Return
            expectation : ``float``\n
                Expected value of the cost function.

        """
        prog = QProg()
        prog << self.build_ansatz(params)

        qubits = list(range(self.n_qubits))
        prog << measure(qubits, qubits)

        qvm = CPUQVM()
        qvm.run(prog, shots)
        result = qvm.result().get_prob_dict(qubits)

        expectation = 0.0
        k = self.coefficient_bits
        for state_str, prob in result.items():
            coeffs = []
            for i in range(self.dimension):
                coeff_bits = state_str[i * k:(i + 1) * k]
                coeffs.append(int(coeff_bits, 2))

            if all(c == 0 for c in coeffs):
                continue

            vector = np.array(coeffs) @ self.basis
            length_sq = int(np.sum(vector ** 2))
            expectation += prob * length_sq

        return expectation

    def run(self, maxiter=200, learning_rate=0.1, shots=1000):
        """
        Run VQLS optimization using gradient descent.

        Parameters
            maxiter : ``int``, ``optional``\n
                Maximum number of iterations. Default is 200.
            learning_rate : ``float``, ``optional``\n
                Learning rate for gradient descent. Default is 0.1.
            shots : ``int``, ``optional``\n
                Number of measurements per iteration. Default is 1000.

        Return
            result : ``dict``\n
                Dictionary containing:
                - 'best_cost': Best cost function value found
                - 'best_params': Best parameters found
                - 'best_state': Best quantum state measurement
                - 'cost_history': Cost function history
                - 'iterations': Total iterations performed

        Examples
            Run VQLS optimization:

        >>> from pyqpanda_alg.QLattice.vqls import VQLS
        >>> vqls = VQLS([[2, 1], [1, 3]], coefficient_bits=3)
        >>> result = vqls.run(maxiter=100)
        >>> print(f"Best cost: {result['best_cost']:.4f}")

        """
        params = np.random.uniform(0, 2 * np.pi, self.n_params)

        cost_history = []
        best_cost = float('inf')
        best_params = None

        for iteration in range(maxiter):
            cost = self.compute_expectation(params, shots)
            cost_history.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()

            gradient = np.zeros_like(params)
            epsilon = 0.01
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += epsilon
                params_minus = params.copy()
                params_minus[i] -= epsilon
                gradient[i] = (self.compute_expectation(params_plus) -
                               self.compute_expectation(params_minus)) / (2 * epsilon)

            params = params - learning_rate * gradient

            if iteration > 10 and abs(cost_history[-1] - cost_history[-10]) < 0.01:
                break

        best_state = self._find_best_state(best_params, shots)

        return {
            'best_cost': best_cost,
            'best_params': best_params.tolist(),
            'best_state': best_state,
            'cost_history': cost_history,
            'iterations': len(cost_history)
        }

    def _find_best_state(self, params, shots):
        """Find the most probable quantum state for given parameters."""
        prog = QProg()
        prog << self.build_ansatz(params)

        qubits = list(range(self.n_qubits))
        prog << measure(qubits, qubits)

        qvm = CPUQVM()
        qvm.run(prog, shots)
        result = qvm.result().get_prob_dict(qubits)

        sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        best_state_str = sorted_result[0][0]
        best_prob = sorted_result[0][1]

        coeffs = []
        k = self.coefficient_bits
        for i in range(self.dimension):
            coeff_bits = best_state_str[i * k:(i + 1) * k]
            coeffs.append(int(coeff_bits, 2))

        vector = np.array(coeffs) @ self.basis
        length_sq = int(np.sum(vector ** 2))

        return {
            'state': best_state_str,
            'probability': best_prob,
            'coefficients': coeffs,
            'vector': vector.tolist(),
            'length_squared': length_sq
        }
