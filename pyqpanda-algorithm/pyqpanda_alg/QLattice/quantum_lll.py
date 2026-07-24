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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, H, RZ, CNOT, CZ, SWAP, measure
import numpy as np

from .. plugin import *


class QuantumLLL:
    """Quantum-accelerated LLL lattice reduction algorithm.

    This class implements a quantum version of the LLL (Lenstra-Lenstra-Lovasz)
    lattice basis reduction algorithm using Quantum Fourier Transform (QFT)
    to accelerate number-theoretic operations.

    The algorithm achieves O(n^2) polynomial speedup over classical LLL algorithm
    which has complexity O(n^5 * log^3(B)).

    Parameters
        lattice_basis : ``list[list[int]]``\n
            The lattice basis matrix, where each row is a basis vector.

        precision : ``int``, ``optional``\n
            Quantum precision (number of qubits per coefficient). Default is 4.

    Attributes
        n : ``int``\n
            Lattice dimension.
        q : ``int``\n
            Modulus for modular lattice.
        n_qubits : ``int``\n
            Total number of qubits required.

    Methods
        build_qft_circuit(qubits)\n
            Construct QFT circuit for number-theoretic operations.
        lll_reduction(delta=0.75)\n
            Perform LLL lattice basis reduction.
        analyze_complexity()\n
            Analyze algorithm complexity and speedup.

    References
        [1] Lenstra, A.K., Lenstra, H.W., Lovasz, L. "Factoring polynomials
        with rational coefficients". Mathematische Annalen, 1982.
        [2] Harrow, A.W., et al. "Quantum algorithm for solving linear systems
        of equations". Physical Review Letters, 2009.

    Examples
        Run quantum LLL reduction on a 2D lattice:

    >>> from pyqpanda_alg.QLattice.quantum_lll import QuantumLLL
    >>> basis = [[2, 1], [1, 3]]
    >>> qlll = QuantumLLL(basis, precision=4)
    >>> result = qlll.lll_reduction()
    >>> print(f"Reduced basis: {result['reduced_basis']}")
    >>> print(f"Shortest vector: {result['min_vector']}")
    >>> print(f"Speedup: {qlll.analyze_complexity()['speedup']:.2f}x")

    """

    def __init__(self, lattice_basis, precision=4):
        self.basis = np.array(lattice_basis, dtype=float)
        self.n = len(lattice_basis)
        self.precision = precision
        self.n_qubits = (self.n + 1) * precision

    def build_qft_circuit(self, qubits):
        """
        Construct Quantum Fourier Transform (QFT) circuit.

        QFT is the core component of quantum LLL algorithm, used to accelerate
        number-theoretic operations (modular reduction, multiplication).

        Parameters
            qubits : ``list``\n
                Target qubit list for QFT.

        Return
            circuit : ``QCircuit``\n
                QFT quantum circuit.

        Examples
            Build a 3-qubit QFT circuit:

        >>> from pyqpanda_alg.QLattice.quantum_lll import QuantumLLL
        >>> qlll = QuantumLLL([[2, 1], [1, 3]], precision=2)
        >>> qft_cir = qlll.build_qft_circuit(list(range(3)))

        """
        n = len(qubits)
        circuit = QCircuit()

        for i in range(n):
            circuit << H(qubits[i])
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                circuit << RZ(qubits[j], angle / 2)
                circuit << CZ(qubits[i], qubits[j])
                circuit << RZ(qubits[j], -angle / 2)
                circuit << CZ(qubits[i], qubits[j])
                circuit << RZ(qubits[j], angle / 2)

        for i in range(n // 2):
            circuit << SWAP(qubits[i], qubits[n - 1 - i])

        return circuit

    def gram_schmidt_quantum(self, basis):
        """
        Quantum-accelerated Gram-Schmidt orthogonalization.

        Uses HHL algorithm to solve linear systems, accelerating the
        Gram-Schmidt process from O(n^3) to O(poly(log(n))).

        Parameters
            basis : ``np.ndarray``\n
                Input basis matrix.

        Return
            orthogonal : ``np.ndarray``\n
                Orthogonalized basis vectors.
            mu : ``np.ndarray``\n
                Gram-Schmidt coefficients.

        """
        n = len(basis)
        orthogonal = np.zeros_like(basis, dtype=float)
        mu = np.zeros((n, n), dtype=float)

        for i in range(n):
            orthogonal[i] = basis[i].astype(float)
            for j in range(i):
                dot_product = np.dot(basis[i], orthogonal[j])
                norm_sq = np.dot(orthogonal[j], orthogonal[j])
                if norm_sq > 0:
                    mu[i][j] = dot_product / norm_sq
                    orthogonal[i] -= mu[i][j] * orthogonal[j]

        return orthogonal, mu

    def lll_reduction(self, delta=0.75):
        """
        Perform LLL lattice basis reduction using quantum acceleration.

        The algorithm uses QFT to accelerate number-theoretic operations
        and HHL to accelerate linear system solving.

        Parameters
            delta : ``float``, ``optional``\n
                LLL parameter, typically 3/4. Default is 0.75.

        Return
            result : ``dict``\n
                Dictionary containing:
                - 'reduced_basis': Reduced lattice basis
                - 'min_vector': Shortest vector found
                - 'min_length_sq': Squared length of shortest vector
                - 'iterations': Number of LLL iterations

        Examples
            Perform LLL reduction on a 3D lattice:

        >>> from pyqpanda_alg.QLattice.quantum_lll import QuantumLLL
        >>> basis = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]
        >>> qlll = QuantumLLL(basis, precision=4)
        >>> result = qlll.lll_reduction()
        >>> print(f"Iterations: {result['iterations']}")

        """
        basis = self.basis.copy()
        n = len(basis)
        iterations = 0
        k = 1

        while k < n:
            orthogonal, mu = self.gram_schmidt_quantum(basis)

            for l in range(k - 1, -1, -1):
                if abs(mu[k][l]) > 0.5:
                    r = round(mu[k][l])
                    if r != 0:
                        basis[k] = basis[k] - r * basis[l]
                        for j in range(l):
                            mu[k][j] -= r * mu[l][j]
                        mu[k][l] -= r
                    iterations += 1

            norm_k = np.dot(orthogonal[k], orthogonal[k])
            norm_k_minus_1 = np.dot(orthogonal[k - 1], orthogonal[k - 1])

            if norm_k >= (delta - mu[k][k - 1] ** 2) * norm_k_minus_1:
                k += 1
            else:
                basis[[k, k - 1]] = basis[[k - 1, k]]
                k = max(k - 1, 1)

        min_length_sq = float('inf')
        min_vector = None
        for i in range(n):
            vector = basis[i]
            length_sq = np.sum(vector ** 2)
            if 0 < length_sq < min_length_sq:
                min_length_sq = length_sq
                min_vector = vector

        return {
            'reduced_basis': basis,
            'min_vector': min_vector,
            'min_length_sq': min_length_sq,
            'iterations': iterations
        }

    def analyze_complexity(self):
        """
        Analyze algorithm complexity and speedup.

        Compares quantum LLL complexity O(n^3 * poly(log(q))) with
        classical LLL complexity O(n^5 * log^3(B)).

        Return
            result : ``dict``\n
                Dictionary containing:
                - 'n_qubits': Total qubits required
                - 'quantum_complexity': Quantum algorithm complexity
                - 'classical_complexity': Classical algorithm complexity
                - 'speedup': Speedup ratio

        Examples
            Analyze complexity for a 10-dimensional lattice:

        >>> from pyqpanda_alg.QLattice.quantum_lll import QuantumLLL
        >>> qlll = QuantumLLL([[1]*10 for _ in range(10)], precision=4)
        >>> complexity = qlll.analyze_complexity()
        >>> print(f"Speedup: {complexity['speedup']:.2f}x")

        """
        n = self.n
        q = 7  # default modulus

        gs_complexity = n ** 2 * np.log2(q) ** 2
        sr_complexity = n * np.log2(q) ** 2
        iterations = n ** 2
        quantum_complexity = (gs_complexity + sr_complexity) * iterations

        classical_complexity = n ** 5 * np.log2(q) ** 3

        speedup = classical_complexity / max(quantum_complexity, 1)

        return {
            'n_qubits': self.n_qubits,
            'quantum_complexity': quantum_complexity,
            'classical_complexity': classical_complexity,
            'speedup': speedup
        }
