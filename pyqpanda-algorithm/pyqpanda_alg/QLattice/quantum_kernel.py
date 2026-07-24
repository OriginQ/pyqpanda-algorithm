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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, RX, CZ, RZ, measure
import numpy as np

from .. plugin import *


class QuantumKernel:
    """Quantum kernel method for Module Learning With Errors (MLWE) problem.

    This class implements a quantum kernel approach to solve the MLWE problem
    by mapping samples to high-dimensional Hilbert space using SWAP test.

    The quantum kernel function K(x,y) = |<psi(x)|psi(y)>|^2 is computed
    using the SWAP test circuit, leveraging the exponential dimension
    advantage of quantum states.

    Parameters
        n_features : ``int``\n
            Number of feature qubits for sample encoding.

    Attributes
        n_qubits : ``int``\n
            Total number of qubits required (2 * n_features + 1).

    Methods
        encode_sample(sample, qubits)\n
            Encode classical sample into quantum state.
        build_swap_test(sample1, sample2)\n
            Construct SWAP test circuit.
        compute_kernel(sample1, sample2, shots)\n
            Compute quantum kernel function value.

    References
        [1] Havlicek, V., et al. "Supervised learning with quantum-enhanced
        feature spaces". Nature, 2019.

    Examples
        Compute quantum kernel between two samples:

    >>> from pyqpanda_alg.QLattice.quantum_kernel import QuantumKernel
    >>> qk = QuantumKernel(n_features=4)
    >>> sample1 = np.random.uniform(0, np.pi, 4)
    >>> sample2 = np.random.uniform(0, np.pi, 4)
    >>> kernel_value = qk.compute_kernel(sample1, sample2, shots=1000)

    """

    def __init__(self, n_features):
        self.n_features = n_features
        self.n_qubits = 2 * n_features + 1

    def encode_sample(self, sample, qubits):
        """
        Encode classical sample into quantum state using angle encoding.

        |psi(x)> = tensor_i RX(x_i) |0>

        Parameters
            sample : ``np.ndarray``\n
                Classical sample vector.
            qubits : ``list``\n
                Target qubit list for encoding.

        Return
            circuit : ``QCircuit``\n
                Encoding circuit.

        Examples
            Encode a 4-dimensional sample:

        >>> from pyqpanda_alg.QLattice.quantum_kernel import QuantumKernel
        >>> qk = QuantumKernel(n_features=4)
        >>> sample = np.array([0.1, 0.2, 0.3, 0.4])
        >>> circuit = qk.encode_sample(sample, list(range(4)))

        """
        circuit = QCircuit()
        sample_norm = sample / np.max(np.abs(sample)) * np.pi

        for i, qubit in enumerate(qubits):
            if i < len(sample_norm):
                circuit << RX(qubit, float(sample_norm[i]))

        return circuit

    def build_swap_test(self, sample1, sample2):
        """
        Construct SWAP test circuit for quantum kernel computation.

        The SWAP test estimates |<psi(x)|psi(y)>|^2 by measuring the
        probability of the ancilla qubit being |0>.

        Parameters
            sample1 : ``np.ndarray``\n
                First sample vector.
            sample2 : ``np.ndarray``\n
                Second sample vector.

        Return
            circuit : ``QCircuit``\n
                SWAP test circuit.

        Examples
            Build SWAP test for two samples:

        >>> from pyqpanda_alg.QLattice.quantum_kernel import QuantumKernel
        >>> qk = QuantumKernel(n_features=4)
        >>> circuit = qk.build_swap_test(sample1, sample2)

        """
        n = self.n_features
        circuit = QCircuit()

        qubits1 = list(range(n))
        qubits2 = list(range(n, 2 * n))

        circuit << self.encode_sample(sample1, qubits1)
        circuit << self.encode_sample(sample2, qubits2)

        for i in range(n):
            circuit << CZ(qubits1[i], qubits2[i])
            circuit << RZ(qubits2[i], np.pi / 4)
            circuit << CZ(qubits1[i], qubits2[i])
            circuit << RZ(qubits2[i], -np.pi / 4)

        return circuit

    def compute_kernel(self, sample1, sample2, shots=1000):
        """
        Compute quantum kernel function value K(x,y) = |<psi(x)|psi(y)>|^2.

        Parameters
            sample1 : ``np.ndarray``\n
                First sample vector.
            sample2 : ``np.ndarray``\n
                Second sample vector.
            shots : ``int``, ``optional``\n
                Number of measurements. Default is 1000.

        Return
            kernel_value : ``float``\n
                Quantum kernel function value.

        Examples
            Compute kernel between two random samples:

        >>> from pyqpanda_alg.QLattice.quantum_kernel import QuantumKernel
        >>> qk = QuantumKernel(n_features=4)
        >>> s1 = np.random.uniform(0, np.pi, 4)
        >>> s2 = np.random.uniform(0, np.pi, 4)
        >>> K = qk.compute_kernel(s1, s2, shots=1000)
        >>> print(f"Kernel value: {K:.4f}")

        """
        prog = QProg()
        prog << self.build_swap_test(sample1, sample2)

        qubits = list(range(self.n_qubits))
        prog << measure(qubits, qubits)

        qvm = CPUQVM()
        qvm.run(prog, shots)
        result = qvm.result().get_prob_dict(qubits)

        prob_0 = 0.0
        for state_str, prob in result.items():
            if state_str[-1] == '0':
                prob_0 += prob

        kernel_value = 2 * prob_0 - 1
        return max(0, kernel_value)

    def compute_kernel_matrix(self, samples, shots=1000):
        """
        Compute kernel matrix for a set of samples.

        Parameters
            samples : ``list[np.ndarray]``\n
                List of sample vectors.
            shots : ``int``, ``optional``\n
                Number of measurements per kernel computation. Default is 1000.

        Return
            K : ``np.ndarray``\n
                Kernel matrix of shape (n_samples, n_samples).

        Examples
            Compute kernel matrix for 5 samples:

        >>> from pyqpanda_alg.QLattice.quantum_kernel import QuantumKernel
        >>> qk = QuantumKernel(n_features=4)
        >>> samples = [np.random.uniform(0, np.pi, 4) for _ in range(5)]
        >>> K = qk.compute_kernel_matrix(samples)

        """
        n_samples = len(samples)
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = self.compute_kernel(samples[i], samples[j], shots)
                K[j, i] = K[i, j]

        return K
