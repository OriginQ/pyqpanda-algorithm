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

import unittest
import numpy as np
from pyqpanda_alg.QLattice.quantum_lll import QuantumLLL


class TestQuantumLLL(unittest.TestCase):
    """Test cases for QuantumLLL algorithm."""

    def test_init(self):
        """Test QuantumLLL initialization."""
        basis = [[2, 1], [1, 3]]
        qlll = QuantumLLL(basis, precision=4)
        self.assertEqual(qlll.n, 2)
        self.assertEqual(qlll.precision, 4)
        self.assertEqual(qlll.n_qubits, 12)

    def test_gram_schmidt(self):
        """Test Gram-Schmidt orthogonalization."""
        basis = [[2, 1], [1, 3]]
        qlll = QuantumLLL(basis, precision=4)
        orthogonal, mu = qlll.gram_schmidt_quantum(np.array(basis, dtype=float))

        # Check orthogonality
        dot_product = np.dot(orthogonal[0], orthogonal[1])
        self.assertAlmostEqual(dot_product, 0, places=10)

    def test_lll_reduction(self):
        """Test LLL reduction."""
        basis = [[2, 1], [1, 3]]
        qlll = QuantumLLL(basis, precision=4)
        result = qlll.lll_reduction()

        self.assertIn('reduced_basis', result)
        self.assertIn('min_vector', result)
        self.assertIn('min_length_sq', result)
        self.assertIn('iterations', result)
        self.assertGreater(result['min_length_sq'], 0)

    def test_complexity_analysis(self):
        """Test complexity analysis."""
        basis = [[2, 1], [1, 3]]
        qlll = QuantumLLL(basis, precision=4)
        complexity = qlll.analyze_complexity()

        self.assertIn('n_qubits', complexity)
        self.assertIn('quantum_complexity', complexity)
        self.assertIn('classical_complexity', complexity)
        self.assertIn('speedup', complexity)
        self.assertGreater(complexity['speedup'], 1)

    def test_qft_circuit(self):
        """Test QFT circuit construction."""
        basis = [[2, 1], [1, 3]]
        qlll = QuantumLLL(basis, precision=2)
        qft_cir = qlll.build_qft_circuit(list(range(4)))
        self.assertIsNotNone(qft_cir)


if __name__ == '__main__':
    unittest.main()
