import unittest

import numpy as np

from pyqpanda_alg.HadamardTest import reference_expectation


class TestHadamardTestReference(unittest.TestCase):
    def test_complex_rx_expectation_for_plus_state(self):
        theta = np.pi / 3
        state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        rx = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

        result = reference_expectation(state, rx)

        self.assertAlmostEqual(result.real, np.cos(theta / 2), places=12)
        self.assertAlmostEqual(result.imag, -np.sin(theta / 2), places=12)

    def test_rejects_non_unitary_matrix(self):
        state = np.array([1.0, 0.0], dtype=np.complex128)
        bad = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            reference_expectation(state, bad)

    def test_rejects_non_normalized_state(self):
        state = np.array([1.0, 1.0], dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)

        with self.assertRaises(ValueError):
            reference_expectation(state, identity)


if __name__ == "__main__":
    unittest.main()
