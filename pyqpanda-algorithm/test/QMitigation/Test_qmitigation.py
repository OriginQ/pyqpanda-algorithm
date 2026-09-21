import unittest

import numpy as np

from pyqpanda_alg.QMitigation import (
    assignment_matrix_from_calibration,
    factorized_assignment_matrix,
    global_fold_sequence,
    mitigate_probabilities,
    polynomial_extrapolate,
    project_probability_simplex,
    richardson_coefficients,
    richardson_extrapolate,
)


class TestQuantumErrorMitigation(unittest.TestCase):
    def test_assignment_matrix_and_mitigation(self):
        assignment = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        )
        ideal = np.array([0.7, 0.3])
        observed = assignment @ ideal

        got = mitigate_probabilities(observed, assignment)
        np.testing.assert_allclose(got, ideal, atol=1e-12)

    def test_calibration_counts_build_columns(self):
        calibration = {
            "0": {"0": 90, "1": 10},
            "1": {"0": 20, "1": 80},
        }
        got = assignment_matrix_from_calibration(calibration)
        expected = np.array(
            [
                [0.9, 0.2],
                [0.1, 0.8],
            ]
        )
        np.testing.assert_allclose(got, expected)

    def test_factorized_assignment_matrix(self):
        first = np.array([[0.9, 0.1], [0.1, 0.9]])
        second = np.array([[0.8, 0.2], [0.2, 0.8]])
        got = factorized_assignment_matrix([first, second])
        np.testing.assert_allclose(got, np.kron(first, second))

    def test_simplex_projection(self):
        got = project_probability_simplex([0.8, 0.3, -0.1])
        self.assertAlmostEqual(float(got.sum()), 1.0)
        self.assertTrue(np.all(got >= 0.0))

    def test_global_fold_scale_three(self):
        operations = ("A", "B")
        got = global_fold_sequence(
            operations,
            3,
            inverse=lambda operation: operation + "_dagger",
        )
        self.assertEqual(
            got,
            ("A", "B", "B_dagger", "A_dagger", "A", "B"),
        )

    def test_richardson_cancels_linear_noise(self):
        scales = [1.0, 3.0]
        noisy = np.array([0.7 + 0.05 * scale for scale in scales])
        coefficients = richardson_coefficients(scales)
        self.assertAlmostEqual(float(coefficients.sum()), 1.0)
        self.assertAlmostEqual(
            float(richardson_extrapolate(scales, noisy)),
            0.7,
            places=12,
        )

    def test_polynomial_extrapolation(self):
        scales = np.array([1.0, 2.0, 3.0, 4.0])
        noisy = 0.61 + 0.03 * scales - 0.002 * scales**2
        fit = polynomial_extrapolate(scales, noisy, degree=2)
        self.assertAlmostEqual(float(fit.value), 0.61, places=12)
        self.assertLess(fit.residual_norm, 1e-12)


if __name__ == "__main__":
    unittest.main()
