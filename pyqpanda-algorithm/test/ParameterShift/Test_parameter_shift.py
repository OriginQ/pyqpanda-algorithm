import unittest

import numpy as np

from pyqpanda_alg.ParameterShift import (
    ParameterShiftRule,
    build_shift_schedule,
    evaluate_shift_batch,
    parameter_shift_gradient,
    parameter_shift_jacobian,
    parameter_shift_vjp,
)


class TestParameterShift(unittest.TestCase):
    def test_standard_rule_matches_analytic_gradient(self):
        theta = np.array([0.31, -0.27])

        def objective(x):
            return np.cos(x[0]) + 0.4 * np.sin(x[1])

        got = parameter_shift_gradient(objective, theta)
        expected = np.array([-np.sin(theta[0]), 0.4 * np.cos(theta[1])])
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_frequency_two_rule(self):
        theta = np.array([0.13])
        rules = [ParameterShiftRule.two_eigenvalue(2.0)]

        got = parameter_shift_gradient(
            lambda x: np.cos(2.0 * x[0]),
            theta,
            rules=rules,
        )
        expected = np.array([-2.0 * np.sin(2.0 * theta[0])])
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_vector_jacobian(self):
        theta = np.array([0.2, 0.4])

        def vector_output(x):
            return np.array(
                [
                    np.cos(x[0]),
                    np.sin(x[0]) * np.cos(x[1]),
                ]
            )

        got = parameter_shift_jacobian(vector_output, theta)
        expected = np.array(
            [
                [-np.sin(theta[0]), np.cos(theta[0]) * np.cos(theta[1])],
                [0.0, -np.sin(theta[0]) * np.sin(theta[1])],
            ]
        )
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_remote_batch_schedule_reconstructs(self):
        theta = np.array([0.2, -0.4])
        schedule = build_shift_schedule(theta, trainable=[1])

        def batch(points):
            return [np.cos(point[1]) for point in points]

        values = evaluate_shift_batch(batch, schedule)
        got = schedule.reconstruct(values)
        expected = np.array([0.0, np.sin(0.4)])
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_vjp(self):
        theta = np.array([0.2, -0.4])
        cotangent = np.array([2.0, -3.0])

        def vector_output(x):
            return np.array([np.cos(x[0]), np.sin(x[1])])

        got = parameter_shift_vjp(vector_output, theta, cotangent)
        expected = np.array(
            [
                -2.0 * np.sin(theta[0]),
                -3.0 * np.cos(theta[1]),
            ]
        )
        np.testing.assert_allclose(got, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
