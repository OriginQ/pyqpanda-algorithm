import unittest

import numpy as np

from pyqpanda_alg.QNG import (
    block_diagonal_metric,
    fubini_study_metric,
    natural_gradient_step,
    solve_natural_gradient,
)


class TestQuantumNaturalGradient(unittest.TestCase):
    def test_single_rotation_fubini_study_metric(self):
        theta = 0.37
        state = np.array(
            [
                np.cos(theta / 2.0),
                -1j * np.sin(theta / 2.0),
            ]
        )
        jacobian = np.array(
            [
                [
                    -0.5 * np.sin(theta / 2.0),
                    -0.5j * np.cos(theta / 2.0),
                ]
            ]
        )

        metric = fubini_study_metric(state, jacobian)
        np.testing.assert_allclose(metric, [[0.25]], atol=1e-12)

    def test_global_phase_direction_projects_to_zero(self):
        state = np.array([1.0 + 0.0j, 0.0 + 0.0j])
        jacobian = np.array([[1.0j, 0.0j]])

        metric = fubini_study_metric(state, jacobian)
        np.testing.assert_allclose(metric, [[0.0]], atol=1e-12)

    def test_regularized_solve_and_diagnostics(self):
        metric = np.diag([0.25, 1.0])
        gradient = np.array([1.0, 2.0])

        result = solve_natural_gradient(
            metric,
            gradient,
            damping=0.0,
        )
        np.testing.assert_allclose(result.direction, [4.0, 2.0], atol=1e-12)
        self.assertEqual(result.diagnostics.rank, 2)
        self.assertAlmostEqual(result.diagnostics.condition_number, 4.0)

    def test_block_diagonal_metric(self):
        metric = np.array(
            [
                [1.0, 0.2, 0.3],
                [0.2, 2.0, 0.4],
                [0.3, 0.4, 3.0],
            ]
        )
        got = block_diagonal_metric(metric, [[0, 1], [2]])
        expected = np.array(
            [
                [1.0, 0.2, 0.0],
                [0.2, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        np.testing.assert_allclose(got, expected)

    def test_natural_gradient_step(self):
        theta = np.array([0.4])
        state = np.array(
            [
                np.cos(theta[0] / 2.0),
                -1j * np.sin(theta[0] / 2.0),
            ]
        )
        jacobian = np.array(
            [
                [
                    -0.5 * np.sin(theta[0] / 2.0),
                    -0.5j * np.cos(theta[0] / 2.0),
                ]
            ]
        )
        gradient = np.array([0.1])

        step = natural_gradient_step(
            theta,
            state,
            jacobian,
            gradient,
            learning_rate=0.5,
            damping=0.0,
        )
        np.testing.assert_allclose(step.parameters, [0.2], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
