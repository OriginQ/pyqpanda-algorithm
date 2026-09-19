"""Train-only preprocessing, independently checked circuits, and score semantics."""

import builtins
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation.forecast_regression import (
    _pauli_z_expectations, _vqc_statevector, make_regressor, regression_metrics,
)


class RidgeTests(unittest.TestCase):
    def test_coefficients_match_independent_regularized_normal_equations(self):
        features = np.array([[0, 10], [1, 15], [2, 11], [3, 20]], dtype=float)
        target = np.array([2, 3, 8, 7], dtype=float)
        model = make_regressor(alpha=0.2).fit(features, target)
        scaled_x = (features - features.min(axis=0)) / np.ptp(features, axis=0)
        scaled_y = (target - target.min()) / np.ptp(target)
        design = np.column_stack([np.ones(len(features)), scaled_x])
        penalty = np.diag([0, 0.2, 0.2])
        expected = np.linalg.solve(design.T @ design + penalty, design.T @ scaled_y)
        report = model.describe()
        assert_allclose([report["intercept"], *report["coefficients"]], expected)
        assert_allclose(model.predict(features), (design @ expected) * np.ptp(target) + target.min())

    def test_future_extremes_do_not_refit_scalers_or_clip(self):
        model = make_regressor(alpha=0).fit([[0, 5], [1, 5], [2, 5]], [10, 20, 30])
        before = model.describe()
        assert_allclose(model.predict([[100, 500], [-100, -500]]), [1010, -990])
        self.assertEqual(before, model.describe())
        self.assertEqual(before["feature_scaler"]["minimum"], [0, 5])
        self.assertEqual(before["feature_scaler"]["maximum"], [2, 5])
        self.assertEqual(before["feature_scaler"]["scale"], [2, 1])
        self.assertEqual(before["target_scaler"]["minimum"], 10)
        self.assertEqual(before["target_scaler"]["maximum"], 30)
        before["feature_scaler"]["minimum"][0] = -1000
        self.assertEqual(model.describe()["feature_scaler"]["minimum"][0], 0)

    def test_constant_target_and_one_sample_are_exact_and_json_serializable(self):
        for features, target in [([[3, 9], [3, 9]], [42, 42]), ([[3, 9]], [[42]])]:
            model = make_regressor().fit(features, target)
            assert_array_equal(model.predict([[0, 0], [100, -100]]), [42, 42])
            json.dumps(model.describe(), allow_nan=False)

    def test_ridge_does_not_require_scipy(self):
        original_import = builtins.__import__

        def no_scipy(name, *args, **kwargs):
            if name.startswith("scipy"):
                raise AssertionError("Ridge imported SciPy")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_scipy):
            result = make_regressor().fit([[1], [2]], [3, 4]).predict([[3]])
        self.assertTrue(np.isfinite(result).all())

    def test_input_validation(self):
        for features, target in [([], []), ([1, 2], [1, 2]), ([[1]], [1, 2]),
                                 ([[1], [2]], [[1, 2], [3, 4]]),
                                 ([[math.nan]], [1]), ([[1]], [math.inf])]:
            with self.subTest(features=features, target=target), self.assertRaises(ValueError):
                make_regressor().fit(features, target)
        for options in [{"method": "fake"}, {"seed": -1}, {"seed": True},
                        {"maxiter": 0}, {"maxiter": 1.5}, {"alpha": -1},
                        {"alpha": math.inf}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_regressor(**options)
        model = make_regressor()
        with self.assertRaises(ValueError):
            model.predict([[1]])
        with self.assertRaises(ValueError):
            model.describe()
        model.fit([[1], [2]], [3, 4])
        for features in [[[1, 2]], [[math.inf]], []]:
            with self.subTest(features=features), self.assertRaises(ValueError):
                model.predict(features)


class CircuitTests(unittest.TestCase):
    def test_zero_state_and_ring_entanglement_have_known_expectations(self):
        zero = _vqc_statevector(np.zeros((1, 4)), np.zeros((4, 3)))
        assert_array_equal(_pauli_z_expectations(zero), [[1, 1, 1, 1]])
        # RY(pi/2) on q0 followed by the ring produces
        # (|0000> + |1110>) / sqrt(2), in q3 q2 q1 q0 order.
        entangled = _vqc_statevector([[np.pi / 2, 0, 0, 0]], np.zeros((4, 3)))[0]
        expected = np.zeros(16)
        expected[[0, 14]] = 1 / np.sqrt(2)
        assert_allclose(entangled, expected, atol=1e-14)
        assert_allclose(_pauli_z_expectations(entangled[None]), [[1, 0, 0, 0]], atol=1e-14)
        # Trace all but q1. Its mixed state proves this is not a product state.
        reduced_q1 = np.zeros((2, 2), dtype=complex)
        for low in range(16):
            for high in range(16):
                if (low & ~2) == (high & ~2):
                    reduced_q1[(low >> 1) & 1, (high >> 1) & 1] += entangled[low] * entangled[high].conjugate()
        self.assertAlmostEqual(float(np.trace(reduced_q1 @ reduced_q1).real), 0.5)

    def test_complex_rotation_state_matches_independent_dense_simulator(self):
        rng = np.random.default_rng(32)
        encoded, parameters = rng.normal(size=(3, 4)), rng.normal(size=(4, 3))
        actual = _vqc_statevector(encoded, parameters)
        assert_allclose(np.sum(np.abs(actual) ** 2, axis=1), 1, atol=1e-14)
        paulis = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
                  np.diag([1, -1])]

        def rotation(angle, pauli):
            return np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * pauli

        def dense_gate(local, qubit):
            result = np.ones((1, 1))
            for index in reversed(range(4)):
                result = np.kron(result, local if index == qubit else np.eye(2))
            return result

        for row, angles in enumerate(encoded):
            state = np.eye(16, dtype=complex)[:, 0]
            for qubit in range(4):
                state = dense_gate(rotation(angles[qubit], paulis[1]), qubit) @ state
            for qubit in range(4):
                for axis in range(3):
                    state = dense_gate(rotation(parameters[qubit, axis], paulis[axis]), qubit) @ state
            for control, target in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                cnot = np.zeros((16, 16))
                for basis in range(16):
                    bits = list(f"{basis:04b}")
                    if bits[3 - control] == "1":
                        bits[3 - target] = "0" if bits[3 - target] == "1" else "1"
                    cnot[int("".join(bits), 2), basis] = 1
                state = cnot @ state
            assert_allclose(actual[row], state, atol=1e-14)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy required for VQC optimization")
class VQCTests(unittest.TestCase):
    def test_angles_train_loss_falls_and_seed_reproduces_model(self):
        rng = np.random.default_rng(71)
        features = rng.uniform(size=(20, 4))
        target = np.sin(2 * features[:, 0]) + features[:, 1] ** 2 - features[:, 3]
        first = make_regressor("vqc", seed=9, maxiter=6).fit(features, target)
        second = make_regressor("vqc", seed=9, maxiter=6).fit(features, target)
        report = first.describe()
        self.assertGreater(np.linalg.norm(np.array(report["angles"]) - report["initial_angles"]), 1e-4)
        self.assertLess(report["optimization"]["training_loss"], report["optimization"]["initial_loss"])
        self.assertLessEqual(report["optimization"]["nit"], 6)
        self.assertGreater(report["optimization"]["nfev"], 1)
        assert_allclose(first.predict(features), second.predict(features), atol=1e-12)
        assert_allclose(report["angles"], second.describe()["angles"], atol=1e-10)
        self.assertEqual(report["initial_angles"], second.describe()["initial_angles"])
        self.assertEqual(report["optimization"]["nit"], second.describe()["optimization"]["nit"])
        json.dumps(report, allow_nan=False)

    def test_projection_is_seeded_fixed_and_scalers_remain_train_only(self):
        features = np.arange(48, dtype=float).reshape(8, 6)
        first = make_regressor("vqc", seed=3, maxiter=1).fit(features, [5] * 8)
        second = make_regressor("vqc", seed=3, maxiter=1).fit(features, [5] * 8)
        other = make_regressor("vqc", seed=4, maxiter=1).fit(features, [5] * 8)
        before = first.describe()
        assert_array_equal(before["projection"], second.describe()["projection"])
        self.assertFalse(np.array_equal(before["projection"], other.describe()["projection"]))
        assert_allclose(np.sum(np.abs(before["projection"]), axis=0), 1)
        assert_array_equal(first.predict(np.full((2, 6), 5000)), [5, 5])
        self.assertEqual(before, first.describe())
        self.assertEqual(before["optimization"]["nit"], 0)
        self.assertEqual(before["optimization"]["training_loss"], 0)


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        report = regression_metrics([1, 2, 3], [2, 4, 6])
        self.assertEqual(report["mae"], 2)
        self.assertAlmostEqual(report["rmse"], math.sqrt(14 / 3))
        self.assertAlmostEqual(report["nse"], -6)
        self.assertAlmostEqual(report["kge"], 1 - math.sqrt(2))
        perfect = regression_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(perfect, {"mae": 0, "rmse": 0, "nse": 1, "kge": 1})

    def test_undefined_scores_are_none_and_serializable(self):
        for observed, predicted, nse in [([2, 2], [2, 2], None),
                                        ([2, 2], [1, 3], None),
                                        ([1, 2, 3], [2, 2, 2], 0),
                                        ([-1, 1], [-2, 2], 0), ([2], [3], None)]:
            with self.subTest(observed=observed, predicted=predicted):
                report = regression_metrics(observed, predicted)
                self.assertEqual(report["nse"], nse)
                self.assertIsNone(report["kge"])
                json.dumps(report, allow_nan=False)

    def test_zero_prediction_mean_is_valid_for_original_kge(self):
        # beta = 0 is valid; this is the standard-deviation-ratio KGE definition.
        report = regression_metrics([1, 2, 3], [-1, 0, 1])
        self.assertAlmostEqual(report["kge"], 0)

    def test_invalid_and_nonfinite_inputs_are_rejected(self):
        for observed, predicted in [([], []), ([1], [1, 2]), ([math.nan], [1]),
                                     ([1], [math.inf]), ([[1, 2]], [[1, 2]])]:
            with self.subTest(observed=observed, predicted=predicted), self.assertRaises(ValueError):
                regression_metrics(observed, predicted)


if __name__ == "__main__":
    unittest.main()
