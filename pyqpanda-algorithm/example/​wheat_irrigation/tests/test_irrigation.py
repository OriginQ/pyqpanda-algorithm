"""Independent checks of water-balance and QUBO equations on tiny instances.

Run from the project directory with ``python -m unittest discover -s tests -v``.
The reference objective deliberately uses a daily water-balance loop instead of
the cumulative-weight algebra used to construct the QUBO.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irrigation_model import IrrigationInstance, build_qubo


def example_data(budget=2):
    return {
        "name": "independent mathematical test",
        "source_note": "Synthetic values for equation verification only.",
        "days": 3,
        "budget_events": budget,
        "dose_mm": 9.0,
        "water_cost": 0.4,
        "spatial_weight": 1.7,
        "timing_weight": 2.3,
        "zone_area_ha": 1.0,
        "zones": [
            {
                "name": "A",
                "taw_mm": 40.0,
                "initial_moisture_mm": 17.0,
                "window_days": [1, 2, 3],
                "target_mm": [24.0, 25.0, 23.0],
                "stress_weights": [0.7, 1.2, 0.9],
                "et0_mm": [3.0, 5.0, 4.0],
                "crop_coefficient": [1.0, 1.2, 0.8],
                "rainfall_mm": [1.0, 0.0, 2.0],
                "effective_rain_fraction": 0.6,
                "capillary_rise_mm": [0.3, 0.4, 0.5],
            },
            {
                "name": "B",
                "taw_mm": 35.0,
                "initial_moisture_mm": 13.0,
                "window_days": [2, 3],
                "target_mm": 22.0,
                "stress_weights": 1.1,
                "et0_mm": [4.0, 3.0, 5.0],
                "crop_coefficient": 1.1,
                "rainfall_mm": [0.0, 2.0, 1.0],
                "effective_rain_fraction": 0.8,
                "capillary_rise_mm": 0.6,
            },
        ],
        "adjacency": [["A", "B"]],
    }


def binary_vectors(width):
    """Little-endian order, matching statevector basis indices."""
    indices = np.arange(1 << width, dtype=np.uint64)
    return ((indices[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(int)


def on_day(value, day):
    return value[day] if isinstance(value, (list, tuple)) else value


def independent_physics(data, decisions):
    """Evaluate daily physical costs without reading constructed coefficients."""
    days = data["days"]
    schedule = np.zeros((len(data["zones"]), days), dtype=int)
    cursor = 0
    for zi, zone in enumerate(data["zones"]):
        for day in sorted(zone["window_days"]):
            schedule[zi, day - 1] = decisions[cursor]
            cursor += 1
    moisture = np.empty_like(schedule, dtype=float)
    stress = 0.0
    for zi, zone in enumerate(data["zones"]):
        stored = zone["initial_moisture_mm"]
        for day in range(days):
            stored += (
                data["dose_mm"] * schedule[zi, day]
                + zone["rainfall_mm"][day] * zone["effective_rain_fraction"]
                + on_day(zone["capillary_rise_mm"], day)
                - zone["et0_mm"][day] * on_day(zone["crop_coefficient"], day)
            )
            moisture[zi, day] = stored
            stress += on_day(zone["stress_weights"], day) * (
                stored - on_day(zone["target_mm"], day)
            ) ** 2
    names = {zone["name"]: zi for zi, zone in enumerate(data["zones"])}
    spatial = sum(
        int(np.dot(schedule[names[left]], schedule[names[right]]))
        for left, right in data["adjacency"]
    )
    timing = int(np.sum(schedule[:, :-1] * schedule[:, 1:]))
    objective = (
        data["water_cost"] * data["dose_mm"] * int(schedule.sum())
        + stress
        + data["spatial_weight"] * spatial
        + data["timing_weight"] * timing
    )
    return objective, schedule, moisture


class IrrigationModelTests(unittest.TestCase):
    def setUp(self):
        self.data = example_data()
        self.model = build_qubo(IrrigationInstance.from_dict(self.data))

    def test_all_bitstrings_match_daily_water_balance_and_budget(self):
        model = self.model
        bits = binary_vectors(model.n_variables)
        expected = []
        for state in bits:
            decisions = state[: model.n_decisions]
            objective, _, _ = independent_physics(self.data, decisions)
            slack = np.dot(state[model.n_decisions :], model.slack_weights)
            residual = int(decisions.sum()) + slack - self.data["budget_events"]
            expected.append(objective + model.budget_penalty * residual**2)
        assert_allclose(model.energy(bits), expected, atol=1e-9, rtol=1e-12)
        assert_allclose(model.energy(bits[-1]), expected[-1], atol=1e-9)
        self.assertEqual(model.n_decisions, 5)
        self.assertEqual(model.n_variables, 7)
        assert_array_equal(np.tril(model.quadratic), 0)

    def test_all_decisions_decode_and_preserve_soil_memory(self):
        for decisions in binary_vectors(self.model.n_decisions):
            _, schedule, moisture = independent_physics(self.data, decisions)
            assert_array_equal(self.model.decode_schedule(decisions), schedule)
            assert_allclose(self.model.moisture_trajectory(decisions), moisture, atol=1e-12)
            self.assertEqual(schedule[1, 0], 0)  # B's canal is closed on day 1.

    def test_linear_bucket_deliberately_does_not_clip(self):
        data = example_data()
        data["dose_mm"] = 100.0
        data["zones"][0]["initial_moisture_mm"] = 0.0
        data["zones"][0]["rainfall_mm"] = [0.0] * 3
        data["zones"][0]["capillary_rise_mm"] = 0.0
        model = build_qubo(IrrigationInstance.from_dict(data))
        zeros = np.zeros(model.n_decisions, dtype=int)
        self.assertLess(model.moisture_trajectory(zeros)[0, 0], 0.0)
        zeros[0] = 1
        self.assertGreater(model.moisture_trajectory(zeros)[0, 0], data["zones"][0]["taw_mm"])

    def test_ising_energy_matches_qubo_for_every_basis_state(self):
        model = self.model
        h, couplings, offset = model.to_ising()
        bits = binary_vectors(model.n_variables)
        spins = 1 - 2 * bits
        ising_energies = offset + spins @ h + np.einsum("bi,ij,bj->b", spins, couplings, spins)
        assert_allclose(ising_energies, model.energy(bits), atol=1e-9, rtol=1e-12)
        assert_array_equal(np.tril(couplings), 0)

    def test_bounded_slack_and_encode_cover_every_feasible_schedule(self):
        for budget, weights in ((0, []), (2, [1, 1]), (4, [1, 2, 1])):
            with self.subTest(budget=budget):
                data = example_data(budget)
                model = build_qubo(IrrigationInstance.from_dict(data))
                assert_array_equal(model.slack_weights, weights)
                represented = set((binary_vectors(len(weights)) @ model.slack_weights).tolist())
                self.assertEqual(represented, set(range(budget + 1)))
                for decisions in binary_vectors(model.n_decisions):
                    if decisions.sum() > budget:
                        with self.assertRaises(ValueError):
                            model.encode_schedule(decisions)
                    else:
                        full_bits = model.encode_schedule(decisions)
                        assert_array_equal(full_bits[: model.n_decisions], decisions)
                        slack = np.dot(full_bits[model.n_decisions :], model.slack_weights)
                        self.assertEqual(decisions.sum() + slack, budget)
                        objective, _, _ = independent_physics(data, decisions)
                        self.assertAlmostEqual(model.energy(full_bits), objective, places=8)

    def test_certified_penalty_makes_all_global_minima_feasible(self):
        for budget in (0, 2, 4):
            with self.subTest(budget=budget):
                data = example_data(budget)
                model = build_qubo(IrrigationInstance.from_dict(data))
                zero_objective = independent_physics(data, np.zeros(model.n_decisions))[0]
                objective_linear = []
                for decision in np.eye(model.n_decisions, dtype=int):
                    objective_linear.append(independent_physics(data, decision)[0] - zero_objective)
                bound = max(0.0, -min(objective_linear))
                self.assertGreater(model.budget_penalty, bound)
                bits = binary_vectors(model.n_variables)
                energies = model.energy(bits)
                minima = bits[np.isclose(energies, energies.min(), atol=1e-8, rtol=0)]
                self.assertTrue(np.all(minima[:, : model.n_decisions].sum(axis=1) <= budget))
                self.assertTrue(np.all(
                    minima[:, : model.n_decisions].sum(axis=1)
                    + minima[:, model.n_decisions :] @ model.slack_weights == budget
                ))

    def test_spatial_and_timing_terms_are_soft_costs(self):
        data = example_data(budget=5)
        model = build_qubo(IrrigationInstance.from_dict(data))
        decisions = np.ones(model.n_decisions, dtype=int)
        full_bits = model.encode_schedule(decisions)
        objective, schedule, _ = independent_physics(data, decisions)
        # All five open-window events can coexist, despite these soft costs.
        self.assertAlmostEqual(model.energy(full_bits), objective, places=8)
        baseline = copy.deepcopy(data)
        baseline["spatial_weight"] = 0.0
        baseline["timing_weight"] = 0.0
        baseline_model = build_qubo(IrrigationInstance.from_dict(baseline))
        difference = model.energy(full_bits) - baseline_model.energy(baseline_model.encode_schedule(decisions))
        expected = 2 * data["spatial_weight"] + 3 * data["timing_weight"]
        self.assertAlmostEqual(difference, expected, places=8)
        self.assertEqual(int(schedule.sum()), 5)

    def test_invalid_instance_values_are_rejected(self):
        mutations = [
            ("negative budget", lambda d: d.update(budget_events=-1)),
            ("noninteger budget", lambda d: d.update(budget_events=1.5)),
            ("zero horizon", lambda d: d.update(days=0)),
            ("zero dose", lambda d: d.update(dose_mm=0)),
            ("negative water cost", lambda d: d.update(water_cost=-1)),
            ("negative spatial weight", lambda d: d.update(spatial_weight=-1)),
            ("negative timing weight", lambda d: d.update(timing_weight=-1)),
            ("NaN dose", lambda d: d.update(dose_mm=float("nan"))),
            ("unknown adjacent zone", lambda d: d.update(adjacency=[["A", "missing"]])),
            ("window outside horizon", lambda d: d["zones"][0].update(window_days=[1, 4])),
            ("duplicate window day", lambda d: d["zones"][0].update(window_days=[1, 1])),
            ("negative stress weight", lambda d: d["zones"][0].update(stress_weights=-0.1)),
            ("ET0 length mismatch", lambda d: d["zones"][0].update(et0_mm=[1.0])),
            ("invalid effective rain fraction", lambda d: d["zones"][0].update(effective_rain_fraction=1.2)),
            ("infinite forcing", lambda d: d["zones"][0].update(capillary_rise_mm=float("inf"))),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                data = example_data()
                mutate(data)
                with self.assertRaises(ValueError):
                    IrrigationInstance.from_dict(data)

    def test_nonbinary_and_wrong_width_bitstrings_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.energy(np.zeros(self.model.n_variables - 1))
        invalid = np.zeros(self.model.n_variables)
        invalid[0] = 0.5
        with self.assertRaises(ValueError):
            self.model.energy(invalid)
        with self.assertRaises(ValueError):
            self.model.decode_schedule(np.zeros(self.model.n_decisions + 1))


if __name__ == "__main__":
    unittest.main()
