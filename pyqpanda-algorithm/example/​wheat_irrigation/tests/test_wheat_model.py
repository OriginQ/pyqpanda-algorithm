"""Independent supply-accounting and winter-wheat trigger regressions."""

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation.model import IrrigationInstance, build_qubo, greedy_schedule


def wheat_data():
    """Synthetic three-day example with a hand-calculable canal allocation."""
    return {
        "name": "Wheat model unit fixture",
        "region": "黄河流域缺水灌区",
        "crop": "winter_wheat",
        "start_date": "2026-04-10",
        "days": 3,
        "dose_mm": 10,
        "zone_area_ha": 2,
        "irrigation_efficiency": 0.8,
        "water_budget_m3": 650,
        "water_cost": 0.3,
        "spatial_weight": 0,
        "timing_weight": 2,
        "zones": [{
            "name": "Wheat A",
            "taw_mm": 100,
            "initial_moisture_mm": 60,
            "window_days": [1, 2, 3],
            "target_mm": [60, 65, 70],
            "stress_weights": [1, 3, 5],
            "et0_mm": [5, 10, 5],
            "crop_coefficient": 1,
            "rainfall_mm": 0,
            "effective_rain_fraction": 0.8,
            "capillary_rise_mm": 0,
            "growth_stage": ["jointing", "jointing", "heading"],
            "allowable_depletion_fraction": [0.5, 0.4, 0.3],
        }],
    }


class WheatSupplyTests(unittest.TestCase):
    def test_gross_supply_derives_event_limit_and_accounts_for_losses(self):
        instance = IrrigationInstance.from_dict(wheat_data())
        self.assertEqual(instance.budget_events, 2)
        self.assertEqual(instance.event_net_water_m3, 200)
        self.assertEqual(instance.event_gross_water_m3, 250)
        self.assertEqual(instance.effective_budget_gross_water_m3, 500)
        self.assertEqual(instance.water_budget_m3, 650.0)
        self.assertIsInstance(instance.water_budget_m3, float)
        report = build_qubo(instance).evaluate([0, 1, 1])
        self.assertTrue(report["budget_feasible"])
        self.assertEqual(report["net_water_m3"], 400)
        self.assertEqual(report["gross_water_m3"], 500)
        self.assertEqual(report["conveyance_loss_m3"], 100)
        self.assertEqual(report["remaining_water_m3"], 150)
        self.assertAlmostEqual(report["water_budget_utilization"], 500 / 650)
        assert_allclose(report["moisture_mm"], [[55, 55, 60]])

    def test_event_and_volume_caps_both_apply_and_volume_only_cap_is_bounded(self):
        for event_cap, volume, expected in [(1, 650, 1), (3, 650, 2), (0, 650, 0)]:
            with self.subTest(event_cap=event_cap):
                data = wheat_data()
                data.update(budget_events=event_cap, water_budget_m3=volume)
                instance = IrrigationInstance.from_dict(data)
                self.assertEqual(instance.budget_events, expected)
                self.assertEqual(instance.water_budget_m3, volume)
        data = wheat_data()
        data["water_budget_m3"] = 10000
        self.assertEqual(IrrigationInstance.from_dict(data).budget_events, 3)

    def test_efficiency_changes_withdrawal_without_changing_root_zone_dose(self):
        data = wheat_data()
        first = build_qubo(IrrigationInstance.from_dict(data))
        data["irrigation_efficiency"] = 0.5
        second = build_qubo(IrrigationInstance.from_dict(data))
        self.assertEqual(second.instance.budget_events, 1)
        first_report, second_report = first.evaluate([1, 0, 0]), second.evaluate([1, 0, 0])
        self.assertEqual(first_report["net_water_m3"], second_report["net_water_m3"])
        self.assertEqual(second_report["gross_water_m3"], 400)
        self.assertEqual(first_report["objective"], second_report["objective"])
        self.assertEqual(first_report["moisture_mm"], second_report["moisture_mm"])

    def test_decimal_boundary_does_not_lose_an_event_or_allow_real_overspend(self):
        data = wheat_data()
        data.update(dose_mm=0.01, zone_area_ha=1, irrigation_efficiency=1,
                    water_budget_m3=0.3)
        model = build_qubo(IrrigationInstance.from_dict(data))
        self.assertEqual(model.instance.budget_events, 3)
        self.assertEqual(model.evaluate([1, 1, 1])["remaining_water_m3"], 0)
        data["water_budget_m3"] = 0.3 - 1e-10
        self.assertEqual(IrrigationInstance.from_dict(data).budget_events, 2)

    def test_zero_budget_and_infeasible_schedule_have_explicit_diagnostics(self):
        data = wheat_data()
        data["water_budget_m3"] = 0
        model = build_qubo(IrrigationInstance.from_dict(data))
        self.assertEqual(model.instance.budget_events, 0)
        empty = model.evaluate([0, 0, 0])
        self.assertEqual(empty["water_budget_utilization"], 0)
        self.assertEqual(empty["remaining_water_m3"], 0)
        infeasible = model.evaluate([1, 0, 0])
        self.assertFalse(infeasible["budget_feasible"])
        self.assertEqual(infeasible["remaining_water_m3"], -250)
        self.assertEqual(infeasible["water_budget_utilization"], 0)

    def test_legacy_event_only_input_gets_compatible_unit_efficiency_defaults(self):
        data = wheat_data()
        for field in ("region", "crop", "start_date", "water_budget_m3", "irrigation_efficiency"):
            data.pop(field)
        data["budget_events"] = 2
        data["zones"][0].pop("growth_stage")
        data["zones"][0].pop("allowable_depletion_fraction")
        instance = IrrigationInstance.from_dict(data)
        self.assertEqual(instance.region, "generic")
        self.assertEqual(instance.crop, "winter_wheat")
        self.assertIsNone(instance.start_date)
        self.assertEqual(instance.water_budget_m3, 400)
        self.assertEqual(instance.irrigation_efficiency, 1)
        self.assertEqual(instance.zones[0].growth_stage, ("unspecified",) * 3)
        assert_allclose(instance.zones[0].moisture_threshold_mm, [45, 45, 45])


class WheatStageTests(unittest.TestCase):
    def test_stage_thresholds_drive_greedy_and_are_reported_per_zone_and_stage(self):
        model = build_qubo(IrrigationInstance.from_dict(wheat_data()))
        assert_allclose(model.instance.zones[0].moisture_threshold_mm, [50, 60, 70])
        assert_array_equal(greedy_schedule(model), [0, 1, 1])
        assert_array_equal(greedy_schedule(model, depletion_fraction=0.65), [0, 0, 0])
        report = model.evaluate([0, 1, 1])
        self.assertEqual(report["below_threshold_zone_days"], 2)
        diagnostic = report["zone_diagnostics"][0]
        self.assertEqual(diagnostic["below_threshold_days"], [2, 3])
        self.assertEqual(diagnostic["minimum_moisture_mm"], 55)
        self.assertEqual(diagnostic["stage_summary"], [
            {"stage": "jointing", "days": [1, 2], "irrigation_events": 1,
             "below_threshold_days": [2]},
            {"stage": "heading", "days": [3], "irrigation_events": 1,
             "below_threshold_days": [3]},
        ])

    def test_stage_labels_and_trigger_fractions_do_not_change_qubo_objective(self):
        data = wheat_data()
        first = build_qubo(IrrigationInstance.from_dict(data))
        data["zones"][0].update(growth_stage="grain_filling", allowable_depletion_fraction=0.2)
        second = build_qubo(IrrigationInstance.from_dict(data))
        self.assertEqual(second.instance.zones[0].growth_stage, ("grain_filling",) * 3)
        assert_array_equal(first.linear, second.linear)
        assert_array_equal(first.quadratic, second.quadratic)
        self.assertEqual(first.constant, second.constant)

    def test_volume_budget_qubo_matches_independent_daily_cost_for_all_states(self):
        model = build_qubo(IrrigationInstance.from_dict(wheat_data()))
        for state in itertools.product((0, 1), repeat=model.n_variables):
            decisions = state[:3]
            stored, stress = 60, 0
            for event, demand, target, weight in zip(decisions, [5, 10, 5], [60, 65, 70], [1, 3, 5]):
                stored += 10 * event - demand
                stress += weight * (stored - target) ** 2
            physical_cost = 3 * sum(decisions) + stress + 2 * (
                decisions[0] * decisions[1] + decisions[1] * decisions[2])
            residual = sum(decisions) + int(np.dot(state[3:], model.slack_weights)) - 2
            self.assertAlmostEqual(model.energy(state),
                                   physical_cost + model.budget_penalty * residual ** 2, places=8)
            self.assertAlmostEqual(model.evaluate(decisions)["objective"], physical_cost, places=8)


class WheatValidationTests(unittest.TestCase):
    def test_invalid_volume_efficiency_dates_and_metadata_are_rejected(self):
        invalid_fields = {
            "water_budget_m3": [-1, True, None, "650", float("inf"), float("nan")],
            "irrigation_efficiency": [0, -0.1, 1.01, True, None, float("inf"), float("nan")],
            "start_date": ["2026-02-30", "20260410", "10/04/2026", "9999-12-31", 20260410, True],
            "region": ["", " ", None, 1],
            "crop": ["", None, 1],
        }
        for field, values in invalid_fields.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    data = wheat_data()
                    data[field] = value
                    IrrigationInstance.from_dict(data)
        data = wheat_data()
        data.pop("water_budget_m3")
        with self.assertRaisesRegex(ValueError, "budget_events or water_budget_m3"):
            IrrigationInstance.from_dict(data)

    def test_invalid_stage_series_and_depletion_fractions_are_rejected(self):
        invalid_fields = {
            "growth_stage": ["", " ", None, ["jointing"], ["jointing", 1, "heading"]],
            "allowable_depletion_fraction": [-0.1, 1.01, True, [0.5], [0.5, 0.5, 1.1], float("nan")],
        }
        for field, values in invalid_fields.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    data = wheat_data()
                    data["zones"][0][field] = value
                    IrrigationInstance.from_dict(data)
        model = build_qubo(IrrigationInstance.from_dict(wheat_data()))
        for value in (-0.1, 1.1, True, "0.5", float("nan")):
            with self.subTest(override=value), self.assertRaises(ValueError):
                greedy_schedule(model, depletion_fraction=value)


if __name__ == "__main__":
    unittest.main()
