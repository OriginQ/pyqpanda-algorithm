"""Independent numerical and unit-boundary checks for forecast preparation."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation.forecast_physics import SoilRetentionCurve, daily_fao56


def weather_record():
    return {
        "tmin_c": 15.0,
        "tmax_c": 25.0,
        "relative_humidity_min_pct": 40.0,
        "relative_humidity_max_pct": 80.0,
        "wind_speed_2m_m_s": 2.0,
        "net_radiation_mj_m2_day": 12.0,
    }


class DailyFAO56Tests(unittest.TestCase):
    def test_independently_tabulated_daily_short_grass_fixture(self):
        # A separate 45-digit decimal evaluation gives e0(Tmin)=1.7053462321,
        # e0(Tmax)=3.1677777175, ea=1.3156940363, Delta=0.1447401881,
        # gamma=0.0673645. This fixture detects RH pairing, use of Tmean in
        # Delta rather than es, and the daily short-grass coefficients.
        self.assertAlmostEqual(daily_fao56(weather_record()), 4.546159953098535, places=12)

    def test_saturated_zero_degree_still_air_has_only_radiation_demand(self):
        weather = weather_record()
        weather.update(tmin_c=0, tmax_c=0, relative_humidity_min_pct=100,
                       relative_humidity_max_pct=100, wind_speed_2m_m_s=0,
                       net_radiation_mj_m2_day=10)
        # At 0 C, e0 is exactly 0.6108 and Delta=4098*0.6108/237.3**2;
        # humidity makes the aerodynamic term zero.
        self.assertAlmostEqual(daily_fao56(weather), 1.6219447486506344, places=12)

    def test_pressure_and_heat_flux_defaults_and_signed_fluxes(self):
        weather = weather_record()
        original = daily_fao56(weather)
        weather.update(atmospheric_pressure_kpa=101.3, soil_heat_flux_mj_m2_day=0)
        self.assertEqual(daily_fao56(weather), original)
        # Rn and G enter through their difference, and each may be negative.
        weather.update(net_radiation_mj_m2_day=-2, soil_heat_flux_mj_m2_day=-14)
        self.assertEqual(daily_fao56(weather), original)
        weather["soil_heat_flux_mj_m2_day"] = -12
        self.assertLess(daily_fao56(weather), original)
        before_pressure_change = daily_fao56(weather)
        weather["atmospheric_pressure_kpa"] = 85
        self.assertNotEqual(daily_fao56(weather), before_pressure_change)

    def test_negative_reference_demand_is_zero(self):
        weather = weather_record()
        weather.update(relative_humidity_min_pct=100, relative_humidity_max_pct=100,
                       net_radiation_mj_m2_day=-1)
        self.assertEqual(daily_fao56(weather), 0.0)

    def test_missing_fields_and_non_object_records_are_rejected(self):
        for name in weather_record():
            weather = weather_record()
            del weather[name]
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, name):
                daily_fao56(weather)
        for value in (None, [], "weather", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                daily_fao56(value)

    def test_every_numeric_field_rejects_boolean_nonfinite_and_nonnumeric_values(self):
        fields = list(weather_record()) + ["atmospheric_pressure_kpa", "soil_heat_flux_mj_m2_day"]
        for field in fields:
            for value in (True, np.bool_(False), None, "2.0", [], float("nan"),
                          float("inf"), -float("inf"), 10 ** 1000):
                weather = weather_record()
                weather[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    daily_fao56(weather)

    def test_invalid_ranges_and_extrema_order_are_rejected(self):
        invalid = [
            {"tmin_c": -81}, {"tmax_c": 61}, {"tmin_c": 26},
            {"relative_humidity_min_pct": -1}, {"relative_humidity_max_pct": 101},
            {"relative_humidity_min_pct": 81}, {"wind_speed_2m_m_s": -0.1},
            {"atmospheric_pressure_kpa": 0}, {"atmospheric_pressure_kpa": -1},
        ]
        for change in invalid:
            weather = weather_record()
            weather.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                daily_fao56(weather)

    def test_finite_input_overflow_is_rejected(self):
        weather = weather_record()
        weather.update(atmospheric_pressure_kpa=1e308, wind_speed_2m_m_s=1e308)
        with self.assertRaisesRegex(ValueError, "overflow"):
            daily_fao56(weather)


class SoilRetentionCurveTests(unittest.TestCase):
    def setUp(self):
        # Synthetic calibration in available root-zone water, not theta.
        self.data = {"tension_kpa": [10, 30, 90], "available_water_mm": [120, 80, 0]}
        self.curve = SoilRetentionCurve.from_dict(self.data, taw_mm=120)

    def test_piecewise_linear_interpolation_and_inverse(self):
        for tension, water in ((10, 120), (20, 100), (30, 80), (60, 40), (90, 0)):
            with self.subTest(tension=tension):
                self.assertAlmostEqual(self.curve.water_from_tension(tension), water)
                self.assertAlmostEqual(self.curve.tension_from_water(water), tension)
        self.assertEqual(self.curve.taw_mm, 120)

    def test_out_of_domain_queries_do_not_silently_clip_or_extrapolate(self):
        for tension in (-1, 0, 9.999, 90.001, 1e6):
            with self.subTest(tension=tension), self.assertRaisesRegex(ValueError, "domain"):
                self.curve.water_from_tension(tension)
        for water in (-0.001, 120.001):
            with self.subTest(water=water), self.assertRaisesRegex(ValueError, "domain"):
                self.curve.tension_from_water(water)

    def test_conversion_rejects_boolean_nonfinite_and_nonnumeric_queries(self):
        for method in (self.curve.water_from_tension, self.curve.tension_from_water):
            for value in (True, np.bool_(True), None, "30", [], float("nan"), float("inf")):
                with self.subTest(method=method.__name__, value=value), self.assertRaises(ValueError):
                    method(value)

    def test_bad_calibration_shapes_values_and_endpoints_are_rejected(self):
        invalid = [
            None, [], {}, {"tension_kpa": [10, 90]},
            {"tension_kpa": [10], "available_water_mm": [120]},
            {"tension_kpa": [10, 30, 90], "available_water_mm": [120, 0]},
            {"tension_kpa": [10, 10, 90], "available_water_mm": [120, 80, 0]},
            {"tension_kpa": [10, 90, 30], "available_water_mm": [120, 80, 0]},
            {"tension_kpa": [-1, 30, 90], "available_water_mm": [120, 80, 0]},
            {"tension_kpa": [10, 30, 90], "available_water_mm": [120, 120, 0]},
            {"tension_kpa": [10, 30, 90], "available_water_mm": [120, -1, 0]},
            {"tension_kpa": [10, 90], "available_water_mm": [121, 0]},
            {"tension_kpa": [10, 90], "available_water_mm": [119, 0]},
            {"tension_kpa": [10, 90], "available_water_mm": [120, 1]},
            {"tension_kpa": [10, True], "available_water_mm": [120, 0]},
            {"tension_kpa": [10, float("inf")], "available_water_mm": [120, 0]},
            {"tension_kpa": [10, 90], "available_water_mm": [120, float("nan")]},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                SoilRetentionCurve.from_dict(data, 120)
        for taw in (0, -1, True, None, "120", float("inf"), float("nan")):
            with self.subTest(taw=taw), self.assertRaises(ValueError):
                SoilRetentionCurve.from_dict(self.data, taw)

    def test_two_point_calibration_and_zero_tension_are_supported(self):
        curve = SoilRetentionCurve.from_dict(
            {"tension_kpa": [0, 1500], "available_water_mm": [120, 0]}, 120)
        self.assertEqual(curve.water_from_tension(750), 60)
        self.assertEqual(curve.tension_from_water(60), 750)

    def test_original_calibration_mutation_does_not_change_the_curve(self):
        self.data["tension_kpa"][1] = 80
        self.data["available_water_mm"][1] = 1
        self.assertEqual(self.curve.water_from_tension(30), 80)


if __name__ == "__main__":
    unittest.main()
