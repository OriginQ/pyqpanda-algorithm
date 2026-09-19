"""Offline evidence validation; fixtures never access credentials or the cloud."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation.hardware import prepare_qaoa_circuit
from wheat_irrigation.hardware_batch_analysis import analyze_probability_result
from wheat_irrigation.model import IrrigationInstance, build_qubo


class HardwareBatchAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.input = {
            "name": "tiny batch fixture", "source_note": "Synthetic offline data",
            "days": 2, "budget_events": 1, "dose_mm": 9.0,
            "water_cost": 0.4, "spatial_weight": 0.0, "timing_weight": 1.0,
            "zone_area_ha": 1.0, "adjacency": [],
            "zones": [{
                "name": "A", "taw_mm": 40.0, "initial_moisture_mm": 17.0,
                "window_days": [1, 2], "target_mm": [24.0, 25.0],
                "stress_weights": [0.7, 1.2], "et0_mm": [3.0, 5.0],
                "crop_coefficient": 1.0, "rainfall_mm": [0.0, 0.0],
                "effective_rain_fraction": 0.6, "capillary_rise_mm": 0.0,
            }],
        }
        self.model = build_qubo(IrrigationInstance.from_dict(self.input))
        circuit = prepare_qaoa_circuit(self.model, [0.0], [0.0], max_qubits=8)
        self.manifest = {
            "schema_version": 1, "circuit": circuit,
            "report": {
                "input_data": self.input,
                "configuration": {"max_qubits": 8, "shots": 1024, "depth": 1},
                "hardware": {"chip": "fixture-chip", "shots": 1024, "job_id": None,
                             "circuit_sha256": circuit["sha256"]},
                "qaoa_reference": {"parameters": {"gammas": [0.0], "betas": [0.0]}},
            },
        }
        self.response = {
            "success": True, "code": 10000,
            "obj": {"taskId": "fixture-job", "chipId": "fixture-chip", "taskState": "3",
                    "measureQubitSize": [3], "errCode": 0,
                    "taskResult": [json.dumps({"key": ["0x0", "0x1", "0x3"], "value": [0.1, 0.2, 0.7]})]},
        }

    def analyze(self, response=None, manifest=None, job="fixture-job"):
        return analyze_probability_result(manifest or self.manifest, response or self.response, job)

    def distribution(self, keys, values):
        response = deepcopy(self.response)
        response["obj"]["taskResult"] = [json.dumps({"key": keys, "value": values})]
        return response

    def test_known_metrics_preserve_hardware_evidence_without_resampling(self):
        before = deepcopy(self.manifest), deepcopy(self.response)
        with patch("numpy.random.default_rng", side_effect=AssertionError("No resampling")):
            analysis = self.analyze()
        self.assertEqual((self.manifest, self.response), before)
        metrics = analysis["normalized_export_metrics"]
        self.assertAlmostEqual(metrics["budget_feasible_probability"], 0.3)
        self.assertAlmostEqual(metrics["zero_slack_residual_probability"], 0.2)
        self.assertAlmostEqual(metrics["optimal_full_qubo_state_probability"], 0.2)
        self.assertAlmostEqual(analysis["ideal_metrics"]["budget_feasible_probability"], 0.75)
        self.assertAlmostEqual(analysis["total_variation_distance_normalized_export_vs_ideal"], 0.65)
        self.assertFalse(analysis["raw_integer_counts_available"])
        self.assertFalse(analysis["counts_synthesized"])
        candidate = analysis["best_observed_budget_feasible_candidate"]
        self.assertEqual(candidate["observed_state_index"], 1)
        self.assertEqual(candidate["observed_binary_key"], "001")
        self.assertEqual(candidate["decision_bits_x0_first"], "10")
        self.assertEqual(candidate["absolute_gap_to_exact"], 0)
        self.assertEqual(candidate["events_by_zone"], {"A": [1]})
        self.assertEqual(analysis["reported_probabilities_by_state_index"], {"0": 0.1, "1": 0.2, "3": 0.7})
        json.dumps(analysis, allow_nan=False)

    def test_mass_bound_handles_truncation_and_discloses_normalization(self):
        response = self.distribution(["0", "1", "3"], [0.0999999, 0.2, 0.7])
        result = self.analyze(response)
        self.assertEqual(result["reported_probability_sum_decimal"], "0.9999999")
        self.assertEqual(result["probability_mass_tolerance_decimal"], "3E-7")
        self.assertAlmostEqual(result["raw_export_weighted_metrics"]["budget_feasible_probability"], 0.2999999)
        self.assertAlmostEqual(result["normalized_export_metrics"]["budget_feasible_probability"], 0.2999999 / 0.9999999)
        for values in ([0.0999996, 0.2, 0.7], [0.1000004, 0.2, 0.7], [0, 0, 0]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "mass"):
                self.analyze(self.distribution(["0", "1", "3"], values))

    def test_no_feasible_observation_never_falls_back_to_the_classical_optimum(self):
        result = self.analyze(self.distribution(["1", "3"], [0, 1]))
        self.assertIsNone(result["best_observed_budget_feasible_candidate"])
        self.assertEqual(result["unique_observed_states"], 1)
        self.assertEqual(result["unique_budget_feasible_observed_states"], 0)
        self.assertEqual(result["normalized_export_metrics"]["budget_feasible_probability"], 0)
        self.assertTrue(result["classical_optimal_candidate"]["budget_feasible"])

    def test_observed_slack_is_not_repaired(self):
        result = self.analyze(self.distribution(["0"], [1]))
        candidate = result["best_observed_budget_feasible_candidate"]
        self.assertEqual(candidate["observed_state_index"], 0)
        self.assertEqual(candidate["observed_slack_residual"], -1)
        self.assertGreater(candidate["absolute_gap_to_exact"], 0)
        self.assertEqual(result["normalized_export_metrics"]["budget_feasible_probability"], 1)
        self.assertEqual(result["normalized_export_metrics"]["zero_slack_residual_probability"], 0)

    def test_mixed_tasks_chip_width_and_noncompleted_responses_are_rejected(self):
        mutations = [
            lambda r: r["obj"].update(taskId="other-job"),
            lambda r: r["obj"].update(chipId="other-chip"),
            lambda r: r["obj"].update(measureQubitSize=[4]),
            lambda r: r["obj"].update(measureQubitSize=[3, 3]),
            lambda r: r["obj"].update(measureQubitSize=[True]),
            lambda r: r["obj"].update(taskState="2"),
            lambda r: r["obj"].update(taskState=3.0),
            lambda r: r["obj"].update(errCode=7),
            lambda r: r.update(success=False),
            lambda r: r.update(code="10000"),
            lambda r: r["obj"].update(taskResult=r["obj"]["taskResult"] * 2),
        ]
        for mutate in mutations:
            response = deepcopy(self.response)
            mutate(response)
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.analyze(response)
        manifest = deepcopy(self.manifest)
        manifest["report"]["hardware"]["job_id"] = "different-saved-job"
        with self.assertRaisesRegex(ValueError, "job ID"):
            self.analyze(manifest=manifest)

    def test_bad_keys_probabilities_and_duplicate_hex_aliases_are_rejected(self):
        for keys, values in [
            (["0", "0"], [0.5, 0.5]), (["0x1", "01"], [0.5, 0.5]),
            (["8"], [1]), (["-1"], [1]), ([" 1"], [1]), ([1], [1]),
            (["0"], [float("nan")]), (["0"], [float("inf")]),
            (["0", "1"], [-0.1, 1.1]), (["0"], [True]), (["0"], ["1"]),
            (["0"], [1.0000001]), (["0"], [10 ** 1000]), (["0"], []), ([], []),
        ]:
            with self.subTest(keys=keys, values=values), self.assertRaises(ValueError):
                self.analyze(self.distribution(keys, values))

    def test_duplicate_json_fields_and_invalid_payloads_are_rejected(self):
        for payload in ['{"key":["0"],"value":[1],"value":[1]}', '[]', '{', 'null']:
            response = deepcopy(self.response)
            response["obj"]["taskResult"] = [payload]
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.analyze(response)

    def test_prepared_circuit_and_saved_parameters_are_verified(self):
        mutations = [
            lambda m: m["circuit"].update(originir="QINIT 3\n"),
            lambda m: m["report"]["qaoa_reference"]["parameters"].update(gammas=[0.1]),
            lambda m: m["report"]["input_data"].update(dose_mm=8.0),
            lambda m: m["report"]["hardware"].update(circuit_sha256="wrong"),
            lambda m: m["report"]["hardware"].update(shots=99),
        ]
        for mutate in mutations:
            manifest = deepcopy(self.manifest)
            mutate(manifest)
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                self.analyze(manifest=manifest)


if __name__ == "__main__":
    unittest.main()
