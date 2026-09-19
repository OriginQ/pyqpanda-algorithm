"""Native batch analysis uses only synthetic offline evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation.hardware import prepare_qaoa_circuit
from wheat_irrigation.hardware_native_batch_analysis import analyze_native_batch_results
from wheat_irrigation.model import IrrigationInstance, build_qubo


def make_plan(scenario, days, gamma):
    data = {
        "name": scenario, "source_note": "Synthetic offline fixture",
        "days": days, "budget_events": 1, "dose_mm": 9.0,
        "water_cost": 0.4, "spatial_weight": 0.0, "timing_weight": 1.0,
        "zone_area_ha": 1.0, "adjacency": [],
        "zones": [{
            "name": "A", "taw_mm": 40.0, "initial_moisture_mm": 17.0,
            "window_days": list(range(1, days + 1)),
            "target_mm": [24.0] * days, "stress_weights": [0.7] * days,
            "et0_mm": [3.0] * days, "crop_coefficient": 1.0,
            "rainfall_mm": [0.0] * days, "effective_rain_fraction": 0.6,
            "capillary_rise_mm": 0.0,
        }],
    }
    model = build_qubo(IrrigationInstance.from_dict(data))
    circuit = prepare_qaoa_circuit(model, [gamma], [0.0], max_qubits=8)
    return {
        "schema_version": 1, "circuit": circuit,
        "report": {
            "input_data": data,
            "configuration": {"scenario": scenario, "max_qubits": 8, "shots": 1024, "depth": 1},
            "hardware": {"chip": "fixture-chip", "shots": 1024, "job_id": None,
                         "circuit_sha256": circuit["sha256"]},
            "qaoa_reference": {"parameters": {"gammas": [gamma], "betas": [0.0]}},
        },
    }


class NativeBatchAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.names = ["jointing", "heading", "grain_filling"]
        self.plans = [make_plan(name, days, gamma) for name, days, gamma
                      in zip(self.names, (2, 3, 2), (0.0, 0.1, 0.2))]
        self.response = {
            "success": True, "code": 10000,
            "obj": {"taskId": "parent-job", "chipId": "fixture-chip", "taskState": "3",
                    "errCode": 0, "taskType": 17, "taskCalMethod": "2",
                    "measureQubitSize": [plan["circuit"]["n_qubits"] for plan in self.plans],
                    "taskResult": [json.dumps({"key": [key], "value": [1.0]})
                                   for key in ("0x0", "0x1", "0x3")]},
        }

    def analyze(self, plans=None, response=None, job_id="parent-job", names=None):
        return analyze_native_batch_results(
            self.plans if plans is None else plans,
            self.response if response is None else response,
            job_id, self.names if names is None else names,
        )

    def test_complete_batch_preserves_order_parent_identity_and_probability_evidence(self):
        original = deepcopy((self.plans, self.response, self.names))
        with patch("numpy.random.default_rng", side_effect=AssertionError("No synthesized counts")):
            result = self.analyze()
        self.assertEqual((self.plans, self.response, self.names), original)
        self.assertEqual(result["parent_job_id"], "parent-job")
        self.assertEqual(result["result_count"], 3)
        self.assertEqual(result["platform_task_type"], 17)
        self.assertEqual(result["platform_task_cal_method"], "2")
        self.assertFalse(result["circuit_identity_verified_by_cloud"])
        self.assertFalse(result["counts_synthesized"])
        self.assertIn("SDK input/result sequence", result["ordering_basis"])
        for index, row in enumerate(result["circuit_results"]):
            self.assertEqual(row["result_index"], index)
            self.assertEqual(row["scenario_name"], self.names[index])
            self.assertEqual(row["parent_job_id"], "parent-job")
            analysis = row["analysis"]
            self.assertEqual(analysis["job_id"], "parent-job")
            self.assertEqual(analysis["circuit_sha256"], self.plans[index]["circuit"]["sha256"])
            self.assertEqual(analysis["reported_probabilities_by_state_index"],
                             {str((0, 1, 3)[index]): 1.0})
            self.assertFalse(analysis["raw_integer_counts_available"])
            self.assertFalse(analysis["counts_synthesized"])
        json.dumps(result, allow_nan=False)

    def test_probability_content_is_not_used_to_reorder_or_repair_results(self):
        response = deepcopy(self.response)
        response["obj"]["taskResult"].reverse()
        result = self.analyze(response=response)
        self.assertEqual(result["circuit_results"][0]["analysis"]["reported_probabilities_by_state_index"],
                         {"3": 1.0})
        self.assertEqual(result["circuit_results"][2]["analysis"]["reported_probabilities_by_state_index"],
                         {"0": 1.0})
        self.assertFalse(result["circuit_identity_verified_by_cloud"])

    def test_missing_or_extra_results_and_measurement_widths_are_rejected(self):
        mutations = [
            lambda obj: obj.update(taskResult=obj["taskResult"][:-1]),
            lambda obj: obj.update(taskResult=obj["taskResult"] + obj["taskResult"][:1]),
            lambda obj: obj.update(taskResult=[]),
            lambda obj: obj.update(taskResult=[{}, {}, {}]),
            lambda obj: obj.update(measureQubitSize=obj["measureQubitSize"][:-1]),
            lambda obj: obj.update(measureQubitSize=[4, 3, 3]),
            lambda obj: obj.update(measureQubitSize=[3, True, 3]),
        ]
        for mutate in mutations:
            response = deepcopy(self.response)
            mutate(response["obj"])
            with self.subTest(obj=response["obj"]), self.assertRaises(ValueError):
                self.analyze(response=response)

    def test_cross_task_chip_and_noncompleted_responses_are_rejected(self):
        mutations = [
            lambda obj: obj.update(taskId="other-parent-job"),
            lambda obj: obj.update(chipId="other-chip"),
            lambda obj: obj.update(taskState="2"),
            lambda obj: obj.update(taskState=3.0),
            lambda obj: obj.update(errCode=8),
            lambda obj: obj.update(errCode=False),
            lambda obj: obj.pop("errCode"),
        ]
        for mutate in mutations:
            response = deepcopy(self.response)
            mutate(response["obj"])
            with self.subTest(obj=response["obj"]), self.assertRaises(ValueError):
                self.analyze(response=response)
        for field, value in (("job_id", "unrelated-job"), ("chip", "other-chip")):
            plans = deepcopy(self.plans)
            plans[1]["report"]["hardware"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.analyze(plans=plans)

    def test_plan_or_scenario_order_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Scenario"):
            self.analyze(names=list(reversed(self.names)))
        with self.assertRaisesRegex(ValueError, "Scenario"):
            self.analyze(plans=list(reversed(self.plans)))
        plans = [self.plans[1], self.plans[0], self.plans[2]]
        names = [self.names[1], self.names[0], self.names[2]]
        with self.assertRaisesRegex(ValueError, "width sequence"):
            self.analyze(plans=plans, names=names)

    def test_plan_integrity_and_each_probability_payload_are_still_checked(self):
        plans = deepcopy(self.plans)
        plans[1]["circuit"]["originir"] += "invalid circuit\n"
        with self.assertRaisesRegex(ValueError, "Prepared circuit"):
            self.analyze(plans=plans)
        response = deepcopy(self.response)
        response["obj"]["taskResult"][1] = json.dumps({"key": ["0x0"], "value": [0.5]})
        with self.assertRaisesRegex(ValueError, "mass"):
            self.analyze(response=response)

    def test_absent_or_unknown_platform_enum_is_not_invented_or_interpreted(self):
        response = deepcopy(self.response)
        response["obj"].pop("taskType")
        result = self.analyze(response=response)
        self.assertIsNone(result["platform_task_type"])
        self.assertFalse(result["platform_task_type_present"])
        response["obj"]["taskType"] = "099"
        result = self.analyze(response=response)
        self.assertEqual(result["platform_task_type"], "099")
        for value in (True, -1, "native-batch", 2.5, {"secret": "not metadata"}):
            response["obj"]["taskType"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "enum"):
                self.analyze(response=response)

    def test_invalid_input_shapes_and_inconsistent_shots_are_rejected(self):
        for kwargs in ({"plans": []}, {"names": []}, {"names": ["", "heading", "grain_filling"]},
                       {"job_id": ""}, {"response": {}}, {"plans": [None, None, None]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.analyze(**kwargs)
        plans = deepcopy(self.plans)
        plans[1]["report"]["configuration"]["shots"] = 2048
        with self.assertRaisesRegex(ValueError, "same requested shots"):
            self.analyze(plans=plans)


if __name__ == "__main__":
    unittest.main()
