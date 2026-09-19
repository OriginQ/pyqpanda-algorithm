"""Offline circuit equivalence and mocked cloud lifecycle checks (no QPU jobs)."""

from contextlib import ExitStack
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation import hardware
from wheat_irrigation.model import IrrigationInstance, build_qubo
from wheat_irrigation.solvers import qaoa_probabilities
from test_irrigation import example_data


def small_model():
    data = example_data(budget=0)
    data["zones"] = data["zones"][:1]
    data["adjacency"] = []
    return build_qubo(IrrigationInstance.from_dict(data))


def simulate_exported_ir(originir):
    """Independent gate-level reference, omitting only terminal measurements."""
    lines = originir.splitlines()
    n = int(lines[0].split()[1])
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1
    for line in lines[2:]:
        name = line.split()[0]
        qubits = [int(q) for q in re.findall(r"q\[(\d+)\]", line)]
        if name == "MEASURE":
            continue
        if name == "CNOT":
            control, target = qubits
            old = state.copy()
            for index in range(len(state)):
                source = index ^ (1 << target) if index & (1 << control) else index
                state[index] = old[source]
            continue
        if name == "H":
            matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        else:
            angle = float(re.search(r"\((.*)\)", line)[1])
            if name == "RZ":
                matrix = np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])
            elif name == "RX":
                c, s = np.cos(angle / 2), -1j * np.sin(angle / 2)
                matrix = np.array([[c, s], [s, c]])
            else:
                raise AssertionError(name)
        q = qubits[0]
        for index in range(len(state)):
            if not index & (1 << q):
                pair = [index, index | (1 << q)]
                state[pair] = matrix @ state[pair]
    return np.abs(state) ** 2


class CircuitTests(unittest.TestCase):
    def test_optional_native_sdk_parses_and_simulates_exported_originir(self):
        try:
            from pyqpanda3.core import CPUQVM
            from pyqpanda3.intermediate_compiler import convert_originir_string_to_qprog
        except (ImportError, OSError):
            self.skipTest("pyqpanda3 native SDK unavailable")
        model = small_model()
        gammas, betas = [0.37, 0.81], [0.29, -0.18]
        circuit = hardware.prepare_qaoa_circuit(model, gammas, betas)
        # Parse the full measured export to validate its syntax, then evaluate
        # unmeasured amplitudes for an exact comparison without shot noise.
        convert_originir_string_to_qprog(circuit["originir"])
        unmeasured = "\n".join(line for line in circuit["originir"].splitlines()
                                if not line.startswith("MEASURE")) + "\n"
        prog = convert_originir_string_to_qprog(unmeasured)
        simulator = CPUQVM()
        simulator.run(prog, 1)
        probabilities = simulator.result().get_prob_list(list(range(model.n_variables)))
        assert_allclose(probabilities,
                        qaoa_probabilities(model, gammas, betas, backend="numpy"),
                        atol=1e-12, rtol=1e-11)

    def test_export_matches_local_qaoa_with_negative_beta_and_multiple_layers(self):
        model = small_model()
        gammas, betas = [0.31, 1.17], [-0.2, 0.51]
        with patch.object(hardware, "_load_sdk", side_effect=AssertionError("must stay offline")), \
                patch.object(hardware.os.environ, "get", side_effect=AssertionError("no credentials")):
            circuit = hardware.prepare_qaoa_circuit(model, gammas, betas)
        assert_allclose(simulate_exported_ir(circuit["originir"]),
                        qaoa_probabilities(model, gammas, betas, backend="numpy"),
                        atol=1e-13, rtol=1e-12)
        self.assertEqual(circuit["n_qubits"], 3)
        self.assertEqual(circuit["gate_counts"]["MEASURE"], 3)
        self.assertEqual(circuit["gate_counts"]["RX"], 6)
        self.assertTrue(circuit["originir"].endswith("MEASURE q[2],c[2]\n"))

    def test_constant_model_keeps_inactive_qubits(self):
        model = SimpleNamespace(n_variables=3, to_ising=lambda: (np.zeros(3), np.zeros((3, 3)), 7))
        circuit = hardware.prepare_qaoa_circuit(model, [0.7], [-0.3])
        self.assertEqual(circuit["normalization"], 1)
        self.assertEqual(circuit["gate_counts"], {"H": 3, "MEASURE": 3, "RX": 3})
        assert_allclose(simulate_exported_ir(circuit["originir"]), np.full(8, 1 / 8), atol=1e-14)

    def test_invalid_model_and_angles(self):
        with self.assertRaises(ValueError):
            hardware.prepare_qaoa_circuit(small_model(), [], [])
        model = SimpleNamespace(n_variables=2, to_ising=lambda: ([1, 2], [[0, 0], [1, 0]], 0))
        with self.assertRaises(ValueError):
            hardware.prepare_qaoa_circuit(model, [1], [1])


class CountTests(unittest.TestCase):
    def test_binary_keys_are_little_endian_integers(self):
        self.assertEqual(hardware.decode_counts({"001": 7, "100": 2, "000": 0}, 3, 9), {1: 7, 4: 2})

    def test_invalid_keys_probabilities_and_wrong_shot_totals_are_rejected(self):
        invalid = ({"1": 10}, {"0x1": 10}, {"002": 10}, {1: 10},
                   {"001": 1.0}, {"001": True}, {"001": -1}, {}, {"001": 9})
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(hardware.HardwareError):
                hardware.decode_counts(value, 3, 10)


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.circuit = hardware.prepare_qaoa_circuit(small_model(), [0.4], [-0.2])
        self.options = Mock()
        self.result = Mock()
        self.result.job_status.return_value = SimpleNamespace(name="FINISHED")
        self.result.get_counts.return_value = {"001": 7, "100": 3}
        self.job = Mock()
        self.job.job_id.return_value = "test-job-001"
        self.job.query.return_value = self.result
        self.backend = Mock()
        self.backend.chip_info.return_value.available_qubits.return_value = [10, 12, 25]
        self.backend.run.return_value = self.job
        self.service = Mock()
        self.service.backends.return_value = {"test-chip": True, "full_amplitude": True}
        self.service.backend.return_value = self.backend
        self.sdk = SimpleNamespace(
            QCloudService=Mock(return_value=self.service),
            QCloudOptions=Mock(return_value=self.options),
            QCloudJob=Mock(return_value=self.job),
            DataBase=SimpleNamespace(Binary="binary"),
            LogOutput=SimpleNamespace(FILE="file"),
        )
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(hardware, "_load_sdk", return_value=self.sdk))
        self.parser = self.stack.enter_context(patch.object(hardware, "_parse_originir", return_value="qprog"))
        self.stack.enter_context(patch.dict(hardware.os.environ, {"TEST_QCLOUD_KEY": "test-secret-never-print"}))

    def run_cloud(self, **kwargs):
        parameters = dict(chip="test-chip", shots=10, token_env="TEST_QCLOUD_KEY", timeout=0,
                          on_submitted=Mock())
        parameters.update(kwargs)
        return hardware.run_hardware(self.circuit, **parameters)

    def test_submission_persists_before_query_and_requests_unamended_counts(self):
        callback = Mock()
        self.job.query.side_effect = lambda: (callback.assert_called_once_with("test-job-001") or self.result)
        report = self.run_cloud(on_submitted=callback)
        self.assertEqual(report["status"], "finished")
        self.assertEqual(report["counts"], {1: 7, 4: 3})
        self.assertEqual(report["raw_counts"], {"001": 7, "100": 3})
        self.backend.run.assert_called_once_with("qprog", 10, self.options)
        self.options.set_amend.assert_called_once_with(False)
        self.options.set_is_prob_counts.assert_called_once_with(True)
        self.options.set_mapping.assert_called_once_with(True)
        self.service.setup_logging.assert_called_once_with("file", hardware.os.devnull)
        self.result.get_counts.assert_called_once_with("binary")
        self.result.get_probs.assert_not_called()
        self.job.result.assert_not_called()

    def test_resume_does_not_submit_parse_or_require_chip_currently_online(self):
        report = self.run_cloud(job_id="existing-job", on_submitted=None)
        self.assertEqual(report["job_id"], "existing-job")
        self.sdk.QCloudJob.assert_called_once_with("existing-job")
        self.backend.run.assert_not_called()
        self.service.backends.assert_not_called()
        self.parser.assert_not_called()

    def test_zero_timeout_returns_pending_without_waiting(self):
        self.result.job_status.return_value = SimpleNamespace(name="QUEUING")
        with patch.object(hardware.time, "sleep", side_effect=AssertionError("unexpected sleep")):
            report = self.run_cloud()
        self.assertEqual(report["status"], "pending")
        self.assertEqual(report["counts"], {})
        self.job.query.assert_called_once()
        self.result.get_counts.assert_not_called()

    def test_polling_stops_at_timeout(self):
        self.result.job_status.return_value = SimpleNamespace(name="COMPUTING")
        with patch.object(hardware.time, "monotonic", side_effect=[0, 1, 5]), \
                patch.object(hardware.time, "sleep") as sleep:
            report = self.run_cloud(timeout=5, poll_interval=10)
        self.assertEqual(report["status"], "pending")
        self.assertEqual(self.job.query.call_count, 2)
        sleep.assert_called_once_with(4.0)

    def test_failed_job_does_not_expose_server_error_or_counts(self):
        self.result.job_status.return_value = SimpleNamespace(name="FAILED")
        report = self.run_cloud()
        self.assertEqual(report["status"], "failed")
        self.result.get_counts.assert_not_called()
        self.result.error_message.assert_not_called()

    def test_persistence_failure_retains_id_and_stops_polling(self):
        with self.assertRaisesRegex(hardware.HardwareError, "test-job-001"):
            self.run_cloud(on_submitted=Mock(side_effect=OSError("disk full")))
        self.job.query.assert_not_called()

    def test_secret_in_sdk_exception_is_redacted(self):
        self.job.query.side_effect = RuntimeError("test-secret-never-print")
        with self.assertRaises(hardware.HardwareError) as caught:
            self.run_cloud()
        self.assertNotIn("test-secret-never-print", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)

    def test_guards_reject_before_submission(self):
        invalid = ({"chip": "full_amplitude"}, {"timeout": -1}, {"poll_interval": 0},
                   {"shots": True}, {"job_id": ""}, {"on_submitted": None})
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.run_cloud(**values)
        self.backend.run.assert_not_called()

    def test_tampered_circuit_is_rejected_before_sdk_import(self):
        self.circuit["originir"] += "H q[0]\n"
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_cloud()
        self.parser.assert_not_called()

    def test_offline_backend_and_insufficient_capacity_do_not_submit(self):
        self.service.backends.return_value = {"test-chip": False}
        with self.assertRaises(hardware.HardwareError):
            self.run_cloud()
        self.service.backends.return_value = {"test-chip": True}
        self.backend.chip_info.return_value.available_qubits.return_value = [0, 1]
        with self.assertRaises(hardware.HardwareError):
            self.run_cloud()
        self.backend.run.assert_not_called()

    def test_missing_token_rejects_cloud_access(self):
        with patch.object(hardware.os.environ, "get", return_value=""), \
                self.assertRaisesRegex(hardware.HardwareError, "TEST_QCLOUD_KEY"):
            self.run_cloud()
        self.sdk.QCloudService.assert_not_called()


if __name__ == "__main__":
    unittest.main()
