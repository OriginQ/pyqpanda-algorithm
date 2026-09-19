"""Validate circuit evolution and finite-shot reporting independently."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irrigation import main, summarize_distribution
from irrigation_model import IrrigationInstance, build_qubo
from solvers import enumerate_spectrum, qaoa_probabilities, solve_qaoa
from test_irrigation import example_data


def three_qubit_model():
    data = example_data(budget=0)
    data["zones"] = data["zones"][:1]
    data["adjacency"] = []
    return build_qubo(IrrigationInstance.from_dict(data))


class QAOATests(unittest.TestCase):
    def test_zero_angles_are_uniform_and_indices_are_little_endian(self):
        model = three_qubit_model()
        bits, _ = enumerate_spectrum(model)
        self.assertEqual(bits[1].tolist(), [1, 0, 0])
        self.assertEqual(bits[4].tolist(), [0, 0, 1])
        assert_allclose(qaoa_probabilities(model, [0.0], [0.0], backend="numpy"),
                        np.full(8, 1 / 8), atol=1e-14)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy not installed")
    def test_two_layer_evolution_matches_independent_dense_matrix_exponential(self):
        from scipy.linalg import expm
        model = three_qubit_model()
        bits, energies = enumerate_spectrum(model)
        h, J, offset = model.to_ising()
        scale = max(np.max(np.abs(h)), np.max(np.abs(J)))
        cost = np.diag((energies - offset) / scale)
        mixer = np.zeros((len(bits), len(bits)))
        for index in range(len(bits)):
            for qubit in range(model.n_variables):
                mixer[index ^ (1 << qubit), index] += 1
        gammas, betas = [0.37, 0.81], [0.29, -0.18]
        state = np.ones(len(bits), dtype=complex) / np.sqrt(len(bits))
        for gamma, beta in zip(gammas, betas):
            state = expm(-1j * beta * mixer) @ expm(-1j * gamma * cost) @ state
        assert_allclose(qaoa_probabilities(model, gammas, betas, backend="numpy"),
                        np.abs(state) ** 2, atol=1e-13, rtol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy not installed")
    def test_optimizer_and_sampling_are_seeded_and_energy_keeps_constant(self):
        model = three_qubit_model()
        first = solve_qaoa(model, depth=1, maxiter=20, restarts=1, shots=120, seed=19, backend="numpy")
        second = solve_qaoa(model, depth=1, maxiter=20, restarts=1, shots=120, seed=19, backend="numpy")
        self.assertEqual(first["counts"], second["counts"])
        self.assertEqual(first["parameters"], second["parameters"])
        self.assertEqual(sum(first["counts"].values()), 120)
        self.assertEqual(first["evaluations"], sum(run["nfev"] for run in first["optimizer_runs"]))
        self.assertAlmostEqual(float(first["probabilities"].sum()), 1)
        _, energies = enumerate_spectrum(model)
        self.assertAlmostEqual(first["expectation"], float(first["probabilities"] @ energies), places=10)

    def test_size_limits_and_invalid_angles_fail_before_simulation(self):
        model = three_qubit_model()
        for limit in (2, 19, 0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                enumerate_spectrum(model, max_qubits=limit)
        for gamma, beta in (([], []), ([0.1], [0.2, 0.3]), ([float("nan")], [0.2])):
            with self.assertRaises(ValueError):
                qaoa_probabilities(model, gamma, beta)

class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.model = build_qubo(IrrigationInstance.from_dict(example_data()))
        self.bits, self.energies = enumerate_spectrum(self.model)
        self.uniform = np.full(len(self.bits), 1 / len(self.bits))

    def test_no_feasible_shots_do_not_get_replaced_by_exact_solution(self):
        counts = {len(self.bits) - 1: 10}
        report = summarize_distribution(self.model, self.bits, self.energies,
                                        self.uniform, counts, float(self.energies.min()))
        self.assertIsNone(report["best_budget_feasible_sample"])
        self.assertEqual(report["sampled"]["budget_feasible_probability"], 0)

    def test_budget_feasible_and_correct_slack_are_distinct_and_unrepaired(self):
        # No irrigation, all-zero slack: agricultural budget feasible but the
        # equality N+s=K fails. It is the only observed candidate, even though
        # better schedules exist in the exact spectrum.
        report = summarize_distribution(self.model, self.bits, self.energies,
                                        self.uniform, {0: 10}, float(self.energies.min()))
        self.assertEqual(report["sampled"]["budget_feasible_probability"], 1)
        self.assertEqual(report["sampled"]["zero_slack_residual_probability"], 0)
        candidate = report["best_budget_feasible_sample"]
        self.assertEqual(candidate["events"], 0)
        self.assertEqual(candidate["sampled_state_index"], 0)
        self.assertEqual(candidate["sampled_slack_residual"], -2)
        self.assertAlmostEqual(candidate["sampled_qubo_energy"] - candidate["objective"],
                               4 * self.model.budget_penalty)

    def test_classical_cli_exports_reusable_files(self):
        import contextlib
        import io
        import json
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--backend", "classical", "--no-plot", "--output", directory, "--shots", "32"])
            self.assertEqual(code, 0)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["model"]["total_qubits"], 11)
            self.assertNotIn("qaoa", summary)
            self.assertTrue(summary["solutions"]["Exact"]["budget_feasible"])
            for name in ("summary.json", "qubo.json", "schedules.csv", "moisture.csv"):
                self.assertGreater((Path(directory) / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
