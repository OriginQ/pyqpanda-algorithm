"""Dependency isolation and full-register pyqpanda_alg integration regressions."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheat_irrigation.cli import main
from wheat_irrigation.solvers import enumerate_spectrum, qaoa_probabilities, solve_qaoa


NATIVE_AVAILABLE = all(importlib.util.find_spec(name) is not None
                       for name in ("pyqpanda3", "pyqpanda_alg"))
NATIVE_ENGINE = "pyqpanda_alg.QAOA.qaoa.QAOA.run_qaoa_circuit"


class TinyIsingModel:
    """Small energy oracle with independently specified Ising coefficients."""

    def __init__(self, h, j=None, offset=2.3):
        self.h = np.asarray(h, dtype=float)
        self.n_variables = len(self.h)
        self.j = (np.zeros((self.n_variables, self.n_variables))
                  if j is None else np.asarray(j, dtype=float))
        self.offset = offset

    def to_ising(self):
        return self.h, self.j, self.offset

    def energy(self, bits):
        z = 1.0 - 2.0 * np.asarray(bits, dtype=float)
        return z @ self.h + np.einsum("...i,ij,...j->...", z, self.j, z) + self.offset


@contextmanager
def fake_quantum_library(probabilities=None):
    """Expose the public library API, recording the actual circuit calls.

    The fake performs no circuit simulation: a deliberately asymmetric
    distribution reveals endian errors and accidental NumPy fallbacks.
    """
    instances = []

    class PauliOperator:
        def __init__(self, terms):
            self.terms = dict(terms)

    class QAOA:
        def __init__(self, problem, init_circuit=None, mixer_circuit=None, norm=False):
            self.problem = problem
            self.norm = norm
            # The library infers an incomplete register for inactive variables.
            self.problem_dimension = 0
            self.calls = []
            instances.append(self)

        def run_qaoa_circuit(self, gammas, betas, shots=-1):
            self.calls.append((np.array(gammas, copy=True), np.array(betas, copy=True), shots))
            if probabilities is not None:
                return dict(probabilities)
            width = self.problem_dimension
            return {format(index, f"0{width}b"): 1 / (1 << width)
                    for index in range(1 << width)}

    modules = {name: ModuleType(name) for name in (
        "pyqpanda_alg", "pyqpanda_alg.QAOA", "pyqpanda_alg.QAOA.qaoa",
        "pyqpanda3", "pyqpanda3.hamiltonian",
    )}
    modules["pyqpanda_alg.QAOA.qaoa"].QAOA = QAOA
    modules["pyqpanda3.hamiltonian"].PauliOperator = PauliOperator
    with patch.dict(sys.modules, modules):
        yield instances


class LibraryRoutingTests(unittest.TestCase):
    def test_numpy_remains_available_without_quantum_dependencies(self):
        with patch.dict(sys.modules, {"pyqpanda3": None, "pyqpanda_alg": None}):
            probabilities = qaoa_probabilities(TinyIsingModel([1.0]), [0.0], [0.0], backend="numpy")
        assert_allclose(probabilities, [0.5, 0.5], atol=1e-14)

    def test_missing_dependencies_fail_without_silent_fallback(self):
        for dependency in ("pyqpanda3.hamiltonian", "pyqpanda_alg.QAOA.qaoa"):
            for backend in ("pyqpanda_alg", "pyqpanda"):
                with self.subTest(dependency=dependency, backend=backend), fake_quantum_library():
                    with patch.dict(sys.modules, {dependency: None}):
                        with self.assertRaises(RuntimeError) as caught:
                            qaoa_probabilities(TinyIsingModel([1.0]), [0.2], [0.3], backend=backend)
                    message = str(caught.exception)
                    self.assertIn("pyqpanda", message)
                    self.assertIn("pip install", message)
                    self.assertIn("numpy", message)

    def test_default_uses_library_with_normalized_coefficients_and_full_register(self):
        j = np.zeros((4, 4))
        j[0, 2] = 0.6
        model = TinyIsingModel([2.0, 0.0, -0.4, 0.0], j, offset=83.0)
        with fake_quantum_library({"0001": 0.25, "1000": 0.75}) as instances:
            actual = qaoa_probabilities(model, [0.31, 0.72], [0.42, -0.13])
        self.assertEqual(len(instances), 1)
        engine = instances[0]
        self.assertFalse(engine.norm)
        self.assertEqual(engine.problem_dimension, 4)
        self.assertEqual(engine.problem.terms, {"Z0": 1.0, "Z2": -0.2, "Z0 Z2": 0.3})
        self.assertEqual(len(engine.calls), 1)
        gammas, betas, shots = engine.calls[0]
        assert_allclose(gammas, [0.31, 0.72])
        assert_allclose(betas, [-0.42, 0.13])
        self.assertEqual(shots, -1)
        expected = np.zeros(16)
        expected[1], expected[8] = 0.25, 0.75
        assert_allclose(actual, expected)

    def test_legacy_backend_name_still_invokes_library(self):
        with fake_quantum_library({"0": 0.2, "1": 0.8}) as instances:
            actual = qaoa_probabilities(TinyIsingModel([1.0]), [0.2], [0.3], backend="pyqpanda")
        self.assertEqual(len(instances[0].calls), 1)
        assert_allclose(actual, [0.2, 0.8])

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy not installed")
    def test_default_optimization_evaluates_library_and_samples_reproducibly(self):
        model = TinyIsingModel([1.0, -0.3])
        with fake_quantum_library({"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}) as instances:
            first = solve_qaoa(model, depth=2, maxiter=12, restarts=2, shots=100, seed=19)
            second = solve_qaoa(model, depth=2, maxiter=12, restarts=2, shots=100, seed=19)
        self.assertEqual(len(instances), 2)
        self.assertEqual(first["backend"], "pyqpanda_alg")
        self.assertEqual(first["engine"], NATIVE_ENGINE)
        self.assertEqual(len(instances[0].calls), first["evaluations"] + 1)
        self.assertEqual(first["circuit_evaluations"], len(instances[0].calls))
        self.assertTrue(all(call[2] == -1 for call in instances[0].calls))
        self.assertEqual(first["counts"], second["counts"])
        self.assertEqual(sum(first["counts"].values()), 100)
        _, energies = enumerate_spectrum(model)
        self.assertAlmostEqual(first["expectation"], float(np.array([0.1, 0.2, 0.3, 0.4]) @ energies))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy not installed")
    def test_default_cli_routes_through_library_and_exports_engine(self):
        with tempfile.TemporaryDirectory() as directory, fake_quantum_library() as instances:
            with redirect_stdout(io.StringIO()):
                code = main(["--no-plot", "--depth", "1", "--maxiter", "8",
                             "--restarts", "1", "--shots", "32", "--output", directory])
            summary = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(summary["configuration"]["backend"], "pyqpanda_alg")
        self.assertEqual(summary["qaoa"]["engine"], NATIVE_ENGINE)
        self.assertEqual(instances[0].problem_dimension, summary["model"]["total_qubits"])
        self.assertEqual(len(instances[0].calls), summary["qaoa"]["evaluations"] + 1)
        self.assertEqual(sum(summary["qaoa"]["counts_by_state_index"].values()), 32)

    def test_default_cli_reports_dependency_error_and_does_not_export_results(self):
        with tempfile.TemporaryDirectory() as directory, fake_quantum_library():
            output = Path(directory) / "result"
            error = io.StringIO()
            with patch.dict(sys.modules, {"pyqpanda_alg.QAOA.qaoa": None}):
                with redirect_stdout(io.StringIO()), redirect_stderr(error):
                    with self.assertRaises(SystemExit) as caught:
                        main(["--no-plot", "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("pyqpanda", error.getvalue())
            self.assertFalse((output / "summary.json").exists())


@unittest.skipUnless(NATIVE_AVAILABLE, "pyqpanda3 and pyqpanda_alg are required for native integration")
class NativeLibraryTests(unittest.TestCase):
    def test_two_layer_native_circuit_matches_independent_dense_exponential(self):
        from scipy.linalg import expm

        j = np.zeros((3, 3))
        j[0, 2] = 0.63
        model = TinyIsingModel([1.7, -0.2, -0.4], j)
        bits, energies = enumerate_spectrum(model)
        cost = np.diag((energies - model.offset) / 1.7)
        mixer = np.zeros((len(bits), len(bits)))
        for index in range(len(bits)):
            for qubit in range(model.n_variables):
                mixer[index ^ (1 << qubit), index] = 1
        gammas, betas = [0.31, 0.72], [0.42, -0.13]
        state = np.ones(len(bits), dtype=complex) / np.sqrt(len(bits))
        for gamma, beta in zip(gammas, betas):
            state = expm(-1j * beta * mixer) @ expm(-1j * gamma * cost) @ state
        actual = qaoa_probabilities(model, gammas, betas)
        assert_allclose(actual, np.abs(state) ** 2, atol=1e-8, rtol=1e-8)

    def test_preserves_middle_and_high_inactive_qubits(self):
        j = np.zeros((4, 4))
        j[0, 2] = 0.63
        model = TinyIsingModel([1.7, 0.0, -0.4, 0.0], j)
        gammas, betas = [0.31, 0.72], [0.42, -0.13]
        expected = qaoa_probabilities(model, gammas, betas, backend="numpy")
        actual = qaoa_probabilities(model, gammas, betas)
        self.assertEqual(actual.shape, (16,))
        assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)

    def test_qubit_indices_above_nine_keep_numeric_variable_order(self):
        h = np.zeros(12)
        h[[0, 2, 10]] = [1.7, -0.4, 0.9]
        j = np.zeros((12, 12))
        j[0, 10] = 0.63
        model = TinyIsingModel(h, j)
        gammas, betas = [0.31, 0.72], [0.42, -0.13]
        expected = qaoa_probabilities(model, gammas, betas, backend="numpy")
        actual = qaoa_probabilities(model, gammas, betas)
        self.assertEqual(actual.shape, (4096,))
        assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)

    def test_handles_constant_hamiltonian(self):
        model = TinyIsingModel([0.0, 0.0, 0.0])
        actual = qaoa_probabilities(model, [0.37, -0.12], [0.9, 0.15])
        assert_allclose(actual, np.full(8, 1 / 8), atol=1e-8, rtol=1e-8)

    def test_native_optimization_preserves_energy_and_seeded_sampling(self):
        model = TinyIsingModel([1.2, -0.3, 0.5], offset=7.9)
        first = solve_qaoa(model, depth=1, maxiter=12, restarts=1, shots=128, seed=7)
        second = solve_qaoa(model, depth=1, maxiter=12, restarts=1, shots=128, seed=7)
        self.assertEqual(first["counts"], second["counts"])
        self.assertEqual(sum(first["counts"].values()), 128)
        _, energies = enumerate_spectrum(model)
        params = first["parameters"]
        reference = qaoa_probabilities(model, params["gammas"], params["betas"], backend="numpy")
        assert_allclose(first["probabilities"], reference, atol=1e-8, rtol=1e-8)
        self.assertAlmostEqual(first["expectation"], float(reference @ energies), places=7)


if __name__ == "__main__":
    unittest.main()
