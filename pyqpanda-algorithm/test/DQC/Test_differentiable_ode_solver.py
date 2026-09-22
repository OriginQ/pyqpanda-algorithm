"""Tests for the differentiable quantum-circuit ODE application."""

import numpy as np

from pyqpanda_alg.DQC import ExperimentConfig, make_cases, run_experiments


class TestDifferentiableODESolver:
    def test_benchmark_definitions_have_independent_references(self):
        cases = make_cases()
        assert set(cases) == {"lambda20", "coupled", "riccati"}
        for case in cases.values():
            x = np.linspace(case.initial[0] * 0.0, 0.2, 3)
            reference = np.asarray(case.reference(x))
            assert reference.shape[0] == x.size
            assert np.all(np.isfinite(reference))

    def test_numpy_smoke_preserves_initial_value_constraint(self):
        config = ExperimentConfig(
            n_qubits=2,
            depth=1,
            collocation_points=8,
            holdout_points=9,
            steps=0,
            seed=3,
        )
        results = run_experiments("coupled", config=config, backend="numpy")
        metrics = results["coupled"]["metrics"]
        assert metrics["backend"] == "numpy"
        assert metrics["training_uses_reference"] is False
        assert np.isfinite(metrics["aggregate_rmse"])

