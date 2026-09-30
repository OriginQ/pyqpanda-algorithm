import importlib.util
import numpy as np
import pytest

from pyqpanda_alg.BFDCQO import BFDCQO, CVaRBFDCQO, AdaptiveShotBFDCQO
from pyqpanda_alg.BFDCQO.core.feedback import cvar_feedback_from_counts
from pyqpanda_alg.BFDCQO.benchmark.instances import default_maxcut_instances
from pyqpanda_alg.BFDCQO.benchmark.max3sat import default_max3sat_instances
from pyqpanda_alg.BFDCQO.benchmark.family import run_family_benchmark, run_adaptive_efficiency_benchmark


def weighted6():
    return next(x for x in default_maxcut_instances() if x.name == "weighted6").to_ising()


def frustrated_hubo():
    return next(x for x in default_max3sat_instances() if x.name == "frustrated5_unsat").to_native_ising()


def test_cvar_count_selection_keeps_lowest_energy_samples():
    # 2-qubit basis states 00,01,10,11 in integer order.
    counts = np.array([2, 2, 2, 2])
    energies = np.array([0.0, 1.0, 2.0, 3.0])
    info = cvar_feedback_from_counts(counts, energies, n=2, alpha=0.25)
    assert info.effective_samples == 2
    assert info.energy_threshold == 0.0
    assert np.allclose(info.bias, [1.0, 1.0])


def test_cvar_alpha_one_exactly_recovers_basic_feedback():
    p = weighted6()
    kwargs = dict(problem=p, n_trotter=3, dt=0.1, iterations=5, shots=512, seed=9, backend="numpy")
    a = BFDCQO(**kwargs).solve()
    b = CVaRBFDCQO(**kwargs, cvar_alpha=1.0).solve()
    assert a.best_solution == b.best_solution
    assert np.isclose(a.best_energy, b.best_energy)
    assert a.shot_history == b.shot_history
    for x, y in zip(a.history, b.history):
        assert np.allclose(x.bias, y.bias, atol=1e-15)
        assert np.isclose(x.expected_energy, y.expected_energy, atol=1e-15)
        assert np.isclose(x.ground_state_probability, y.ground_state_probability, atol=1e-15)


def test_cvar_effective_samples_matches_alpha_ceiling():
    p = weighted6()
    r = CVaRBFDCQO(p, iterations=3, shots=257, seed=1, cvar_alpha=0.10).solve()
    assert all(h.effective_feedback_shots == 26 for h in r.history)
    assert all(np.isclose(h.feedback_alpha, 0.10) for h in r.history)


def test_adaptive_strict_budget_and_nonuniform_schedule():
    p = weighted6()
    r = AdaptiveShotBFDCQO(
        p, total_shot_budget=6144, max_iterations=6,
        min_shots=256, max_shots=2048, initial_shots=1024,
        seed=4, early_stop=False,
    ).solve()
    assert r.total_shots == 6144
    assert sum(r.shot_history) == 6144
    assert len(r.shot_history) == 6
    assert all(x >= 256 for x in r.shot_history)
    assert len(set(r.shot_history)) > 1
    assert r.variant == "adaptive"


def test_adaptive_native_hubo_reaches_exact_optimum():
    p = frustrated_hubo()
    r = AdaptiveShotBFDCQO(
        p, total_shot_budget=6144, max_iterations=6,
        min_shots=256, max_shots=2048, initial_shots=1024,
        seed=3, early_stop=False,
    ).solve()
    assert r.total_shots == 6144
    assert r.success
    assert r.best_solution in r.exact_ground_states


def test_adaptive_early_stop_never_exceeds_budget():
    p = weighted6()
    r = AdaptiveShotBFDCQO(
        p, total_shot_budget=6144, max_iterations=6,
        min_shots=256, max_shots=2048, initial_shots=1024,
        seed=0, early_stop=True, min_iterations=2,
        uncertainty_tol=1.0, bias_delta_tol=1.0,
    ).solve()
    assert r.stopped_early
    assert r.stop_reason == "bias_converged"
    assert r.total_shots < 6144
    assert r.total_shots == sum(r.shot_history)


def test_family_benchmark_has_strict_matched_budgets_and_both_problem_types(tmp_path):
    out = run_family_benchmark(
        budgets=(1536, 3072), seeds=(0,), cvar_alpha=0.10, output_dir=tmp_path
    )
    assert len(out["rows"]) == 2 * 2 * 1 * 3
    assert {r["problem_kind"] for r in out["rows"]} == {"QUBO", "HUBO-3"}
    assert {r["variant"] for r in out["rows"]} == {"basic", "cvar", "adaptive"}
    for r in out["rows"]:
        assert r["total_shots"] == r["shot_budget"]
        assert sum(r["shot_history"]) == r["shot_budget"]
        assert 0.0 <= r["p_opt"] <= 1.0 + 1e-12
        assert 0.0 <= r["expected_approx_ratio"] <= 1.0 + 1e-10
    assert (tmp_path / "family_benchmark.json").exists()
    assert (tmp_path / "family_raw.csv").exists()
    assert (tmp_path / "family_summary.csv").exists()


@pytest.mark.skipif(importlib.util.find_spec("pyqpanda3") is None, reason="pyqpanda3 not installed")
def test_family_benchmarks_report_real_cpuqvm_backend():
    matched = run_family_benchmark(
        budgets=(192,), seeds=(0,), cvar_alpha=0.10,
        backend="pyqpanda",
    )
    assert matched["config"]["backend"] == "pyqpanda3.CPUQVM"
    assert {row["backend"] for row in matched["rows"]} == {
        "pyqpanda3.CPUQVM"
    }

    adaptive = run_adaptive_efficiency_benchmark(
        budget_caps=(192,), seeds=(0,), cvar_alpha=0.10,
        backend="pyqpanda",
    )
    assert adaptive["config"]["backend"] == "pyqpanda3.CPUQVM"
    assert {row["backend"] for row in adaptive["rows"]} == {
        "pyqpanda3.CPUQVM"
    }


def test_cvar_degenerate_threshold_is_basis_order_invariant():
    # alpha=0.5 cuts through a degenerate E=0 group. The two states must be
    # retained proportionally (80/20), not by basis index.
    counts = np.array([8, 2, 0, 0])
    energies = np.array([0.0, 0.0, 1.0, 1.0])
    info = cvar_feedback_from_counts(counts, energies, n=2, alpha=0.5)
    assert info.effective_samples == 5
    assert np.allclose(info.bias, [0.6, 1.0], atol=1e-12)


def test_adaptive_cvar_efficiency_never_exceeds_cap(tmp_path):
    out = run_adaptive_efficiency_benchmark(
        budget_caps=(1536,), seeds=(0, 1), cvar_alpha=0.10, output_dir=tmp_path
    )
    assert len(out["rows"]) == 2 * 1 * 2 * 2
    for r in out["rows"]:
        assert r["total_shots"] <= r["budget_cap"]
        if r["variant"] == "cvar_fixed":
            assert r["total_shots"] == r["budget_cap"]
    assert (tmp_path / "adaptive_efficiency_benchmark.json").exists()
    assert (tmp_path / "adaptive_efficiency_summary.csv").exists()


@pytest.mark.skipif(importlib.util.find_spec("pyqpanda3") is None, reason="pyqpanda3 not installed")
def test_pyqpanda_probability_engine_matches_numpy_before_feedback():
    p = frustrated_hubo()
    a = BFDCQO(p, n_trotter=3, dt=0.1, iterations=1, shots=None, backend="numpy")
    b = BFDCQO(p, n_trotter=3, dt=0.1, iterations=1, shots=None, backend="pyqpanda")
    pa, _ = a._probabilities(np.zeros(p.n))
    pb, _ = b._probabilities(np.zeros(p.n))
    assert np.allclose(pa, pb, atol=1e-8)
