import importlib.util
import numpy as np
import pytest

from pyqpanda_alg.BFDCQO import MWISProblem, BFDCQO, HybridBFDCQO
from pyqpanda_alg.BFDCQO.local_search import (
    uncertain_variable_order,
    confidence_guided_local_search,
    random_block_local_search,
)
from pyqpanda_alg.BFDCQO.benchmark.mwis import default_mwis_instances, run_mwis_hybrid_benchmark


def test_mwis_qubo_matches_direct_objective_for_all_assignments():
    app = default_mwis_instances()[0]
    p = app.to_ising()
    for idx in range(1 << app.n):
        bits = [(idx >> q) & 1 for q in range(app.n)]
        assert np.isclose(app.qubo_energy_bits(bits), p.energy_index(idx), atol=1e-12)


def test_mwis_exact_ising_ground_state_matches_combinatorial_optimum():
    for app in default_mwis_instances():
        opt_weight, opt_states = app.exact_mwis()
        e, states = app.to_ising().exact_ground_states()
        assert np.isclose(e, -opt_weight, atol=1e-12)
        assert set(states) == set(opt_states)
        assert all(app.is_independent_bitstring(s) for s in states)


def test_mwis_penalty_rejects_unsafe_value():
    with pytest.raises(ValueError):
        MWISProblem(np.array([1.0, 2.0]), ((0, 1),), penalty=2.0)


def test_uncertain_order_uses_absolute_bias_confidence():
    bias = np.array([0.9, -0.1, 0.4, -0.7])
    assert uncertain_variable_order(bias) == (1, 2, 3, 0)


def test_confidence_block_search_is_monotone_and_budgeted():
    app = default_mwis_instances()[0]
    p = app.to_ising()
    q = BFDCQO(p, n_trotter=2, dt=0.1, iterations=3, shots=24, seed=2).solve()
    local = confidence_guided_local_search(
        p, q.best_solution, q.history[-1].bias, neighborhood_size=4, rounds=1
    )
    assert local.best_energy <= local.start_energy + 1e-12
    assert local.evaluations == 16
    assert len(local.steps) == 1
    assert len(local.steps[0].variables) == 4


def test_hybrid_confidence_can_recover_known_lowshot_mwis_optimum():
    app = default_mwis_instances()[0]
    p = app.to_ising()
    base = BFDCQO(p, n_trotter=2, dt=0.1, iterations=3, shots=24, seed=2).solve()
    hybrid = HybridBFDCQO(
        p, base_variant="basic", local_method="confidence",
        neighborhood_size=4, local_rounds=1,
        n_trotter=2, dt=0.1, max_iterations=3, shots=24, seed=2,
    ).solve()
    assert not base.success
    assert hybrid.success
    assert hybrid.total_shots == base.total_shots == 72
    assert hybrid.local_evaluations == 16
    assert hybrid.best_energy < base.best_energy - 1e-12


def test_random_and_confidence_ablation_have_equal_classical_budget():
    app = default_mwis_instances()[1]
    p = app.to_ising()
    hc = HybridBFDCQO(
        p, base_variant="basic", local_method="confidence",
        neighborhood_size=4, local_rounds=1,
        n_trotter=2, dt=0.1, max_iterations=3, shots=24, seed=0,
    ).solve()
    hr = HybridBFDCQO(
        p, base_variant="basic", local_method="random", local_seed=321,
        neighborhood_size=4, local_rounds=1,
        n_trotter=2, dt=0.1, max_iterations=3, shots=24, seed=0,
    ).solve()
    assert hc.total_shots == hr.total_shots == 72
    assert hc.local_evaluations == hr.local_evaluations == 16
    # Their quantum stages are identical under the same seed/configuration.
    assert hc.quantum_result.best_solution == hr.quantum_result.best_solution
    assert np.allclose(hc.quantum_result.history[-1].bias, hr.quantum_result.history[-1].bias)


def test_mwis_benchmark_budget_and_solver_matrix(tmp_path):
    out = run_mwis_hybrid_benchmark(
        seeds=(0,), quantum_budget=72,
        local_neighborhood_size=4, local_rounds=1,
        qaoa_backend="numpy", output_dir=tmp_path,
    )
    assert len(out["rows"]) == 3 * 1 * 4
    assert {r["solver"] for r in out["rows"]} == {
        "BF-basic", "Hybrid-random", "Hybrid-confidence", "QAOA-p1-reference"
    }
    for r in out["rows"]:
        assert r["total_shots"] == 72
        assert 0.0 <= r["weight_ratio"] <= 1.0 + 1e-12
        if r["solver"].startswith("Hybrid"):
            assert r["local_evaluations"] == 16
        else:
            assert r["local_evaluations"] == 0
    assert (tmp_path / "mwis_hybrid_benchmark.json").exists()
    assert (tmp_path / "mwis_hybrid_summary.csv").exists()


def test_mwis_upstream_qaoa_cpuqvm_smoke_reports_observed_budget():
    app = default_mwis_instances()[0]
    result = run_mwis_hybrid_benchmark(
        instances=(app,), seeds=(0,), quantum_budget=96,
        local_neighborhood_size=4, local_rounds=1,
        bfdcqo_backend="pyqpanda",
        qaoa_implementation="upstream",
    )
    assert len(result["rows"]) == 4
    assert {
        row["backend"] for row in result["rows"]
        if row["solver"] != "QAOA-p1-upstream"
    } == {"pyqpanda3.CPUQVM"}
    qaoa = next(row for row in result["rows"] if row["solver"] == "QAOA-p1-upstream")
    assert qaoa["backend"] == "pyqpanda_alg.QAOA.CPUQVM"
    assert len(qaoa["best_solution"]) == app.n
    assert qaoa["feasible"] == app.is_independent_bitstring(qaoa["best_solution"])
    assert qaoa["violations"] == app.violations_bitstring(qaoa["best_solution"])
    assert 0.0 <= qaoa["weight_ratio"] <= 1.0 + 1e-12
    assert 0.0 <= qaoa["p_opt"] <= 1.0
    assert qaoa["total_shots"] > 0
    assert qaoa["target_shot_budget"] == 96
    assert qaoa["budget_delta"] == (
        qaoa["total_shots"] - qaoa["target_shot_budget"]
    )
    assert result["config"]["qaoa_final_shots"] is None


def test_mwis_rejects_unknown_qaoa_implementation():
    with pytest.raises(ValueError, match="qaoa_implementation"):
        run_mwis_hybrid_benchmark(qaoa_implementation="unknown")


@pytest.mark.skipif(importlib.util.find_spec("pyqpanda3") is None, reason="pyqpanda3 not installed")
def test_pyqpanda_mwis_probability_engine_matches_numpy():
    app = default_mwis_instances()[0]
    p = app.to_ising()
    a = BFDCQO(p, n_trotter=2, dt=0.1, iterations=1, shots=None, backend="numpy")
    b = BFDCQO(p, n_trotter=2, dt=0.1, iterations=1, shots=None, backend="pyqpanda")
    pa, _ = a._probabilities(np.zeros(p.n))
    pb, _ = b._probabilities(np.zeros(p.n))
    assert np.allclose(pa, pb, atol=1e-8)


def test_cvar_hybrid_uses_unfiltered_raw_bias_without_extra_shots():
    app = default_mwis_instances()[0]
    p = app.to_ising()
    h = HybridBFDCQO(
        p, base_variant="cvar", local_method="confidence",
        cvar_alpha=0.10, neighborhood_size=4, local_rounds=1,
        n_trotter=2, dt=0.1, max_iterations=3, shots=24, seed=2,
    ).solve()
    rec = h.quantum_result.history[-1]
    assert rec.raw_bias is not None
    assert h.total_shots == 72
    assert h.local_evaluations == 16
    # Raw confidence uses all 24 samples; CVaR feedback uses only ceil(2.4)=3.
    assert rec.effective_feedback_shots == 3
    assert np.all(np.abs(rec.raw_bias) <= 1.0 + 1e-12)


def test_hybrid_adaptive_variant_respects_quantum_budget():
    app = default_mwis_instances()[1]
    p = app.to_ising()
    h = HybridBFDCQO(
        p, base_variant="adaptive", local_method="confidence",
        total_shot_budget=96, max_iterations=3, shots=32,
        min_shots=16, max_shots=64, initial_shots=32,
        neighborhood_size=3, local_rounds=1,
        n_trotter=2, dt=0.1, seed=1,
    ).solve()
    assert h.total_shots == 96
    assert h.local_evaluations == 8
