import importlib.util
import numpy as np
import pytest

from pyqpanda_alg.BFDCQO import ExactSolver, QAOASolver
from pyqpanda_alg.BFDCQO.problems import MaxCutProblem
from pyqpanda_alg.BFDCQO.benchmark import run_maxcut_benchmark


def test_maxcut_ising_energy_is_negative_cut():
    app = MaxCutProblem.from_edges(4, [(0,1),(1,2),(2,3),(3,0),(0,2)], name="g")
    p = app.to_ising()
    for idx in range(1 << app.n):
        bits = [(idx >> q) & 1 for q in range(app.n)]
        assert np.isclose(p.energy_index(idx), -app.cut_value_bits(bits), atol=1e-12)


def test_exact_solver_matches_maxcut_ground_truth():
    app = MaxCutProblem.from_edges(5, [(i,j) for i in range(5) for j in range(i+1,5)], name="k5")
    result = ExactSolver().solve(app.to_ising())
    assert np.isclose(-result.best_energy, 6.0)
    assert result.states_evaluated == 32


def test_qaoa_numpy_reference_budget_and_quality():
    app = MaxCutProblem.from_edges(6, [(i,(i+1)%6) for i in range(6)], name="cycle6")
    q = QAOASolver(
        app.to_ising(), p=1, backend="numpy", shots_per_eval=256,
        max_evals=20, final_shots=1024, total_shot_budget=6144,
        seed=7, method="COBYLA",
    ).solve()
    assert q.total_shots == 6144
    assert q.circuit_evaluations == q.objective_evaluations + 1
    assert q.best_energy >= q.exact_ground_energy - 1e-12
    assert 0.0 <= q.ground_state_probability <= 1.0
    assert np.isfinite(q.expected_energy)


def test_benchmark_equal_shot_budget(tmp_path):
    result = run_maxcut_benchmark(seeds=(3,), qaoa_backend="numpy", output_dir=tmp_path)
    assert len(result["summary"]) == 6
    assert (tmp_path / "benchmark_raw.json").exists()
    for row in result["rows"]:
        assert row["total_shots"] == 6144
        assert 0.0 <= row["approx_ratio"] <= 1.0 + 1e-12
        assert 0.0 <= row["p_opt"] <= 1.0 + 1e-12


def test_maxcut_cpuqvm_benchmark_reports_observed_upstream_budget():
    result = run_maxcut_benchmark(
        seeds=(3,),
        bfdcqo_backend="pyqpanda",
        qaoa_implementation="upstream",
        qaoa_shots=32,
        qaoa_max_evals=4,
    )
    assert result["config"]["bfdcqo_backend"] == "pyqpanda3.CPUQVM"
    assert result["config"]["qaoa_backend"] == "pyqpanda_alg.QAOA.CPUQVM"
    assert {row["solver"] for row in result["rows"]} == {
        "BF-DCQO", "QAOA-p1-upstream"
    }
    qaoa_rows = [
        row for row in result["rows"]
        if row["solver"] == "QAOA-p1-upstream"
    ]
    assert qaoa_rows
    for row in qaoa_rows:
        assert row["backend"] == "pyqpanda_alg.QAOA.CPUQVM"
        assert row["target_shot_budget"] == 192
        assert row["total_shots"] == row["circuit_evaluations"] * 32
        assert row["budget_delta"] == row["total_shots"] - 192
    assert result["config"]["seeds"] == [3]
    assert result["config"]["qaoa_final_shots"] is None


def test_maxcut_rejects_unknown_qaoa_implementation():
    with pytest.raises(ValueError, match="qaoa_implementation"):
        run_maxcut_benchmark(qaoa_implementation="unknown")


@pytest.mark.skipif(importlib.util.find_spec("pyqpanda3") is None, reason="pyqpanda3 not installed")
def test_pyqpanda_qaoa_matches_numpy_exact_objective():
    app = MaxCutProblem.from_edges(4, [(0,1),(1,2),(2,3),(3,0)], name="cycle4")
    p = app.to_ising()
    a = QAOASolver(p, p=1, backend="numpy", shots_per_eval=None, max_evals=15, final_shots=2048, seed=5).solve()
    b = QAOASolver(p, p=1, backend="pyqpanda", shots_per_eval=None, max_evals=15, final_shots=2048, seed=5).solve()
    assert np.isclose(a.expected_energy, b.expected_energy, atol=1e-7)
    assert np.isclose(a.ground_state_probability, b.ground_state_probability, atol=1e-7)


def test_qaoa_cycle6_known_p1_expectation():
    # For an even cycle with the convention H=-Cut, beta=3pi/8 and gamma=pi/4
    # reaches the standard p=1 expected cut value 3n/4 = 4.5 for n=6.
    app = MaxCutProblem.from_edges(6, [(i,(i+1)%6) for i in range(6)], name="cycle6")
    solver = QAOASolver(app.to_ising(), p=1, backend="numpy", shots_per_eval=None, max_evals=1, final_shots=10, seed=0)
    prob = solver._numpy_prob(np.array([3*np.pi/8, np.pi/4]))
    expected_energy = float(prob @ app.to_ising().energies())
    assert np.isclose(-expected_energy, 4.5, atol=1e-10)
