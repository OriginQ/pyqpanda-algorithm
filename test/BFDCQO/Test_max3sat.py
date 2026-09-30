import importlib.util
import numpy as np
import pytest

from pyqpanda_alg.BFDCQO.applications.max3sat import (
    Max3SATProblem,
    max3sat_to_qubo,
)
from pyqpanda_alg.BFDCQO.benchmark.max3sat import default_max3sat_instances, run_max3sat_benchmark
from pyqpanda_alg.BFDCQO.problems.polynomial import binary_hubo_energy
from pyqpanda_alg.BFDCQO.solvers import ExactSolver, BFDCQO, QAOASolver


def test_single_clause_binary_hubo_matches_truth_table():
    # (x1 OR NOT x2 OR x3) is false only at x=[0,1,0].
    app = Max3SATProblem.from_clauses(3, [(1, -2, 3)], name="single")
    offset, terms = app.binary_hubo()
    assert any(len(t) == 3 for t in terms)
    for idx in range(8):
        bits = [(idx >> q) & 1 for q in range(3)]
        direct = app.unsatisfied_weight_bits(bits)
        poly = binary_hubo_energy(3, terms, bits, offset=offset)
        assert np.isclose(direct, poly, atol=1e-12)
    assert app.unsatisfied_weight_bits([0, 1, 0]) == 1.0
    assert app.unsatisfied_weight_bits([1, 1, 0]) == 0.0


def test_native_ising_matches_max3sat_unsat_cost_for_all_assignments():
    for app in default_max3sat_instances():
        native = app.to_native_ising()
        assert native.degree == 3
        for idx in range(1 << app.n):
            bits = [(idx >> q) & 1 for q in range(app.n)]
            assert np.isclose(native.energy_index(idx), app.unsatisfied_weight_bits(bits), atol=1e-12)


def test_rosenberg_quadratization_preserves_energy_for_every_logical_assignment():
    for app in default_max3sat_instances():
        offset, terms = app.binary_hubo()
        red = max3sat_to_qubo(app, penalty_margin=1.0)
        assert red.total_n == app.n + sum(1 for t in terms if len(t) == 3)
        assert all(len(t) <= 2 for t in red.terms)
        for idx in range(1 << app.n):
            bits = [(idx >> q) & 1 for q in range(app.n)]
            native = app.unsatisfied_weight_bits(bits)
            reduced, anc = red.min_energy_for_logical_bits(bits)
            assert np.isclose(native, reduced, atol=1e-12), (app.name, bits, native, reduced, anc)


def test_exact_native_and_quadratized_ground_states_solve_same_application():
    for app in default_max3sat_instances():
        opt, _ = app.exact_solution()
        native_exact = ExactSolver().solve(app.to_native_ising())
        red = max3sat_to_qubo(app)
        q_exact = ExactSolver().solve(red.to_ising())
        assert np.isclose(app.total_weight - native_exact.best_energy, opt, atol=1e-12)
        logical = q_exact.best_solutions[0][-app.n:]
        assert np.isclose(app.satisfied_weight_bitstring(logical), opt, atol=1e-12)


def test_native_hubo_bfdcqo_hits_exact_optimum_on_max3sat_instances():
    for app in default_max3sat_instances():
        r = BFDCQO(app.to_native_ising(), n_trotter=3, dt=0.1, iterations=6, shots=2048, seed=7).solve()
        opt, _ = app.exact_solution()
        assert np.isclose(app.total_weight - r.best_energy, opt, atol=1e-9)


def test_max3sat_benchmark_budget_and_reduction_metadata(tmp_path):
    r = run_max3sat_benchmark(seeds=(0,), qaoa_backend="numpy", output_dir=tmp_path)
    assert len(r["reductions"]) == 4
    assert len(r["rows"]) == 8
    for red in r["reductions"]:
        assert red["native_degree"] == 3
        assert red["ancillas"] > 0
        assert red["qubo_n"] == red["logical_n"] + red["ancillas"]
    for row in r["rows"]:
        assert row["total_shots"] == 6144
        assert 0.0 <= row["p_opt_logical"] <= 1.0 + 1e-12
        assert 0.0 <= row["expected_approx_ratio"] <= 1.0 + 1e-10
        assert row["best_approx_ratio"] <= 1.0 + 1e-10
    assert (tmp_path / "max3sat_benchmark.json").exists()
    assert (tmp_path / "max3sat_summary.csv").exists()
    assert (tmp_path / "max3sat_reduction_summary.csv").exists()


def test_max3sat_upstream_qaoa_uses_quadratized_cpuqvm_problem():
    app = default_max3sat_instances()[0]
    result = run_max3sat_benchmark(
        instances=(app,), seeds=(0,),
        bfdcqo_backend="pyqpanda",
        qaoa_implementation="upstream",
        qaoa_shots=32,
        qaoa_max_evals=4,
    )
    assert len(result["rows"]) == 2
    qaoa = next(row for row in result["rows"] if row["solver"] == "QAOA-p1-upstream")
    assert qaoa["backend"] == "pyqpanda_alg.QAOA.CPUQVM"
    assert qaoa["formulation"] == "Rosenberg-quadratized-QUBO"
    assert qaoa["solver_n"] > qaoa["logical_n"]
    assert qaoa["target_shot_budget"] == 192
    assert qaoa["total_shots"] > 0
    assert qaoa["budget_delta"] == qaoa["total_shots"] - 192
    assert len(qaoa["best_logical_solution"]) == app.n
    assert np.isclose(
        app.satisfied_weight_bitstring(qaoa["best_logical_solution"]),
        qaoa["best_satisfied_weight"],
    )
    assert 0.0 <= qaoa["p_opt_logical"] <= 1.0
    assert 0.0 <= qaoa["best_approx_ratio"] <= 1.0 + 1e-12
    assert 0.0 <= qaoa["expected_approx_ratio"] <= 1.0 + 1e-12
    assert result["config"]["seeds"] == [0]
    assert result["config"]["qaoa_final_shots"] is None


def test_max3sat_rejects_unknown_qaoa_implementation():
    with pytest.raises(ValueError, match="qaoa_implementation"):
        run_max3sat_benchmark(qaoa_implementation="unknown")


@pytest.mark.skipif(importlib.util.find_spec("pyqpanda3") is None, reason="pyqpanda3 not installed")
def test_quadratized_qaoa_pyqpanda_matches_numpy_distribution():
    app = default_max3sat_instances()[0]
    red = max3sat_to_qubo(app)
    p = red.to_ising()
    params = np.array([0.27, 0.63])  # beta, gamma for p=1
    a = QAOASolver(p, p=1, backend="numpy", shots_per_eval=None, max_evals=2, final_shots=32)
    b = QAOASolver(p, p=1, backend="pyqpanda", shots_per_eval=None, max_evals=2, final_shots=32)
    pa = a.probabilities(params)
    pb = b.probabilities(params)
    assert np.allclose(pa, pb, atol=1e-8)
