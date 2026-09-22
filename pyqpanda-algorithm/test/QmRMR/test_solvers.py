import numpy as np

from pyqpanda_alg.QmRMR.constrained import (
    ConstrainedQAOAQmRMR,
    ExchangeAnsatzSelector,
    QmRMRFeatureSelection,
    QmRMRProblem,
)
from pyqpanda_alg.QmRMR.constrained.problem import expectation_from_probabilities


def make_problem():
    rng = np.random.default_rng(11)
    q = rng.uniform(0.0, 1.0, size=(4, 4))
    q = 0.5 * (q + q.T)
    u = -rng.uniform(0.0, 1.0, size=4)
    return QmRMRProblem(q, u, 2)


def test_explicit_bit_order_matches_pyqpanda3_display_order():
    problem = QmRMRProblem(np.zeros((2, 2)), [1.0, 2.0], 1)
    selector = ExchangeAnsatzSelector(problem, bit_order="qubit")
    probabilities = {"00": 0.0, "01": 1.0, "10": 0.0, "11": 0.0}
    # pyQPanda3 displays the highest-index qubit on the left: raw ``01`` is
    # feature-index state ``10`` when feature i is represented by qubit i.
    value = expectation_from_probabilities(problem, probabilities, reverse_key=True)
    np.testing.assert_allclose(value, 1.0, atol=1e-12)
    assert selector.bit_order == "qubit"


def test_exchange_solver_returns_finite_feasible_result():
    problem = make_problem()
    result = ExchangeAnsatzSelector(problem).optimize(
        optimizer="slsqp",
        maxiter=4,
        seed=3,
    )
    assert result.selected_state.count(1) == 2
    assert result.feasible_probability > 1.0 - 1e-12
    assert np.isfinite(result.selected_value)
    assert result.parameters.size == result.metadata["effective_parameter_count"]


def test_constrained_qaoa_returns_finite_feasible_result():
    problem = make_problem()
    result = ConstrainedQAOAQmRMR(
        problem,
        layers=1,
        initializer="amplitude",
    ).optimize(
        optimizer="slsqp",
        maxiter=5,
        seed=3,
    )
    assert result.selected_state.count(1) == 2
    assert result.feasible_probability > 1.0 - 1e-12
    assert np.isfinite(result.selected_value)
    assert result.parameters.size == 2
    assert result.gate_count


def test_constrained_qaoa_supports_lower_tail_cvar_objective():
    problem = make_problem()
    result = ConstrainedQAOAQmRMR(
        problem,
        layers=1,
        initializer="amplitude",
        objective_mode="cvar",
        cvar_alpha=0.5,
    ).optimize(
        optimizer="slsqp",
        maxiter=2,
        seed=4,
    )
    assert result.metadata["optimization_objective"] == "cvar"
    np.testing.assert_allclose(result.metadata["cvar_alpha"], 0.5)
    assert np.isfinite(result.metadata["optimization_value"])
    assert np.isfinite(result.expected_value)
    assert result.feasible_probability > 1.0 - 1e-12


def test_progressive_qaoa_uses_lifted_warm_start_and_restarts():
    problem = make_problem()
    lifted = ConstrainedQAOAQmRMR.lift_parameters([0.1, -0.2], 3)
    np.testing.assert_allclose(lifted, [0.1, -0.2, 0.1, -0.2, 0.1, -0.2])
    stages = ConstrainedQAOAQmRMR(
        problem,
        layers=2,
        initializer="amplitude",
    ).optimize_progressive(
        optimizer="slsqp",
        maxiter=2,
        seed=5,
        restarts=2,
    )
    assert len(stages) == 2
    assert stages[0].metadata["warm_start_used"] is False
    assert stages[1].metadata["warm_start_used"] is True
    assert stages[0].metadata["restart_count"] == 2
    assert np.isfinite(stages[1].expected_value)


def test_high_level_facade_returns_feature_index_order_tuple():
    problem = make_problem()
    facade = QmRMRFeatureSelection(
        problem.quadratic,
        -problem.linear,
        problem.select_num,
        ansatz="constrained_qaoa",
        objective_convention="canonical_mrmr",
        layers=1,
        initializer="amplitude",
        maxiter=2,
        seed=9,
    )
    history, choice, top = facade.get_his_res()
    assert history
    assert len(choice) == problem.n_features
    assert sum(choice) == problem.select_num
    assert top
    assert facade.last_result is not None


def test_high_level_facade_exposes_exchange_option():
    problem = make_problem()
    facade = QmRMRFeatureSelection(
        problem.quadratic,
        -problem.linear,
        problem.select_num,
        ansatz="exchange",
        objective_convention="canonical_mrmr",
        maxiter=2,
        seed=13,
        restarts=2,
    )
    result = facade.optimize()
    assert result.selected_state.count(1) == problem.select_num
    assert result.feasible_probability > 1.0 - 1e-12
    assert result.metadata["restart_count"] == 2


def test_high_level_facade_exposes_cvar_configuration():
    problem = make_problem()
    facade = QmRMRFeatureSelection(
        problem.quadratic,
        -problem.linear,
        problem.select_num,
        ansatz="constrained_qaoa",
        optimization_objective="cvar",
        cvar_alpha=0.5,
        layers=1,
        maxiter=1,
        seed=15,
    )
    result = facade.optimize()
    assert result.summary()["optimization_objective"] == "cvar"
    np.testing.assert_allclose(result.summary()["cvar_alpha"], 0.5)

