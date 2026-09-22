from pyqpanda_alg.UnitCommitment import (
    Case5CommitmentQuboBuilder,
    audit_case5_qubo,
    load_case5_day_ahead,
    load_case5_uc,
    solve_milp_uc,
)
import pytest


def test_case5_classical_reference_is_feasible():
    instance = load_case5_uc()
    result = solve_milp_uc(instance, evaluate_method="linprog", enforce_network=True)
    assert result.success
    assert result.schedule.success
    assert result.schedule.total_cost == pytest.approx(28439.307374, abs=1.0e-3)


def test_case5_qubo_audit_has_quadratic_degree_and_stable_bit_order():
    instance = load_case5_uc()
    model = Case5CommitmentQuboBuilder(capacity_weight=200.0).build(instance)
    report = audit_case5_qubo(model, samples=32, seed=7)
    assert report.passed
    assert report.degree == 2
    assert report.energy_identity_max_error < 1.0e-8


def test_case5_day_ahead_reference_has_24_feasible_hours():
    instance = load_case5_day_ahead()
    assert instance.time_periods == 24
    assert len(instance.demand_mw) == 24
    result = solve_milp_uc(instance, evaluate_method="linprog", enforce_network=True)
    assert result.success
    assert result.schedule.success
    assert len(result.commitments) == 24
    assert len(result.schedule.dispatch) == 24

