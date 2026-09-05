import numpy as np

from pyqpanda_alg.QSEncode import (
    DEFAULT_METHODS,
    PreparationMethod,
    ResourceAudit,
    SelectionDecision,
)
from pyqpanda_alg.QSEncode._error_budget import find_k_star
from pyqpanda_alg.QSEncode._selection import generate_candidate_grid, run_resource_selection
from pyqpanda_alg.QSEncode._transforms import normalized_fourier, normalized_fwht


PROBABILITIES = np.array(
    [0.0006917643261373052, 0.015724004731018214, 0.1261730210273901,
     0.3574112099154543, 0.3574112099154544, 0.1261730210273902,
     0.01572400473101823, 0.0006917643261373052]
)


def test_candidate_grid_preserves_every_k_method_and_same_compiler_profile():
    coefficients = normalized_fwht(np.sqrt(PROBABILITIES))
    budget = find_k_star(coefficients, 0.99)
    grid = generate_candidate_grid(coefficients, basis="walsh", error_budget=budget)

    assert grid.dense_full.role == "dense_full"
    assert grid.dense_full.resource_audit.valid
    assert len(grid.candidates) == len(budget.candidate_k) * len(DEFAULT_METHODS)
    assert {(candidate.k, candidate.method) for candidate in grid.candidates} == {
        (k, method) for k in budget.candidate_k for method in DEFAULT_METHODS
    }
    fingerprints = {
        candidate.resource_audit.compiler_profile_fingerprint
        for candidate in (grid.dense_full, *grid.candidates)
        if candidate.resource_audit is not None
    }
    assert len(fingerprints) == 1


def test_end_to_end_resource_selection_returns_explainable_result_and_attribution():
    coefficients = normalized_fwht(np.sqrt(PROBABILITIES))
    budget = find_k_star(coefficients, 0.99)
    result = run_resource_selection(coefficients, basis="walsh", error_budget=budget)

    assert result.selection.decision in {SelectionDecision.COMPRESS, SelectionDecision.DO_NOT_COMPRESS}
    assert result.selection.baseline_resource.valid
    assert len(result.grid.candidates) == len(budget.candidate_k) * len(DEFAULT_METHODS)
    if result.selection.decision is SelectionDecision.COMPRESS:
        assert result.attribution is not None
        assert result.attribution.two_qubit_identity_error == 0
        assert result.attribution.depth_identity_error == 0


def test_resource_invalid_method_candidate_is_retained_in_grid(monkeypatch):
    coefficients = normalized_fwht(np.sqrt(PROBABILITIES))
    budget = find_k_star(coefficients, 0.99)

    def synthetic_audit(build, **kwargs):
        failed = build.method is PreparationMethod.DS_QUANTUM_STATE_PREPARATION
        return ResourceAudit(
            compiled_two_qubit_gates=None if failed else 10.0,
            compiled_depth=None if failed else 20.0,
            compiled_total_gates=None if failed else 30.0,
            required_qubits=build.required_qubits,
            allocated_qubits=6,
            repetitions=5,
            ancillas=build.ancillas,
            successful_attempts=0 if failed else 5,
            failed_attempts=5 if failed else 0,
            valid=not failed,
            status="compile_failure" if failed else "valid",
        )

    monkeypatch.setattr(
        "pyqpanda_alg.QSEncode._selection.audit_build_resources", synthetic_audit
    )
    grid = generate_candidate_grid(coefficients, basis="walsh", error_budget=budget)

    ds_candidates = [
        candidate for candidate in grid.candidates
        if candidate.method is PreparationMethod.DS_QUANTUM_STATE_PREPARATION
    ]
    assert len(ds_candidates) == len(budget.candidate_k)
    fidelity_eligible_ds = [
        candidate for candidate in ds_candidates
        if candidate.retained_fidelity >= 0.99 - 1e-12
    ]
    assert fidelity_eligible_ds
    assert all(candidate.status == "resource_invalid" for candidate in fidelity_eligible_ds)
    assert all(candidate.eligible is False for candidate in ds_candidates)
    assert all(candidate.resource_audit is not None for candidate in ds_candidates)


def test_frozen_fourier_case_compresses_with_exact_attribution_identity():
    coefficients = normalized_fourier(np.sqrt(PROBABILITIES))
    budget = find_k_star(coefficients, 0.99)
    result = run_resource_selection(
        coefficients, basis="fourier", error_budget=budget
    )

    assert result.selection.decision is SelectionDecision.COMPRESS
    assert result.selection.selected_candidate_id == "compressed__k4__sparse_isometry"
    assert result.attribution is not None
    assert result.attribution.two_qubit_identity_error == 0
    assert result.attribution.depth_identity_error == 0
    ds_k4 = next(
        candidate for candidate in result.grid.candidates
        if candidate.k == 4
        and candidate.method is PreparationMethod.DS_QUANTUM_STATE_PREPARATION
    )
    assert ds_k4.status == "capability_incompatible"
    assert ds_k4 in result.grid.candidates
