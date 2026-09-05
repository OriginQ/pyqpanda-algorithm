"""Frozen six-key resource selection, refusal, and attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .config import (
    DEFAULT_METHODS,
    CompilerConfig,
    PreparationMethod,
    SelectionDecision,
)
from .exceptions import BaselineConstructionError, InternalInvariantError
from .models import (
    AttributionReport,
    CandidateResult,
    ErrorBudgetResult,
    SelectionResult,
)
from ._capability import assess_capability
from ._compiler import FROZEN_COMPILER_PROFILE
from ._error_budget import MINIMALITY_TOLERANCE, retained_energy_ratio, top_k_coefficients
from ._preparation import adapt_preparation_input
from ._resources import audit_build_resources


METHOD_ORDER = {method: index for index, method in enumerate(DEFAULT_METHODS)}


@dataclass(frozen=True, slots=True)
class CandidateGrid:
    dense_full: CandidateResult
    candidates: tuple[CandidateResult, ...]


@dataclass(frozen=True, slots=True)
class ResourceSelectionRun:
    grid: CandidateGrid
    selection: SelectionResult
    attribution: AttributionReport | None


def _candidate_from_state(
    coefficients: np.ndarray,
    *,
    basis: str,
    method: PreparationMethod,
    k: int,
    role: str,
    retained_fidelity: float,
    fidelity_target: float,
    profile: CompilerConfig,
) -> CandidateResult:
    prepared = adapt_preparation_input(coefficients)
    allocated = profile.physical_capacity_multiplier * prepared.logical_output_qubits
    capability = assess_capability(
        method, prepared, available_qubits=allocated
    )
    audit = None
    if capability.report.compatible and capability.build is not None:
        audit = audit_build_resources(
            capability.build, basis=basis, profile=profile
        )

    if role == "dense_full":
        eligible = bool(capability.report.compatible and audit is not None and audit.valid)
        reason = "mandatory_baseline" if eligible else "baseline_unavailable"
        status = "baseline_valid" if eligible else "baseline_failure"
    elif retained_fidelity < fidelity_target - MINIMALITY_TOLERANCE:
        eligible, reason, status = False, "fidelity_below_target", "fidelity_ineligible"
    elif not capability.report.compatible:
        eligible, reason, status = False, "capability_incompatible", "capability_incompatible"
    elif audit is None or not audit.valid:
        eligible, reason, status = False, "resource_audit_invalid", "resource_invalid"
    else:
        eligible, reason, status = True, "eligible", "eligible"

    return CandidateResult(
        candidate_id=(
            "dense_full__amplitude_encode"
            if role == "dense_full"
            else f"compressed__k{k}__{method.value}"
        ),
        method=method,
        k=k,
        status=status,
        verified_fidelity=capability.report.logical_fidelity,
        resource_audit=audit,
        failure_reason=None if eligible else reason,
        capability=capability.report,
        retained_fidelity=float(retained_fidelity),
        eligible=eligible,
        eligibility_reason=reason,
        role=role,
    )


def generate_candidate_grid(
    coefficients: ArrayLike,
    *,
    basis: str,
    error_budget: ErrorBudgetResult,
    profile: CompilerConfig = FROZEN_COMPILER_PROFILE,
    methods: Sequence[PreparationMethod] = DEFAULT_METHODS,
) -> CandidateGrid:
    values = np.asarray(coefficients)
    dense_full = _candidate_from_state(
        values,
        basis=basis,
        method=PreparationMethod.AMPLITUDE_ENCODE,
        k=int(values.size),
        role="dense_full",
        retained_fidelity=1.0,
        fidelity_target=error_budget.fidelity_target,
        profile=profile,
    )
    candidates: list[CandidateResult] = []
    for k in error_budget.candidate_k:
        retained = retained_energy_ratio(values, k)
        selected = top_k_coefficients(values, k, normalize=True)
        for method in methods:
            candidates.append(
                _candidate_from_state(
                    selected,
                    basis=basis,
                    method=method,
                    k=k,
                    role=("dense_compressed" if method is PreparationMethod.AMPLITUDE_ENCODE else "method_candidate"),
                    retained_fidelity=retained,
                    fidelity_target=error_budget.fidelity_target,
                    profile=profile,
                )
            )
    return CandidateGrid(dense_full=dense_full, candidates=tuple(candidates))


def selector_key(candidate: CandidateResult) -> tuple[float, float, int, int, int, int]:
    audit = candidate.resource_audit
    if not candidate.eligible or audit is None or not audit.valid:
        raise InternalInvariantError(code="selector_received_ineligible_candidate")
    if audit.compiled_two_qubit_gates is None or audit.compiled_depth is None:
        raise InternalInvariantError(code="selector_missing_primary_resource")
    return (
        audit.compiled_two_qubit_gates,
        audit.compiled_depth,
        audit.required_qubits,
        len(audit.ancillas),
        candidate.k,
        METHOD_ORDER[candidate.method],
    )


def select_best_eligible(candidates: Iterable[CandidateResult]) -> CandidateResult | None:
    eligible = [
        candidate for candidate in candidates
        if candidate.eligible
        and candidate.resource_audit is not None
        and candidate.resource_audit.valid
    ]
    return min(eligible, key=selector_key) if eligible else None


def select_resource_candidate(
    baseline: CandidateResult,
    candidates: Iterable[CandidateResult],
) -> SelectionResult:
    baseline_audit = baseline.resource_audit
    if (
        baseline_audit is None
        or not baseline_audit.valid
        or baseline_audit.compiled_two_qubit_gates is None
        or baseline_audit.compiled_depth is None
    ):
        raise BaselineConstructionError(code="dense_full_resource_invalid")
    best = select_best_eligible(candidates)
    if best is None:
        return SelectionResult(
            decision=SelectionDecision.DO_NOT_COMPRESS,
            reason_code="no_eligible_compressed_candidate",
            reason="No compressed candidate satisfied fidelity, capability, and resource validity.",
            baseline_resource=baseline_audit,
        )
    best_audit = best.resource_audit
    if best_audit is None or best_audit.compiled_two_qubit_gates is None or best_audit.compiled_depth is None:
        raise InternalInvariantError(code="best_candidate_missing_resource")
    delta_twoq = baseline_audit.compiled_two_qubit_gates - best_audit.compiled_two_qubit_gates
    delta_depth = baseline_audit.compiled_depth - best_audit.compiled_depth
    improves = delta_twoq > 0 or delta_depth > 0
    return SelectionResult(
        decision=(SelectionDecision.COMPRESS if improves else SelectionDecision.DO_NOT_COMPRESS),
        reason_code=(
            "compressed_candidate_improves_primary_resource"
            if improves else "no_primary_resource_improvement"
        ),
        reason=(
            "At least one primary compiled resource strictly improves."
            if improves else "Neither compiled two-qubit gates nor depth improves."
        ),
        selected_candidate_id=best.candidate_id if improves else None,
        method=best.method if improves else None,
        k=best.k if improves else None,
        baseline_resource=baseline_audit,
        best_compressed_candidate_id=best.candidate_id,
        best_compressed_resource=best_audit,
        comparison_metrics={
            "two_qubit_difference": delta_twoq,
            "depth_difference": delta_depth,
        },
    )


def _valid_primary(candidate: CandidateResult) -> tuple[float, float]:
    audit = candidate.resource_audit
    if audit is None or not audit.valid or audit.compiled_two_qubit_gates is None or audit.compiled_depth is None:
        raise InternalInvariantError(code="attribution_resource_invalid")
    return audit.compiled_two_qubit_gates, audit.compiled_depth


def compute_attribution(
    dense_full: CandidateResult,
    dense_compressed: CandidateResult,
    selected: CandidateResult,
) -> AttributionReport:
    full_twoq, full_depth = _valid_primary(dense_full)
    dense_twoq, dense_depth = _valid_primary(dense_compressed)
    selected_twoq, selected_depth = _valid_primary(selected)
    total_twoq = full_twoq - selected_twoq
    trunc_twoq = full_twoq - dense_twoq
    prep_twoq = dense_twoq - selected_twoq
    total_depth = full_depth - selected_depth
    trunc_depth = full_depth - dense_depth
    prep_depth = dense_depth - selected_depth
    return AttributionReport(
        truncation_two_qubit_gain=(100.0 * trunc_twoq / full_twoq if full_twoq else None),
        preparation_two_qubit_gain=(100.0 * prep_twoq / full_twoq if full_twoq else None),
        truncation_depth_gain=(100.0 * trunc_depth / full_depth if full_depth else None),
        preparation_depth_gain=(100.0 * prep_depth / full_depth if full_depth else None),
        total_two_qubit_difference=total_twoq,
        truncation_two_qubit_difference=trunc_twoq,
        preparation_two_qubit_difference=prep_twoq,
        total_depth_difference=total_depth,
        truncation_depth_difference=trunc_depth,
        preparation_depth_difference=prep_depth,
        two_qubit_identity_error=abs(total_twoq - (trunc_twoq + prep_twoq)),
        depth_identity_error=abs(total_depth - (trunc_depth + prep_depth)),
    )


def run_resource_selection(
    coefficients: ArrayLike,
    *,
    basis: str,
    error_budget: ErrorBudgetResult,
    profile: CompilerConfig = FROZEN_COMPILER_PROFILE,
    methods: Sequence[PreparationMethod] = DEFAULT_METHODS,
) -> ResourceSelectionRun:
    grid = generate_candidate_grid(
        coefficients,
        basis=basis,
        error_budget=error_budget,
        profile=profile,
        methods=methods,
    )
    selection = select_resource_candidate(grid.dense_full, grid.candidates)
    attribution = None
    if selection.decision is SelectionDecision.COMPRESS:
        selected = next(
            candidate for candidate in grid.candidates
            if candidate.candidate_id == selection.selected_candidate_id
        )
        dense = next(
            candidate for candidate in grid.candidates
            if candidate.k == selected.k
            and candidate.method is PreparationMethod.AMPLITUDE_ENCODE
        )
        attribution = compute_attribution(grid.dense_full, dense, selected)
    return ResourceSelectionRun(grid=grid, selection=selection, attribution=attribution)


__all__ = [
    "METHOD_ORDER", "CandidateGrid", "ResourceSelectionRun", "generate_candidate_grid",
    "selector_key", "select_best_eligible", "select_resource_candidate",
    "compute_attribution", "run_resource_selection",
]
