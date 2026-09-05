"""Compiled semantic verification for an already frozen resource selection.

This module migrates the constrained mapping rules validated by Final
Technical Hardening and the Locked Generalization runner.  It never compiles,
ranks, selects, or substitutes a candidate.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from pyqpanda3.core import CPUQVM, QProg

from .config import PreparationMethod, VerificationLevel
from .exceptions import InternalInvariantError
from .models import SemanticVerification, SemanticVerificationAttempt
from ._compiler import TECHNICAL_REPETITIONS, CompilationAttempt
from ._compiler import compose_end_to_end_program
from ._error_budget import top_k_coefficients
from ._preparation import adapt_preparation_input, build_preparation
from ._selection import ResourceSelectionRun


SEMANTIC_TOLERANCE = 1e-10
_FIDELITY_TIE_TOLERANCE = 1e-13


def _simulate(program: QProg) -> np.ndarray:
    machine = CPUQVM()
    machine.run(program, 1)
    state = np.asarray(machine.result().get_state_vector(), dtype=np.complex128)
    if state.ndim != 1 or state.size < 2 or state.size & (state.size - 1):
        raise InternalInvariantError(code="invalid_semantic_statevector")
    if not np.all(np.isfinite(state)):
        raise InternalInvariantError(code="nonfinite_semantic_statevector")
    return state


def _reduced_density(state: np.ndarray, ordered_output: tuple[int, ...]) -> np.ndarray:
    total = int(round(math.log2(int(state.size))))
    if (
        len(set(ordered_output)) != len(ordered_output)
        or any(qubit < 0 or qubit >= total for qubit in ordered_output)
    ):
        raise InternalInvariantError(code="invalid_semantic_output_register")
    ancillas = tuple(qubit for qubit in range(total) if qubit not in ordered_output)
    tensor = state.reshape([2] * total, order="F")
    reordered = np.transpose(tensor, axes=ordered_output + ancillas)
    matrix = reordered.reshape(
        (2 ** len(ordered_output), 2 ** len(ancillas)), order="F"
    )
    return matrix @ matrix.conj().T


def _logical_target(
    logical_state: np.ndarray, output_qubits: tuple[int, ...]
) -> tuple[np.ndarray, float]:
    density = _reduced_density(logical_state, output_qubits)
    values, vectors = np.linalg.eigh(density)
    target = vectors[:, int(np.argmax(values))]
    purity = float(np.real(np.trace(density @ density)))
    return target, purity


def _fidelity(target: np.ndarray, state: np.ndarray, output: tuple[int, ...]) -> float:
    density = _reduced_density(state, output)
    value = float(np.real(np.vdot(target, density @ target)))
    if not math.isfinite(value):
        raise InternalInvariantError(code="nonfinite_semantic_fidelity")
    return min(1.0, max(0.0, value))


def _best_within_register(
    target: np.ndarray, state: np.ndarray, output: tuple[int, ...]
) -> dict[str, Any]:
    best = -1.0
    mapping: tuple[int, ...] = ()
    ties = 0
    tested = 0
    for permutation in itertools.permutations(output):
        tested += 1
        value = _fidelity(target, state, permutation)
        if value > best + _FIDELITY_TIE_TOLERANCE:
            best, mapping, ties = value, permutation, 1
        elif abs(value - best) <= _FIDELITY_TIE_TOLERANCE:
            ties += 1
    return {
        "fidelity": best,
        "mapping": mapping,
        "mapping_method": "output_register_constrained_permutation",
        "mapping_ties": ties,
        "mappings_tested": tested,
        "pure_subsets": None,
    }


def _best_ordered_output_subset(
    target: np.ndarray, state: np.ndarray, output_size: int
) -> dict[str, Any]:
    total = int(round(math.log2(int(state.size))))
    pure_subsets: list[tuple[int, ...]] = []
    for subset in itertools.combinations(range(total), output_size):
        density = _reduced_density(state, subset)
        purity = float(np.real(np.trace(density @ density)))
        if purity >= 1.0 - SEMANTIC_TOLERANCE:
            pure_subsets.append(subset)
    best = -1.0
    mapping: tuple[int, ...] = ()
    ties = 0
    tested = 0
    for subset in pure_subsets:
        for permutation in itertools.permutations(subset):
            tested += 1
            value = _fidelity(target, state, permutation)
            if value > best + _FIDELITY_TIE_TOLERANCE:
                best, mapping, ties = value, permutation, 1
            elif abs(value - best) <= _FIDELITY_TIE_TOLERANCE:
                ties += 1
    return {
        "fidelity": best,
        "mapping": mapping,
        "mapping_method": "ordered_output_subset_with_ancilla_trace",
        "mapping_ties": ties,
        "mappings_tested": tested,
        "pure_subsets": len(pure_subsets),
    }


def certify_compiled_attempt(
    logical_program: QProg,
    compiled_program: QProg,
    *,
    output_qubits: tuple[int, ...],
    method: PreparationMethod,
    attempt_index: int,
) -> SemanticVerificationAttempt:
    """Certify one existing compiled attempt without recompilation."""

    logical_state = _simulate(logical_program)
    compiled_state = _simulate(compiled_program)
    target, logical_purity = _logical_target(logical_state, output_qubits)
    evidence = _best_within_register(target, compiled_state, output_qubits)
    if (
        evidence["fidelity"] < 1.0 - SEMANTIC_TOLERANCE
        and method is PreparationMethod.DS_QUANTUM_STATE_PREPARATION
    ):
        evidence = _best_ordered_output_subset(
            target, compiled_state, len(output_qubits)
        )
    if logical_purity < 1.0 - SEMANTIC_TOLERANCE:
        status = "uncertified"
        reason = "logical_output_purity_below_tolerance"
    elif evidence["fidelity"] >= 1.0 - SEMANTIC_TOLERANCE:
        status = "certified_pass"
        reason = "constrained_semantic_fidelity_passed"
    else:
        status = "certified_fail"
        reason = "constrained_semantic_fidelity_below_tolerance"
    diagnostics = {
        "logical_output_purity": logical_purity,
        "mapping_ties": evidence["mapping_ties"],
        "mappings_tested": evidence["mappings_tested"],
        "pure_subsets": evidence["pure_subsets"],
        "reason": reason,
        "provenance": "migrated_frozen_constrained_mapping_v1",
    }
    return SemanticVerificationAttempt(
        attempt_index=attempt_index,
        status=status,
        fidelity=float(evidence["fidelity"]),
        mapping_method=str(evidence["mapping_method"]),
        output_qubits=output_qubits,
        mapping=tuple(int(qubit) for qubit in evidence["mapping"]),
        ancilla_treatment="reduced_density_partial_trace",
        diagnostics=diagnostics,
    )


def standard_verification() -> SemanticVerification:
    return SemanticVerification(
        level=VerificationLevel.STANDARD,
        status="not_run_by_standard",
        recommendation_valid=True,
        minimum_fidelity=None,
        technical_repetitions=0,
        attempts=(),
    )


def audit_compiled_attempts(
    *,
    logical_program: QProg,
    attempts: tuple[CompilationAttempt, ...],
    output_qubits: tuple[int, ...],
    method: PreparationMethod,
    selected_candidate_id: str,
) -> SemanticVerification:
    if len(attempts) != TECHNICAL_REPETITIONS:
        raise InternalInvariantError(code="audit_requires_exactly_five_attempts")
    if tuple(item.attempt_index for item in attempts) != tuple(range(5)):
        raise InternalInvariantError(code="audit_attempt_index_mismatch")

    records: list[SemanticVerificationAttempt] = []
    for attempt in attempts:
        if not attempt.success or attempt.compiled_program is None:
            records.append(
                SemanticVerificationAttempt(
                    attempt_index=attempt.attempt_index,
                    status="compile_unavailable",
                    fidelity=None,
                    mapping_method=None,
                    output_qubits=output_qubits,
                    exception_type=attempt.exception_type,
                    exception_message=attempt.exception_message,
                    diagnostics={"compiler_status": attempt.status},
                )
            )
            continue
        try:
            records.append(
                certify_compiled_attempt(
                    logical_program,
                    attempt.compiled_program,
                    output_qubits=output_qubits,
                    method=method,
                    attempt_index=attempt.attempt_index,
                )
            )
        except InternalInvariantError:
            raise
        except Exception as error:
            records.append(
                SemanticVerificationAttempt(
                    attempt_index=attempt.attempt_index,
                    status="uncertified",
                    fidelity=None,
                    mapping_method=None,
                    output_qubits=output_qubits,
                    exception_type=type(error).__name__,
                    exception_message=str(error),
                    diagnostics={"reason": "semantic_verifier_exception"},
                )
            )

    statuses = {record.status for record in records}
    if statuses == {"certified_pass"}:
        overall = "certified_pass"
    elif "certified_fail" in statuses:
        overall = "certified_fail"
    else:
        overall = "uncertified"
    fidelities = [record.fidelity for record in records if record.fidelity is not None]
    return SemanticVerification(
        level=VerificationLevel.AUDIT,
        status=overall,
        recommendation_valid=overall == "certified_pass",
        minimum_fidelity=min(fidelities) if fidelities else None,
        technical_repetitions=TECHNICAL_REPETITIONS,
        selected_candidate_id=selected_candidate_id,
        attempts=tuple(records),
    )


def verify_resource_selection(
    coefficients: np.ndarray,
    *,
    basis: str,
    run: ResourceSelectionRun,
    level: VerificationLevel,
) -> SemanticVerification:
    """Verify the already selected candidate using its existing five attempts."""

    if level is VerificationLevel.STANDARD:
        return standard_verification()
    if level is not VerificationLevel.AUDIT:
        raise InternalInvariantError(code="unsupported_verification_level")

    if run.selection.decision.value == "compress":
        candidate = next(
            (
                item for item in run.grid.candidates
                if item.candidate_id == run.selection.selected_candidate_id
            ),
            None,
        )
        if candidate is None:
            raise InternalInvariantError(code="selected_candidate_missing")
        selected_coefficients = top_k_coefficients(
            coefficients, candidate.k, normalize=True
        )
    else:
        candidate = run.grid.dense_full
        selected_coefficients = np.asarray(coefficients)

    audit = candidate.resource_audit
    if audit is None:
        raise InternalInvariantError(code="verification_resource_audit_missing")
    attempts = tuple(audit.compilation_attempts)
    prepared = adapt_preparation_input(selected_coefficients)
    build = build_preparation(candidate.method, prepared)
    logical = compose_end_to_end_program(build, basis=basis)
    return audit_compiled_attempts(
        logical_program=logical.program,
        attempts=attempts,
        output_qubits=logical.output_qubits,
        method=candidate.method,
        selected_candidate_id=candidate.candidate_id,
    )


__all__ = [
    "SEMANTIC_TOLERANCE",
    "certify_compiled_attempt",
    "standard_verification",
    "audit_compiled_attempts",
    "verify_resource_selection",
]
