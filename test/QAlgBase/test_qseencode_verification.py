import numpy as np
import pytest
from pyqpanda3.core import H, QProg

from pyqpanda_alg.QSEncode import PreparationMethod, VerificationLevel
from pyqpanda_alg.QSEncode._compiler import CompilationAttempt
from pyqpanda_alg.QSEncode._verification import (
    SEMANTIC_TOLERANCE,
    audit_compiled_attempts,
    certify_compiled_attempt,
    verify_resource_selection,
)


def _program_with_h(qubit_count=2):
    program = QProg(qubit_count)
    program << H(qubit_count - 1)
    return program


def _attempt(index, program, *, success=True):
    return CompilationAttempt(
        attempt_index=index,
        success=success,
        status="success" if success else "compile_failure",
        compiled_program=program if success else None,
        compiled_originir="originir" if success else None,
        originir_sha256=f"hash-{index}" if success else None,
        compiled_depth=1 if success else None,
        compiled_total_gates=1 if success else None,
        compiled_one_qubit_gates=1 if success else None,
        compiled_two_qubit_gates=0 if success else None,
        compiled_cnot_gates=0 if success else None,
        required_qubits=1,
        allocated_qubits=2,
        ancilla_count=0,
        compiler_profile_fingerprint="profile",
        compiler_profile={},
    )


def test_compiled_semantic_certification_is_phase_sensitive_and_global_phase_safe():
    logical = _program_with_h()
    compiled = _program_with_h()

    evidence = certify_compiled_attempt(
        logical,
        compiled,
        output_qubits=(1,),
        method=PreparationMethod.AMPLITUDE_ENCODE,
        attempt_index=0,
    )

    assert evidence.status == "certified_pass"
    assert evidence.fidelity >= 1.0 - SEMANTIC_TOLERANCE
    assert evidence.mapping_method == "output_register_constrained_permutation"


def test_standard_mode_does_not_call_compiled_semantic_verifier(monkeypatch):
    import pyqpanda_alg.QSEncode._verification as module

    monkeypatch.setattr(
        module,
        "certify_compiled_attempt",
        lambda *args, **kwargs: pytest.fail("standard must not run semantic sweep"),
    )
    result = module.standard_verification()

    assert result.level is VerificationLevel.STANDARD
    assert result.status == "not_run_by_standard"
    assert result.recommendation_valid is True
    assert result.technical_repetitions == 0
    assert result.attempts == ()


def test_audit_uses_exactly_existing_five_attempts_and_never_compiles(monkeypatch):
    import pyqpanda_alg.QSEncode._verification as module

    calls = []
    monkeypatch.setattr(
        module,
        "certify_compiled_attempt",
        lambda *args, attempt_index, **kwargs: calls.append(attempt_index)
        or module.SemanticVerificationAttempt(
            attempt_index=attempt_index,
            status="certified_pass",
            fidelity=1.0,
            mapping_method="synthetic",
            output_qubits=(1,),
            mapping=(1,),
            ancilla_treatment="partial_trace",
        ),
    )
    program = _program_with_h()
    attempts = tuple(_attempt(index, program) for index in range(5))
    verification = module.audit_compiled_attempts(
        logical_program=program,
        attempts=attempts,
        output_qubits=(1,),
        method=PreparationMethod.AMPLITUDE_ENCODE,
        selected_candidate_id="winner",
    )

    assert calls == [0, 1, 2, 3, 4]
    assert verification.status == "certified_pass"
    assert verification.recommendation_valid is True
    assert verification.technical_repetitions == 5


def test_any_audit_failure_invalidates_recommendation_without_reselection(monkeypatch):
    import pyqpanda_alg.QSEncode._verification as module

    program = _program_with_h()
    attempts = tuple(_attempt(index, program) for index in range(5))

    def synthetic(*args, attempt_index, **kwargs):
        return module.SemanticVerificationAttempt(
            attempt_index=attempt_index,
            status="certified_fail" if attempt_index == 2 else "certified_pass",
            fidelity=0.5 if attempt_index == 2 else 1.0,
            mapping_method="synthetic",
            output_qubits=(1,),
            mapping=(1,),
            ancilla_treatment="partial_trace",
        )

    monkeypatch.setattr(module, "certify_compiled_attempt", synthetic)
    verification = module.audit_compiled_attempts(
        logical_program=program,
        attempts=attempts,
        output_qubits=(1,),
        method=PreparationMethod.SPARSE_ISOMETRY,
        selected_candidate_id="compressed__k4__sparse_isometry",
    )

    assert verification.status == "certified_fail"
    assert verification.recommendation_valid is False
    assert verification.selected_candidate_id == "compressed__k4__sparse_isometry"
    assert len(verification.attempts) == 5


def test_compile_failure_is_retained_as_uncertified_attempt():
    program = _program_with_h()
    attempts = tuple(
        _attempt(index, program, success=index != 3) for index in range(5)
    )
    verification = audit_compiled_attempts(
        logical_program=program,
        attempts=attempts,
        output_qubits=(1,),
        method=PreparationMethod.AMPLITUDE_ENCODE,
        selected_candidate_id="baseline",
    )

    assert verification.status == "uncertified"
    assert verification.recommendation_valid is False
    assert verification.attempts[3].status == "compile_unavailable"


def test_internal_invariant_error_propagates(monkeypatch):
    import pyqpanda_alg.QSEncode._verification as module
    from pyqpanda_alg.QSEncode import InternalInvariantError

    program = _program_with_h()
    attempts = tuple(_attempt(index, program) for index in range(5))

    def fail(*args, **kwargs):
        raise InternalInvariantError(code="synthetic_internal_failure")

    monkeypatch.setattr(module, "certify_compiled_attempt", fail)
    with pytest.raises(InternalInvariantError) as error:
        module.audit_compiled_attempts(
            logical_program=program,
            attempts=attempts,
            output_qubits=(1,),
            method=PreparationMethod.AMPLITUDE_ENCODE,
            selected_candidate_id="baseline",
        )
    assert error.value.code == "synthetic_internal_failure"


def test_ds_uses_ordered_output_subset_with_ancilla_trace_when_needed():
    logical = QProg(4)
    logical << H(2)
    compiled = QProg(4)
    compiled << H(0)

    evidence = certify_compiled_attempt(
        logical,
        compiled,
        output_qubits=(2, 3),
        method=PreparationMethod.DS_QUANTUM_STATE_PREPARATION,
        attempt_index=0,
    )

    assert evidence.status == "certified_pass"
    assert evidence.mapping_method == "ordered_output_subset_with_ancilla_trace"
    assert evidence.fidelity >= 1.0 - SEMANTIC_TOLERANCE


def test_frozen_small_cases_reuse_resource_attempts_for_audit():
    from pyqpanda_alg.QSEncode import SelectionDecision
    from pyqpanda_alg.QSEncode._error_budget import find_k_star
    from pyqpanda_alg.QSEncode._selection import run_resource_selection
    from pyqpanda_alg.QSEncode._transforms import normalized_fourier, normalized_fwht

    probabilities = np.array([
        0.0006917643261373052, 0.015724004731018214,
        0.1261730210273901, 0.3574112099154543,
        0.3574112099154544, 0.1261730210273902,
        0.01572400473101823, 0.0006917643261373052,
    ])
    cases = (
        ("walsh", normalized_fwht(np.sqrt(probabilities)), SelectionDecision.DO_NOT_COMPRESS),
        ("fourier", normalized_fourier(np.sqrt(probabilities)), SelectionDecision.COMPRESS),
    )
    for basis, coefficients, expected_decision in cases:
        budget = find_k_star(coefficients, 0.99)
        run = run_resource_selection(
            coefficients, basis=basis, error_budget=budget
        )
        selection_before = run.selection
        verification = verify_resource_selection(
            coefficients,
            basis=basis,
            run=run,
            level=VerificationLevel.AUDIT,
        )

        assert run.selection is selection_before
        assert run.selection.decision is expected_decision
        assert verification.status == "certified_pass"
        assert verification.recommendation_valid is True
        assert len(verification.attempts) == 5
        assert verification.minimum_fidelity >= 1.0 - SEMANTIC_TOLERANCE
