from dataclasses import replace

import numpy as np
import pytest

from pyqpanda_alg.QSEncode import DEFAULT_METHODS, PreparationMethod
from pyqpanda_alg.QSEncode.exceptions import InternalInvariantError
from pyqpanda_alg.QSEncode._capability import (
    CAPABILITY_REASON_CODES,
    assess_all_capabilities,
    assess_capability,
)
from pyqpanda_alg.QSEncode._preparation import (
    adapt_preparation_input,
    build_preparation,
)


def test_capability_reports_are_retained_in_default_method_order():
    prepared_input = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])
    assessments = assess_all_capabilities(prepared_input)

    assert tuple(item.report.method for item in assessments) == DEFAULT_METHODS
    assert len(assessments) == 3
    assert all(item.report.reason_code in CAPABILITY_REASON_CODES for item in assessments)


def test_mixed_compatible_and_incompatible_candidates_are_not_dropped():
    prepared_input = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])

    assessments = assess_all_capabilities(prepared_input, available_qubits=3)

    assert len(assessments) == 3
    assert [item.report.compatible for item in assessments] == [True, True, False]
    assert assessments[-1].report.method is PreparationMethod.DS_QUANTUM_STATE_PREPARATION
    assert assessments[-1].report.reason_code == "insufficient_qubits"


def test_insufficient_qubits_is_static_structured_incompatibility():
    prepared_input = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])
    assessment = assess_capability(
        PreparationMethod.DS_QUANTUM_STATE_PREPARATION,
        prepared_input,
        available_qubits=3,
    )

    assert assessment.build is None
    assert assessment.report.compatible is False
    assert assessment.report.reason_code == "insufficient_qubits"
    assert assessment.report.failure_stage == "static"
    assert assessment.report.required_qubits == 4


def test_invalid_sparse_keys_are_reported_without_top_level_exception():
    prepared_input = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])
    malformed = replace(prepared_input, sparse_items=(("2x", 1.0),))

    assessment = assess_capability(PreparationMethod.SPARSE_ISOMETRY, malformed)

    assert assessment.report.compatible is False
    assert assessment.report.reason_code == "invalid_sparse_keys"
    assert assessment.report.failure_stage == "static"
    assert assessment.build is None


def test_backend_constructor_failure_is_structured(monkeypatch):
    prepared_input = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])

    def reject(*args, **kwargs):
        raise RuntimeError("synthetic constructor rejection")

    monkeypatch.setattr("pyqpanda_alg.QSEncode._capability.build_preparation", reject)
    assessment = assess_capability(PreparationMethod.SPARSE_ISOMETRY, prepared_input)

    assert assessment.report.compatible is False
    assert assessment.report.reason_code == "constructor_rejected_input"
    assert assessment.report.failure_stage == "construction"
    assert assessment.report.exception_type == "RuntimeError"
    assert "synthetic" in assessment.report.exception_message


def test_mandatory_amplitude_baseline_failure_is_preserved_for_future_escalation(monkeypatch):
    prepared_input = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])

    def reject(*args, **kwargs):
        raise ValueError("synthetic baseline rejection")

    monkeypatch.setattr("pyqpanda_alg.QSEncode._capability.build_preparation", reject)
    assessment = assess_capability(PreparationMethod.AMPLITUDE_ENCODE, prepared_input)

    assert assessment.report.method is PreparationMethod.AMPLITUDE_ENCODE
    assert assessment.report.compatible is False
    assert assessment.report.reason_code == "constructor_rejected_input"
    assert assessment.report.exception_type == "ValueError"
    assert assessment.build is None


def test_logical_fidelity_mismatch_is_a_structured_candidate_result(monkeypatch):
    prepared_input = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        "pyqpanda_alg.QSEncode._capability._logical_state_fidelity",
        lambda *args, **kwargs: 0.5,
    )

    assessment = assess_capability(PreparationMethod.SPARSE_ISOMETRY, prepared_input)

    assert assessment.report.compatible is False
    assert assessment.report.reason_code == "logical_fidelity_mismatch"
    assert assessment.report.failure_stage == "correctness"
    assert assessment.report.logical_fidelity == 0.5
    assert assessment.build is not None


def test_internal_invariant_error_from_build_propagates_unchanged(monkeypatch):
    prepared_input = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])
    internal_failure = InternalInvariantError(code="synthetic_internal_failure")

    def fail_with_internal_invariant(*args, **kwargs):
        raise internal_failure

    monkeypatch.setattr(
        "pyqpanda_alg.QSEncode._capability.build_preparation",
        fail_with_internal_invariant,
    )

    with pytest.raises(InternalInvariantError) as exc_info:
        assess_capability(PreparationMethod.AMPLITUDE_ENCODE, prepared_input)

    assert exc_info.value is internal_failure
    assert exc_info.value.code == "synthetic_internal_failure"


def test_real_duplicate_output_metadata_is_rejected_by_capability_gate(monkeypatch):
    prepared_input = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])
    valid_build = build_preparation(
        PreparationMethod.AMPLITUDE_ENCODE, prepared_input
    )
    malformed_build = replace(
        valid_build,
        output_qubits=(0, 1, 0, 1),
        diagnostics={
            **valid_build.diagnostics,
            "backend_reported_output_qubits": (0, 1, 0, 1),
        },
    )
    monkeypatch.setattr(
        "pyqpanda_alg.QSEncode._capability.build_preparation",
        lambda *args, **kwargs: malformed_build,
    )

    assessment = assess_capability(
        PreparationMethod.AMPLITUDE_ENCODE, prepared_input
    )

    assert assessment.report.compatible is False
    assert assessment.report.reason_code == "unexpected_output_register"
    assert assessment.report.failure_stage == "construction"
