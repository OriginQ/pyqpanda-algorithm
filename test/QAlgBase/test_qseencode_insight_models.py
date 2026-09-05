from dataclasses import FrozenInstanceError, fields

import pytest

from pyqpanda_alg.QSEncode import (
    AttributionReport,
    CandidateResult,
    CapabilityReport,
    ErrorBudgetResult,
    EvidenceScope,
    EvidenceScopeStatus,
    InputSummary,
    InsightResult,
    PreparationArtifact,
    ResourceAudit,
    SelectionResult,
    SemanticVerification,
    SemanticVerificationAttempt,
    TransformDiagnostics,
)


MODEL_TYPES = (
    InputSummary,
    TransformDiagnostics,
    ErrorBudgetResult,
    CapabilityReport,
    CandidateResult,
    SelectionResult,
    ResourceAudit,
    SemanticVerification,
    SemanticVerificationAttempt,
    AttributionReport,
    EvidenceScope,
    InsightResult,
    PreparationArtifact,
)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_all_schema_models_use_frozen_slotted_dataclasses(model_type):
    assert model_type.__dataclass_params__.frozen is True
    assert "__slots__" in model_type.__dict__


def test_input_summary_is_frozen_slotted_and_has_required_hash_fields():
    names = {field.name for field in fields(InputSummary)}
    assert {
        "original_length",
        "padded_length",
        "original_sum",
        "normalized",
        "padding_count",
        "original_input_sha256",
        "effective_probability_sha256",
    } <= names
    summary = InputSummary(
        original_length=2,
        padded_length=2,
        original_sum=1.0,
        normalized=False,
        padding_count=0,
        original_input_sha256="a" * 64,
        effective_probability_sha256="b" * 64,
    )
    assert not hasattr(summary, "__dict__")
    with pytest.raises(FrozenInstanceError):
        summary.original_length = 3


def test_insight_result_schema_version_and_config_provenance_are_contract_fields():
    names = {field.name for field in fields(InsightResult)}
    assert "schema_version" in names
    assert "analysis_config_fingerprint" in names
    assert "analysis_config" in names
    assert InsightResult.SCHEMA_VERSION == "qseencode-insight-v1"


def test_evidence_scope_uses_stable_status_enum():
    scope = EvidenceScope(
        status=EvidenceScopeStatus.OUTSIDE_VALIDATED_SCOPE,
        reasons=("phase1_contract_only",),
        reference=None,
    )
    assert scope.status.value == "outside_validated_scope"
    assert scope.reasons == ("phase1_contract_only",)


def test_phase3_capability_report_reserves_explanatory_and_register_metadata():
    names = {field.name for field in fields(CapabilityReport)}
    assert {
        "method",
        "compatible",
        "reason_code",
        "reason",
        "required_qubits",
        "ancillas",
        "input_constraints",
        "observed_output_qubits",
        "exception_type",
        "exception_message",
        "diagnostics",
        "failure_stage",
        "logical_fidelity",
    } <= names
