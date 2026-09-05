import json
from dataclasses import replace

import numpy as np
import pytest

from pyqpanda_alg.QSEncode import (
    EvidenceScopeStatus,
    QSEncodeInsight,
    ResultBindingError,
    SelectionDecision,
    SemanticVerification,
    UncertifiedSelectionError,
    VerificationLevel,
)


PROBABILITIES = np.array([
    0.0006917643261373052, 0.015724004731018214,
    0.1261730210273901, 0.3574112099154543,
    0.3574112099154544, 0.1261730210273902,
    0.01572400473101823, 0.0006917643261373052,
])


@pytest.fixture(scope="module")
def walsh_result():
    return QSEncodeInsight(basis="walsh").analyze(PROBABILITIES)


@pytest.fixture(scope="module")
def fourier_result():
    return QSEncodeInsight(basis="fourier").analyze(PROBABILITIES)


def test_walsh_public_example_is_complete_dnc_and_prepares_baseline(walsh_result):
    engine = QSEncodeInsight(basis="walsh")

    assert walsh_result.selection.decision is SelectionDecision.DO_NOT_COMPRESS
    assert walsh_result.transform is not None
    assert walsh_result.error_budget is not None
    assert walsh_result.capabilities
    assert walsh_result.candidates
    assert walsh_result.semantic_verification.status == "not_run_by_standard"
    assert walsh_result.evidence_scope.status is EvidenceScopeStatus.VALIDATED_DEFAULT
    artifact = engine.prepare(PROBABILITIES, result=walsh_result)

    assert artifact.selected_candidate_id == "dense_full__amplitude_encode"
    assert artifact.decision is SelectionDecision.DO_NOT_COMPRESS
    assert artifact.k is None
    assert artifact.verification_status == "standard_validated"


def test_fourier_public_example_compresses_and_prepares_sparse_winner(fourier_result):
    engine = QSEncodeInsight(basis="fourier")

    assert fourier_result.selection.decision is SelectionDecision.COMPRESS
    assert fourier_result.selection.selected_candidate_id == "compressed__k4__sparse_isometry"
    assert fourier_result.evidence_scope.status is EvidenceScopeStatus.VALIDATED_DEFAULT
    artifact = engine.prepare(PROBABILITIES, result=fourier_result)

    assert artifact.selected_candidate_id == "compressed__k4__sparse_isometry"
    assert artifact.decision is SelectionDecision.COMPRESS
    assert artifact.k == 4
    assert artifact.output_qubits == (3, 4, 5)


def test_nondefault_target_runs_but_is_outside_validated_scope():
    result = QSEncodeInsight(
        basis="walsh", fidelity_target=0.98
    ).analyze(PROBABILITIES)

    assert result.evidence_scope.status is EvidenceScopeStatus.OUTSIDE_VALIDATED_SCOPE
    assert "fidelity_target_not_0.99" in result.evidence_scope.reasons


def test_prepare_binding_rejects_input_before_program_build(walsh_result, monkeypatch):
    import pyqpanda_alg.QSEncode.insight as module

    monkeypatch.setattr(
        module,
        "build_preparation",
        lambda *args, **kwargs: pytest.fail("binding must precede program build"),
    )
    with pytest.raises(ResultBindingError) as error:
        QSEncodeInsight(basis="walsh").prepare(
            np.roll(PROBABILITIES, 1), result=walsh_result
        )
    assert error.value.code == "input_mismatch"


def test_prepare_binding_rejects_configuration_mismatch(walsh_result):
    with pytest.raises(ResultBindingError) as error:
        QSEncodeInsight(basis="fourier").prepare(
            PROBABILITIES, result=walsh_result
        )
    assert error.value.code == "configuration_mismatch"


def test_audit_pass_has_exactly_five_certifications_and_prepares_winner():
    engine = QSEncodeInsight(basis="fourier", verification="audit")
    result = engine.analyze(PROBABILITIES)

    assert result.semantic_verification.status == "certified_pass"
    assert result.semantic_verification.recommendation_valid is True
    assert len(result.semantic_verification.attempts) == 5
    artifact = engine.prepare(PROBABILITIES, result=result)
    assert artifact.verification_status == "audit_certified_5_of_5"


def test_audit_failure_blocks_prepare_and_explicit_fallback_returns_baseline(monkeypatch):
    import pyqpanda_alg.QSEncode.insight as module

    original = module.verify_resource_selection

    def invalidate(*args, **kwargs):
        verified = original(*args, **kwargs)
        return SemanticVerification(
            level=VerificationLevel.AUDIT,
            status="certified_fail",
            recommendation_valid=False,
            minimum_fidelity=0.5,
            technical_repetitions=5,
            selected_candidate_id=verified.selected_candidate_id,
            attempts=verified.attempts,
        )

    monkeypatch.setattr(module, "verify_resource_selection", invalidate)
    engine = QSEncodeInsight(basis="fourier", verification="audit")
    result = engine.analyze(PROBABILITIES)
    selected_before = result.selection

    with pytest.raises(UncertifiedSelectionError):
        engine.prepare(PROBABILITIES, result=result)
    artifact = engine.prepare(
        PROBABILITIES, result=result, fallback_to_baseline=True
    )

    assert result.selection is selected_before
    assert artifact.selected_candidate_id == "dense_full__amplitude_encode"
    assert artifact.provenance["artifact_kind"] == "fallback_from_uncertified_selection"
    assert artifact.provenance["original_selected_candidate_id"] == selected_before.selected_candidate_id


def test_result_nested_snapshots_are_immutable_and_json_stable(fourier_result):
    with pytest.raises(TypeError):
        fourier_result.analysis_config["basis"] = "walsh"
    with pytest.raises(TypeError):
        fourier_result.selection.comparison_metrics["depth_difference"] = 0
    candidate = next(item for item in fourier_result.candidates if item.resource_audit)
    with pytest.raises(TypeError):
        candidate.resource_audit.compiler_profile["topology"] = "all_to_all"

    first = fourier_result.to_json()
    second = fourier_result.to_json()
    payload = json.loads(first)
    assert first == second
    assert payload["selection"]["decision"] == "compress"
    assert payload["semantic_verification"]["status"] == "not_run_by_standard"
    assert "compiled_program" not in first
    assert "compiled_originir" not in first


def test_result_defensively_snapshots_caller_owned_dicts_and_lists(fourier_result):
    source_config = {"nested": {"items": [1, 2]}}
    source_candidates = list(fourier_result.candidates)
    snapshot = replace(
        fourier_result,
        analysis_config=source_config,
        candidates=source_candidates,
    )

    source_config["nested"]["items"].append(3)
    source_candidates.clear()

    assert snapshot.analysis_config["nested"]["items"] == (1, 2)
    assert len(snapshot.candidates) == len(fourier_result.candidates)
    with pytest.raises(TypeError):
        snapshot.analysis_config["nested"]["new"] = "value"


def test_prepare_result_none_reuses_analyze(monkeypatch):
    engine = QSEncodeInsight(basis="walsh")
    calls = []
    original = engine.analyze

    def track(probabilities):
        calls.append(1)
        return original(probabilities)

    monkeypatch.setattr(engine, "analyze", track)
    artifact = engine.prepare(PROBABILITIES)

    assert calls == [1]
    assert artifact.decision is SelectionDecision.DO_NOT_COMPRESS
