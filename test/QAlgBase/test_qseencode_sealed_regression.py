"""Small sealed parity suite for the contest submission.

The fixture is the preregistered Gaussian N=8 case.  It does not read or run
the 480-cell Locked benchmark.
"""

import numpy as np
import pytest

from pyqpanda_alg.QSEncode import (
    EvidenceScopeStatus,
    PreparationMethod,
    QSEncodeInsight,
    ResultBindingError,
    SelectionDecision,
)
from pyqpanda_alg.QSEncode._capability import assess_capability
from pyqpanda_alg.QSEncode._preparation import adapt_preparation_input


PROBABILITIES = np.array([
    0.0006917643261373052,
    0.015724004731018214,
    0.1261730210273901,
    0.3574112099154543,
    0.3574112099154544,
    0.1261730210273902,
    0.01572400473101823,
    0.0006917643261373052,
])


@pytest.fixture(scope="module")
def sealed_standard_results():
    return {
        basis: QSEncodeInsight(basis=basis).analyze(PROBABILITIES)
        for basis in ("walsh", "fourier")
    }


def test_sealed_walsh_refuses_compression_and_prepares_dense_baseline(
    sealed_standard_results,
):
    result = sealed_standard_results["walsh"]
    baseline = result.selection.baseline_resource

    assert result.error_budget.k_star == 4
    assert result.selection.decision is SelectionDecision.DO_NOT_COMPRESS
    assert baseline.compiled_two_qubit_gates == 21.0
    assert baseline.compiled_depth == 37.0
    assert result.evidence_scope.status is EvidenceScopeStatus.VALIDATED_DEFAULT

    artifact = QSEncodeInsight(basis="walsh").prepare(
        PROBABILITIES, result=result
    )
    assert artifact.selected_candidate_id == "dense_full__amplitude_encode"
    assert artifact.k is None


def test_sealed_fourier_winner_resources_and_attribution(sealed_standard_results):
    result = sealed_standard_results["fourier"]
    baseline = result.selection.baseline_resource
    dense = next(
        candidate
        for candidate in result.candidates
        if candidate.candidate_id == "compressed__k4__amplitude_encode"
    )
    winner = next(
        candidate
        for candidate in result.candidates
        if candidate.candidate_id == "compressed__k4__sparse_isometry"
    )

    assert result.error_budget.k_star == 4
    assert result.selection.decision is SelectionDecision.COMPRESS
    assert result.selection.selected_candidate_id == winner.candidate_id
    assert baseline.compiled_two_qubit_gates == 114.0
    assert abs(baseline.compiled_depth - 205.0) <= 1.0
    assert dense.resource_audit.compiled_two_qubit_gates == 73.0
    assert abs(dense.resource_audit.compiled_depth - 123.0) <= 1.0
    assert winner.resource_audit.compiled_two_qubit_gates == 37.0
    # PyQPanda3 transpilation is technically non-deterministic.  The frozen
    # five-repeat median is 62, while the same validated environment can emit
    # 61 without changing the selected method or the resource direction.
    assert abs(winner.resource_audit.compiled_depth - 62.0) <= 1.0
    attribution = result.attribution
    assert (
        attribution.total_two_qubit_difference,
        attribution.truncation_two_qubit_difference,
        attribution.preparation_two_qubit_difference,
    ) == (77.0, 41.0, 36.0)
    assert attribution.total_depth_difference == (
        baseline.compiled_depth - winner.resource_audit.compiled_depth
    )
    assert attribution.truncation_depth_difference == (
        baseline.compiled_depth - dense.resource_audit.compiled_depth
    )
    assert attribution.preparation_depth_difference == (
        dense.resource_audit.compiled_depth - winner.resource_audit.compiled_depth
    )
    assert attribution.two_qubit_identity_error == 0.0
    assert attribution.depth_identity_error == 0.0


def test_sealed_ds_preparation_register_semantics():
    compatible_state = np.array(
        [1 / np.sqrt(3), 0, 0, 1j / np.sqrt(3), 0, -1 / np.sqrt(3), 0, 0]
    )
    prepared = adapt_preparation_input(compatible_state)
    assessment = assess_capability(
        PreparationMethod.DS_QUANTUM_STATE_PREPARATION,
        prepared,
        available_qubits=6,
    )

    assert assessment.report.required_qubits == 6
    assert assessment.report.compatible
    assert assessment.build.output_qubits == (3, 4, 5)
    assert assessment.build.ancillas == (0, 1, 2)
    assert assessment.report.logical_fidelity >= 1.0 - 1e-10


@pytest.mark.parametrize("basis", ["walsh", "fourier"])
def test_sealed_audit_is_exactly_five_of_five(basis):
    result = QSEncodeInsight(basis=basis, verification="audit").analyze(
        PROBABILITIES
    )

    assert result.semantic_verification.status == "certified_pass"
    assert result.semantic_verification.recommendation_valid is True
    assert result.semantic_verification.technical_repetitions == 5
    assert len(result.semantic_verification.attempts) == 5
    assert {attempt.attempt_index for attempt in result.semantic_verification.attempts} == set(range(5))
    assert result.semantic_verification.minimum_fidelity >= 1.0 - 1e-10


def test_sealed_binding_and_evidence_scope(sealed_standard_results):
    result = sealed_standard_results["walsh"]
    with pytest.raises(ResultBindingError) as error:
        QSEncodeInsight(basis="walsh").prepare(
            np.roll(PROBABILITIES, 1), result=result
        )
    assert error.value.code == "input_mismatch"

    outside = QSEncodeInsight(
        basis="walsh", fidelity_target=0.98
    ).analyze(PROBABILITIES)
    assert outside.evidence_scope.status is EvidenceScopeStatus.OUTSIDE_VALIDATED_SCOPE
