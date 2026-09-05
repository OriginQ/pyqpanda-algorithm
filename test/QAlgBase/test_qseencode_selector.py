import pytest

from pyqpanda_alg.QSEncode import (
    CandidateResult,
    PreparationMethod,
    ResourceAudit,
    SelectionDecision,
)
from pyqpanda_alg.QSEncode._selection import (
    select_best_eligible,
    select_resource_candidate,
)
from pyqpanda_alg.QSEncode.exceptions import BaselineConstructionError


def _audit(twoq, depth, required=2, ancillas=()):
    return ResourceAudit(
        compiled_two_qubit_gates=float(twoq),
        compiled_depth=float(depth),
        compiled_total_gates=float(twoq + depth),
        required_qubits=required,
        allocated_qubits=4,
        repetitions=5,
        ancillas=tuple(ancillas),
        compiled_cnot_gates=float(twoq),
        valid=True,
        status="valid",
        successful_attempts=5,
    )


def _candidate(name, method, k, twoq, depth, required=2, ancillas=(), fidelity=0.99, eligible=True):
    return CandidateResult(
        candidate_id=name,
        method=method,
        k=k,
        status="eligible" if eligible else "fidelity_ineligible",
        retained_fidelity=fidelity,
        eligible=eligible,
        eligibility_reason="eligible" if eligible else "fidelity_below_target",
        resource_audit=_audit(twoq, depth, required, ancillas),
    )


@pytest.mark.parametrize(
    ("left", "right", "winner"),
    [
        (_candidate("a", PreparationMethod.DS_QUANTUM_STATE_PREPARATION, 9, 1, 999, 8, range(4)), _candidate("b", PreparationMethod.AMPLITUDE_ENCODE, 1, 2, 1), "a"),
        (_candidate("a", PreparationMethod.DS_QUANTUM_STATE_PREPARATION, 9, 2, 3, 8, range(4)), _candidate("b", PreparationMethod.AMPLITUDE_ENCODE, 1, 2, 4), "a"),
        (_candidate("a", PreparationMethod.DS_QUANTUM_STATE_PREPARATION, 9, 2, 3, 3, range(2)), _candidate("b", PreparationMethod.AMPLITUDE_ENCODE, 1, 2, 3, 4), "a"),
        (_candidate("a", PreparationMethod.DS_QUANTUM_STATE_PREPARATION, 9, 2, 3, 4, (0,)), _candidate("b", PreparationMethod.AMPLITUDE_ENCODE, 1, 2, 3, 4, (0, 1)), "a"),
        (_candidate("a", PreparationMethod.DS_QUANTUM_STATE_PREPARATION, 2, 2, 3, 4, (0,)), _candidate("b", PreparationMethod.AMPLITUDE_ENCODE, 3, 2, 3, 4, (0,)), "a"),
        (_candidate("a", PreparationMethod.AMPLITUDE_ENCODE, 2, 2, 3, 4, (0,)), _candidate("b", PreparationMethod.SPARSE_ISOMETRY, 2, 2, 3, 4, (0,)), "a"),
    ],
    ids=["twoq", "depth", "required", "ancilla", "k", "method_order"],
)
def test_exact_six_key_lexicographic_selector(left, right, winner):
    assert select_best_eligible((right, left)).candidate_id == winner


def test_fidelity_is_eligibility_not_resource_tiebreak():
    lower_resource = _candidate("low", PreparationMethod.AMPLITUDE_ENCODE, 2, 1, 1, fidelity=0.9901)
    higher_fidelity = _candidate("high-f", PreparationMethod.SPARSE_ISOMETRY, 2, 2, 2, fidelity=1.0)
    assert select_best_eligible((higher_fidelity, lower_resource)).candidate_id == "low"


def test_ineligible_candidate_never_participates_even_with_zero_resources():
    ineligible = _candidate("bad", PreparationMethod.AMPLITUDE_ENCODE, 1, 0, 0, eligible=False)
    eligible = _candidate("good", PreparationMethod.SPARSE_ISOMETRY, 2, 5, 5)
    assert select_best_eligible((ineligible, eligible)).candidate_id == "good"


@pytest.mark.parametrize(
    ("twoq", "depth", "expected"),
    [
        (10, 20, SelectionDecision.DO_NOT_COMPRESS),
        (9, 99, SelectionDecision.COMPRESS),
        (99, 19, SelectionDecision.COMPRESS),
    ],
)
def test_dnc_requires_no_improvement_in_both_primary_resources(twoq, depth, expected):
    baseline = _candidate("dense_full", PreparationMethod.AMPLITUDE_ENCODE, 8, 10, 20)
    compressed = _candidate("compressed", PreparationMethod.SPARSE_ISOMETRY, 3, twoq, depth)
    selection = select_resource_candidate(baseline, (compressed,))
    assert selection.decision is expected
    if expected is SelectionDecision.COMPRESS:
        assert selection.selected_candidate_id == "compressed"
    else:
        assert selection.selected_candidate_id is None
        assert selection.best_compressed_candidate_id == "compressed"


def test_no_eligible_candidate_is_normal_do_not_compress_result():
    baseline = _candidate("dense_full", PreparationMethod.AMPLITUDE_ENCODE, 8, 10, 20)
    bad = _candidate("bad", PreparationMethod.SPARSE_ISOMETRY, 3, 1, 1, eligible=False)
    result = select_resource_candidate(baseline, (bad,))
    assert result.decision is SelectionDecision.DO_NOT_COMPRESS
    assert result.reason_code == "no_eligible_compressed_candidate"


def test_invalid_dense_full_baseline_is_not_disguised_as_dnc():
    baseline = _candidate("dense_full", PreparationMethod.AMPLITUDE_ENCODE, 8, 10, 20)
    baseline = CandidateResult(
        candidate_id=baseline.candidate_id,
        method=baseline.method,
        k=baseline.k,
        status="baseline_failure",
        resource_audit=ResourceAudit(
            None, None, None, 3, 6, 5, valid=False, status="compile_failure"
        ),
    )
    with pytest.raises(BaselineConstructionError) as exc_info:
        select_resource_candidate(baseline, ())
    assert exc_info.value.code == "dense_full_resource_invalid"
