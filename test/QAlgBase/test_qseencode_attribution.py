from pyqpanda_alg.QSEncode import CandidateResult, PreparationMethod, ResourceAudit
from pyqpanda_alg.QSEncode._selection import compute_attribution


def _candidate(name, method, k, twoq, depth):
    audit = ResourceAudit(
        compiled_two_qubit_gates=float(twoq), compiled_depth=float(depth),
        compiled_total_gates=float(twoq + depth), required_qubits=2,
        allocated_qubits=4, repetitions=5, valid=True, status="valid",
    )
    return CandidateResult(name, method, k, "eligible", resource_audit=audit, eligible=True)


def test_attribution_absolute_identity_is_exact_for_both_resources():
    baseline = _candidate("full", PreparationMethod.AMPLITUDE_ENCODE, 8, 100, 200)
    dense = _candidate("dense-k", PreparationMethod.AMPLITUDE_ENCODE, 4, 70, 140)
    selected = _candidate("sparse-k", PreparationMethod.SPARSE_ISOMETRY, 4, 40, 90)
    result = compute_attribution(baseline, dense, selected)

    assert result.total_two_qubit_difference == 60
    assert result.truncation_two_qubit_difference == 30
    assert result.preparation_two_qubit_difference == 30
    assert result.total_depth_difference == 110
    assert result.truncation_depth_difference == 60
    assert result.preparation_depth_difference == 50
    assert result.two_qubit_identity_error == 0
    assert result.depth_identity_error == 0


def test_amplitude_winner_has_exact_zero_preparation_contribution():
    baseline = _candidate("full", PreparationMethod.AMPLITUDE_ENCODE, 8, 100, 200)
    selected = _candidate("dense-k", PreparationMethod.AMPLITUDE_ENCODE, 4, 70, 140)
    result = compute_attribution(baseline, selected, selected)
    assert result.preparation_two_qubit_difference == 0
    assert result.preparation_depth_difference == 0
    assert result.two_qubit_identity_error == 0
    assert result.depth_identity_error == 0
