import numpy as np
import pytest

from pyqpanda_alg.QSEncode import PreparationMethod
from pyqpanda_alg.QSEncode._capability import assess_capability
from pyqpanda_alg.QSEncode._error_budget import top_k_coefficients
from pyqpanda_alg.QSEncode._preparation import adapt_preparation_input
from pyqpanda_alg.QSEncode._resources import audit_build_resources
from pyqpanda_alg.QSEncode._transforms import normalized_fwht


PROBABILITIES = np.array(
    [
        0.0006917643261373052,
        0.015724004731018214,
        0.1261730210273901,
        0.3574112099154543,
        0.3574112099154544,
        0.1261730210273902,
        0.01572400473101823,
        0.0006917643261373052,
    ]
)


def _audit(coefficients, method, basis="walsh"):
    assessment = assess_capability(method, adapt_preparation_input(coefficients))
    assert assessment.report.compatible and assessment.build is not None
    return audit_build_resources(assessment.build, basis=basis)


def test_dense_full_dense_compressed_and_compatible_methods_are_auditable():
    coefficients = normalized_fwht(np.sqrt(PROBABILITIES))
    compressed = top_k_coefficients(coefficients, 4, normalize=True)

    dense_full = _audit(coefficients, PreparationMethod.AMPLITUDE_ENCODE)
    dense_compressed = _audit(compressed, PreparationMethod.AMPLITUDE_ENCODE)
    sparse = _audit(compressed, PreparationMethod.SPARSE_ISOMETRY)
    ds = _audit(compressed, PreparationMethod.DS_QUANTUM_STATE_PREPARATION)

    assert all(audit.valid for audit in (dense_full, dense_compressed, sparse, ds))
    assert all(audit.repetitions == 5 for audit in (dense_full, dense_compressed, sparse, ds))
    assert all(audit.allocated_qubits == 6 for audit in (dense_full, dense_compressed, sparse, ds))
    assert [dense_compressed.required_qubits, sparse.required_qubits, ds.required_qubits] == [3, 3, 6]
    assert [len(dense_compressed.ancillas), len(sparse.ancillas), len(ds.ancillas)] == [0, 0, 3]


def test_frozen_gaussian_n8_walsh_resource_direction_and_scale():
    """Frozen provenance: Probe 4 SHA F28C... helper, gaussian/N=8/k=4."""

    coefficients = normalized_fwht(np.sqrt(PROBABILITIES))
    compressed = top_k_coefficients(coefficients, 4, normalize=True)
    medians = {
        method: _audit(compressed, method).compiled_two_qubit_gates
        for method in PreparationMethod
    }

    assert medians[PreparationMethod.DS_QUANTUM_STATE_PREPARATION] > medians[PreparationMethod.AMPLITUDE_ENCODE]
    assert medians[PreparationMethod.DS_QUANTUM_STATE_PREPARATION] > medians[PreparationMethod.SPARSE_ISOMETRY]
    historical = {
        PreparationMethod.AMPLITUDE_ENCODE: 21.0,
        PreparationMethod.SPARSE_ISOMETRY: 22.0,
        PreparationMethod.DS_QUANTUM_STATE_PREPARATION: 116.0,
    }
    for method, value in medians.items():
        assert 0.5 <= value / historical[method] <= 2.0


def test_valid_resource_audit_contains_medians_ranges_and_width_metrics():
    audit = _audit([1.0, 0.0, 0.0, 0.0], PreparationMethod.AMPLITUDE_ENCODE)

    assert audit.valid and audit.status == "valid"
    assert audit.successful_attempts == 5 and audit.failed_attempts == 0
    assert audit.two_qubit_range[0] <= audit.compiled_two_qubit_gates <= audit.two_qubit_range[1]
    assert audit.depth_range[0] <= audit.compiled_depth <= audit.depth_range[1]
    assert audit.q_required_times_depth == audit.required_qubits * audit.compiled_depth
    assert audit.q_allocated_times_depth == audit.allocated_qubits * audit.compiled_depth
    assert audit.compiled_two_qubit_gates == audit.compiled_cnot_gates
