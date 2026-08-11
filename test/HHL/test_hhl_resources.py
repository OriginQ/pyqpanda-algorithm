"""Resource estimation tests for the HHL solver.

Covers the deterministic qubit accounting of
:func:`estimate_hhl_resources` on a 2x2 system (the plan's verbatim
fixture), the exact controlled-evolution and tomography circuit
counts, the documented default shot request, the approximation labels
separating estimates from post-transpilation counts, padding of a 3x3
system into a two-qubit data register, and fail-fast rejection of
invalid input before any estimation.
"""

import numpy as np
import pytest

from pyqpanda_alg.HHL import HHLConfig
from pyqpanda_alg.HHL.resources import HHLResourceEstimate, estimate_hhl_resources
from pyqpanda_alg.execution import AlgorithmInputError


def test_two_by_two_resource_estimate():
    estimate = estimate_hhl_resources(
        np.diag([1.0, 2.0]),
        np.array([1.0, 1.0]),
        HHLConfig(phase_qubits=3),
    )
    assert estimate.data_qubits == 1
    assert estimate.phase_qubits == 3
    assert estimate.ancilla_qubits == 1
    assert estimate.total_qubits == 5


def test_controlled_evolutions_match_phase_register_size():
    estimate = estimate_hhl_resources(
        np.diag([1.0, 2.0]), np.array([1.0, 1.0]), HHLConfig(phase_qubits=3)
    )
    assert estimate.controlled_evolutions == 3


def test_tomography_circuits_are_pauli_bases_over_data_register():
    estimate = estimate_hhl_resources(
        np.diag([1.0, 2.0]), np.array([1.0, 1.0]), HHLConfig(phase_qubits=3)
    )
    assert estimate.tomography_circuits == 3 ** estimate.data_qubits


def test_shots_report_the_documented_execution_default():
    estimate = estimate_hhl_resources(
        np.diag([1.0, 2.0]), np.array([1.0, 1.0]), HHLConfig(phase_qubits=3)
    )
    assert estimate.shots == 1000


def test_synthesized_gates_are_labeled_approximate():
    estimate = estimate_hhl_resources(
        np.diag([1.0, 2.0]), np.array([1.0, 1.0]), HHLConfig(phase_qubits=3)
    )
    assert estimate.synthesized_gates > 0
    assert "synthesized_gates" in estimate.approximate_labels
    assert "total_qubits" not in estimate.approximate_labels


def test_three_dimensional_system_pads_data_register_to_two_qubits():
    estimate = estimate_hhl_resources(
        np.diag([1.0, 2.0, 3.0]), np.ones(3), HHLConfig(phase_qubits=2)
    )
    assert estimate.data_qubits == 2
    assert estimate.total_qubits == 5
    assert estimate.tomography_circuits == 9
    assert estimate.controlled_evolutions == 2


def test_single_row_system_leaves_no_data_qubit():
    with pytest.raises(AlgorithmInputError, match="data qubit"):
        estimate_hhl_resources(np.array([[1.0]]), np.array([1.0]), HHLConfig())


def test_invalid_system_is_rejected_before_estimation():
    matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(AlgorithmInputError, match="Hermitian"):
        estimate_hhl_resources(matrix, np.array([1.0, 0.0]), HHLConfig())


def test_resource_estimate_rejects_inconsistent_totals():
    with pytest.raises(ValueError, match="total_qubits"):
        HHLResourceEstimate(
            data_qubits=1,
            phase_qubits=3,
            ancilla_qubits=1,
            total_qubits=4,
            controlled_evolutions=3,
            synthesized_gates=35,
            tomography_circuits=3,
        )
