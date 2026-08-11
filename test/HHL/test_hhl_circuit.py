"""Structural and exact-unitary tests for HHL circuit synthesis.

Covers the plan's verbatim structural fixture
(:func:`build_hhl_circuit` on the 2x2 identity returns a
:class:`QProg` with the expected register layout), the build metadata
(eigenvalue bounds, evolution time, reciprocal scale), and exact
unitary verification: the post-selected data-register state of small
on-grid systems (diagonal 2x2 and a representative 4x4 Hamiltonian)
must match the direction of the exact ``A^{-1} b`` solution.
"""

import numpy as np
from pyqpanda3.core import QProg

from pyqpanda_alg.HHL import HHLConfig
from pyqpanda_alg.HHL.circuit import HHLCircuitBuild, build_hhl_circuit
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def test_identity_system_builds_expected_registers():
    build = build_hhl_circuit(np.eye(2), np.array([0.6, 0.8]), HHLConfig(phase_qubits=2))
    assert isinstance(build.program, QProg)
    assert len(build.data_qubits) == 1
    assert len(build.phase_qubits) == 2
    assert build.success_qubit not in build.data_qubits


def test_build_records_evolution_time_and_reciprocal_scale():
    build = build_hhl_circuit(np.eye(2), np.array([0.6, 0.8]), HHLConfig(phase_qubits=2))
    assert build.evolution_time > 0
    assert build.reciprocal_scale > 0
    assert isinstance(build, HHLCircuitBuild)


def test_identity_two_by_two_matches_exact_solution_direction():
    matrix = np.eye(2)
    vector = np.array([0.6, 0.8])
    build = build_hhl_circuit(matrix, vector, HHLConfig(phase_qubits=2))
    post = _postselected_data_state(build)
    expected = np.linalg.solve(matrix, vector)
    _assert_same_direction(post, expected)


def test_diagonal_two_by_two_matches_exact_solution_direction():
    matrix = np.diag([1.0, 2.0])
    vector = np.array([1.0, 1.0])
    build = build_hhl_circuit(matrix, vector, HHLConfig(phase_qubits=2))
    post = _postselected_data_state(build)
    expected = np.linalg.solve(matrix, vector)
    _assert_same_direction(post, expected)


def test_four_by_four_hamiltonian_matches_exact_solution_direction():
    matrix = np.array(
        [
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ]
    )
    vector = np.array([1.0, 2.0, 3.0, 4.0])
    build = build_hhl_circuit(matrix, vector, HHLConfig(phase_qubits=3))
    post = _postselected_data_state(build)
    expected = np.linalg.solve(matrix, vector)
    _assert_same_direction(post, expected)


def _postselected_data_state(build) -> np.ndarray:
    """Run the build and return the data-register state of the success branch.

    The circuit is measurement-free, so the full statevector is
    available; the success branch is the block where the success qubit
    is |1> and the (exactly uncomputed) phase register is |0...0>.
    """
    backend = LocalBackend()
    task = backend.submit_statevector(build.program, options=ExecutionOptions())
    statevector = np.asarray(task.result().single_statevector())
    data_count = len(build.data_qubits)
    phase_count = len(build.phase_qubits)
    dimension = 1 << (data_count + phase_count + 1)
    index = np.arange(dimension)
    success = ((index >> build.success_qubit) & 1) == 1
    zero_phase = ((index >> data_count) & ((1 << phase_count) - 1)) == 0
    return statevector[success & zero_phase]


def _assert_same_direction(post, expected) -> None:
    """Assert ``post`` points along the exact solution, up to a phase."""
    post = post / np.linalg.norm(post)
    expected = expected / np.linalg.norm(expected)
    overlap = abs(np.vdot(post, expected))
    assert overlap > 1.0 - 1e-6
