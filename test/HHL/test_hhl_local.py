"""Local-backend solving tests for the HHL object API.

Covers the plan's verbatim exact-direction fixture — solving
``diag([1, 2])`` against ``[1, 1]`` through :meth:`HHL.run` on the
local backend must reproduce the direction of the exact solution with
a small residual against the original unpadded system — plus a 4x4
tridiagonal system, a padded 3x3 system whose padding must not leak
into the truncated solution, and the default run contract (success
probability without full-vector reconstruction).
"""

import numpy as np

from pyqpanda_alg.HHL import HHL
from pyqpanda_alg.HHL.circuit import HHLCircuitBuild


def normalized(vector):
    return vector / np.linalg.norm(vector)


def test_hhl_local_matches_diagonal_solution_direction():
    matrix = np.diag([1.0, 2.0])
    vector = np.array([1.0, 1.0])
    result = HHL(matrix, vector, precision=1e-3).run(reconstruct=True)
    expected = normalized(np.linalg.solve(matrix, vector))
    assert abs(np.vdot(expected, normalized(result.classical_vector))) > 0.99
    assert result.residual < 0.05


def test_hhl_local_four_by_four_tridiagonal_matches_solution_direction():
    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 4.0],
        ]
    )
    vector = np.array([5.0, 6.0, 6.0, 5.0])
    result = HHL(matrix, vector, precision=1e-2).run(reconstruct=True)
    expected = normalized(np.linalg.solve(matrix, vector))
    assert abs(np.vdot(expected, normalized(result.classical_vector))) > 0.99
    assert result.residual < 0.05


def test_hhl_local_signed_eigenvalues_match_solution_direction():
    # A Hermitian matrix with a negative eigenvalue exercises the
    # signed-eigenvalue handling of the phase estimation and reciprocal
    # rotation; the recovered direction must still match the exact
    # solution of the original system.
    matrix = np.diag([-1.0, 2.0])
    vector = np.array([1.0, 1.0])
    result = HHL(matrix, vector, precision=1e-3).run(reconstruct=True)
    expected = normalized(np.linalg.solve(matrix, vector))
    assert abs(np.vdot(expected, normalized(result.classical_vector))) > 0.99
    assert result.residual < 0.05


def test_hhl_local_preserves_negative_maximum_magnitude_eigenvalue():
    matrix = np.diag([-1.0, 0.5])
    vector = np.array([1.0, 1.0])
    result = HHL(matrix, vector, precision=6.25e-2).run(reconstruct=True)
    expected = normalized(np.linalg.solve(matrix, vector))

    assert abs(np.vdot(expected, normalized(result.classical_vector))) > 0.99
    assert result.residual < 0.05


def test_hhl_local_padded_three_by_three_ignores_padding():
    matrix = np.diag([1.0, 2.0, 3.0])
    vector = np.array([1.0, 2.0, 3.0])
    result = HHL(matrix, vector, precision=1e-3).run(reconstruct=True)
    expected = normalized(np.linalg.solve(matrix, vector))
    assert abs(np.vdot(expected, normalized(result.classical_vector))) > 0.99
    assert result.residual < 0.05


def test_hhl_local_default_run_reports_success_probability_only():
    result = HHL(np.diag([1.0, 2.0]), np.array([1.0, 1.0])).run()
    assert result.classical_vector is None
    assert result.residual is None
    assert 0.0 < result.success_probability <= 1.0


def test_hhl_build_circuit_returns_build_with_register_metadata():
    build = HHL(np.diag([1.0, 2.0]), np.array([1.0, 1.0])).build_circuit()
    assert isinstance(build, HHLCircuitBuild)
    assert len(build.data_qubits) == 1
    assert len(build.phase_qubits) >= 1
    assert build.success_qubit not in build.data_qubits
