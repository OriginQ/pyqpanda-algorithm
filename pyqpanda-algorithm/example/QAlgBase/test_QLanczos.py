import numpy as np
import pytest

from pyqpanda_alg.QLanczos import (
    QuantumLanczos,
    dense_hamiltonian_element,
    imaginary_time_evolver,
    run_quantum_lanczos,
    solve_generalized_hermitian,
    statevector_overlap,
)


def test_three_dimensional_krylov_space_recovers_dense_spectrum():
    hamiltonian = np.asarray(
        [[-1.1, 0.2, 0.0], [0.2, 0.3, -0.15], [0.0, -0.15, 1.2]],
        dtype=np.complex128,
    )
    seed = np.asarray([1.0, 0.8, -0.3], dtype=np.complex128)
    seed /= np.linalg.norm(seed)

    result = QuantumLanczos(
        seed,
        imaginary_time_evolver(hamiltonian, 0.4),
        statevector_overlap,
        dense_hamiltonian_element(hamiltonian),
        order=3,
        num_roots=3,
        overlap_cutoff=1e-12,
    ).run()

    np.testing.assert_allclose(
        result.energies,
        np.linalg.eigvalsh(hamiltonian),
        atol=1e-9,
    )
    assert result.overlap_rank == 3
    assert np.max(result.residual_norms) < 1e-9


def test_rank_revealing_solver_discards_duplicate_basis_direction():
    basis = (
        np.asarray([1.0, 0.0], dtype=np.complex128),
        np.asarray([1.0, 0.0], dtype=np.complex128),
        np.asarray([0.0, 1.0], dtype=np.complex128),
    )
    hamiltonian = np.diag([-0.75, 0.5]).astype(np.complex128)

    result = run_quantum_lanczos(
        basis,
        statevector_overlap,
        dense_hamiltonian_element(hamiltonian),
        num_roots=2,
    )

    np.testing.assert_allclose(result.energies, [-0.75, 0.5], atol=1e-12)
    assert result.overlap_rank == 2
    assert result.effective_rank == 2
    assert result.discarded_directions == 1


def test_regularization_can_stabilize_a_singular_overlap_matrix():
    hamiltonian = np.diag([0.0, 1.0]).astype(np.complex128)
    overlap = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)

    result = solve_generalized_hermitian(
        hamiltonian,
        overlap,
        num_roots=2,
        regularization=1e-5,
    )

    assert result.overlap_rank == 1
    assert result.effective_rank == 2
    assert np.all(np.isfinite(result.energies))
    assert np.all(np.isfinite(result.residual_norms))


def test_rejects_non_hermitian_hamiltonian():
    hamiltonian = np.asarray(
        [[0.0, 1.0], [0.0, 1.0]], dtype=np.complex128
    )
    overlap = np.eye(2, dtype=np.complex128)

    with pytest.raises(ValueError, match="Hermitian"):
        solve_generalized_hermitian(hamiltonian, overlap)
