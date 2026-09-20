# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantum Lanczos / Krylov-subspace spectral estimation.

The implementation is backend agnostic: quantum backends only need to provide
overlap and Hamiltonian matrix-element callbacks for a sequence of Krylov
states. This keeps shot scheduling and hardware execution outside the stable
linear-algebra core.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np


MatrixElement = Callable[[Any, Any], complex]
Evolution = Callable[[Any, int], Any]


@dataclass(frozen=True)
class QuantumLanczosResult:
    """Result of a generalized Hermitian Krylov solve."""

    energies: np.ndarray
    coefficients: np.ndarray
    residual_norms: np.ndarray
    hamiltonian_matrix: np.ndarray
    overlap_matrix: np.ndarray
    effective_overlap_matrix: np.ndarray
    overlap_rank: int
    effective_rank: int
    discarded_directions: int
    condition_number: float
    basis_states: Tuple[Any, ...] = ()


def _as_complex_square(matrix: Any, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("%s must be a square matrix" % name)
    if value.shape[0] == 0:
        raise ValueError("%s must not be empty" % name)
    if not np.all(np.isfinite(value.real)) or not np.all(np.isfinite(value.imag)):
        raise ValueError("%s contains non-finite values" % name)
    return value


def _validated_hermitian(matrix: Any, name: str, atol: float) -> np.ndarray:
    value = _as_complex_square(matrix, name)
    if not np.allclose(value, value.conj().T, atol=atol, rtol=atol):
        raise ValueError("%s must be Hermitian" % name)
    return 0.5 * (value + value.conj().T)


def krylov_measurement_pairs(order: int) -> Tuple[Tuple[int, int], ...]:
    """Return the unique upper-triangle matrix elements needed for a solve."""
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    return tuple((i, j) for i in range(order) for j in range(i, order))


def build_krylov_basis(
    seed_state: Any,
    evolve: Evolution,
    order: int,
) -> Tuple[Any, ...]:
    """Build Krylov states by repeatedly applying evolve.

    evolve(state, step) receives step numbers beginning at one. The state can
    be a PyQPanda circuit descriptor, remote-job handle, state vector, or any
    user-defined object understood by the matrix-element callbacks.
    """
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    if not callable(evolve):
        raise TypeError("evolve must be callable")

    states = [seed_state]
    current = seed_state
    for step in range(1, order):
        current = evolve(current, step)
        states.append(current)
    return tuple(states)


def assemble_krylov_matrices(
    basis_states: Sequence[Any],
    overlap: MatrixElement,
    hamiltonian: MatrixElement,
) -> Tuple[np.ndarray, np.ndarray]:
    """Measure and assemble Hamiltonian and overlap matrices.

    Only unique upper-triangle callback evaluations are requested. The lower
    triangle is filled by Hermitian conjugation, halving backend work and
    enabling deterministic remote batching.
    """
    states = tuple(basis_states)
    if not states:
        raise ValueError("basis_states must not be empty")
    if not callable(overlap) or not callable(hamiltonian):
        raise TypeError("overlap and hamiltonian must be callable")

    size = len(states)
    h_matrix = np.zeros((size, size), dtype=np.complex128)
    s_matrix = np.zeros((size, size), dtype=np.complex128)

    for i, j in krylov_measurement_pairs(size):
        sij = complex(overlap(states[i], states[j]))
        hij = complex(hamiltonian(states[i], states[j]))
        if not np.isfinite(sij.real) or not np.isfinite(sij.imag):
            raise ValueError(
                "overlap callback returned a non-finite value at (%d, %d)" % (i, j)
            )
        if not np.isfinite(hij.real) or not np.isfinite(hij.imag):
            raise ValueError(
                "hamiltonian callback returned a non-finite value at (%d, %d)" % (i, j)
            )
        s_matrix[i, j] = sij
        h_matrix[i, j] = hij
        if i != j:
            s_matrix[j, i] = np.conjugate(sij)
            h_matrix[j, i] = np.conjugate(hij)

    s_matrix[np.diag_indices(size)] = np.real(np.diag(s_matrix))
    h_matrix[np.diag_indices(size)] = np.real(np.diag(h_matrix))
    return h_matrix, s_matrix


def solve_generalized_hermitian(
    hamiltonian_matrix: Any,
    overlap_matrix: Any,
    num_roots: Optional[int] = None,
    overlap_cutoff: float = 1e-10,
    regularization: float = 0.0,
    hermitian_atol: float = 1e-10,
) -> QuantumLanczosResult:
    """Solve H c = E S c with rank-revealing symmetric orthogonalization.

    Near-linear dependencies in the measured Krylov basis are removed by an
    overlap eigenvalue cutoff. regularization adds lambda I to the overlap
    matrix before the effective solve. The unregularized numerical rank is
    reported separately so callers can diagnose a collapsing basis.
    """
    if overlap_cutoff <= 0.0:
        raise ValueError("overlap_cutoff must be positive")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if hermitian_atol <= 0.0:
        raise ValueError("hermitian_atol must be positive")

    h_matrix = _validated_hermitian(
        hamiltonian_matrix, "hamiltonian_matrix", hermitian_atol
    )
    s_matrix = _validated_hermitian(
        overlap_matrix, "overlap_matrix", hermitian_atol
    )
    if h_matrix.shape != s_matrix.shape:
        raise ValueError(
            "hamiltonian_matrix and overlap_matrix must have the same shape"
        )

    raw_values = np.linalg.eigvalsh(s_matrix)
    raw_scale = max(1.0, float(np.max(np.abs(raw_values))))
    raw_threshold = max(
        overlap_cutoff * raw_scale,
        np.finfo(float).eps * h_matrix.shape[0],
    )
    if float(np.min(raw_values)) < -max(raw_threshold, hermitian_atol):
        raise ValueError("overlap_matrix must be positive semidefinite")
    raw_rank = int(np.count_nonzero(raw_values > raw_threshold))

    effective_s = s_matrix + regularization * np.eye(
        s_matrix.shape[0], dtype=np.complex128
    )
    s_values, s_vectors = np.linalg.eigh(effective_s)
    scale = max(1.0, float(np.max(np.abs(s_values))))
    threshold = max(
        overlap_cutoff * scale,
        np.finfo(float).eps * h_matrix.shape[0],
    )
    keep = s_values > threshold
    effective_rank = int(np.count_nonzero(keep))
    if effective_rank == 0:
        raise ValueError("Krylov basis has zero numerical overlap rank")

    if num_roots is None:
        roots = effective_rank
    else:
        if not isinstance(num_roots, int) or num_roots < 1:
            raise ValueError("num_roots must be a positive integer or None")
        if num_roots > effective_rank:
            raise ValueError(
                "num_roots (%d) exceeds effective Krylov rank (%d)"
                % (num_roots, effective_rank)
            )
        roots = num_roots

    kept_values = s_values[keep]
    kept_vectors = s_vectors[:, keep]
    whitener = kept_vectors / np.sqrt(kept_values)[np.newaxis, :]

    reduced_h = whitener.conj().T.dot(h_matrix).dot(whitener)
    reduced_h = 0.5 * (reduced_h + reduced_h.conj().T)
    energies, reduced_vectors = np.linalg.eigh(reduced_h)

    order = np.argsort(np.real(energies))[:roots]
    energies = np.real_if_close(energies[order]).astype(float)
    coefficients = whitener.dot(reduced_vectors[:, order])

    for column in range(coefficients.shape[1]):
        vector = coefficients[:, column]
        norm_sq = np.vdot(vector, effective_s.dot(vector))
        norm = float(np.sqrt(max(0.0, np.real(norm_sq))))
        if norm == 0.0:
            raise ValueError("generalized eigenvector has zero overlap norm")
        coefficients[:, column] = vector / norm

    residuals = np.asarray(
        [
            np.linalg.norm(
                h_matrix.dot(coefficients[:, column])
                - energies[column] * effective_s.dot(coefficients[:, column])
            )
            for column in range(coefficients.shape[1])
        ],
        dtype=float,
    )

    condition_number = float(np.max(kept_values) / np.min(kept_values))
    return QuantumLanczosResult(
        energies=energies,
        coefficients=coefficients,
        residual_norms=residuals,
        hamiltonian_matrix=h_matrix,
        overlap_matrix=s_matrix,
        effective_overlap_matrix=effective_s,
        overlap_rank=raw_rank,
        effective_rank=effective_rank,
        discarded_directions=h_matrix.shape[0] - effective_rank,
        condition_number=condition_number,
    )


def run_quantum_lanczos(
    basis_states: Sequence[Any],
    overlap: MatrixElement,
    hamiltonian: MatrixElement,
    num_roots: Optional[int] = 1,
    overlap_cutoff: float = 1e-10,
    regularization: float = 0.0,
    hermitian_atol: float = 1e-10,
) -> QuantumLanczosResult:
    """Assemble measured matrices and solve the quantum Krylov problem."""
    states = tuple(basis_states)
    h_matrix, s_matrix = assemble_krylov_matrices(states, overlap, hamiltonian)
    result = solve_generalized_hermitian(
        h_matrix,
        s_matrix,
        num_roots=num_roots,
        overlap_cutoff=overlap_cutoff,
        regularization=regularization,
        hermitian_atol=hermitian_atol,
    )
    return QuantumLanczosResult(
        energies=result.energies,
        coefficients=result.coefficients,
        residual_norms=result.residual_norms,
        hamiltonian_matrix=result.hamiltonian_matrix,
        overlap_matrix=result.overlap_matrix,
        effective_overlap_matrix=result.effective_overlap_matrix,
        overlap_rank=result.overlap_rank,
        effective_rank=result.effective_rank,
        discarded_directions=result.discarded_directions,
        condition_number=result.condition_number,
        basis_states=states,
    )


class QuantumLanczos:
    """Build and solve a quantum Krylov/Lanczos subspace end to end."""

    def __init__(
        self,
        seed_state: Any,
        evolve: Evolution,
        overlap: MatrixElement,
        hamiltonian: MatrixElement,
        order: int,
        num_roots: Optional[int] = 1,
        overlap_cutoff: float = 1e-10,
        regularization: float = 0.0,
        hermitian_atol: float = 1e-10,
    ) -> None:
        if not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")
        self.seed_state = seed_state
        self.evolve = evolve
        self.overlap = overlap
        self.hamiltonian = hamiltonian
        self.order = order
        self.num_roots = num_roots
        self.overlap_cutoff = overlap_cutoff
        self.regularization = regularization
        self.hermitian_atol = hermitian_atol

    def basis(self) -> Tuple[Any, ...]:
        return build_krylov_basis(self.seed_state, self.evolve, self.order)

    def run(self) -> QuantumLanczosResult:
        return run_quantum_lanczos(
            self.basis(),
            self.overlap,
            self.hamiltonian,
            num_roots=self.num_roots,
            overlap_cutoff=self.overlap_cutoff,
            regularization=self.regularization,
            hermitian_atol=self.hermitian_atol,
        )


def statevector_overlap(left: Any, right: Any) -> complex:
    """Exact-state reference overlap callback for examples and simulators."""
    lhs = np.asarray(left, dtype=np.complex128).reshape(-1)
    rhs = np.asarray(right, dtype=np.complex128).reshape(-1)
    if lhs.shape != rhs.shape:
        raise ValueError("state vectors must have matching dimensions")
    return complex(np.vdot(lhs, rhs))


def dense_hamiltonian_element(hamiltonian_matrix: Any) -> MatrixElement:
    """Create an exact-state callback returning <left|H|right>."""
    matrix = _validated_hermitian(
        hamiltonian_matrix, "hamiltonian_matrix", 1e-10
    )

    def element(left: Any, right: Any) -> complex:
        lhs = np.asarray(left, dtype=np.complex128).reshape(-1)
        rhs = np.asarray(right, dtype=np.complex128).reshape(-1)
        if lhs.shape != rhs.shape or lhs.shape[0] != matrix.shape[0]:
            raise ValueError("state-vector dimension does not match Hamiltonian")
        return complex(np.vdot(lhs, matrix.dot(rhs)))

    return element


def imaginary_time_evolver(
    hamiltonian_matrix: Any,
    delta_tau: float,
    normalize: bool = True,
) -> Evolution:
    """Return an exact reference exp(-delta_tau H) Krylov evolution."""
    if delta_tau <= 0.0:
        raise ValueError("delta_tau must be positive")
    matrix = _validated_hermitian(
        hamiltonian_matrix, "hamiltonian_matrix", 1e-10
    )
    values, vectors = np.linalg.eigh(matrix)
    factors = np.exp(-delta_tau * values)
    propagator = (vectors * factors[np.newaxis, :]).dot(vectors.conj().T)

    def evolve(state: Any, step: int) -> np.ndarray:
        del step
        vector = np.asarray(state, dtype=np.complex128).reshape(-1)
        if vector.shape[0] != matrix.shape[0]:
            raise ValueError("state-vector dimension does not match Hamiltonian")
        next_state = propagator.dot(vector)
        if normalize:
            norm = float(np.linalg.norm(next_state))
            if norm == 0.0:
                raise ValueError("imaginary-time evolution produced a zero vector")
            next_state = next_state / norm
        return next_state

    return evolve
