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

"""qDRIFT randomized Hamiltonian simulation.

This module separates the qDRIFT sampling algorithm from any specific quantum
backend. It produces signed Pauli-evolution schedules that can be translated to
PyQPanda circuits or executed by the included exact small-system reference.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np


_PAULI = {
    "I": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


@dataclass(frozen=True)
class PauliTerm:
    """One real-coefficient Hermitian Pauli term."""

    coefficient: float
    pauli: str

    def __post_init__(self) -> None:
        coefficient = float(self.coefficient)
        pauli = str(self.pauli).replace(" ", "").upper()
        if not np.isfinite(coefficient):
            raise ValueError("Pauli coefficient must be finite")
        if not pauli:
            raise ValueError("Pauli string must not be empty")
        invalid = sorted(set(pauli) - set("IXYZ"))
        if invalid:
            raise ValueError("invalid Pauli symbols: %s" % "".join(invalid))
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "pauli", pauli)


@dataclass(frozen=True)
class QDriftRotation:
    """One sampled evolution exp(-i * exponent * P)."""

    term_index: int
    pauli: str
    exponent: float


@dataclass(frozen=True)
class QDriftPlan:
    """A reproducible qDRIFT schedule."""

    terms: Tuple[PauliTerm, ...]
    probabilities: Tuple[float, ...]
    rotations: Tuple[QDriftRotation, ...]
    time: float
    lambda_norm: float
    segment_count: int
    diamond_error_bound: float


@dataclass(frozen=True)
class QDriftObservableResult:
    """Monte-Carlo observable estimate over independent qDRIFT trajectories."""

    mean: float
    standard_error: float
    samples: np.ndarray
    segment_count: int
    diamond_error_bound: float


def _coerce_terms(terms: Iterable[Any]) -> Tuple[PauliTerm, ...]:
    normalized = []
    for item in terms:
        if isinstance(item, PauliTerm):
            term = item
        else:
            try:
                coefficient, pauli = item
            except Exception as exc:
                raise TypeError(
                    "terms must contain PauliTerm values or (coefficient, pauli) pairs"
                ) from exc
            term = PauliTerm(coefficient, pauli)
        if term.coefficient != 0.0:
            normalized.append(term)

    if not normalized:
        raise ValueError("Hamiltonian must contain at least one non-zero Pauli term")

    width = len(normalized[0].pauli)
    if any(len(term.pauli) != width for term in normalized):
        raise ValueError("all Pauli strings must have the same qubit width")
    return tuple(normalized)


def pauli_l1_norm(terms: Iterable[Any]) -> float:
    """Return lambda = sum_j |h_j| for H = sum_j h_j P_j."""
    normalized = _coerce_terms(terms)
    return float(sum(abs(term.coefficient) for term in normalized))


def qdrift_segment_count(
    terms: Iterable[Any],
    time: float,
    epsilon: float,
) -> int:
    """Return the Campbell qDRIFT segment bound.

    The first-order qDRIFT diamond-norm bound is
    2 * lambda^2 * time^2 / N, so N is chosen to make that value no larger
    than epsilon.
    """
    time = float(time)
    epsilon = float(epsilon)
    if not np.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if time == 0.0:
        return 0
    lam = pauli_l1_norm(terms)
    return max(1, int(np.ceil(2.0 * (lam * time) ** 2 / epsilon)))


def qdrift_error_bound(
    terms: Iterable[Any],
    time: float,
    segments: int,
) -> float:
    """Return 2 * lambda^2 * time^2 / N for a chosen schedule size."""
    time = float(time)
    if not np.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and non-negative")
    if not isinstance(segments, int) or segments < 0:
        raise ValueError("segments must be a non-negative integer")
    if time == 0.0:
        return 0.0
    if segments == 0:
        raise ValueError("non-zero time requires at least one segment")
    lam = pauli_l1_norm(terms)
    return float(2.0 * (lam * time) ** 2 / segments)


def sample_qdrift(
    terms: Iterable[Any],
    time: float,
    epsilon: Optional[float] = None,
    segments: Optional[int] = None,
    seed: Optional[Any] = None,
) -> QDriftPlan:
    """Sample a qDRIFT Pauli-evolution schedule.

    Exactly one of epsilon or segments may be supplied. If epsilon is given,
    the Campbell bound selects the number of segments. Each sampled Pauli term
    is drawn with probability |h_j| / lambda and evolves for signed exponent
    sign(h_j) * lambda * time / N.
    """
    normalized = _coerce_terms(terms)
    time = float(time)
    if not np.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and non-negative")
    if epsilon is not None and segments is not None:
        raise ValueError("provide epsilon or segments, not both")
    if epsilon is None and segments is None:
        raise ValueError("epsilon or segments is required")

    lam = float(sum(abs(term.coefficient) for term in normalized))
    probabilities = np.asarray(
        [abs(term.coefficient) / lam for term in normalized],
        dtype=float,
    )

    if segments is None:
        segment_count = qdrift_segment_count(normalized, time, float(epsilon))
    else:
        if not isinstance(segments, int) or segments < 0:
            raise ValueError("segments must be a non-negative integer")
        segment_count = segments
        if time > 0.0 and segment_count == 0:
            raise ValueError("non-zero time requires at least one segment")

    if segment_count == 0:
        rotations = ()
        error_bound = 0.0
    else:
        generator = np.random.default_rng(seed)
        choices = generator.choice(
            len(normalized),
            size=segment_count,
            replace=True,
            p=probabilities,
        )
        base_exponent = lam * time / segment_count
        rotations = tuple(
            QDriftRotation(
                term_index=int(index),
                pauli=normalized[int(index)].pauli,
                exponent=float(
                    np.sign(normalized[int(index)].coefficient) * base_exponent
                ),
            )
            for index in choices
        )
        error_bound = float(2.0 * (lam * time) ** 2 / segment_count)

    return QDriftPlan(
        terms=normalized,
        probabilities=tuple(float(value) for value in probabilities),
        rotations=rotations,
        time=time,
        lambda_norm=lam,
        segment_count=segment_count,
        diamond_error_bound=error_bound,
    )


def pauli_matrix(pauli: str) -> np.ndarray:
    """Return the dense matrix for a left-to-right Pauli string."""
    value = str(pauli).replace(" ", "").upper()
    if not value:
        raise ValueError("Pauli string must not be empty")
    invalid = sorted(set(value) - set("IXYZ"))
    if invalid:
        raise ValueError("invalid Pauli symbols: %s" % "".join(invalid))

    matrix = np.asarray([[1.0]], dtype=np.complex128)
    for symbol in value:
        matrix = np.kron(matrix, _PAULI[symbol])
    return matrix


def dense_hamiltonian(terms: Iterable[Any]) -> np.ndarray:
    """Build the exact dense Hamiltonian for reference-sized problems."""
    normalized = _coerce_terms(terms)
    dimension = 2 ** len(normalized[0].pauli)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for term in normalized:
        matrix += term.coefficient * pauli_matrix(term.pauli)
    return 0.5 * (matrix + matrix.conj().T)


def apply_qdrift_plan(
    state: Any,
    plan: QDriftPlan,
) -> np.ndarray:
    """Apply a sampled qDRIFT schedule to an exact state-vector reference."""
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    dimension = 2 ** len(plan.terms[0].pauli)
    if vector.shape[0] != dimension:
        raise ValueError("state-vector dimension does not match Pauli width")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("state vector must not be zero")
    vector = vector / norm

    cache = {}
    identity = np.eye(dimension, dtype=np.complex128)
    for rotation in plan.rotations:
        matrix = cache.get(rotation.pauli)
        if matrix is None:
            matrix = pauli_matrix(rotation.pauli)
            cache[rotation.pauli] = matrix
        alpha = rotation.exponent
        unitary = np.cos(alpha) * identity - 1.0j * np.sin(alpha) * matrix
        vector = unitary.dot(vector)
    return vector


def exact_time_evolution(
    terms: Iterable[Any],
    state: Any,
    time: float,
) -> np.ndarray:
    """Exact dense exp(-i H t) reference for small-system comparisons."""
    time = float(time)
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    hamiltonian = dense_hamiltonian(terms)
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vector.shape[0] != hamiltonian.shape[0]:
        raise ValueError("state-vector dimension does not match Hamiltonian")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("state vector must not be zero")
    vector = vector / norm

    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1.0j * time * energies)
    unitary = (eigenvectors * phases[np.newaxis, :]).dot(
        eigenvectors.conj().T
    )
    return unitary.dot(vector)


def state_fidelity(left: Any, right: Any) -> float:
    """Return pure-state fidelity |<left|right>|^2."""
    lhs = np.asarray(left, dtype=np.complex128).reshape(-1)
    rhs = np.asarray(right, dtype=np.complex128).reshape(-1)
    if lhs.shape != rhs.shape:
        raise ValueError("state vectors must have matching dimensions")
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        raise ValueError("state vectors must not be zero")
    amplitude = np.vdot(lhs / lhs_norm, rhs / rhs_norm)
    return float(min(1.0, max(0.0, abs(amplitude) ** 2)))


def expectation_value(state: Any, observable: Any) -> float:
    """Return the real expectation value of a Hermitian dense observable."""
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    matrix = np.asarray(observable, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("observable must be square")
    if matrix.shape[0] != vector.shape[0]:
        raise ValueError("observable dimension does not match state")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError("observable must be Hermitian")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("state vector must not be zero")
    vector = vector / norm
    value = np.vdot(vector, matrix.dot(vector))
    return float(np.real_if_close(value))


def estimate_observable(
    terms: Iterable[Any],
    initial_state: Any,
    observable: Any,
    time: float,
    trajectories: int,
    epsilon: Optional[float] = None,
    segments: Optional[int] = None,
    seed: Optional[Any] = None,
) -> QDriftObservableResult:
    """Estimate an observable by averaging independent qDRIFT trajectories."""
    if not isinstance(trajectories, int) or trajectories < 1:
        raise ValueError("trajectories must be a positive integer")
    normalized = _coerce_terms(terms)
    generator = np.random.default_rng(seed)

    first_plan = sample_qdrift(
        normalized,
        time,
        epsilon=epsilon,
        segments=segments,
        seed=generator,
    )
    values = [
        expectation_value(
            apply_qdrift_plan(initial_state, first_plan),
            observable,
        )
    ]

    for _ in range(1, trajectories):
        plan = sample_qdrift(
            normalized,
            time,
            segments=first_plan.segment_count,
            seed=generator,
        )
        values.append(
            expectation_value(
                apply_qdrift_plan(initial_state, plan),
                observable,
            )
        )

    samples = np.asarray(values, dtype=float)
    if trajectories == 1:
        standard_error = 0.0
    else:
        standard_error = float(
            np.std(samples, ddof=1) / np.sqrt(float(trajectories))
        )

    return QDriftObservableResult(
        mean=float(np.mean(samples)),
        standard_error=standard_error,
        samples=samples,
        segment_count=first_plan.segment_count,
        diamond_error_bound=first_plan.diamond_error_bound,
    )
