"""Quantum state tomography utilities.

This module implements full Pauli-basis quantum state tomography without tying
the reconstruction step to a particular simulator or hardware backend. A caller
executes the measurement settings returned by :func:`pauli_measurement_plan`,
passes the resulting count/probability dictionaries to
:func:`linear_inversion_tomography`, and receives a density matrix.

Bit strings are interpreted in the same left-to-right order as the basis
characters. For example, basis ``"XY"`` means X on the first qubit and Y on
the second; outcome ``"01"`` means the +1 X eigenstate on the first qubit and
the -1 Y eigenstate on the second.
"""

from itertools import product
from typing import Mapping, Optional, Sequence

import numpy as np


_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

_SQRT2 = np.sqrt(2.0)
_EIGENVECTORS = {
    "X": {
        "0": np.array([1.0, 1.0], dtype=complex) / _SQRT2,
        "1": np.array([1.0, -1.0], dtype=complex) / _SQRT2,
    },
    "Y": {
        "0": np.array([1.0, 1.0j], dtype=complex) / _SQRT2,
        "1": np.array([1.0, -1.0j], dtype=complex) / _SQRT2,
    },
    "Z": {
        "0": np.array([1.0, 0.0], dtype=complex),
        "1": np.array([0.0, 1.0], dtype=complex),
    },
}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _validate_basis(basis: str, num_qubits: Optional[int] = None) -> str:
    if not isinstance(basis, str):
        raise TypeError("basis must be a string")
    canonical = basis.upper()
    if not canonical or any(axis not in "XYZ" for axis in canonical):
        raise ValueError("basis must contain only X, Y, and Z")
    if num_qubits is not None and len(canonical) != num_qubits:
        raise ValueError(
            f"basis {basis!r} has {len(canonical)} axes; expected {num_qubits}"
        )
    return canonical


def _normalize_distribution(
    distribution: Mapping[str, float], num_qubits: int
) -> dict[str, float]:
    if not isinstance(distribution, Mapping) or not distribution:
        raise ValueError("each measurement distribution must be a non-empty mapping")

    normalized: dict[str, float] = {}
    total = 0.0
    for outcome, weight in distribution.items():
        if not isinstance(outcome, str):
            raise TypeError("measurement outcomes must be bit strings")
        if len(outcome) != num_qubits or any(bit not in "01" for bit in outcome):
            raise ValueError(
                f"invalid outcome {outcome!r}; expected a {num_qubits}-bit string"
            )
        value = float(weight)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("measurement weights must be finite and non-negative")
        normalized[outcome] = value
        total += value

    if total <= 0.0:
        raise ValueError("measurement distribution has zero total weight")
    return {outcome: weight / total for outcome, weight in normalized.items()}


def _kron_all(operators: Sequence[np.ndarray]) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for operator in operators:
        result = np.kron(result, operator)
    return result


def pauli_measurement_plan(num_qubits: int) -> tuple[str, ...]:
    """Return the complete X/Y/Z product-basis plan for full tomography.

    The plan contains ``3 ** num_qubits`` settings. Each string is ordered from
    the first logical qubit to the last logical qubit.
    """
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")
    return tuple("".join(axes) for axes in product("XYZ", repeat=num_qubits))


def expectation_from_distribution(
    distribution: Mapping[str, float],
    active_positions: Optional[Sequence[int]] = None,
) -> float:
    """Return a product-Pauli expectation from one eigenbasis distribution.

    Outcome bit ``0`` contributes eigenvalue +1 and bit ``1`` contributes -1.
    ``active_positions`` can omit qubits whose Pauli factor is the identity.
    Counts and probabilities are both accepted because weights are normalized.
    """
    if not isinstance(distribution, Mapping) or not distribution:
        raise ValueError("distribution must be a non-empty mapping")

    first_outcome = next(iter(distribution))
    if not isinstance(first_outcome, str):
        raise TypeError("measurement outcomes must be bit strings")
    num_qubits = len(first_outcome)
    weights = _normalize_distribution(distribution, num_qubits)

    if active_positions is None:
        positions = tuple(range(num_qubits))
    else:
        positions = tuple(int(position) for position in active_positions)
        if len(set(positions)) != len(positions):
            raise ValueError("active_positions must not contain duplicates")
        if any(position < 0 or position >= num_qubits for position in positions):
            raise ValueError("active_positions contains an out-of-range index")

    expectation = 0.0
    for outcome, probability in weights.items():
        eigenvalue = 1.0
        for position in positions:
            if outcome[position] == "1":
                eigenvalue = -eigenvalue
        expectation += eigenvalue * probability
    return float(expectation)


def project_density_matrix(matrix: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """Project a Hermitian estimate onto the positive, trace-one state space.

    Negative eigenvalues introduced by finite-shot linear inversion are clipped
    to zero, then the spectrum is renormalized to unit trace.
    """
    candidate = np.asarray(matrix, dtype=complex)
    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError("density matrix must be square")
    if not _is_power_of_two(candidate.shape[0]):
        raise ValueError("density-matrix dimension must be a power of two")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("density matrix contains non-finite values")

    hermitian = (candidate + candidate.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    eigenvalues = np.clip(eigenvalues.real, 0.0, None)
    weight = float(np.sum(eigenvalues))
    if weight <= atol:
        raise ValueError("density matrix has no positive spectral weight")

    projected = (eigenvectors * eigenvalues[np.newaxis, :]) @ eigenvectors.conj().T
    projected /= weight
    return (projected + projected.conj().T) / 2.0


def probabilities_from_density_matrix(
    matrix: np.ndarray, basis: str, atol: float = 1e-12
) -> dict[str, float]:
    """Calculate ideal product-basis probabilities for a density matrix.

    This helper is useful for examples, simulators, and generating deterministic
    reference data. Hardware callers normally supply measured count dictionaries
    directly to :func:`linear_inversion_tomography`.
    """
    canonical_basis = _validate_basis(basis)
    num_qubits = len(canonical_basis)
    dimension = 1 << num_qubits
    density = np.asarray(matrix, dtype=complex)
    if density.shape != (dimension, dimension):
        raise ValueError(
            f"matrix shape {density.shape} does not match {num_qubits} qubits"
        )
    if not np.all(np.isfinite(density)):
        raise ValueError("density matrix contains non-finite values")
    if np.linalg.norm(density - density.conj().T) > max(atol, 1e-10):
        raise ValueError("density matrix must be Hermitian")

    trace = np.trace(density)
    if abs(float(trace.imag)) > max(atol, 1e-10) or float(trace.real) <= atol:
        raise ValueError("density matrix must have a positive real trace")
    density = density / float(trace.real)

    probabilities: dict[str, float] = {}
    for bits in product("01", repeat=num_qubits):
        outcome = "".join(bits)
        local_projectors = []
        for axis, bit in zip(canonical_basis, bits):
            vector = _EIGENVECTORS[axis][bit]
            local_projectors.append(np.outer(vector, vector.conj()))
        projector = _kron_all(local_projectors)
        value = float(np.trace(density @ projector).real)
        if value < -max(atol, 1e-10):
            raise ValueError("density matrix produces a negative probability")
        probabilities[outcome] = max(0.0, value)

    total = float(sum(probabilities.values()))
    if total <= atol:
        raise ValueError("basis probabilities have zero total weight")
    return {outcome: value / total for outcome, value in probabilities.items()}


def linear_inversion_tomography(
    measurements: Mapping[str, Mapping[str, float]],
    num_qubits: Optional[int] = None,
    physical: bool = True,
    atol: float = 1e-12,
) -> np.ndarray:
    """Reconstruct a density matrix from complete Pauli-basis measurements.

    Parameters
    ----------
    measurements:
        Mapping from basis strings (for example ``"XZ"``) to outcome weights.
        Weights may be raw counts or probabilities.
    num_qubits:
        Number of qubits. When omitted, it is inferred from the first basis key.
    physical:
        If true, project the linear-inversion estimate onto the positive,
        trace-one density-matrix set.
    atol:
        Numerical tolerance used by the physical projection.
    """
    if not isinstance(measurements, Mapping) or not measurements:
        raise ValueError("measurements must be a non-empty mapping")

    first_basis = next(iter(measurements))
    inferred = len(first_basis) if isinstance(first_basis, str) else 0
    if num_qubits is None:
        num_qubits = inferred
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")

    normalized: dict[str, dict[str, float]] = {}
    for basis, distribution in measurements.items():
        canonical_basis = _validate_basis(basis, num_qubits)
        if canonical_basis in normalized:
            raise ValueError(f"duplicate basis after normalization: {canonical_basis}")
        normalized[canonical_basis] = _normalize_distribution(distribution, num_qubits)

    plan = pauli_measurement_plan(num_qubits)
    missing = [basis for basis in plan if basis not in normalized]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(f"missing {len(missing)} tomography bases: {preview}{suffix}")

    dimension = 1 << num_qubits
    density = np.zeros((dimension, dimension), dtype=complex)

    for factors in product("IXYZ", repeat=num_qubits):
        if all(factor == "I" for factor in factors):
            coefficient = 1.0
        else:
            measurement_basis = "".join(
                "Z" if factor == "I" else factor for factor in factors
            )
            active_positions = [
                index for index, factor in enumerate(factors) if factor != "I"
            ]
            coefficient = expectation_from_distribution(
                normalized[measurement_basis], active_positions
            )

        pauli_word = _kron_all([_PAULI[factor] for factor in factors])
        density += coefficient * pauli_word

    density /= float(dimension)
    density = (density + density.conj().T) / 2.0
    if physical:
        return project_density_matrix(density, atol=atol)
    return density


def pure_state_fidelity(matrix: np.ndarray, state: np.ndarray) -> float:
    """Return ``<psi|rho|psi>`` for a pure target state."""
    density = np.asarray(matrix, dtype=complex)
    vector = np.asarray(state, dtype=complex).reshape(-1)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density matrix must be square")
    if density.shape[0] != vector.size:
        raise ValueError("state-vector dimension does not match density matrix")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("state vector must have a finite non-zero norm")
    vector = vector / norm
    value = np.vdot(vector, density @ vector)
    return float(np.clip(value.real, 0.0, 1.0))


__all__ = [
    "expectation_from_distribution",
    "linear_inversion_tomography",
    "pauli_measurement_plan",
    "probabilities_from_density_matrix",
    "project_density_matrix",
    "pure_state_fidelity",
]
