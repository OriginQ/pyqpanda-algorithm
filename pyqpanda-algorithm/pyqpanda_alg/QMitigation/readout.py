"""Readout-error mitigation utilities.

Bitstrings use normal binary display order: the left-most character is the
most-significant bit. Assignment matrices use rows for measured states and
columns for prepared states.
"""

from typing import Mapping, Optional, Sequence, Union

import numpy as np


CountsLike = Mapping[str, float]


def _validate_bitstring(bitstring: str, qubit_count: int) -> str:
    value = str(bitstring)
    if len(value) != qubit_count or any(bit not in "01" for bit in value):
        raise ValueError(
            f"expected a {qubit_count}-bit binary string, got {bitstring!r}"
        )
    return value


def _infer_qubit_count(keys: Sequence[str]) -> int:
    if not keys:
        raise ValueError("at least one bitstring is required")
    lengths = {len(str(key)) for key in keys}
    if len(lengths) != 1:
        raise ValueError("all bitstrings must have the same length")
    qubit_count = lengths.pop()
    if qubit_count <= 0:
        raise ValueError("bitstrings may not be empty")
    for key in keys:
        _validate_bitstring(str(key), qubit_count)
    return qubit_count


def counts_to_probabilities(
    counts: CountsLike,
    qubit_count: Optional[int] = None,
) -> np.ndarray:
    """Convert sparse bitstring counts/probabilities into a dense vector."""

    if qubit_count is None:
        qubit_count = _infer_qubit_count(list(counts.keys()))
    qubit_count = int(qubit_count)
    if qubit_count <= 0:
        raise ValueError("qubit_count must be positive")

    dimension = 1 << qubit_count
    vector = np.zeros(dimension, dtype=float)
    for bitstring, weight in counts.items():
        normalized = _validate_bitstring(bitstring, qubit_count)
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("count/probability weights must be finite and non-negative")
        vector[int(normalized, 2)] += weight

    total = float(vector.sum())
    if total <= 0:
        raise ValueError("counts must contain positive total weight")
    return vector / total


def probabilities_to_dict(
    probabilities,
    *,
    atol: float = 0.0,
) -> dict:
    """Return a dense probability vector as fixed-width bitstring mapping."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    qubit_count = int(round(np.log2(values.size)))
    if (1 << qubit_count) != values.size:
        raise ValueError("probability vector length must be a power of two")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite")

    return {
        format(index, f"0{qubit_count}b"): float(value)
        for index, value in enumerate(values)
        if abs(float(value)) > atol
    }


def assignment_matrix_from_calibration(
    calibration_counts: Mapping[str, CountsLike],
    qubit_count: Optional[int] = None,
) -> np.ndarray:
    """Build P(measured=i | prepared=j) from complete basis calibration."""

    prepared = list(calibration_counts.keys())
    if qubit_count is None:
        qubit_count = _infer_qubit_count(prepared)
    qubit_count = int(qubit_count)
    dimension = 1 << qubit_count
    expected = {format(index, f"0{qubit_count}b") for index in range(dimension)}
    normalized_prepared = {
        _validate_bitstring(bitstring, qubit_count) for bitstring in prepared
    }
    missing = expected - normalized_prepared
    extra = normalized_prepared - expected
    if missing or extra or len(prepared) != dimension:
        raise ValueError(
            "calibration must contain exactly one entry for every prepared basis state"
        )

    matrix = np.zeros((dimension, dimension), dtype=float)
    for prepared_state, measured_counts in calibration_counts.items():
        column = int(_validate_bitstring(prepared_state, qubit_count), 2)
        matrix[:, column] = counts_to_probabilities(
            measured_counts,
            qubit_count=qubit_count,
        )
    return matrix


def factorized_assignment_matrix(single_qubit_matrices: Sequence[object]) -> np.ndarray:
    """Kronecker-compose 2x2 assignment matrices in MSB-to-LSB order."""

    if not single_qubit_matrices:
        raise ValueError("at least one single-qubit assignment matrix is required")

    result = np.array([[1.0]])
    for matrix in single_qubit_matrices:
        value = np.asarray(matrix, dtype=float)
        if value.shape != (2, 2):
            raise ValueError("every single-qubit assignment matrix must be 2x2")
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError("assignment matrix entries must be finite and non-negative")
        if not np.allclose(value.sum(axis=0), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("each assignment-matrix column must sum to one")
        result = np.kron(result, value)
    return result


def project_probability_simplex(values) -> np.ndarray:
    """Euclidean projection onto p_i >= 0, sum_i p_i = 1."""

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values must be a non-empty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must be finite")

    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, vector.size + 1)
    valid = sorted_values - cumulative / indices > 0
    if not np.any(valid):
        return np.full(vector.size, 1.0 / vector.size)
    rho = int(np.nonzero(valid)[0][-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    projected /= projected.sum()
    return projected


def mitigate_probabilities(
    observed: Union[CountsLike, Sequence[float], np.ndarray],
    assignment_matrix,
    *,
    rcond: float = 1e-10,
    project: bool = True,
) -> np.ndarray:
    """Invert a readout assignment matrix and optionally project to probabilities."""

    matrix = np.asarray(assignment_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("assignment_matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("assignment_matrix must be finite")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("assignment_matrix columns must sum to one")

    dimension = matrix.shape[0]
    qubit_count = int(round(np.log2(dimension)))
    if (1 << qubit_count) != dimension:
        raise ValueError("assignment_matrix dimension must be a power of two")

    if isinstance(observed, Mapping):
        vector = counts_to_probabilities(observed, qubit_count=qubit_count)
    else:
        vector = np.asarray(observed, dtype=float)
        if vector.shape != (dimension,):
            raise ValueError("observed vector length must match assignment_matrix")
        if not np.all(np.isfinite(vector)) or np.any(vector < 0):
            raise ValueError("observed probabilities must be finite and non-negative")
        total = float(vector.sum())
        if total <= 0:
            raise ValueError("observed probabilities must have positive total weight")
        vector = vector / total

    rcond = float(rcond)
    if not np.isfinite(rcond) or rcond < 0:
        raise ValueError("rcond must be a non-negative finite number")

    corrected = np.linalg.pinv(matrix, rcond=rcond) @ vector
    if project:
        corrected = project_probability_simplex(corrected)
    return corrected
