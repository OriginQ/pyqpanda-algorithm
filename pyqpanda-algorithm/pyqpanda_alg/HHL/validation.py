"""Input normalization for the HHL solver: validation and padding.

Padding rule (reversible)
-------------------------
Systems whose dimension is not a power of two are padded to the next
power of two by block-diagonal augmentation::

    A_padded = [[A, 0], [0, I]]      b_padded = [b, 0, ..., 0]

The identity block contributes only unit eigenvalues, so padding never
introduces a zero eigenvalue into an invertible system.  The rule is
reversible: truncate a computed solution to its first
``original_dimension`` entries (the padded ``b`` contributes nothing
beyond the first ``n`` entries) and scale by ``original_vector_norm``
to recover the solution of the original unnormalized system.

Every rejection raised here is an :class:`AlgorithmInputError` with a
message naming the failing class, so callers can fail fast before any
task submission.
"""

import numpy as np

from pyqpanda_alg.execution import AlgorithmInputError

from .model import HHLConfig, NormalizedLinearSystem

_HERMITIAN_RTOL = 1e-8
_HERMITIAN_ATOL = 1e-10


def normalize_linear_system(
    matrix, vector, config: HHLConfig
) -> NormalizedLinearSystem:
    """Validate a linear system and produce its padded, normalized form.

    ``matrix`` must be a finite Hermitian square matrix given as a
    2-D array or as a flat list whose length is a perfect square
    (legacy flattened form); ``vector`` must be a matching finite
    1-D sequence with nonzero norm.  The system must be invertible and
    its condition number (largest/smallest singular value) at most
    ``config.max_condition_number``.  The returned system carries the
    padded Hermitian matrix and the unit-norm padded vector, plus the
    original dimension and vector norm needed to undo the padding.
    """
    matrix = _coerce_matrix(matrix)
    vector = _coerce_vector(vector, matrix.shape[0])
    dimension = matrix.shape[0]

    if not np.all(np.isfinite(matrix)):
        raise AlgorithmInputError("matrix must contain only finite values")
    if not np.all(np.isfinite(vector)):
        raise AlgorithmInputError("vector must contain only finite values")

    if not np.allclose(
        matrix, matrix.conj().T, rtol=_HERMITIAN_RTOL, atol=_HERMITIAN_ATOL
    ):
        raise AlgorithmInputError("matrix must be Hermitian (A == A.conj().T)")

    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        raise AlgorithmInputError("vector must have nonzero norm")

    singular_values = np.abs(np.linalg.eigvalsh(matrix))
    largest = singular_values.max()
    smallest = singular_values.min()
    singular_floor = dimension * np.finfo(np.float64).eps * largest
    if smallest <= singular_floor:
        raise AlgorithmInputError(
            "matrix is singular (has a zero eigenvalue within working precision)"
        )
    condition_number = largest / smallest
    if condition_number > config.max_condition_number:
        raise AlgorithmInputError(
            f"matrix condition number {condition_number:.3g} exceeds "
            f"max_condition_number {config.max_condition_number:g}"
        )

    if np.iscomplexobj(matrix):
        matrix = matrix.astype(np.complex128)
    else:
        matrix = matrix.astype(np.float64)
    if np.iscomplexobj(vector):
        vector = vector.astype(np.complex128)
    else:
        vector = vector.astype(np.float64)

    padded_dimension = _next_power_of_two(dimension)
    padded_matrix = np.zeros((padded_dimension, padded_dimension), dtype=matrix.dtype)
    padded_matrix[:dimension, :dimension] = matrix
    padding_size = padded_dimension - dimension
    if padding_size:
        padded_matrix[dimension:, dimension:] = np.eye(padding_size, dtype=matrix.dtype)
    padded_vector = np.zeros(padded_dimension, dtype=vector.dtype)
    padded_vector[:dimension] = vector
    padded_vector /= vector_norm

    return NormalizedLinearSystem(
        original_dimension=dimension,
        padded_dimension=padded_dimension,
        matrix=padded_matrix,
        vector=padded_vector,
        original_vector_norm=vector_norm,
    )


def _coerce_matrix(matrix) -> np.ndarray:
    """Turn an array or flat legacy list into a square 2-D array."""
    try:
        matrix = np.asarray(matrix)
    except ValueError as exc:
        raise AlgorithmInputError(
            "matrix must be a square array or a flat list of numbers"
        ) from exc
    if matrix.ndim == 1:
        size = matrix.shape[0]
        root = int(np.sqrt(size))
        if root * root != size:
            raise AlgorithmInputError(
                f"flattened matrix list of length {size} is not a perfect square"
            )
        matrix = matrix.reshape(root, root)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AlgorithmInputError(f"matrix must be square, got shape {matrix.shape}")
    return matrix


def _coerce_vector(vector, dimension: int) -> np.ndarray:
    """Turn an array or flat list into a 1-D vector of the right size."""
    try:
        vector = np.asarray(vector)
    except ValueError as exc:
        raise AlgorithmInputError(
            "vector must be a one-dimensional sequence of numbers"
        ) from exc
    if vector.ndim != 1:
        raise AlgorithmInputError(f"vector must be one-dimensional, got shape {vector.shape}")
    if vector.shape[0] != dimension:
        raise AlgorithmInputError(
            f"vector length {vector.shape[0]} does not match matrix dimension {dimension}"
        )
    return vector


def _next_power_of_two(dimension: int) -> int:
    """Smallest power of two greater than or equal to ``dimension``."""
    return 1 << (dimension - 1).bit_length()
