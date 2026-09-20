"""Quantum Natural Gradient utilities based on the Fubini-Study metric."""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricDiagnostics:
    """Numerical diagnostics for a regularized quantum metric solve."""

    rank: int
    condition_number: float
    eigenvalues: Tuple[float, ...]
    cutoff: float


@dataclass(frozen=True)
class NaturalGradientResult:
    """Natural-gradient direction and the metric used to obtain it."""

    direction: np.ndarray
    metric: np.ndarray
    diagnostics: MetricDiagnostics


@dataclass(frozen=True)
class NaturalGradientStep:
    """One optimizer step together with the natural-gradient solve details."""

    parameters: np.ndarray
    result: NaturalGradientResult


def _as_finite_vector(values, name: str, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def fubini_study_metric(
    state,
    state_jacobian,
    *,
    normalization_atol: float = 1e-8,
) -> np.ndarray:
    """Construct the pure-state Fubini-Study metric.

    Args:
        state: Normalized state vector |psi>.
        state_jacobian: Array with shape (n_parameters, hilbert_dimension).
            Row i is d|psi>/d theta_i.
        normalization_atol: Absolute tolerance for the state normalization gate.

    Returns:
        A real symmetric positive-semidefinite metric up to floating-point
        roundoff:

            g_ij = Re(<d_i psi|d_j psi>
                      - <d_i psi|psi><psi|d_j psi>)

    The projection term removes unobservable global-phase directions.
    """

    psi = _as_finite_vector(state, "state", dtype=complex)
    norm = float(np.linalg.norm(psi))
    if not np.isclose(norm, 1.0, atol=normalization_atol, rtol=0.0):
        raise ValueError("state must be normalized before metric construction")

    jacobian = np.asarray(state_jacobian, dtype=complex)
    if jacobian.ndim != 2:
        raise ValueError("state_jacobian must have shape (n_parameters, state_dim)")
    if jacobian.shape[1] != psi.size:
        raise ValueError("state_jacobian state dimension does not match state")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("state_jacobian must contain only finite values")

    derivative_overlap = jacobian.conj() @ jacobian.T
    derivative_state = jacobian.conj() @ psi
    quantum_geometric_tensor = derivative_overlap - np.outer(
        derivative_state,
        derivative_state.conj(),
    )
    metric = np.real(quantum_geometric_tensor)
    return 0.5 * (metric + metric.T)


def block_diagonal_metric(
    metric,
    blocks: Iterable[Sequence[int]],
) -> np.ndarray:
    """Keep only requested parameter blocks of a metric tensor.

    This provides a scalable approximation for large ansaetze while preserving
    exact within-block geometry.
    """

    matrix = np.asarray(metric, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("metric must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("metric must contain only finite values")

    n = matrix.shape[0]
    result = np.zeros_like(matrix)
    seen = set()
    for block in blocks:
        indices = tuple(int(index) for index in block)
        if not indices:
            raise ValueError("metric blocks may not be empty")
        if len(set(indices)) != len(indices):
            raise ValueError("a metric block may not repeat an index")
        if any(index < 0 or index >= n for index in indices):
            raise IndexError("metric block index is out of range")
        overlap = seen.intersection(indices)
        if overlap:
            raise ValueError("metric blocks must be disjoint")
        seen.update(indices)
        selector = np.ix_(indices, indices)
        result[selector] = matrix[selector]

    return result


def solve_natural_gradient(
    metric,
    gradient,
    *,
    damping: float = 1e-6,
    rcond: float = 1e-10,
) -> NaturalGradientResult:
    """Solve (g + damping I) v = gradient with a stable eigenspace inverse."""

    matrix = np.asarray(metric, dtype=float)
    grad = _as_finite_vector(gradient, "gradient", dtype=float)

    if matrix.ndim != 2 or matrix.shape != (grad.size, grad.size):
        raise ValueError("metric shape must be (n_parameters, n_parameters)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("metric must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError("metric must be symmetric")

    damping = float(damping)
    rcond = float(rcond)
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be a non-negative finite number")
    if not np.isfinite(rcond) or rcond < 0:
        raise ValueError("rcond must be a non-negative finite number")

    regularized = 0.5 * (matrix + matrix.T) + damping * np.eye(grad.size)
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    cutoff = rcond * scale

    if np.any(eigenvalues < -cutoff):
        raise ValueError("regularized metric is not positive semidefinite")

    keep = eigenvalues > cutoff
    rank = int(np.count_nonzero(keep))
    projected = eigenvectors.T @ grad
    inverse_projected = np.zeros_like(projected)
    inverse_projected[keep] = projected[keep] / eigenvalues[keep]
    direction = eigenvectors @ inverse_projected

    if rank == 0:
        condition = float("inf")
    else:
        retained = eigenvalues[keep]
        condition = float(np.max(retained) / np.min(retained))

    diagnostics = MetricDiagnostics(
        rank=rank,
        condition_number=condition,
        eigenvalues=tuple(float(value) for value in eigenvalues),
        cutoff=float(cutoff),
    )
    return NaturalGradientResult(
        direction=direction,
        metric=regularized,
        diagnostics=diagnostics,
    )


def quantum_natural_gradient(
    state,
    state_jacobian,
    gradient,
    *,
    damping: float = 1e-6,
    rcond: float = 1e-10,
    blocks: Optional[Iterable[Sequence[int]]] = None,
) -> NaturalGradientResult:
    """Build the Fubini-Study metric and return the natural-gradient direction."""

    metric = fubini_study_metric(state, state_jacobian)
    if blocks is not None:
        metric = block_diagonal_metric(metric, blocks)
    return solve_natural_gradient(
        metric,
        gradient,
        damping=damping,
        rcond=rcond,
    )


def natural_gradient_step(
    parameters,
    state,
    state_jacobian,
    gradient,
    *,
    learning_rate: float = 0.1,
    damping: float = 1e-6,
    rcond: float = 1e-10,
    blocks: Optional[Iterable[Sequence[int]]] = None,
) -> NaturalGradientStep:
    """Apply one theta <- theta - learning_rate * g_FS^+ * gradient step."""

    theta = _as_finite_vector(parameters, "parameters", dtype=float)
    grad = _as_finite_vector(gradient, "gradient", dtype=float)
    if theta.size != grad.size:
        raise ValueError("parameters and gradient must have the same length")

    learning_rate = float(learning_rate)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a positive finite number")

    result = quantum_natural_gradient(
        state,
        state_jacobian,
        grad,
        damping=damping,
        rcond=rcond,
        blocks=blocks,
    )
    return NaturalGradientStep(
        parameters=theta - learning_rate * result.direction,
        result=result,
    )
