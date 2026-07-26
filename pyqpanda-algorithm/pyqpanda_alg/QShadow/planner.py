"""Observable-aware local-Pauli measurement planning.

The optimiser minimises a documented diagonal second-moment proxy,

    sum_j w_j c_j^2 / prod_{q in supp(P_j)} beta[q, P_j[q]],

under independent single-qubit basis probabilities ``beta`` and a probability
floor.  This objective follows the locally-biased classical-shadow motivation;
the block-coordinate solver in this module is an independent implementation.
It is a proxy, not a claim of globally optimal state-dependent variance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .observable import PauliObservable, coerce_observables

AXES = "XYZ"
_AXIS_TO_INDEX = {axis: index for index, axis in enumerate(AXES)}


def _validate_probabilities(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_qubits: int | None = None,
    probability_floor: float | None = None,
) -> np.ndarray:
    result: np.ndarray = np.asarray(probabilities, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("basis probabilities must have shape (n_qubits, 3)")
    if n_qubits is not None and result.shape[0] != n_qubits:
        raise ValueError(
            f"probabilities describe {result.shape[0]} qubits, expected {n_qubits}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("basis probabilities must be finite")
    if np.any(result <= 0):
        raise ValueError("every X/Y/Z basis probability must be positive")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("each qubit's X/Y/Z probabilities must sum to one")
    if probability_floor is not None and np.any(result < probability_floor - 1e-12):
        raise ValueError("initial probabilities violate probability_floor")
    return result.copy()


def uniform_probabilities(n_qubits: int) -> np.ndarray:
    """Return uniform local X/Y/Z probabilities for ``n_qubits``."""

    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
        raise TypeError("n_qubits must be an integer")
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    return np.full((n_qubits, 3), 1.0 / 3.0, dtype=float)


def _weighted_terms(
    observables: PauliObservable | Sequence[PauliObservable],
    observable_weights: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[PauliObservable, ...]]:
    checked = coerce_observables(observables)
    weights: np.ndarray
    if observable_weights is None:
        weights = np.ones(len(checked), dtype=float)
    else:
        weights = np.asarray(observable_weights, dtype=float)
        if weights.shape != (len(checked),):
            raise ValueError("observable_weights must match the observable count")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("observable_weights must be finite and non-negative")
        if not np.any(weights > 0):
            raise ValueError("at least one observable weight must be positive")

    axes: list[list[int]] = []
    term_weights: list[float] = []
    for observable, observable_weight in zip(checked, weights):
        for term in observable.terms:
            weight = float(observable_weight * term.coefficient**2)
            if weight == 0.0 or term.is_identity:
                continue
            axes.append([_AXIS_TO_INDEX.get(axis, -1) for axis in term.pauli])
            term_weights.append(weight)
    if not axes:
        return (
            np.empty((0, checked[0].n_qubits), dtype=np.int8),
            np.empty(0, dtype=float),
            checked,
        )
    return np.asarray(axes, dtype=np.int8), np.asarray(term_weights), checked


def _loss_from_arrays(
    probabilities: np.ndarray, term_axes: np.ndarray, term_weights: np.ndarray
) -> float:
    total = 0.0
    for axes, weight in zip(term_axes, term_weights):
        coverage = 1.0
        for qubit, axis in enumerate(axes):
            if axis >= 0:
                coverage *= probabilities[qubit, axis]
        total += weight / coverage
    return float(total)


def planning_loss(
    probabilities: Sequence[Sequence[float]],
    observables: PauliObservable | Sequence[PauliObservable],
    *,
    observable_weights: Sequence[float] | None = None,
) -> float:
    """Evaluate the diagonal second-moment proxy used by the optimiser."""

    term_axes, term_weights, checked = _weighted_terms(
        observables, observable_weights
    )
    checked_probabilities = _validate_probabilities(
        probabilities, n_qubits=checked[0].n_qubits
    )
    return _loss_from_arrays(checked_probabilities, term_axes, term_weights)


@dataclass(frozen=True, slots=True)
class BasisOptimizationResult:
    """Result and convergence diagnostics from basis-probability optimisation."""

    probabilities: np.ndarray
    initial_loss: float
    final_loss: float
    iterations: int
    converged: bool
    loss_history: tuple[float, ...]
    probability_floor: float



def _floor_constrained_inverse_cost(costs: np.ndarray, floor: float) -> np.ndarray:
    """Solve ``min sum_a costs[a]/p[a]`` on a floor-constrained simplex."""

    result: np.ndarray = np.full(3, floor, dtype=float)
    free = [index for index, cost in enumerate(costs) if cost > 0.0]
    if not free:
        return np.full(3, 1.0 / 3.0)

    while free:
        fixed_count = 3 - len(free)
        remaining = 1.0 - fixed_count * floor
        roots: np.ndarray = np.sqrt(costs[free])
        proposal = remaining * roots / roots.sum()
        below_floor = [
            index for index, value in zip(free, proposal) if value < floor
        ]
        if not below_floor:
            result[free] = proposal
            break
        free = [index for index in free if index not in below_floor]

    # Absorb round-off on the largest entry while preserving the floor.
    result[int(np.argmax(result))] += 1.0 - float(result.sum())
    return result


def optimize_basis_probabilities(
    observables: PauliObservable | Sequence[PauliObservable],
    *,
    observable_weights: Sequence[float] | None = None,
    probability_floor: float = 0.02,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
    initial_probabilities: Sequence[Sequence[float]] | None = None,
) -> BasisOptimizationResult:
    """Optimise independent local Pauli-basis probabilities.

    Each coordinate update exactly minimises the proxy with the other qubits
    fixed.  ``probability_floor`` keeps every basis observable and bounds inverse
    propensities during post-processing.
    """

    if not 0.0 < probability_floor < 1.0 / 3.0:
        raise ValueError("probability_floor must lie strictly between 0 and 1/3")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive number")

    term_axes, term_weights, checked = _weighted_terms(
        observables, observable_weights
    )
    n_qubits = checked[0].n_qubits
    probabilities = (
        uniform_probabilities(n_qubits)
        if initial_probabilities is None
        else _validate_probabilities(
            initial_probabilities,
            n_qubits=n_qubits,
            probability_floor=probability_floor,
        )
    )
    initial_loss = _loss_from_arrays(probabilities, term_axes, term_weights)
    history = [initial_loss]
    if term_axes.shape[0] == 0:
        return BasisOptimizationResult(
            probabilities=probabilities,
            initial_loss=0.0,
            final_loss=0.0,
            iterations=0,
            converged=True,
            loss_history=(0.0,),
            probability_floor=probability_floor,
        )

    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        previous_loss = history[-1]
        for qubit in range(n_qubits):
            costs: np.ndarray = np.zeros(3, dtype=float)
            for axes, weight in zip(term_axes, term_weights):
                selected_axis = int(axes[qubit])
                if selected_axis < 0:
                    continue
                other_coverage = 1.0
                for other_qubit, axis in enumerate(axes):
                    if other_qubit != qubit and axis >= 0:
                        other_coverage *= probabilities[other_qubit, axis]
                costs[selected_axis] += weight / other_coverage
            probabilities[qubit] = _floor_constrained_inverse_cost(
                costs, probability_floor
            )

        current_loss = _loss_from_arrays(probabilities, term_axes, term_weights)
        history.append(current_loss)
        iterations = iteration
        improvement = previous_loss - current_loss
        if improvement < -1e-8 * max(1.0, abs(previous_loss)):
            raise ArithmeticError("coordinate update unexpectedly increased planning loss")
        if improvement <= tolerance * max(1.0, abs(previous_loss)):
            converged = True
            break

    return BasisOptimizationResult(
        probabilities=probabilities.copy(),
        initial_loss=initial_loss,
        final_loss=history[-1],
        iterations=iterations,
        converged=converged,
        loss_history=tuple(history),
        probability_floor=probability_floor,
    )


@dataclass(frozen=True, slots=True)
class MeasurementPlan:
    """A reproducible list of q0-first local-Pauli measurement bases.

    ``probabilities`` is one static ``(n_qubits, 3)`` distribution shared by
    every shot in this plan.  To use a different distribution in a later
    round, create and execute another plan, then concatenate the resulting
    :class:`~pyqpanda_alg.QShadow.dataset.ShadowDataset` objects.  The datasets
    retain the correct distribution for every shot.
    """

    bases: tuple[str, ...]
    probabilities: np.ndarray
    seed: int | None = None

    def __post_init__(self) -> None:
        bases = tuple(self.bases)
        if not bases:
            raise ValueError("a measurement plan must contain at least one shot")
        n_qubits = len(bases[0])
        if n_qubits == 0:
            raise ValueError("basis strings cannot be empty")
        for basis in bases:
            if len(basis) != n_qubits or set(basis) - set(AXES):
                raise ValueError("every basis must be an equal-length X/Y/Z string")
        probabilities = _validate_probabilities(
            self.probabilities, n_qubits=n_qubits
        )
        object.__setattr__(self, "bases", bases)
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def shots(self) -> int:
        return len(self.bases)

    @property
    def n_qubits(self) -> int:
        return len(self.bases[0])

    @property
    def grouped_shots(self) -> dict[str, int]:
        """Return identical basis strings compressed into backend jobs."""

        return dict(Counter(self.bases))


def sample_measurement_plan(
    probabilities: Sequence[Sequence[float]],
    shots: int,
    *,
    seed: int | None = None,
) -> MeasurementPlan:
    """Sample a local-Pauli plan from independent per-qubit distributions."""

    checked = _validate_probabilities(probabilities)
    if not isinstance(shots, int) or isinstance(shots, bool):
        raise TypeError("shots must be an integer")
    if shots <= 0:
        raise ValueError("shots must be positive")
    rng = np.random.default_rng(seed)
    sampled: np.ndarray = np.empty((shots, checked.shape[0]), dtype=np.int8)
    for qubit in range(checked.shape[0]):
        sampled[:, qubit] = rng.choice(3, size=shots, p=checked[qubit])
    bases = tuple("".join(AXES[index] for index in row) for row in sampled)
    return MeasurementPlan(bases=bases, probabilities=checked, seed=seed)
