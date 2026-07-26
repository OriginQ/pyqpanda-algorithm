"""Unbiased Pauli-shadow estimators and confidence-budget control."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, sqrt
from typing import Mapping, Sequence

import numpy as np

from .dataset import ShadowDataset
from .observable import PauliObservable, PauliTerm, coerce_observables
from .planner import _AXIS_TO_INDEX


def term_shadow_samples(dataset: ShadowDataset, term: PauliTerm) -> np.ndarray:
    """Return one inverse-propensity shadow estimate per shot for ``term``."""

    if term.n_qubits != dataset.n_qubits:
        raise ValueError("term and dataset qubit counts differ")
    if term.is_identity:
        return np.full(dataset.shots, term.coefficient, dtype=float)

    support: np.ndarray = np.asarray(term.support, dtype=int)
    required_axes: np.ndarray = np.asarray([_AXIS_TO_INDEX[term.pauli[qubit]] for qubit in support], dtype=int)
    matches: np.ndarray = np.all(dataset.bases[:, support] == required_axes, axis=1)
    coverage: np.ndarray = np.ones(dataset.shots, dtype=float)
    signs: np.ndarray = np.ones(dataset.shots, dtype=float)
    for qubit, axis in zip(support, required_axes):
        coverage *= dataset.probabilities[:, qubit, axis]
        signs *= dataset.outcomes[:, qubit]
    samples: np.ndarray = np.zeros(dataset.shots, dtype=float)
    samples[matches] = term.coefficient * signs[matches] / coverage[matches]
    return samples


def observable_shadow_samples(
    dataset: ShadowDataset, observable: PauliObservable
) -> np.ndarray:
    """Return per-shot unbiased estimates of a Pauli observable."""

    if observable.n_qubits != dataset.n_qubits:
        raise ValueError("observable and dataset qubit counts differ")
    samples: np.ndarray = np.zeros(dataset.shots, dtype=float)
    for term in observable.terms:
        samples += term_shadow_samples(dataset, term)
    return samples


def _observable_value_range(
    dataset: ShadowDataset, observable: PauliObservable
) -> float:
    """Conservative common range width for all per-shot estimators."""

    absolute_bound: np.ndarray = np.zeros(dataset.shots, dtype=float)
    for term in observable.terms:
        if term.is_identity or term.coefficient == 0.0:
            continue
        inverse_coverage: np.ndarray = np.ones(dataset.shots, dtype=float)
        for qubit in term.support:
            axis = _AXIS_TO_INDEX[term.pauli[qubit]]
            inverse_coverage /= dataset.probabilities[:, qubit, axis]
        absolute_bound += abs(term.coefficient) * inverse_coverage
    return float(2.0 * np.max(absolute_bound, initial=0.0))


def empirical_bernstein_radius(
    samples: Sequence[float] | np.ndarray,
    *,
    value_range: float,
    failure_probability: float,
) -> float:
    """Return a conservative two-sided empirical Bernstein half-width.

    The implemented form is

    ``sqrt(2 s^2 log(3/delta) / n) + 3 R log(3/delta) / n``,

    where ``s^2`` is the unbiased sample variance and ``R`` is a known range
    width.  The result is capped by ``R``, the deterministic worst-case mean
    deviation.  It is intentionally conservative for high-weight Pauli sums.
    """

    values: np.ndarray = np.asarray(samples, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    if not np.isfinite(value_range) or value_range < 0:
        raise ValueError("value_range must be finite and non-negative")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie between zero and one")
    if value_range == 0.0:
        return 0.0
    if values.size == 1:
        return float(value_range)
    variance = float(np.var(values, ddof=1))
    logarithm = log(3.0 / failure_probability)
    radius = sqrt(2.0 * variance * logarithm / values.size)
    radius += 3.0 * value_range * logarithm / values.size
    return float(min(value_range, radius))


@dataclass(frozen=True, slots=True)
class ShadowEstimate:
    """Point estimate, diagnostics, and a finite-sample confidence interval."""

    observable: str
    value: float
    standard_error: float
    lower: float
    upper: float
    shots: int
    empirical_variance: float
    value_range: float
    failure_probability: float
    minimum_term_matches: int

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0

    @property
    def confidence(self) -> float:
        return 1.0 - self.failure_probability


def _minimum_term_matches(
    dataset: ShadowDataset, observable: PauliObservable
) -> int:
    counts: list[int] = []
    for term in observable.terms:
        if term.is_identity or term.coefficient == 0.0:
            continue
        matched: np.ndarray = np.ones(dataset.shots, dtype=bool)
        for qubit in term.support:
            matched &= dataset.bases[:, qubit] == _AXIS_TO_INDEX[term.pauli[qubit]]
        counts.append(int(matched.sum()))
    return min(counts, default=dataset.shots)


def estimate_observable(
    dataset: ShadowDataset,
    observable: PauliObservable,
    *,
    confidence: float = 0.95,
    failure_probability: float | None = None,
) -> ShadowEstimate:
    """Estimate one observable from any compatible QShadow dataset."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")
    delta = 1.0 - confidence if failure_probability is None else failure_probability
    if not 0.0 < delta < 1.0:
        raise ValueError("failure_probability must lie between zero and one")
    samples = observable_shadow_samples(dataset, observable)
    value = float(np.mean(samples))
    variance = float(np.var(samples, ddof=1)) if dataset.shots > 1 else 0.0
    standard_error = sqrt(variance / dataset.shots)
    value_range = _observable_value_range(dataset, observable)
    radius = empirical_bernstein_radius(
        samples, value_range=value_range, failure_probability=delta
    )
    return ShadowEstimate(
        observable=observable.name,
        value=value,
        standard_error=standard_error,
        lower=value - radius,
        upper=value + radius,
        shots=dataset.shots,
        empirical_variance=variance,
        value_range=value_range,
        failure_probability=delta,
        minimum_term_matches=_minimum_term_matches(dataset, observable),
    )


def estimate_observables(
    dataset: ShadowDataset,
    observables: PauliObservable | Sequence[PauliObservable],
    *,
    family_confidence: float = 0.95,
) -> dict[str, ShadowEstimate]:
    """Estimate many observables with Bonferroni simultaneous confidence."""

    checked = coerce_observables(observables)
    if any(observable.n_qubits != dataset.n_qubits for observable in checked):
        raise ValueError("observable and dataset qubit counts differ")
    if not 0.0 < family_confidence < 1.0:
        raise ValueError("family_confidence must lie between zero and one")
    per_observable_delta = (1.0 - family_confidence) / len(checked)
    return {
        observable.name: estimate_observable(
            dataset,
            observable,
            failure_probability=per_observable_delta,
        )
        for observable in checked
    }


def _required_total_shots(
    *,
    variance: float,
    value_range: float,
    failure_probability: float,
    target_half_width: float,
) -> int:
    if value_range == 0.0:
        return 1
    logarithm = log(3.0 / failure_probability)
    linear = sqrt(max(0.0, 2.0 * variance * logarithm))
    constant = 3.0 * value_range * logarithm
    root_n = (
        linear + sqrt(linear * linear + 4.0 * target_half_width * constant)
    ) / (2.0 * target_half_width)
    return max(2, ceil(root_n * root_n))


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """Decision returned by :class:`ShadowBudgetController`."""

    estimates: Mapping[str, ShadowEstimate]
    complete: bool
    exhausted: bool
    next_batch_shots: int
    reason: str
    current_shots: int
    max_shots: int


class ShadowBudgetController:
    """Translate simultaneous confidence widths into a bounded shot budget.

    The controller does not silently submit jobs.  Callers evaluate a dataset,
    inspect :class:`BudgetReport`, explicitly collect ``next_batch_shots`` more
    records, concatenate them, and evaluate again.  This makes paid QPU use
    auditable and keeps a hard ``max_shots`` ceiling.
    """

    def __init__(
        self,
        observables: PauliObservable | Sequence[PauliObservable],
        *,
        target_half_width: float | Mapping[str, float],
        family_confidence: float = 0.95,
        min_shots: int = 100,
        max_shots: int = 10_000,
        batch_shots: int = 200,
    ) -> None:
        self.observables = coerce_observables(observables)
        if isinstance(target_half_width, Mapping):
            targets = {name: float(value) for name, value in target_half_width.items()}
            expected_names = {observable.name for observable in self.observables}
            if set(targets) != expected_names:
                raise ValueError("target_half_width mapping must match observable names")
        else:
            value = float(target_half_width)
            targets = {observable.name: value for observable in self.observables}
        if any(not np.isfinite(value) or value <= 0 for value in targets.values()):
            raise ValueError("target half-widths must be finite and positive")
        if not 0.0 < family_confidence < 1.0:
            raise ValueError("family_confidence must lie between zero and one")
        for name, value in (
            ("min_shots", min_shots),
            ("max_shots", max_shots),
            ("batch_shots", batch_shots),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if min_shots > max_shots:
            raise ValueError("min_shots cannot exceed max_shots")
        self.targets = targets
        self.family_confidence = family_confidence
        self.min_shots = min_shots
        self.max_shots = max_shots
        self.batch_shots = batch_shots

    def evaluate(self, dataset: ShadowDataset) -> BudgetReport:
        """Assess current confidence widths without performing new measurements."""

        if dataset.n_qubits != self.observables[0].n_qubits:
            raise ValueError("dataset and controller qubit counts differ")
        estimates = estimate_observables(
            dataset, self.observables, family_confidence=self.family_confidence
        )
        widths_reached = all(
            estimate.half_width <= self.targets[name]
            for name, estimate in estimates.items()
        )
        complete = dataset.shots >= self.min_shots and widths_reached
        exhausted = dataset.shots >= self.max_shots and not complete
        if complete:
            return BudgetReport(
                estimates, True, False, 0, "target_reached", dataset.shots, self.max_shots
            )
        if exhausted:
            return BudgetReport(
                estimates, False, True, 0, "max_shots_reached", dataset.shots, self.max_shots
            )

        required = self.min_shots
        for name, estimate in estimates.items():
            required = max(
                required,
                _required_total_shots(
                    variance=estimate.empirical_variance,
                    value_range=estimate.value_range,
                    failure_probability=estimate.failure_probability,
                    target_half_width=self.targets[name],
                ),
            )
        remaining = self.max_shots - dataset.shots
        suggested = max(self.batch_shots, required - dataset.shots)
        next_batch = min(remaining, suggested)
        return BudgetReport(
            estimates,
            False,
            False,
            next_batch,
            "more_shots_required",
            dataset.shots,
            self.max_shots,
        )
