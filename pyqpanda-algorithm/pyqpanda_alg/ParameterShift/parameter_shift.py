"""Parameter-shift differentiation utilities for variational quantum programs.

The module is backend-agnostic: callers provide an evaluator that maps a
parameter vector to a scalar or vector observable. This keeps differentiation
usable with CPUQVM, QCloud, or batched remote execution.
"""

from dataclasses import dataclass
from math import pi
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class ShiftTerm:
    """One weighted shifted evaluation in a parameter-shift rule."""

    coefficient: float
    shift: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.coefficient):
            raise ValueError("shift coefficient must be finite")
        if not np.isfinite(self.shift):
            raise ValueError("shift must be finite")


@dataclass(frozen=True)
class ParameterShiftRule:
    """A linear rule that reconstructs one partial derivative."""

    terms: Tuple[ShiftTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("a parameter-shift rule needs at least one term")

    @classmethod
    def two_eigenvalue(cls, frequency: float = 1.0) -> "ParameterShiftRule":
        """Return the exact two-eigenvalue parameter-shift rule.

        frequency is the positive spectral gap of the generator in the
        convention U(theta) = exp(-i * theta * G). For RX/RY/RZ gates, whose
        generator is P/2, the gap is one and this becomes

            df/dtheta = (f(theta + pi/2) - f(theta - pi/2)) / 2.
        """

        frequency = float(frequency)
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be a positive finite number")
        shift = pi / (2.0 * frequency)
        coefficient = frequency / 2.0
        return cls(
            (
                ShiftTerm(coefficient=coefficient, shift=shift),
                ShiftTerm(coefficient=-coefficient, shift=-shift),
            )
        )


DEFAULT_RULE = ParameterShiftRule.two_eigenvalue()


@dataclass(frozen=True)
class ShiftedEvaluation:
    """One point that must be evaluated to reconstruct a derivative."""

    parameter_index: int
    term_index: int
    coefficient: float
    parameters: Tuple[float, ...]


@dataclass(frozen=True)
class ParameterShiftSchedule:
    """Serializable evaluation schedule suitable for local or remote batching."""

    parameter_count: int
    entries: Tuple[ShiftedEvaluation, ...]

    @property
    def points(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(entry.parameters for entry in self.entries)

    def reconstruct(self, evaluations: Sequence[object]) -> np.ndarray:
        """Reconstruct a full Jacobian from results in schedule order.

        Scalar evaluator outputs produce shape (n_parameters,). Vector or
        tensor outputs produce shape (n_parameters, *output_shape).
        Parameters omitted from trainable are represented by zero rows.
        """

        if len(evaluations) != len(self.entries):
            raise ValueError(
                "evaluation count does not match the parameter-shift schedule"
            )
        if not self.entries:
            return np.zeros((self.parameter_count,), dtype=float)

        values = [np.asarray(value) for value in evaluations]
        output_shape = values[0].shape
        if any(value.shape != output_shape for value in values):
            raise ValueError("all evaluator outputs must have the same shape")
        if any(not np.all(np.isfinite(value)) for value in values):
            raise ValueError("evaluator outputs must be finite")

        dtype = np.result_type(float, *(value.dtype for value in values))
        result = np.zeros((self.parameter_count,) + output_shape, dtype=dtype)
        for entry, value in zip(self.entries, values):
            result[entry.parameter_index] += entry.coefficient * value
        return result


def _normalize_parameters(parameters: ArrayLike) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.ndim != 1:
        raise ValueError("parameters must be a one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must be finite")
    return values


def _normalize_trainable(
    trainable: Optional[Iterable[int]], parameter_count: int
) -> Tuple[int, ...]:
    if trainable is None:
        return tuple(range(parameter_count))

    indices = tuple(int(index) for index in trainable)
    if len(set(indices)) != len(indices):
        raise ValueError("trainable parameter indices must be unique")
    if any(index < 0 or index >= parameter_count for index in indices):
        raise IndexError("trainable parameter index is out of range")
    return indices


def _normalize_rules(
    rules: Optional[Union[ParameterShiftRule, Sequence[ParameterShiftRule]]],
    parameter_count: int,
) -> Tuple[ParameterShiftRule, ...]:
    if rules is None:
        return tuple(DEFAULT_RULE for _ in range(parameter_count))
    if isinstance(rules, ParameterShiftRule):
        return tuple(rules for _ in range(parameter_count))

    normalized = tuple(rules)
    if len(normalized) != parameter_count:
        raise ValueError("rules must provide one entry per parameter")
    if any(not isinstance(rule, ParameterShiftRule) for rule in normalized):
        raise TypeError("every rule must be a ParameterShiftRule")
    return normalized


def build_shift_schedule(
    parameters: ArrayLike,
    rules: Optional[Union[ParameterShiftRule, Sequence[ParameterShiftRule]]] = None,
    trainable: Optional[Iterable[int]] = None,
) -> ParameterShiftSchedule:
    """Build shifted parameter vectors without executing a backend."""

    values = _normalize_parameters(parameters)
    indices = _normalize_trainable(trainable, len(values))
    normalized_rules = _normalize_rules(rules, len(values))

    entries = []
    for parameter_index in indices:
        rule = normalized_rules[parameter_index]
        for term_index, term in enumerate(rule.terms):
            shifted = values.copy()
            shifted[parameter_index] += term.shift
            entries.append(
                ShiftedEvaluation(
                    parameter_index=parameter_index,
                    term_index=term_index,
                    coefficient=term.coefficient,
                    parameters=tuple(float(value) for value in shifted),
                )
            )

    return ParameterShiftSchedule(
        parameter_count=len(values),
        entries=tuple(entries),
    )


def evaluate_shift_schedule(
    evaluator: Callable[[np.ndarray], object],
    schedule: ParameterShiftSchedule,
) -> Tuple[object, ...]:
    """Evaluate a schedule locally, preserving exact schedule order."""

    return tuple(
        evaluator(np.asarray(entry.parameters, dtype=float))
        for entry in schedule.entries
    )


def evaluate_shift_batch(
    batch_evaluator: Callable[[Sequence[Tuple[float, ...]]], Sequence[object]],
    schedule: ParameterShiftSchedule,
) -> Tuple[object, ...]:
    """Submit all shifted points in one backend call."""

    results = tuple(batch_evaluator(schedule.points))
    if len(results) != len(schedule.entries):
        raise ValueError("batch evaluator returned the wrong number of results")
    return results


def parameter_shift_jacobian(
    evaluator: Callable[[np.ndarray], object],
    parameters: ArrayLike,
    rules: Optional[Union[ParameterShiftRule, Sequence[ParameterShiftRule]]] = None,
    trainable: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Return the parameter-shift Jacobian for a scalar/vector evaluator."""

    schedule = build_shift_schedule(parameters, rules=rules, trainable=trainable)
    evaluations = evaluate_shift_schedule(evaluator, schedule)
    return schedule.reconstruct(evaluations)


def parameter_shift_gradient(
    evaluator: Callable[[np.ndarray], object],
    parameters: ArrayLike,
    rules: Optional[Union[ParameterShiftRule, Sequence[ParameterShiftRule]]] = None,
    trainable: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Return the parameter-shift gradient for a scalar evaluator."""

    gradient = parameter_shift_jacobian(
        evaluator,
        parameters,
        rules=rules,
        trainable=trainable,
    )
    if gradient.ndim != 1:
        raise ValueError(
            "parameter_shift_gradient requires a scalar evaluator; "
            "use parameter_shift_jacobian for vector outputs"
        )
    return gradient


def parameter_shift_vjp(
    evaluator: Callable[[np.ndarray], object],
    parameters: ArrayLike,
    cotangent: ArrayLike,
    rules: Optional[Union[ParameterShiftRule, Sequence[ParameterShiftRule]]] = None,
    trainable: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Return a vector-Jacobian product without materializing the Jacobian."""

    cotangent_array = np.asarray(cotangent)
    if not np.all(np.isfinite(cotangent_array)):
        raise ValueError("cotangent must be finite")

    def contracted_evaluator(point: np.ndarray) -> object:
        value = np.asarray(evaluator(point))
        if value.shape != cotangent_array.shape:
            raise ValueError("cotangent shape must match evaluator output shape")
        return np.sum(cotangent_array * value)

    return parameter_shift_gradient(
        contracted_evaluator,
        parameters,
        rules=rules,
        trainable=trainable,
    )
