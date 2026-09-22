"""Problem definition and exact classical reference for QmRMR.

The implementation uses the signed linear coefficient directly:

    f(x) = x.T @ Q @ x + linear.T @ x

For the conventional minimum-redundancy maximum-relevance objective, callers
should pass ``linear=-relevance``. Keeping the sign in the supplied vector
makes the objective used by the solver explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Literal, Sequence

import numpy as np


ObjectiveConvention = Literal["signed_linear", "canonical_mrmr", "plus_relevance"]


@dataclass(frozen=True)
class ExactSolution:
    """Exact fixed-cardinality solution obtained by enumeration."""

    optimal_value: float
    optimal_states: tuple[tuple[int, ...], ...]
    values: dict[str, float]


@dataclass
class QmRMRProblem:
    """A binary quadratic feature-selection problem with exact cardinality."""

    quadratic: np.ndarray | Sequence[Sequence[float]]
    linear: np.ndarray | Sequence[float]
    select_num: int
    objective_convention: ObjectiveConvention | str = "signed_linear"

    def __post_init__(self) -> None:
        q = np.asarray(self.quadratic, dtype=float)
        u = np.asarray(self.linear, dtype=float).reshape(-1)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError("quadratic must be a square matrix")
        if q.shape[0] != u.size:
            raise ValueError("quadratic and linear dimensions must agree")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(u)):
            raise ValueError("quadratic and linear must contain finite values")
        if not isinstance(self.select_num, (int, np.integer)):
            raise TypeError("select_num must be an integer")
        if not 0 <= int(self.select_num) <= q.shape[0]:
            raise ValueError("select_num must satisfy 0 <= select_num <= n_features")
        convention = str(self.objective_convention).lower()
        if convention not in {"signed_linear", "canonical_mrmr", "plus_relevance"}:
            raise ValueError(
                "objective_convention must be 'signed_linear', "
                "'canonical_mrmr', or 'plus_relevance'"
            )

        # x.T @ Q @ x depends only on the symmetric part of Q.
        self.quadratic = 0.5 * (q + q.T)
        self.linear = u
        self.select_num = int(self.select_num)
        self.objective_convention = convention

    @classmethod
    def from_mrmr(
        cls,
        quadratic: np.ndarray | Sequence[Sequence[float]],
        relevance: np.ndarray | Sequence[float],
        select_num: int,
        *,
        convention: Literal["canonical_mrmr", "plus_relevance"] = "canonical_mrmr",
    ) -> "QmRMRProblem":
        """Build a problem from the positive QmRMR relevance vector.

        ``canonical_mrmr`` represents the usual redundancy-minus-relevance
        objective. ``plus_relevance`` is the alternative plus-sign convention
        and is useful when comparing formulations that use a signed relevance
        coefficient.
        """

        relevance_array = np.asarray(relevance, dtype=float).reshape(-1)
        if convention == "canonical_mrmr":
            linear = -relevance_array
        elif convention == "plus_relevance":
            linear = relevance_array
        else:
            raise ValueError(
                "convention must be 'canonical_mrmr' or 'plus_relevance'"
            )
        return cls(
            quadratic,
            linear,
            select_num,
            objective_convention=convention,
        )

    @property
    def objective_label(self) -> str:
        """Return a short human-readable expression label."""

        labels = {
            "signed_linear": "x.T @ Q @ x + linear.T @ x",
            "canonical_mrmr": "x.T @ Q @ x - relevance.T @ x",
            "plus_relevance": "x.T @ Q @ x + relevance.T @ x",
        }
        return labels[self.objective_convention]

    @property
    def relevance_vector(self) -> np.ndarray:
        """Return the positive relevance vector associated with this model."""

        if self.objective_convention == "plus_relevance":
            return self.linear.copy()
        return -self.linear.copy()

    @property
    def n_features(self) -> int:
        return int(self.linear.size)

    def validate_bits(self, bits: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return a validated binary vector in feature-index order."""

        x = np.asarray(bits, dtype=int).reshape(-1)
        if x.size != self.n_features:
            raise ValueError("bit vector length does not match n_features")
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("bit vector must contain only 0 and 1")
        if int(x.sum()) != self.select_num:
            raise ValueError("bit vector does not satisfy the cardinality constraint")
        return x

    @staticmethod
    def state_key(bits: Sequence[int] | np.ndarray) -> str:
        return "".join(str(int(bit)) for bit in bits)

    @staticmethod
    def bits_from_key(key: str) -> tuple[int, ...]:
        if not isinstance(key, str) or not key or any(char not in "01" for char in key):
            raise ValueError("state key must be a non-empty binary string")
        return tuple(int(char) for char in key)

    def objective(self, bits: Sequence[int] | np.ndarray) -> float:
        """Evaluate the signed binary quadratic objective."""

        x = self.validate_bits(bits)
        return float(x @ self.quadratic @ x + self.linear @ x)

    def objective_unconstrained(self, bits: Sequence[int] | np.ndarray) -> float:
        """Evaluate the objective without enforcing the cardinality constraint."""

        x = np.asarray(bits, dtype=float).reshape(-1)
        if x.size != self.n_features:
            raise ValueError("bit vector length does not match n_features")
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("bit vector must contain only 0 and 1")
        return float(x @ self.quadratic @ x + self.linear @ x)

    def iter_feasible_states(self) -> Iterator[tuple[tuple[int, ...], float]]:
        """Yield all feasible states and their exact objective values."""

        for selected in combinations(range(self.n_features), self.select_num):
            bits = [0] * self.n_features
            for index in selected:
                bits[index] = 1
            state = tuple(bits)
            yield state, self.objective(state)

    def exact_solution(self, atol: float = 1e-12) -> ExactSolution:
        """Enumerate the feasible subspace and return all tied minimizers."""

        entries = list(self.iter_feasible_states())
        if not entries:
            raise ValueError("no feasible state exists")
        optimum = min(value for _, value in entries)
        states = tuple(
            state for state, value in entries if abs(value - optimum) <= atol
        )
        values = {self.state_key(state): float(value) for state, value in entries}
        return ExactSolution(float(optimum), states, values)

    def ising_coefficients(
        self,
    ) -> tuple[float, np.ndarray, dict[tuple[int, int], float]]:
        """Map the binary objective to a diagonal Ising polynomial.

        The returned coefficients satisfy

            f(x) = constant + sum_i z_i Z_i + sum_{i<j} zz_ij Z_i Z_j

        under x_i = (1 - Z_i) / 2. The constant is omitted by QAOA circuit
        construction because it contributes only a global phase.
        """

        q = np.asarray(self.quadratic, dtype=float)
        u = np.asarray(self.linear, dtype=float)
        constant = 0.0
        z = np.zeros(self.n_features, dtype=float)
        zz: dict[tuple[int, int], float] = {}

        for i in range(self.n_features):
            diagonal = q[i, i]
            # x_i = (1 - Z_i) / 2, so both the diagonal quadratic and
            # supplied linear coefficient contribute with the same sign.
            constant += 0.5 * diagonal + 0.5 * u[i]
            z[i] += -0.5 * diagonal - 0.5 * u[i]

        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                weight = 2.0 * q[i, j]
                if weight == 0.0:
                    continue
                constant += 0.25 * weight
                z[i] -= 0.25 * weight
                z[j] -= 0.25 * weight
                zz[(i, j)] = 0.25 * weight

        return float(constant), z, zz


def expectation_from_probabilities(
    problem: QmRMRProblem,
    probabilities: dict[str, float],
    *,
    reverse_key: bool = True,
) -> float:
    """Evaluate a diagonal objective from a pyQPanda probability dictionary.

    pyQPanda3 returns the highest-index qubit on the left of a probability key.
    With qubit i representing feature i, reverse_key=True converts that
    presentation order back to feature-index order.
    """

    total = 0.0
    for key, probability in probabilities.items():
        feature_key = key[::-1] if reverse_key else key
        bits = QmRMRProblem.bits_from_key(feature_key)
        if len(bits) != problem.n_features:
            raise ValueError("probability key length does not match problem size")
        total += float(probability) * problem.objective_unconstrained(bits)
    return float(total)


def cvar_from_probabilities(
    problem: QmRMRProblem,
    probabilities: dict[str, float],
    *,
    alpha: float = 0.25,
    reverse_key: bool = True,
) -> float:
    """Return the lower-tail CVaR for a minimisation distribution.

    The feasible states are sorted by objective value and the lowest
    ``alpha`` fraction of their probability mass is averaged.  The feasible
    mass is normalised before taking the tail, which makes the helper useful
    for the fixed-weight circuits in this project and keeps the definition
    independent of tiny simulator leakage outside the constraint sector.
    """

    if not np.isfinite(alpha) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and satisfy 0 < alpha <= 1")

    entries: list[tuple[float, float]] = []
    feasible_mass = 0.0
    for key, probability in probabilities.items():
        p = float(probability)
        if not np.isfinite(p) or p < 0.0:
            raise ValueError("probabilities must be finite and non-negative")
        feature_key = key[::-1] if reverse_key else key
        bits = QmRMRProblem.bits_from_key(feature_key)
        if len(bits) != problem.n_features:
            raise ValueError("probability key length does not match problem size")
        if sum(bits) != problem.select_num:
            continue
        entries.append((problem.objective_unconstrained(bits), p))
        feasible_mass += p

    if not entries or feasible_mass <= 0.0:
        raise RuntimeError("the probability dictionary contains no feasible mass")

    tail_mass = float(alpha) * feasible_mass
    remaining = tail_mass
    weighted_value = 0.0
    for value, probability in sorted(entries, key=lambda item: item[0]):
        if remaining <= 0.0:
            break
        contribution = min(probability, remaining)
        weighted_value += contribution * value
        remaining -= contribution
    return float(weighted_value / tail_mass)

