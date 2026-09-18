# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Budgeted, counts-only assignment for the existing two-feature SWAP encoding.

This optional decision layer does not alter ``QuantumKmeans.fit``. It selects
among measured ancilla-one probabilities, not general Euclidean distances.
The caller supplies the measurement runner; no backend or exact-probability
fallback is selected here.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log, pi, sqrt
from numbers import Integral, Real
from typing import Callable, Iterable, Optional, Tuple

from pyqpanda3.core import CNOT, H, QProg, T, U3, measure

__all__ = ["AssignmentResult", "adaptive_assign", "build_assignment_program"]


@dataclass(frozen=True)
class AssignmentResult:
    """Immutable snapshot, with tuples indexed by original candidate order.

    ``winner`` is an index only for ``separated`` or ``single_candidate``.
    Unmeasured candidates have mean ``None`` and interval ``(0, 1)``.
    Eliminated candidates retain their last cumulative statistics; ``active``
    contains the surviving indices, not a ranking or an estimated winner.
    """
    status: str
    winner: Optional[int]
    means: Tuple[Optional[float], ...]
    intervals: Tuple[Tuple[float, float], ...]
    shots_per_centroid: Tuple[int, ...]
    total_shots: int
    active: Tuple[int, ...]


def _point_snapshot(values, name):
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must contain two real coordinates")
    try:
        coordinates = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must contain two real coordinates") from exc
    if len(coordinates) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    for value in coordinates:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} coordinates must be real numbers, not bools")
        if not -1 <= value <= 1 or not isfinite(value):
            raise ValueError(f"{name} coordinates must be finite and in [-1, 1]")
    return tuple(float(value) for value in coordinates)


def _shot_count(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a bool")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _append_fredkin(program, control, left, right):
    # CX(right,left) CCX(control,left,right) CX(right,left).
    # The CCX is the standard exact (not relative-phase) H/T/CX decomposition.
    # Only single-qubit gates and singly controlled NOT gates are emitted.
    program << CNOT(right, left)
    program << H(right) << CNOT(left, right) << T(right).dagger()
    program << CNOT(control, right) << T(right) << CNOT(left, right)
    program << T(right).dagger() << CNOT(control, right)
    program << T(left) << T(right) << H(right)
    program << CNOT(control, left) << T(control) << T(left).dagger()
    program << CNOT(control, left)
    program << CNOT(right, left)


def build_assignment_program(point: Iterable[Real], centroid: Iterable[Real]) -> QProg:
    """Build a fresh three-qubit SWAP-test program, measuring only q2 into c0.

    Each input is two finite real coordinates ``(x, y)`` in ``[-1, 1]``.
    On q0/q1, prepare ``U3((y+1)*pi/2, (x+1)*pi/2, 0)`` as in the legacy
    implementation. The controlled SWAP uses exact H/T/CNOT decomposition.
    The caller owns the returned program. It needs a fresh |000> preparation
    on every shot, standard gate/dagger support, and only c0 in its counts.
    """
    point = _point_snapshot(point, "point")
    centroid = _point_snapshot(centroid, "centroid")
    program = QProg(3)
    program << H(2)
    for qubit, (x, y) in enumerate((point, centroid)):
        program << U3(qubit, (y + 1) * pi / 2, (x + 1) * pi / 2, 0)
    _append_fredkin(program, 2, 0, 1)
    program << H(2) << measure(2, 0)
    return program


def _ones_from_counts(counts, shots):
    if not isinstance(counts, Mapping):
        raise TypeError("run_counts must return a mapping of single-bit counts")
    total = 0
    ones = 0
    for bit, count in counts.items():
        if not isinstance(bit, str) or bit not in ("0", "1"):
            raise ValueError("count keys must be '0' or '1' for c0 only")
        count = _shot_count(count, "count", 0)
        total += count
        if bit == "1":
            ones = count
    if total != shots:
        raise ValueError("counts must sum exactly to the requested shots")
    return ones


def adaptive_assign(
    point: Iterable[Real],
    centroids: Iterable[Iterable[Real]],
    run_counts: Callable[[QProg, int], Mapping],
    *,
    batch_shots: int = 64,
    max_shots: int = 4096,
    alpha: float = 0.05,
) -> AssignmentResult:
    """Eliminate separated candidates using cumulative fixed-batch counts.

    ``max_shots`` is a global hard cap on requested shots, not a per-candidate
    cap. Active candidates are visited in original-index round-robin order.
    Never request a partial batch. A runner failure or invalid counts raises
    immediately, without retrying or falling back to another backend.

    ``run_counts(program, shots)`` must return a mapping whose only possible
    keys are ``'0'`` and ``'1'`` and whose nonnegative integer (non-bool) values
    sum exactly to ``shots``. Missing outcomes count as zero. Every callback
    receives a freshly built caller-owned program; input coordinates are
    snapshotted before any callback. Statistics are local to this invocation.

    For K candidates and L = floor(max_shots / batch_shots), use the radius
    sqrt(log(2*K*max(1,L)/alpha)/(2*n)) at each candidate's fixed sample sizes
    n = j*batch_shots. The union bound covers all potential K*L looks, so
    adaptive elimination is valid assuming independent stationary Bernoulli
    shots for each fixed candidate. For one assignment, the probability of
    returning an incorrect separated winner is at most alpha. This is not
    a bound conditional on having returned a winner, nor a hardware-fidelity
    guarantee or a statement about Euclidean clustering quality.
    Alpha covers one assignment; multiple points/iterations need a separate
    error-budget allocation. See docs/qkmeans_adaptive_assignment.md.
    """
    point = _point_snapshot(point, "point")
    if isinstance(centroids, (str, bytes, Mapping)):
        raise TypeError("centroids must be a nonempty iterable of points")
    try:
        centers = tuple(_point_snapshot(c, "centroid") for c in centroids)
    except TypeError as exc:
        raise TypeError("centroids must be a nonempty iterable of real points") from exc
    if not centers:
        raise ValueError("centroids must be nonempty")
    batch_shots = _shot_count(batch_shots, "batch_shots", 1)
    max_shots = _shot_count(max_shots, "max_shots", 0)
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real number, not a bool")
    if not 0 < alpha < 1 or not isfinite(alpha):
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be representable as a float strictly between 0 and 1")
    if not callable(run_counts):
        raise TypeError("run_counts must be callable")

    k = len(centers)
    ones = [0] * k
    shots = [0] * k
    means = [None] * k
    intervals = [(0.0, 1.0)] * k
    active = tuple(range(k))
    total = 0

    def result(status, winner=None):
        return AssignmentResult(status, winner, tuple(means), tuple(intervals),
                                tuple(shots), total, active)

    if k == 1:
        return result("single_candidate", 0)
    looks = max_shots // batch_shots
    # Log-space arithmetic avoids overflowing the ratio for very small alpha.
    log_factor = log(2) + log(k) + log(max(1, looks)) - log(alpha)
    while total + batch_shots <= max_shots:
        for index in active:
            # The round's tuple is fixed; an earlier look may eliminate a
            # candidate that has not yet received its turn in this round.
            if index not in active:
                continue
            if total + batch_shots > max_shots:
                return result("undecided")
            program = build_assignment_program(point, centers[index])
            observed = _ones_from_counts(run_counts(program, batch_shots), batch_shots)
            ones[index] += observed
            shots[index] += batch_shots
            total += batch_shots
            mean = ones[index] / shots[index]
            radius = sqrt(log_factor / (2 * shots[index]))
            means[index] = mean
            intervals[index] = (max(0.0, mean - radius), min(1.0, mean + radius))
            best_upper = min(intervals[i][1] for i in active)
            active = tuple(i for i in active if intervals[i][0] <= best_upper)
            if len(active) == 1:
                return result("separated", active[0])
    return result("undecided")
