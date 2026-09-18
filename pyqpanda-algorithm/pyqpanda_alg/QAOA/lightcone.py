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

"""Sampling-only, backward-light-cone QAOA for a Z/ZZ mean objective.

This opt-in executor uses |+> initialization, exp(-i gamma H_cost), and
RX(-2 beta), matching the existing QAOA implementation's default gates.
Only immutable dependency plans are reused. A caller-supplied counts runner
is mandatory; this module never creates a quantum backend or queries its state.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import fsum, isfinite
from numbers import Integral, Real
from types import MappingProxyType

from pyqpanda3.core import CNOT, H, QProg, RX, RZ, measure

__all__ = [
    "QAOALightCone", "ZTerm", "LightConeLayer", "LightConePlan",
    "LightConeProgram", "LightConeEstimate",
]


@dataclass(frozen=True)
class ZTerm:
    """A copied, canonical global support and its finite real coefficient."""

    qubits: tuple[int, ...]
    coefficient: float


@dataclass(frozen=True)
class LightConeLayer:
    """Retained cost terms and mixer qubits, both in global indices."""

    cost_terms: tuple[ZTerm, ...]
    mixer_qubits: tuple[int, ...]


@dataclass(frozen=True)
class LightConePlan:
    """One target's immutable plan; layers are in forward execution order.

    ``global_qubits[local_index]`` is the inverse of ``global_to_local``.
    The plan is structural, not specialized to any gamma or beta value.
    """

    target: ZTerm
    global_qubits: tuple[int, ...]
    layers: tuple[LightConeLayer, ...]

    @property
    def global_to_local(self) -> Mapping[int, int]:
        """Read-only map from original labels to contiguous local indices."""
        return MappingProxyType({q: i for i, q in enumerate(self.global_qubits)})


@dataclass(frozen=True)
class LightConeProgram:
    """A fresh, caller-owned QProg and its immutable interpretation.

    The QProg itself is mutable; it is never stored back into the executor.
    All local qubits are measured to same-index cbits. For width ``w``, a
    counts key must be exactly ``c[w-1]...c[1]c[0]`` (c0 at the right).
    """

    program: QProg
    plan: LightConePlan

    @property
    def local_target(self) -> tuple[int, ...]:
        """The target's qubits after global-to-local remapping."""
        mapping = self.plan.global_to_local
        return tuple(mapping[q] for q in self.plan.target.qubits)

    @property
    def parity_cbits(self) -> tuple[int, ...]:
        """Cbits whose XOR determines the target's +/-1 measurement."""
        return self.local_target

    @property
    def measurement_cbits(self) -> tuple[int, ...]:
        """All measured cbit indices, in increasing order."""
        return tuple(range(len(self.plan.global_qubits)))


@dataclass(frozen=True)
class LightConeEstimate:
    """Sampled mean and shot accounting for one evaluation.

    ``term_means`` contains unweighted parity means in ``executor.plans``
    order. Constants have no entry and require no shots. ``total_shots``
    counts preparations across all programs, not shots of one global circuit.
    """

    mean: float
    term_means: tuple[float, ...]
    shots_per_program: int
    total_shots: int


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite as a float") from error
    if not isfinite(result):
        raise ValueError(f"{name} must be finite as a float")
    return result


def _objective_sum(values: list[float]) -> float:
    """Keep fsum's usual path; rescue finite cancellation after overflow."""
    try:
        result = fsum(values)
    except OverflowError:
        # These are already-rounded finite weighted sample means, not quantum
        # amplitudes. Exact summation here avoids overflowing intermediate sums
        # without scaling away a tiny residual after large cancellation.
        exact = sum((Fraction.from_float(value) for value in values), Fraction(0))
        try:
            result = float(exact)
        except OverflowError as error:
            raise ValueError("sampled objective is outside the finite float range; shots were consumed") from error
    if not isfinite(result):
        raise ValueError("sampled objective is outside the finite float range; shots were consumed")
    return result


def _parameters(values: Iterable[Real], depth: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be an iterable of real numbers")
    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of real numbers") from error
    if len(values) != depth:
        raise ValueError(f"{name} must contain exactly depth={depth} values")
    return tuple(_real(value, f"{name}[{i}]") for i, value in enumerate(values))


def _copy_terms(n_qubits: int, terms: Mapping) -> tuple[ZTerm, ...]:
    if not isinstance(terms, Mapping):
        raise TypeError("terms must map qubit tuples to real coefficients")
    copied = {}
    for support, coefficient in terms.items():
        if not isinstance(support, tuple):
            raise TypeError("each term key must be a tuple of qubit indices")
        if len(support) > 2:
            raise ValueError("only constants, Z fields and ZZ interactions are supported")
        qubits = tuple(_integer(q, "qubit index") for q in support)
        if len(set(qubits)) != len(qubits):
            raise ValueError("a term must not repeat a qubit index")
        if any(q >= n_qubits for q in qubits):
            raise ValueError("qubit indices must be less than n_qubits")
        qubits = tuple(sorted(qubits))
        if qubits in copied:
            raise ValueError("duplicate term after canonicalizing qubit order")
        copied[qubits] = _real(coefficient, "coefficient")
    return tuple(ZTerm(qubits, coefficient) for qubits, coefficient in sorted(copied.items())
                 if coefficient != 0.0)


def _make_plan(target: ZTerm, terms: tuple[ZTerm, ...],
               incident: Mapping[int, list[int]], depth: int) -> LightConePlan:
    support = set(target.qubits)
    backwards = []
    for _ in range(depth):
        # Freeze the entering support before visiting this commuting cost layer.
        # Iteratively growing support while selecting terms would wrongly walk
        # arbitrarily far along a chain within a single layer.
        mixers = tuple(sorted(support))
        selected = sorted({index for q in mixers for index in incident[q]})
        costs = tuple(terms[index] for index in selected)
        backwards.append(LightConeLayer(costs, mixers))
        for term in costs:
            support.update(term.qubits)
    return LightConePlan(target, tuple(sorted(support)), tuple(reversed(backwards)))


def _parity_mean(counts: Mapping[str, int], width: int,
                 parity_cbits: tuple[int, ...], shots: int) -> float:
    if not isinstance(counts, Mapping):
        raise TypeError("run_counts must return a mapping of bitstrings to integer counts")
    total = 0
    signed_total = 0
    for key, count in counts.items():
        if not isinstance(key, str) or len(key) != width or any(bit not in "01" for bit in key):
            raise ValueError(f"counts keys must be {width}-bit binary strings with c0 at the right")
        count = _integer(count, "count")
        total += count
        parity = sum(key[-1 - cbit] == "1" for cbit in parity_cbits) % 2
        signed_total += -count if parity else count
    if total != shots:
        raise ValueError(f"counts sum to {total}, but {shots} shots were requested")
    return signed_total / shots


@dataclass(frozen=True, init=False)
class QAOALightCone:
    """Reusable local sampling plans for ``H = constant + sum a_S Z_S``.

    Parameters
    ----------
    n_qubits : int
        Explicit nonnegative global width. Sparse global labels are allowed.
    terms : Mapping[tuple[int, ...], real]
        ``()`` is the constant; ``(i,)`` is Z_i; ``(i, j)`` is Z_i Z_j.
        Indices must be distinct within a term and in ``range(n_qubits)``.
        Pair order is canonicalized; duplicate canonical supports are rejected.
        Weights are copied as finite floats; zero coefficients are omitted.
        Booleans and complex numbers (even with zero imaginary part) are rejected.
    depth : int
        Nonnegative number of cost-then-mixer layers. At depth zero, nonconstant
        targets are still sampled from local |+> preparations.

    Notes
    -----
    There is one plan per nonzero nonconstant term, in sorted support order.
    No grouping, optimizer, CVaR/Gibbs objective, full bitstring reconstruction,
    backend selection, or expectation-value cache is provided. Changing weights
    or depth requires a new instance; changing angles reuses only the plans.
    """

    n_qubits: int
    terms: tuple[ZTerm, ...]
    depth: int
    constant: float
    plans: tuple[LightConePlan, ...]

    def __init__(self, n_qubits: int, terms: Mapping[tuple[int, ...], Real], depth: int):
        n_qubits = _integer(n_qubits, "n_qubits")
        depth = _integer(depth, "depth")
        copied = _copy_terms(n_qubits, terms)
        nonconstant = tuple(term for term in copied if term.qubits)
        constant = next((term.coefficient for term in copied if not term.qubits), 0.0)
        incident = {}
        for index, term in enumerate(nonconstant):
            for q in term.qubits:
                incident.setdefault(q, []).append(index)
        plans = tuple(_make_plan(target, nonconstant, incident, depth) for target in nonconstant)
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "terms", copied)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "plans", plans)

    def build_programs(self, gammas: Iterable[Real], betas: Iterable[Real]) -> tuple[LightConeProgram, ...]:
        """Build fresh measured local QProgs for exactly ``depth`` angle pairs.

        Each program prepares |+> on every cone qubit, then executes retained
        cost gates (RZ(2 gamma a), or CNOT/RZ/CNOT) and RX(-2 beta) mixers.
        Constants are omitted as global phases. No device is called here.
        All parameters and computed gate angles must be finite real floats.
        """
        gamma_values = _parameters(gammas, self.depth, "gammas")
        beta_values = _parameters(betas, self.depth, "betas")
        mixer_angles = tuple(_real(-2.0 * beta, "mixer angle") for beta in beta_values)
        jobs = []
        for plan in self.plans:
            mapping = plan.global_to_local
            width = len(plan.global_qubits)
            program = QProg(width)
            for q in range(width):
                program << H(q)
            for layer, gamma, mixer_angle in zip(plan.layers, gamma_values, mixer_angles):
                for term in layer.cost_terms:
                    local = tuple(mapping[q] for q in term.qubits)
                    angle = _real((gamma * term.coefficient) * 2.0, "cost angle")
                    if len(local) == 1:
                        program << RZ(local[0], angle)
                    else:
                        program << CNOT(local[0], local[1])
                        program << RZ(local[1], angle)
                        program << CNOT(local[0], local[1])
                for q in layer.mixer_qubits:
                    program << RX(mapping[q], mixer_angle)
            for q in range(width):
                program << measure(q, q)
            jobs.append(LightConeProgram(program, plan))
        return tuple(jobs)

    def estimate(self, gammas: Iterable[Real], betas: Iterable[Real], shots: int,
                 run_counts: Callable[[QProg, int], Mapping[str, int]]) -> LightConeEstimate:
        """Sample the mean objective using only ``run_counts(program, shots)``.

        ``shots`` is a positive integer PER PROGRAM, not a total budget. The
        callback must return nonnegative integer counts summing to exactly
        ``shots``; keys are width-bit strings ``c[w-1]...c[0]``. No probabilities,
        integer keys, whitespace, prefixes, implicit padding, or quasi-counts
        are accepted. Omitted zero-count outcomes are fine.

        The runner must freshly prepare and sample each program. Errors propagate
        without backend fallback or retries. A failure after earlier calls may
        already have consumed shots; no partial estimate is returned. Even a
        constant-only objective requires a callable, but never invokes it.
        """
        shots = _integer(shots, "shots", minimum=1)
        if not callable(run_counts):
            raise TypeError("run_counts must be a callable; no default backend is provided")
        # Validate/build the entire batch before submitting any program.
        jobs = self.build_programs(gammas, betas)
        means = tuple(_parity_mean(run_counts(job.program, shots),
                                   len(job.plan.global_qubits), job.parity_cbits, shots)
                      for job in jobs)
        mean = _objective_sum([self.constant] + [plan.target.coefficient * value
                                      for plan, value in zip(self.plans, means)])
        return LightConeEstimate(mean, means, shots, shots * len(jobs))
