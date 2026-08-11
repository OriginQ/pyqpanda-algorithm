"""Local order finding and factor recovery for the Shor solver.

:class:`Shor` is the orchestration facade of the package.  :meth:`run`
resolves the modulus classically when possible — even, prime, or
perfect-power moduli return an explicit factorization that never claims
a quantum task — and otherwise drives a bounded attempt state machine
on a backend: every attempt draws a base from the injected RNG, applies
gcd preprocessing, submits one order-finding sample task, parses every
nonzero phase sample of the histogram by descending count, recovers a
candidate order with :func:`pyqpanda_alg.Shor.classical.recover_order`,
and derives factors as ``gcd(base**(order // 2) - 1, modulus)`` and
``gcd(base**(order // 2) + 1, modulus)``.

Order reduction
---------------
``recover_order`` may return a proper multiple of the true order (a
convergent denominator can land on a divisor of the true order, or the
sample can alias a multiple of the true order's fraction).  Factor
recovery with such a multiple fails both ways, so :func:`_reduce_order_multiples`
applies the standard halving rule before factor derivation: while the
candidate order is even and ``base**(order // 2)`` is already congruent
to one modulo the modulus, the half-power is redundant and the
candidate is halved.  Only then are the two gcds computed, skipping
candidates whose square root of unity is trivial (odd order, a power of
``-1`` modulo the modulus, or gcds that degenerate to 1 or the modulus).

Provenance
----------
Every attempt is recorded — base, sample task ID, each parsed sample
with its count, candidate order, and rejection reason — in the
``attempts`` entry of the result metadata.  ``used_quantum`` is True
exactly when the returned factorization was derived from a quantum
order-finding task, and ``task_ids`` collects every submitted sample
task ID in submission order.
"""

import math
import random

from pyqpanda_alg.execution import (
    AlgorithmTask,
    CompletedBackendTask,
    ExecutionOptions,
    resolve_backend,
)
from pyqpanda_alg.execution.errors import AlgorithmInputError

from .circuit import build_order_finding_circuit
from .classical import _validate_modulus, classical_preprocess, recover_order
from .model import ShorConfig, ShorResult


def _reduce_order_multiples(order: int, base: int, modulus: int) -> int:
    """Reduce a recovered order that is a proper multiple of the true order.

    While the candidate ``order`` is even and ``base**(order // 2)`` is
    already congruent to one modulo ``modulus``, the half-power is a
    redundant power of the true order and the candidate is halved.  The
    reduced candidate always still satisfies
    ``pow(base, order, modulus) == 1``.  An already-exact order is
    returned unchanged.
    """
    while order % 2 == 0 and pow(base, order // 2, modulus) == 1:
        order //= 2
    return order


def _classical_result(outcome) -> ShorResult:
    """Build the provenance-honest result of a classically resolved run."""
    if outcome.is_prime:
        preprocessing = "prime"
    elif outcome.factors[0] == 2:
        preprocessing = "even"
    else:
        preprocessing = "perfect_power"
    return ShorResult(
        factors=outcome.factors,
        is_prime=outcome.is_prime,
        used_quantum=False,
        task_ids=(),
        metadata={"preprocessing": preprocessing},
    )


def _task_ids(history: list[dict]) -> tuple[str, ...]:
    """Return every submitted sample task ID in submission order."""
    return tuple(record["task_id"] for record in history if record["task_id"])


def _exhausted_result(history: list[dict]) -> ShorResult:
    """Build the result of an attempt budget that ran out without factors."""
    return ShorResult(
        factors=None,
        used_quantum=True,
        task_ids=_task_ids(history),
        order=None,
        metadata={
            "preprocessing": "quantum_order_finding",
            "attempts": history,
            "exhausted": True,
        },
    )


class Shor:
    """Small-scale Shor factorization facade with bounded quantum attempts.

    ``modulus`` is validated at construction.  ``rng`` is the source of
    attempt bases (a :class:`random.Random` by default); injecting a
    seeded instance makes runs reproducible, and a test double that only
    provides ``randrange`` is sufficient.  ``config`` carries the
    bounded attempt budget and the phase-register size.
    """

    def __init__(self, modulus, rng=None, *, config: ShorConfig | None = None) -> None:
        self.modulus = _validate_modulus(modulus)
        if rng is None:
            rng = random.Random()
        if not callable(getattr(rng, "randrange", None)):
            raise AlgorithmInputError("rng must provide randrange()")
        self.rng = rng
        self.config = config if config is not None else ShorConfig()

    def run(self, *, backend=None, execution_options=None) -> ShorResult:
        """Factor ``modulus`` on a backend and return the run snapshot.

        The CPU :class:`~pyqpanda_alg.execution.LocalBackend` is used
        when ``backend`` is None.  Classical fast paths — even, prime,
        and perfect-power moduli — resolve immediately with
        ``used_quantum=False`` and no task IDs.  Odd composites drive
        the bounded attempt loop synchronously, one sample task per
        attempt, until a factor is found or the attempt budget runs
        out; see the module docstring for the per-attempt pipeline and
        the provenance guarantees.
        """
        backend = resolve_backend(backend)
        options = (
            execution_options if execution_options is not None else ExecutionOptions()
        )
        outcome = classical_preprocess(self.modulus)
        if outcome.resolved:
            return _classical_result(outcome)
        history: list[dict] = []
        for _ in range(self.config.max_attempts):
            result = self._attempt_once(backend, options, history)
            if result is not None:
                return result
        return _exhausted_result(history)

    def submit(self, *, backend=None, execution_options=None) -> AlgorithmTask:
        """Return a resumable task running one bounded attempt per poll.

        Each ``poll()`` executes exactly one attempt of the same state
        machine :meth:`run` drives synchronously: draw a base, run gcd
        preprocessing, submit one order-finding sample, and parse its
        phase samples.  Classical fast paths complete on the first poll;
        otherwise the task finishes on the first factorization or when
        the attempt budget is exhausted, and the completed task's result
        is the same :class:`ShorResult` :meth:`run` would return.  The
        local path performs no checkpoints — checkpointed recovery with
        a registered advance factory is the runtime stage of the
        package — so ``backend_task_ids`` and the result metadata are
        the provenance record here.
        """
        backend = resolve_backend(backend)
        options = (
            execution_options if execution_options is not None else ExecutionOptions()
        )
        state = {"classical_done": False, "history": []}

        def advance(state):
            if not state["classical_done"]:
                state["classical_done"] = True
                outcome = classical_preprocess(self.modulus)
                if outcome.resolved:
                    return (
                        CompletedBackendTask(
                            _classical_result(outcome), task_id=""
                        ),
                        True,
                    )
            result = self._attempt_once(backend, options, state["history"])
            if result is not None:
                task_id = result.task_ids[-1] if result.task_ids else ""
                return CompletedBackendTask(result, task_id=task_id), True
            step_task_id = state["history"][-1].get("task_id") or ""
            if len(state["history"]) >= self.config.max_attempts:
                return (
                    CompletedBackendTask(
                        _exhausted_result(state["history"]), task_id=step_task_id
                    ),
                    True,
                )
            return CompletedBackendTask(None, task_id=step_task_id), False

        return AlgorithmTask(
            algorithm="shor",
            initial_state=state,
            advance=advance,
            backend=backend,
        )

    def _attempt_once(self, backend, options, history: list[dict]) -> ShorResult | None:
        """Run one bounded attempt and return the factor result, or None.

        Draws the base from the injected RNG.  A base sharing a factor
        with the modulus resolves the attempt classically.  Otherwise
        one order-finding sample task is submitted and every nonzero
        phase sample of the histogram is parsed by descending count:
        recover a candidate order, reduce multiples of the true order,
        and derive factors from ``gcd(base**(order // 2) - 1, modulus)``
        and ``gcd(base**(order // 2) + 1, modulus)``.
        The record of the attempt — base, task ID, samples, candidate
        orders, rejection reasons — is appended to ``history``.
        """
        base = self.rng.randrange(2, self.modulus - 1)
        common = math.gcd(base, self.modulus)
        if common > 1:
            history.append(
                {
                    "base": base,
                    "task_id": None,
                    "samples": [],
                    "outcome": "classical_gcd",
                }
            )
            return ShorResult(
                factors=_sorted_pair(common, self.modulus // common),
                used_quantum=False,
                task_ids=_task_ids(history),
                order=None,
                metadata={
                    "preprocessing": "classical_gcd",
                    "attempts": history,
                },
            )
        build = build_order_finding_circuit(
            base, self.modulus, self.config.phase_qubits
        )
        sample_task = backend.submit_sample(build.program, options=options)
        counts = sample_task.result().single_counts()
        record = {
            "base": base,
            "task_id": sample_task.id,
            "samples": [],
            "outcome": "rejected",
        }
        history.append(record)
        phase_bits = len(build.phase_qubits)
        samples = sorted(
            ((int(key, 2), count) for key, count in counts.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        for sample, count in samples:
            if sample == 0:
                record["samples"].append(
                    {
                        "sample": sample,
                        "count": count,
                        "candidate_order": None,
                        "rejection": "zero sample carries no phase",
                    }
                )
                continue
            order = recover_order(sample, phase_bits, base, self.modulus)
            if order is None:
                record["samples"].append(
                    {
                        "sample": sample,
                        "count": count,
                        "candidate_order": None,
                        "rejection": "no convergent denominator recovered an order",
                    }
                )
                continue
            order = _reduce_order_multiples(order, base, self.modulus)
            if order % 2 == 1:
                record["samples"].append(
                    {
                        "sample": sample,
                        "count": count,
                        "candidate_order": order,
                        "rejection": "odd order leaves no square root of unity",
                    }
                )
                continue
            half = pow(base, order // 2, self.modulus)
            if half == self.modulus - 1:
                record["samples"].append(
                    {
                        "sample": sample,
                        "count": count,
                        "candidate_order": order,
                        "rejection": "base power is -1 modulo the modulus",
                    }
                )
                continue
            factor_a = math.gcd(half - 1, self.modulus)
            factor_b = math.gcd(half + 1, self.modulus)
            if factor_a in (1, self.modulus) or factor_b in (1, self.modulus):
                record["samples"].append(
                    {
                        "sample": sample,
                        "count": count,
                        "candidate_order": order,
                        "rejection": "trivial gcd factors",
                    }
                )
                continue
            record["samples"].append(
                {
                    "sample": sample,
                    "count": count,
                    "candidate_order": order,
                    "rejection": None,
                }
            )
            record["outcome"] = "factored"
            return ShorResult(
                factors=_sorted_pair(factor_a, factor_b),
                used_quantum=True,
                task_ids=_task_ids(history),
                order=order,
                metadata={
                    "preprocessing": "quantum_order_finding",
                    "attempts": history,
                },
            )
        return None


def _sorted_pair(first: int, second: int) -> tuple[int, int]:
    """Return the pair sorted ascending for a deterministic factor tuple."""
    return (min(first, second), max(first, second))
