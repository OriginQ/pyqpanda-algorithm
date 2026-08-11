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
when the result emerged from the quantum attempt pipeline — an
order-finding success or attempt exhaustion — regardless of the later
classical resolution of the factorization, and ``task_ids`` collects
every submitted sample task ID in submission order.

Recovery
--------
:meth:`Shor.submit` runs the same attempt state machine as a resumable
:class:`~pyqpanda_alg.execution.AlgorithmTask` whose JSON checkpoint
carries the modulus, the config, the RNG draw position, and the full
attempt history (bases, completed task IDs, histograms, candidate
orders, and rejection reasons) — never the backend, credentials, or
live task handles.  :meth:`~pyqpanda_alg.execution.AlgorithmTask.resume`
rebuilds the attempt machine through the factory registered under the
algorithm name ``shor``, continues the exact base sequence, and never
resubmits a completed task.
"""

import math
import random
from typing import Any

from pyqpanda_alg.execution import (
    AlgorithmTask,
    CompletedBackendTask,
    ExecutionOptions,
    register_algorithm,
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


def _result_snapshot(result: ShorResult) -> dict:
    """Return the JSON-safe dict snapshot of ``result`` for task accumulation.

    :class:`AlgorithmTask` accumulates each step's result verbatim and
    checkpoints it, so the accumulated value must be JSON primitives;
    the completed task's ``result()`` and a checkpoint's ``result``
    field therefore carry this snapshot rather than a live
    :class:`ShorResult` (which :meth:`run` still returns directly).
    """
    return {
        "factors": list(result.factors) if result.factors is not None else None,
        "is_prime": result.is_prime,
        "used_quantum": result.used_quantum,
        "task_ids": list(result.task_ids),
        "order": result.order,
        "metadata": result.metadata,
    }


class _ConstantRng:
    """RNG stand-in that always returns one fixed base.

    Rebuilt from checkpoint state for a resumed run whose original RNG
    exposed no ``getstate()``: its constant draw equals the last
    attempted base recorded in the history.
    """

    def __init__(self, value: int) -> None:
        self._value = value

    def randrange(self, a, b):
        return self._value


def _rng_snapshot(rng) -> Any:
    """Return the JSON-safe draw position of ``rng`` for checkpoint state.

    A ``random.Random`` exposes ``getstate()``/``setstate()``; the
    snapshot is its state exactly as the checkpoint layer round-trips
    it (tuples become lists on disk).  A test double providing only
    ``randrange`` (like the runtime tests' ``FixedBaseRng``) has no
    position to capture — its constant draw is recorded as the
    attempted base in the history, which is what a resumed run
    re-draws.
    """
    getstate = getattr(rng, "getstate", None)
    if getstate is None:
        return None
    return getstate()


def _restore_rng(state: dict) -> Any:
    """Rebuild the attempt RNG from the checkpointed snapshot.

    A ``random.Random`` snapshot restores the exact draw position; the
    JSON round trip stores the internal state as a list, which
    :meth:`random.Random.setstate` needs converted back to a tuple.
    A constant double without ``getstate`` has no position: the
    restored rng re-draws the last attempted base from the history.
    A checkpoint with neither a snapshot nor a history carries no draw
    position at all — resuming it would continue with an unseeded
    ``random.Random`` and silently lose reproducibility, so it is
    refused instead.
    """
    snapshot = state.get("rng_state")
    if snapshot is not None:
        version, internal, gauss = snapshot
        rng = random.Random()
        rng.setstate((version, tuple(internal), gauss))
        return rng
    if state["history"]:
        return _ConstantRng(state["history"][-1]["base"])
    raise AlgorithmInputError(
        "the shor checkpoint carries no rng_state and no attempted base "
        "to re-draw, so the resumed run would continue with an unseeded "
        "RNG; resubmit the factorization with an RNG that exposes "
        "getstate()"
    )


def _run_attempt(
    modulus: int,
    rng,
    config: ShorConfig,
    backend,
    options,
    history: list[dict],
) -> ShorResult | None:
    """Run one bounded attempt on ``backend`` and return the factor result.

    Draws the base from ``rng``.  A base sharing a factor with the
    modulus resolves the attempt classically.  Otherwise one
    order-finding sample task is submitted and every nonzero phase
    sample of the histogram is parsed by descending count: recover a
    candidate order, reduce multiples of the true order, and derive
    factors from ``gcd(base**(order // 2) - 1, modulus)`` and
    ``gcd(base**(order // 2) + 1, modulus)``.  The record of the
    attempt — base, task ID, samples, candidate orders, rejection
    reasons — is appended to ``history``.
    """
    base = rng.randrange(2, modulus - 1)
    common = math.gcd(base, modulus)
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
            factors=_sorted_pair(common, modulus // common),
            used_quantum=False,
            task_ids=_task_ids(history),
            order=None,
            metadata={
                "preprocessing": "classical_gcd",
                "attempts": history,
            },
        )
    build = build_order_finding_circuit(base, modulus, config.phase_qubits)
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
        order = recover_order(sample, phase_bits, base, modulus)
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
        order = _reduce_order_multiples(order, base, modulus)
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
        half = pow(base, order // 2, modulus)
        if half == modulus - 1:
            record["samples"].append(
                {
                    "sample": sample,
                    "count": count,
                    "candidate_order": order,
                    "rejection": "base power is -1 modulo the modulus",
                }
            )
            continue
        factor_a = math.gcd(half - 1, modulus)
        factor_b = math.gcd(half + 1, modulus)
        if factor_a in (1, modulus) or factor_b in (1, modulus):
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


def _options_snapshot(options: ExecutionOptions) -> dict:
    """Return the JSON-safe subset of ``options`` persisted in checkpoints.

    Only the committed knobs a resumed run must reproduce — the shot
    count and the timeout — are stored; the runtime-facing flags stay
    out of the durable state.
    """
    return {"shots": options.shots, "timeout": options.timeout}


def _options_from_snapshot(snapshot: Any) -> ExecutionOptions:
    """Rebuild :class:`ExecutionOptions` from a checkpointed snapshot.

    A checkpoint without the snapshot (written by an older build)
    falls back to the defaults; ``shots`` and ``timeout`` are the only
    fields persisted.
    """
    if not isinstance(snapshot, dict):
        return ExecutionOptions()
    return ExecutionOptions(
        shots=snapshot.get("shots", ExecutionOptions.shots),
        timeout=snapshot.get("timeout", ExecutionOptions.timeout),
    )


def _make_shor_advance(exec_backend, options=None, rng=None):
    """Build the one-attempt-per-poll advance of the Shor state machine.

    ``rng`` is the live attempt RNG of a fresh task; a resumed task
    passes None and the advance rebuilds the RNG from the checkpointed
    snapshot in the state.  ``options`` is the live submission options
    of a fresh task; a resumed task passes None and the advance reads
    the checkpointed execution options from the state, so a resumed run
    uses the options its own submit committed — never those of a later
    submit that replaced the registered factory.  Every attempt mutates
    the state (the draw position and the history), so a checkpoint
    written between polls captures exactly what a resumed run needs and
    no completed task is ever resubmitted.
    """

    def advance(state: dict):
        if not state["classical_done"]:
            state["classical_done"] = True
            outcome = classical_preprocess(state["modulus"])
            if outcome.resolved:
                return (
                    CompletedBackendTask(
                        _result_snapshot(_classical_result(outcome)), task_id=""
                    ),
                    True,
                )
        attempt_rng = rng if rng is not None else _restore_rng(state)
        config = ShorConfig(**state["config"])
        run_options = (
            options if options is not None else _options_from_snapshot(state.get("execution_options"))
        )
        result = _run_attempt(
            state["modulus"],
            attempt_rng,
            config,
            exec_backend,
            run_options,
            state["history"],
        )
        state["rng_state"] = _rng_snapshot(attempt_rng)
        if result is not None:
            task_id = result.task_ids[-1] if result.task_ids else ""
            return (
                CompletedBackendTask(_result_snapshot(result), task_id=task_id),
                True,
            )
        step_task_id = state["history"][-1].get("task_id") or ""
        if len(state["history"]) >= config.max_attempts:
            return (
                CompletedBackendTask(
                    _result_snapshot(_exhausted_result(state["history"])),
                    task_id=step_task_id,
                ),
                True,
            )
        return CompletedBackendTask(None, task_id=step_task_id), False

    return advance


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
        the attempt budget is exhausted, and the completed task's
        ``result()`` is the JSON-safe dict snapshot of the
        :class:`ShorResult` :meth:`run` would return (a checkpoint's
        ``result`` field carries the same snapshot — see
        :func:`_result_snapshot`).

        The task is checkpointable: ``task.checkpoint(path)`` persists
        the modulus, config, RNG draw position, the submission options
        (shots and timeout), and the full attempt history (bases,
        completed task IDs, histograms, candidate orders, rejection
        reasons) as JSON, and
        :meth:`~pyqpanda_alg.execution.AlgorithmTask.resume` rebuilds
        the attempt machine through the factory registered under
        ``shor`` — the backend, credentials, and live task handles are
        never serialized — so a resumed run continues the exact base
        sequence and never resubmits a completed task.
        """
        backend = resolve_backend(backend)
        options = (
            execution_options if execution_options is not None else ExecutionOptions()
        )
        state = {
            "modulus": self.modulus,
            "config": {
                "max_attempts": self.config.max_attempts,
                "phase_qubits": self.config.phase_qubits,
            },
            "rng_state": _rng_snapshot(self.rng),
            "classical_done": False,
            "history": [],
            "execution_options": _options_snapshot(options),
        }

        def make_factory(exec_backend):
            # No live options are closed over: the advance reads the
            # submission options from the checkpointed state, so a
            # resumed task uses the options its own submit committed —
            # never those of a later submit that replaced the factory.
            return _make_shor_advance(exec_backend)

        register_algorithm("shor", make_factory)
        return AlgorithmTask(
            algorithm="shor",
            initial_state=state,
            advance=_make_shor_advance(backend, options, rng=self.rng),
            backend=backend,
        )

    def _attempt_once(self, backend, options, history: list[dict]) -> ShorResult | None:
        """Run one bounded attempt and return the factor result, or None.

        Delegates to :func:`_run_attempt` with the solver's modulus,
        RNG, and config; see the module docstring for the per-attempt
        pipeline and the provenance guarantees.
        """
        return _run_attempt(
            self.modulus, self.rng, self.config, backend, options, history
        )


def _sorted_pair(first: int, second: int) -> tuple[int, int]:
    """Return the pair sorted ascending for a deterministic factor tuple."""
    return (min(first, second), max(first, second))
