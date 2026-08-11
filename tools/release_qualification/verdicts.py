"""Deterministic verdict layer: parsed results against committed bounds.

The verdict layer is pure: no IO, no service, no credentials.  Every
verdict consumes only the fixed case contract committed in
:mod:`tools.release_qualification.cases` -- the domain predicate and
the threshold -- and never an alternative bound invented at test time
or after results are seen.

Verdict semantics
-----------------
A case passes exactly when both committed conditions hold:

1. ``case.domain(parsed)`` accepts the natural result shape of the
   case (counts dict for pure-circuit cases, the algorithm's returned
   object for algorithm cases, the fixed ``{"factors": ...}`` dict for
   Shor).
2. The committed threshold bound holds.  For the counts-shaped
   probability-floor cases (bell, Grover) the bound is the probability
   of the committed good outcomes (``P(00) + P(11)`` for bell,
   ``P(11)`` for Grover) against the committed floor.  The qualitative
   cases (Shor, QUBO_GAS, VQE, HHL, ...) encode their acceptance bound
   inside the committed domain predicate itself; the committed
   threshold is then the confidence/tolerance floor the domain
   enforces, and condition 1 is the whole verdict.

:func:`bell_verdict` is the canonical smoke example: it applies the
generic machinery to the committed bell case, so it cannot drift from
the committed floor.
"""

from dataclasses import dataclass
from typing import Any

from .cases import _GOOD_OUTCOMES, case_by_name


@dataclass(frozen=True)
class Verdict:
    """Outcome of one parsed result against the committed case bound.

    ``passed`` is the deterministic decision; ``reason`` describes a
    failure for the record when the runner produced one.
    """

    passed: bool
    reason: str | None = None


def bell_verdict(counts: dict, *, shots: int) -> Verdict:
    """Verdict of the bell smoke case against the committed floor.

    The bell case commits ``P(00) + P(11) >= 0.9`` (see
    ``cases.SMOKE_CASES``); this helper consumes only that committed
    bound via the generic machinery, so it is the canonical smoke
    example of the verdict layer.
    """
    return verdict_for_case(case_by_name("bell"), counts, shots=shots)


def verdict_for_case(case: Any, parsed: Any, *, shots: int | None = None) -> Verdict:
    """Verdict of ``parsed`` against the committed ``case`` contract.

    Passes exactly when the case's domain predicate accepts ``parsed``
    and the committed threshold bound holds (see the module docstring
    for the bound semantics).  ``shots`` is the committed shot count
    needed for the probability-floor comparison.
    """
    if not case.domain(parsed):
        return Verdict(
            passed=False,
            reason=(
                f"{case.algorithm}: parsed result does not satisfy the "
                "committed domain predicate"
            ),
        )
    bound = _threshold_bound(case, parsed, shots)
    if bound is False:
        return Verdict(
            passed=False,
            reason=(
                f"{case.algorithm}: committed threshold {case.threshold} "
                "not met"
            ),
        )
    return Verdict(passed=True)


def _threshold_bound(case: Any, parsed: Any, shots: int | None) -> bool | None:
    """The committed threshold bound of ``parsed``, or None when the
    domain predicate already gates the whole verdict.

    Counts-shaped sampling results (bell, Grover) compare the
    probability of the committed good outcomes against the committed
    floor.  All other cases encode their acceptance bound in the
    committed domain predicate, so no separate numeric bound applies.
    """
    good = _GOOD_OUTCOMES.get(case.algorithm)
    if good is None:
        return None
    if not isinstance(parsed, dict):
        return None
    if not shots:
        # A falsy shot count would silently disable the committed
        # probability floor; refuse it instead.
        raise ValueError(
            f"{case.algorithm} needs a positive shot count for its "
            f"committed probability floor, got {shots!r}"
        )
    probability = sum(parsed.get(key, 0) for key in good) / shots
    return probability >= case.threshold
