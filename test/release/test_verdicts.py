"""Deterministic verdict layer: fixed thresholds, committed before execution.

Plan 7 Task 4: ``bell_verdict`` decides a bell smoke outcome against
the committed floor, and the generic ``verdict_for_case`` applies the
committed case contract (domain predicate + fixed threshold) to any
parsed result.  The first test below is verbatim from the task brief.
"""

import pytest

from tools.release_qualification.cases import (
    SMOKE_CASES,
    _GOOD_OUTCOMES,
    case_by_name,
)
from tools.release_qualification.verdicts import (
    Verdict,
    bell_verdict,
    verdict_for_case,
    _GOOD_OUTCOMES as _VERDICT_GOOD_OUTCOMES,
)


def test_bell_case_passes_fixed_threshold():
    verdict = bell_verdict({"00": 490, "11": 480, "01": 15, "10": 15}, shots=1000)
    assert verdict.passed is True


def test_bell_case_fails_under_the_fixed_threshold():
    verdict = bell_verdict({"00": 250, "11": 250, "01": 250, "10": 250}, shots=1000)
    assert verdict.passed is False


def test_bell_verdict_returns_a_verdict_object():
    verdict = bell_verdict({"00": 490, "11": 480, "01": 15, "10": 15}, shots=1000)
    assert isinstance(verdict, Verdict)
    assert verdict.passed is True


def test_generic_verdict_consumes_the_committed_bell_threshold():
    """The generic path decides bell with the committed case contract only."""
    case = case_by_name("bell")
    passing = verdict_for_case(case, {"00": 490, "11": 480, "01": 15, "10": 15}, shots=case.shots)
    failing = verdict_for_case(case, {"00": 250, "11": 250, "01": 250, "10": 250}, shots=case.shots)
    assert passing.passed is True
    assert failing.passed is False


def test_generic_verdict_rejects_results_outside_the_domain():
    case = case_by_name("bell")
    verdict = verdict_for_case(case, {"00": 490}, shots=case.shots)
    assert verdict.passed is False


def test_generic_verdict_qualitative_cases_gate_on_the_domain():
    """Shor encodes its acceptance bound in the committed domain predicate."""
    case = case_by_name("Shor")
    assert verdict_for_case(case, {"factors": [3, 5]}).passed is True
    assert verdict_for_case(case, {"factors": [3, 7]}).passed is False


def test_smoke_bell_case_is_the_committed_smoke_floor():
    case = case_by_name("bell")
    assert case in SMOKE_CASES
    assert case.threshold == 0.9


def test_verdict_refuses_falsy_shot_counts_for_floor_cases():
    """A falsy shot count would silently disable the committed
    probability floor; the verdict layer must refuse it."""
    case = case_by_name("bell")
    with pytest.raises(ValueError, match="shot count"):
        verdict_for_case(case, {"00": 490, "11": 480}, shots=0)
    with pytest.raises(ValueError, match="shot count"):
        verdict_for_case(case, {"00": 490, "11": 480}, shots=None)


def test_good_outcomes_are_single_sourced_from_cases():
    """The verdict layer must consume the committed good outcomes from
    the case inventory, never a drifting local copy."""
    assert _VERDICT_GOOD_OUTCOMES is _GOOD_OUTCOMES
    assert _VERDICT_GOOD_OUTCOMES == {"bell": ("00", "11"), "Grover": ("11",)}
