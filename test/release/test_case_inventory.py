"""Fixed RC case inventory: completeness, uniqueness, local buildability.

Plan 7 Task 2: the qualification catalog is fixed data committed before
QPU execution -- shots, seeds, thresholds, and attempt budgets are
never retuned after results are seen.  The completeness test below is
verbatim from the task brief: every executable algorithm must have
exactly one fixed case in ``QUALIFICATION_CASES``.
"""

import math

import pytest
from pyqpanda3.core import QCircuit, QProg

from pyqpanda_alg.execution import LocalBackend
from tools.release_qualification.cases import (
    QUALIFICATION_CASES,
    SMOKE_CASES,
    TRANSPILATION_CASES,
    case_by_name,
)

EXECUTABLE_ALGORITHMS = {
    "QAOA", "QARM", "QKmeans", "QPCA", "QSVM", "QUBO_QAOA",
    "QUBO_GAS", "QAE", "QSVD", "QSVR", "Grover",
    "GroverAdaptiveSearch", "QmRMR", "QSEncode", "VQE", "HHL", "Shor",
}


def test_qpu_inventory_covers_every_executable_algorithm():
    assert {case.algorithm for case in QUALIFICATION_CASES} == EXECUTABLE_ALGORITHMS


def test_inventory_has_no_duplicate_case_names():
    names = [case.algorithm for case in QUALIFICATION_CASES]
    assert len(names) == len(set(names))


def test_every_case_builds_a_circuit_or_resource_request_locally():
    for case in QUALIFICATION_CASES:
        request = case.builder()
        assert _is_circuit_or_request(request), (
            f"{case.algorithm}: builder must return a circuit/program, a "
            f"(circuit, observable) pair, or a callable invoke(backend); "
            f"got {type(request).__name__}"
        )


def test_smoke_cases_hold_the_bell_check_and_are_not_qpu_cases():
    smoke_names = {case.algorithm for case in SMOKE_CASES}
    assert "bell" in smoke_names
    assert smoke_names.isdisjoint({case.algorithm for case in QUALIFICATION_CASES})


def test_transpilation_cases_build_circuits_without_a_backend():
    assert TRANSPILATION_CASES
    for case in TRANSPILATION_CASES:
        circuit = case.builder()
        assert isinstance(circuit, (QCircuit, QProg))


def test_fixed_statistical_data_is_committed_before_execution():
    modes = {"sample", "estimate", "variational", "transpile"}
    capabilities = {
        "sampling",
        "estimation",
        "variational_session",
        "statevector",
        "tomography",
    }
    for case in (*QUALIFICATION_CASES, *SMOKE_CASES):
        assert case.shots >= 1
        assert math.isfinite(case.threshold)
        assert case.seed >= 0
        assert case.max_attempts >= 1
        assert case.timeout > 0
        assert case.mode in modes
        assert set(case.required_capabilities) <= capabilities
        if case.fake_backend_required:
            assert case.mode != "transpile"


def test_shor_case_predicate_does_not_encode_factors_or_order():
    case = case_by_name("Shor")
    assert case.domain({"factors": (3, 5)})
    assert case.domain({"factors": (5, 3)})
    assert not case.domain({"factors": (3,)})
    assert not case.domain({"factors": (3, 7)})
    assert not case.domain({"factors": (3, 5, 7)})


def test_hhl_case_is_sampling_mode_without_tomography():
    case = case_by_name("HHL")
    assert case.mode == "sample"
    assert "tomography" not in case.required_capabilities


def test_qae_case_reference_probability_is_0_75():
    """The committed QAE operator (RY(2*pi/3)) must estimate p ~= 0.75."""
    case = case_by_name("QAE")
    invoke = case.builder()
    p_estimated = invoke(LocalBackend())
    assert 0.0 <= p_estimated <= 1.0
    assert abs(p_estimated - 0.75) <= case.threshold


def test_hhl_domain_predicate_accepts_sampled_observables_dict():
    """The sampled HHL path reports observables as a dict keyed by Pauli
    string; the domain predicate must gate that shape, never a list."""
    case = case_by_name("HHL")

    class _Parsed:
        success_probability = 0.8
        metadata = {"observables": {"Z0": 0.25}}

    class _ListShaped:
        success_probability = 0.8
        metadata = {"observables": [0.25]}

    class _NonFinite:
        success_probability = 0.8
        metadata = {"observables": {"Z0": math.inf}}

    class _WrongKey:
        success_probability = 0.8
        metadata = {"observables": {"X0": 0.25}}

    assert case.domain(_Parsed())
    assert not case.domain(_ListShaped())
    assert not case.domain(_NonFinite())
    assert not case.domain(_WrongKey())


def test_case_by_name_resolves_all_inventories():
    assert case_by_name("bell") is not None
    assert case_by_name("Grover").algorithm == "Grover"
    with pytest.raises(KeyError):
        case_by_name("no-such-case")


def _is_circuit_or_request(request):
    """True when ``request`` is a shape the qualification runner can execute."""
    if isinstance(request, (QCircuit, QProg)):
        return True
    if (
        isinstance(request, tuple)
        and len(request) == 2
        and isinstance(request[0], (QCircuit, QProg))
    ):
        return True
    return callable(request)
