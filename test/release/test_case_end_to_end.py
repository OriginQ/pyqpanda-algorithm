"""Every fixed algorithm case, end to end, on both execution paths.

Plan 7: each executable algorithm's case must resolve to its committed
verdict on the QPU runner path (real submission against the service
stand-in) *and* on the preflight path (the device's FakeBackend
stand-in).  The fake outcomes below are pinned to each case's committed
domain predicate: bit widths match the measured qubits, and the QKmeans
sequence mirrors the swap-test distances of the seeded centroids (seed
6, the four fixed points), so the fit converges in two rounds.

The two statevector-gap cases (``QUBO_QAOA``, ``QmRMR``) honestly
declare the ``statevector`` capability runtime backends do not
advertise (see the known Plan 3 gaps in
:mod:`tools.release_qualification.cases`); their end-to-end test
asserts the honest failure, never a fabricated pass.
"""

import pytest

from test.execution.fakes import FakeFakeBackend, FakeRuntimeService
from test.release.conftest import qpu_fake_device
from tools.release_qualification.cases import QUALIFICATION_CASES, SMOKE_CASES
from tools.release_qualification.run_preflight import _qualify
from tools.release_qualification.run_qpu import QPURunner

#: Cases that cannot pass on runtime backends until Plan 3 provides a
#: sampling-based final distribution; they declare ``statevector`` in
#: their committed capabilities, so the honest record is a failed
#: verdict with the capability error.
_STATEVECTOR_GAPS = {"QUBO_QAOA", "QmRMR"}

_CASES = (*QUALIFICATION_CASES, *SMOKE_CASES)


def _qkmeans_distance_probes(probabilities: bool) -> list:
    """QKmeans swap-test distances of the seeded fit, as a 16-entry sequence.

    The distances come from the exact swap-test simulation of the
    seeded centroids (``np.random.seed(6)``, four fixed points): round 0
    against the random initial centroids, round 1 against the first
    data-means update.  Round 1 reproduces round 0's assignment, so the
    fit converges in two rounds (16 distance probes); entries after the
    sequence keep the last value.
    """
    round0 = [[0.233, 0.114], [0.189, 0.091], [0.002, 0.216], [0.005, 0.258]]
    round1 = [[0.283, 0.001], [0.221, 0.001], [0.0, 0.235], [0.0, 0.228]]

    def probe(p: float) -> dict:
        if probabilities:
            return {"1": p, "0": 1.0 - p}
        n = round(p * 1000)
        return {"1": n, "0": 1000 - n}

    return [probe(d) for pair in round0 for d in pair] + [
        probe(d) for pair in round1 for d in pair
    ]


#: Fake runtime sample outcomes that satisfy each case's committed domain
#: predicate on the QPU runner path.  Every entry is a single-element
#: list: all submissions of the case share it (the QKmeans ordered
#: sequence is the one exception).  Cases absent here pass with the
#: service defaults.
_QPU_SAMPLE_RESULTS = {
    "bell": [{"00": 490, "11": 480, "01": 15, "10": 15}],  # P(00)+P(11) = 0.97
    "Grover": [{"11": 950, "00": 50}],  # P(11) = 0.95 >= 0.9
    "QAE": [{"01010110": 1000}],  # ancilla 43: p ~= 0.757 in [0.70, 0.80]
    "QSEncode": [{"0": 500, "1": 500}],  # |+>|+>: [0.5, 0.5]
    "QUBO_GAS": [{"010": 1000}],  # the fixed QUBO's unique minimum -1.0
    "Shor": [{"01000000": 600, "11000000": 400}],  # order of 2 mod 15
    "HHL": [{"10": 600, "11": 300, "00": 50, "01": 50}],  # p_success = 0.9
    "QARM": [
        {
            "00000000000000001": 100,
            "00000000000001001": 100,
            "00000000000000010": 100,
            "00000000000001010": 100,
        }
    ],  # 4 locating numbers: >= 2 transactions per rule
    "QKmeans": _qkmeans_distance_probes(probabilities=False),
}
_QPU_ESTIMATE_RESULTS = {"VQE": [-1.0]}  # Z0 ground energy

#: Fake-backend probabilities for the preflight path.  Only the cases
#: whose measured qubit count differs from the default two-bit outcome
#: need an entry; the preflight verdict is feasibility evidence (the
#: run completes and the circuits transpile), never statistics.
_PREFLIGHT_SAMPLE_RESULTS = {
    "QAE": [{"01010110": 1.0}],
    "QSEncode": [{"0": 0.5, "1": 0.5}],
    "QUBO_GAS": [{"010": 1.0}],
    "QARM": [
        {
            "00000000000000001": 0.25,
            "00000000000001001": 0.25,
            "00000000000000010": 0.25,
            "00000000000001010": 0.25,
        }
    ],
    "QKmeans": _qkmeans_distance_probes(probabilities=True),
}


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.algorithm)
def test_every_case_runs_end_to_end_on_qpu_runner(case, tmp_path):
    """Every fixed case submits, executes, and resolves to its committed
    verdict on the QPU runner path.  The statevector-gap cases record the
    honest failure instead of a fabricated pass."""
    service = FakeRuntimeService()
    if case.algorithm in _QPU_SAMPLE_RESULTS:
        service.sample_results = list(_QPU_SAMPLE_RESULTS[case.algorithm])
    if case.algorithm in _QPU_ESTIMATE_RESULTS:
        service.estimate_results = list(_QPU_ESTIMATE_RESULTS[case.algorithm])
    record = QPURunner(
        service=service, device=qpu_fake_device(), checkpoint_dir=tmp_path / "ckpt"
    ).run_case(case)
    if case.algorithm in _STATEVECTOR_GAPS:
        assert record.verdict == "failed"
        assert "state" in record.parsed_result["error"]
    else:
        assert record.verdict == "passed", record.parsed_result


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.algorithm)
def test_every_case_runs_end_to_end_on_preflight_fake(case):
    """Every fixed case executes on the device's fake backend.  The
    statevector-gap cases record the honest failure."""
    fake = FakeFakeBackend()
    if case.algorithm in _PREFLIGHT_SAMPLE_RESULTS:
        fake.sample_results = list(_PREFLIGHT_SAMPLE_RESULTS[case.algorithm])
    record = _qualify(case, fake)
    if case.algorithm in _STATEVECTOR_GAPS:
        assert record.verdict == "failed"
        assert "state" in record.parsed_result["outcome"]
    else:
        assert record.verdict == "passed", record.parsed_result
