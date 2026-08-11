"""Real-QPU runner: sequential, resumable remote evidence for fixed cases.

Plan 7 Task 4: ``QPURunner`` submits each fixed case against a
credential-free service stand-in, checkpoints every remote task, and
resumes by querying existing task IDs instead of resubmitting.  A
submission failure is recorded as a failed verdict and is never
retried; verdicts consume only the committed case threshold.  The
first test below is verbatim from the task brief.
"""

import pytest

pytest.importorskip("qpanda3_runtime")  # QPandaRuntimeBackend needs the package
from pyqpanda3.core import H
from pyqpanda3.hamiltonian import Hamiltonian

from test.execution.fakes import FakeDevice, FakeRuntimeService
from test.release.conftest import fake_service_with_bad_result
from tools.release_qualification.cases import QualificationCase, case_by_name
from tools.release_qualification.run_qpu import QPURunner

#: Gate set and topology the fake device must advertise so the fixed
#: cases pass the QPandaRuntimeBackend preflight validation (a real
#: device reports its own capabilities through the same surface).
_QPU_GATES = [
    "H", "X", "Y", "Z", "RX", "RY", "RZ", "S", "T", "SDG", "TDG", "SX",
    "CNOT", "CX", "CZ", "SWAP", "CP", "CSWAP", "CCX", "CCU1", "CCCP",
    "CCCU1", "CCH", "CCSWAP", "U1", "CU1", "P", "U2", "U3",
    "CORACLE", "ORACLE",
]


def _qpu_device():
    """Fake device advertising the gates/topology the fixed cases use."""
    device = FakeDevice()
    device.basic_gates.return_value = list(_QPU_GATES)
    device.chip_topo_edges.return_value = [
        [i, j] for i in range(20) for j in range(i + 1, 20)
    ]
    return device


def test_verdict_does_not_retry_after_threshold_failure():
    runner = QPURunner(service=fake_service_with_bad_result())
    runner.run_case(case_by_name("bell"))
    assert runner.submission_count == 1


def test_runner_records_failed_verdict_for_threshold_failure():
    runner = QPURunner(service=fake_service_with_bad_result())
    record = runner.run_case(case_by_name("bell"))
    assert record.verdict == "failed"
    assert record.parsed_result["00"] == 250  # digestable interpretation
    assert record.task_ids


def test_estimate_case_wraps_scalar_parsed_result_as_an_object():
    """The schema requires ``parsed_result`` to be an object; a scalar
    estimate (float) on the circuit+observable path must be recorded as
    ``{"value": ...}`` instead of a schema-invalid bare float."""
    service = FakeRuntimeService()
    runner = QPURunner(service=service, device=_qpu_device())
    case = QualificationCase(
        algorithm="estimate-fixture",
        # a Hamiltonian, not a bare string: the runtime backend requires
        # an observable that exposes qubits()
        builder=lambda: (H(0), Hamiltonian({"Z0": 1.0})),
        domain=lambda parsed: isinstance(parsed, (int, float)),
        mode="estimate",
        required_capabilities=("estimation",),
        shots=1000,
        threshold=0.9,
        seed=1,
        max_attempts=1,
        timeout=60.0,
    )
    record = runner.run_case(case)
    assert record.verdict == "passed"
    assert record.parsed_result == {"value": 0.5}
    assert record.task_ids


def test_shor_qpu_record_has_quantum_provenance():
    """The QPU Shor record must attest a real quantum order-finding run."""
    service = FakeRuntimeService()
    service.sample_results = [{"01000000": 600, "11000000": 400}]
    runner = QPURunner(service=service, device=_qpu_device())
    record = runner.run_case(case_by_name("Shor"))
    assert record.parsed_result["used_quantum"] is True
    assert record.task_ids
    # factors derived after order recovery, never encoded beforehand
    assert record.parsed_result["order"] is not None
    assert set(record.parsed_result["factors"]) == {3, 5}
    assert record.verdict == "passed"


def test_hhl_qpu_record_reports_success_probability_and_observable():
    """The HHL record keeps success probability + requested observable."""
    service = FakeRuntimeService()
    service.sample_results = [{"10": 600, "11": 300, "00": 50, "01": 50}]
    runner = QPURunner(service=service, device=_qpu_device())
    record = runner.run_case(case_by_name("HHL"))
    assert record.parsed_result["success_probability"] == pytest.approx(0.9)
    assert record.parsed_result["metadata"]["observables"]["Z0"] == pytest.approx(1.0 / 3.0)
    assert "statevector" not in record.parsed_result
    assert record.verdict == "passed"
    assert record.task_ids


def test_submission_failure_is_recorded_and_never_retried():
    service = FakeRuntimeService()
    service.submit_error = RuntimeError("device unreachable")
    runner = QPURunner(service=service)
    record = runner.run_case(case_by_name("bell"))
    assert record.verdict == "failed"
    assert runner.submission_count == 1  # one attempt, no retry


def test_runner_checkpoints_and_resumes_by_querying_existing_tasks(tmp_path):
    service = FakeRuntimeService()
    runner = QPURunner(service=service, checkpoint_dir=tmp_path)
    first = runner.run_case(case_by_name("bell"))
    assert runner.submission_count == 1

    resumed = QPURunner(service=service, checkpoint_dir=tmp_path)
    second = resumed.run_case(case_by_name("bell"))
    # the checkpointed remote task is recovered and queried, not resubmitted
    assert resumed.submission_count == 0
    assert service.recovered_task_paths
    assert second.task_ids == first.task_ids


def test_resumed_invoke_case_reuses_checkpointed_tasks(tmp_path):
    """A resumed algorithm run queries checkpointed tasks, then continues."""
    service = FakeRuntimeService()
    service.sample_results = [{"01000000": 600, "11000000": 400}]
    runner = QPURunner(service=service, device=_qpu_device(), checkpoint_dir=tmp_path)
    first = runner.run_case(case_by_name("Shor"))
    assert runner.submission_count == 1

    resumed = QPURunner(service=service, device=_qpu_device(), checkpoint_dir=tmp_path)
    second = resumed.run_case(case_by_name("Shor"))
    assert resumed.submission_count == 0  # existing task queried, not resubmitted
    assert second.parsed_result["used_quantum"] is True
    assert second.task_ids == first.task_ids
