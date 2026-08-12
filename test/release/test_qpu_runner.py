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

from test.execution.fakes import FakeRuntimeService
from test.release.conftest import fake_service_with_bad_result, qpu_fake_device
from tools.release_qualification import cases as cases_module
from tools.release_qualification.cases import QualificationCase, case_by_name
from tools.release_qualification.manifest import contains_credentials
from tools.release_qualification.run_qpu import QPURunner, run_qpu


def _qpu_device():
    """Shared fake device advertising the fixed cases' gates/topology."""
    return qpu_fake_device()


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


def test_run_qpu_creates_missing_checkpoint_dir(tmp_path, monkeypatch):
    """``run_qpu`` must create a missing checkpoint directory before the
    case loop, so ``task.checkpoint()`` never fails with
    ``FileNotFoundError``.  The full fixed case set is not needed to
    prove the directory creation: one real submission (bell) exercises
    the checkpoint write end to end."""
    from tools.release_qualification import run_qpu as run_qpu_module

    monkeypatch.setattr(run_qpu_module, "QUALIFICATION_CASES", ())
    monkeypatch.setattr(run_qpu_module, "SMOKE_CASES", (case_by_name("bell"),))
    checkpoint_dir = tmp_path / "nested" / "checkpoints"
    assert not checkpoint_dir.exists()
    run_qpu(
        FakeRuntimeService(),
        device=_qpu_device(),
        checkpoint_dir=checkpoint_dir,
    )
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "bell-0.json").is_file()


def test_run_qpu_full_case_set_manifests_every_case(tmp_path):
    """``run_qpu`` runs the complete fixed case set end to end: every
    qualification and smoke case is executed against the service, each
    resolves to a verdict record (pass or fail, never a hang or a
    swallowed error), every record is direct QPU execution, and each
    submitted task is checkpointed next to the runner.  The checkpoint
    directory already exists with a stale file, so the run also proves
    the directory bootstrap is idempotent."""
    service = FakeRuntimeService()
    service.sample_results = [{"00": 490, "11": 480, "01": 15, "10": 15}]
    checkpoint_dir = tmp_path / "nested" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "bell-0.json").write_text("{}", encoding="utf-8")

    manifest = run_qpu(service, device=_qpu_device(), checkpoint_dir=checkpoint_dir)

    expected_cases = (*cases_module.QUALIFICATION_CASES, *cases_module.SMOKE_CASES)
    assert len(manifest.cases) == len(expected_cases)
    assert {case.algorithm for case in manifest.cases} == {
        case.algorithm for case in expected_cases
    }
    # direct execution evidence, never a transpile-only record
    assert all(case.execution_mode == "qpu" for case in manifest.cases)
    # every case resolves to a verdict, whether the fake results pass it or not
    assert all(case.verdict in ("passed", "failed") for case in manifest.cases)
    # submitted tasks landed in the checkpoint directory (ordinal per case)
    assert (checkpoint_dir / "bell-0.json").is_file()
    assert (checkpoint_dir / "QAOA-0.json").is_file()
    # no credential-shaped value survives into the artifact
    assert not contains_credentials(manifest.to_dict())


def test_run_qpu_without_checkpoint_dir_writes_nothing(tmp_path, monkeypatch):
    """With ``checkpoint_dir=None`` the runner stays checkpoint-free:
    the run qualifies normally and no task file is written anywhere."""
    from tools.release_qualification import run_qpu as run_qpu_module

    monkeypatch.setattr(run_qpu_module, "QUALIFICATION_CASES", ())
    monkeypatch.setattr(run_qpu_module, "SMOKE_CASES", (case_by_name("bell"),))
    manifest = run_qpu(FakeRuntimeService(), device=_qpu_device())
    assert len(manifest.cases) == 1
    assert manifest.cases[0].algorithm == "bell"
    assert not list(tmp_path.rglob("*.json"))
