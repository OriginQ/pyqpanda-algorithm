"""FakeBackend preflight runner: credential-free preflight evidence.

Plan 7 Task 3: ``run_preflight`` queries the device through the
credential-free service stand-in, constructs the device's FakeBackend,
runs every fixed qualification and transpilation case against it, and
returns a sanitized ``QualificationManifest``.  The test below is
verbatim from the task brief.
"""

from test.execution.fakes import FakeFakeBackend

from tools.release_qualification.manifest import contains_credentials
from tools.release_qualification.run_preflight import run_preflight


def test_preflight_records_transpile_and_fake_results(fake_runtime_service, tmp_path):
    result = run_preflight(fake_runtime_service, "WK_C180", output_dir=tmp_path)
    assert result.device["chip_id"] == "WK_C180"
    assert all(case.transpiled for case in result.cases)
    assert not contains_credentials(result.to_dict())


def test_preflight_shor_record_flags_classical_resolution(fake_runtime_service):
    """The committed Shor draw resolves classically, so the preflight
    record must flag the classical resolution instead of silently
    attesting a quantum run."""
    manifest = run_preflight(fake_runtime_service, "WK_C180")
    shor = next(case for case in manifest.cases if case.algorithm == "Shor")
    assert shor.parsed_result["outcome"] == "ok"
    assert shor.parsed_result["note"]
    assert "classical" in shor.parsed_result["note"]
    assert shor.transpiled is True


def test_preflight_records_failed_transpile_per_case(fake_runtime_service, monkeypatch):
    """A case whose transpile raises (e.g. Shor's 17-qubit multi-controlled
    X on a RPhi+CZ gate set) must not abort the run: the case is recorded
    failed/untranspiled and the remaining cases still qualify."""
    original_transpile = FakeFakeBackend.transpile

    def failing_transpile(self, progs, specified_block=None, is_optimization=True):
        # Only the Shor order-finding program transpiles as a single
        # 17-qubit program (QARM's 17-qubit circuits arrive in a batch
        # of four); every other case transpiles normally.
        if len(progs) == 1 and _qubit_count(progs[0]) == 17:
            raise RuntimeError("transpile boom")
        return original_transpile(self, progs, specified_block, is_optimization)

    monkeypatch.setattr(FakeFakeBackend, "transpile", failing_transpile)
    manifest = run_preflight(fake_runtime_service, "WK_C180")
    shor = next(case for case in manifest.cases if case.algorithm == "Shor")
    assert shor.verdict == "failed"
    assert shor.transpiled is False
    assert "RuntimeError" in shor.parsed_result["outcome"]
    others = [case for case in manifest.cases if case.algorithm != "Shor"]
    assert len(others) == len(manifest.cases) - 1
    assert all(case.transpiled for case in others)
    assert all("transpile boom" not in case.parsed_result["outcome"] for case in others)


def _qubit_count(prog) -> int:
    """Qubit count of a QCircuit/QProg (both surfaces expose methods)."""
    count = getattr(prog, "qubits_num", None)
    return count() if callable(count) else len(prog.qubits())
