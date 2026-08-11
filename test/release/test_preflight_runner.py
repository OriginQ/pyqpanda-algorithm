"""FakeBackend preflight runner: credential-free preflight evidence.

Plan 7 Task 3: ``run_preflight`` queries the device through the
credential-free service stand-in, constructs the device's FakeBackend,
runs every fixed qualification and transpilation case against it, and
returns a sanitized ``QualificationManifest``.  The test below is
verbatim from the task brief.
"""

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
