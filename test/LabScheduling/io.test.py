"""Input snapshot provenance and non-destructive report publication regressions."""

import hashlib
import json
from pathlib import Path

import pyqpanda_alg.LabScheduling.__main__ as cli
import pyqpanda_alg.LabScheduling.report as reports
import pytest
from pyqpanda_alg.LabScheduling import load_problem, parse_problem

DATA = (
    Path(__file__).resolve().parents[2]
    / "pyqpanda-algorithm/example/LabScheduling/data"
)


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hardlink", "directory"])
def test_output_aliases_rejected_before_training_without_changing_input(
    tmp_path, monkeypatch, capsys, alias
):
    source = tmp_path / "input.json"
    original = (DATA / "simple.json").read_bytes()
    source.write_bytes(original)
    output = source if alias == "same_path" else tmp_path / "output.json"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        output.hardlink_to(source)
    elif alias == "directory":
        output.mkdir()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid output must be rejected before running the simulator")

    monkeypatch.setattr(cli, "run_experiment", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main([str(source), "--output", str(output)])
    assert error.value.code == 2
    assert "output" in capsys.readouterr().err
    assert source.read_bytes() == original


@pytest.mark.parametrize("mutation", ["modify", "delete"])
def test_report_hash_describes_the_solved_snapshot_even_if_input_changes(
    tmp_path, monkeypatch, mutation
):
    source = tmp_path / "input.json"
    original = (DATA / "simple.json").read_bytes().replace(b"\n", b"\r\n")
    source.write_bytes(original)
    real_solver = reports.solve_qaoa

    def solve_then_change_file(*args, **kwargs):
        result = real_solver(*args, **kwargs)
        if mutation == "modify":
            changed = json.loads(original)
            changed["name"] = "modified after loading"
            source.write_text(json.dumps(changed), encoding="utf-8")
        else:
            source.unlink()
        return result

    monkeypatch.setattr(reports, "solve_qaoa", solve_then_change_file)
    report = reports.run_experiment(source, shots=2, maxiter=6, restarts=1)
    assert report["problem"] == json.loads(original)["name"]
    assert report["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["exact"]["best"]["objective"] == 5


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_report_write_preserves_previous_report_and_removes_temporary(
    tmp_path, monkeypatch, capsys, failure
):
    source = tmp_path / "input.json"
    source.write_text("input must survive", encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_text("previous complete report", encoding="utf-8")
    monkeypatch.setattr(cli, "run_experiment", lambda *a, **k: {"status": "feasible"})
    if failure == "replace":

        def reject_replace(*args, **kwargs):
            raise OSError("simulated replacement failure")

        monkeypatch.setattr(Path, "replace", reject_replace)
    else:
        real_temporary = cli.NamedTemporaryFile

        def failing_temporary(*args, **kwargs):
            stream = real_temporary(*args, **kwargs)
            write = stream.write

            def partial_write(text):
                write(text[:5])
                raise OSError("simulated full filesystem during write")

            stream.write = partial_write
            return stream

        monkeypatch.setattr(cli, "NamedTemporaryFile", failing_temporary)
    with pytest.raises(SystemExit) as error:
        cli.main([str(source), "--output", str(output)])
    assert error.value.code == 2
    assert "simulated" in capsys.readouterr().err
    assert source.read_text() == "input must survive"
    assert output.read_text() == "previous complete report"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.json", "report.json"]


def test_atomic_report_replacement_preserves_a_valid_output_symlink(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("original input", encoding="utf-8")
    target = tmp_path / "report.json"
    target.write_text("old report", encoding="utf-8")
    link = tmp_path / "report-link.json"
    link.symlink_to(target)
    cli._save_report(source, link, '{"status": "feasible"}')
    assert link.is_symlink()
    assert target.read_text() == '{"status": "feasible"}\n'
    assert source.read_text() == "original input"
    assert len(list(tmp_path.iterdir())) == 3


def test_text_parser_preserves_file_loader_validation():
    text = (DATA / "simple.json").read_text(encoding="utf-8")
    assert parse_problem(text) == load_problem(DATA / "simple.json")
    for invalid in ('{"name":"a","name":"b"}', '{"horizon":NaN}', "[]"):
        with pytest.raises(ValueError):
            parse_problem(invalid)
