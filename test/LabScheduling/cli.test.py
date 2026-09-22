"""CLI contract: reports, exit status, actual samples and no secret configuration."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pyqpanda_alg.LabScheduling.__main__ import main

DATA = (
    Path(__file__).resolve().parents[2]
    / "pyqpanda-algorithm/example/LabScheduling/data"
)


def test_one_command_cli_and_saved_report(tmp_path):
    output = tmp_path / "nested" / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyqpanda_alg.LabScheduling",
            str(DATA / "simple.json"),
            "--maxiter",
            "25",
            "--restarts",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    report = json.loads(process.stdout)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "feasible"
    assert report["absolute_gap"] == 0
    assert (
        report["quantum"]["best"]["objective"]
        == report["exact"]["best"]["objective"]
        == 5
    )
    assert len(report["input_sha256"]) == 64
    assert report["environment"]["pyqpanda3"]
    assert report["uniform_one_hot"]["feasible_probability"] == pytest.approx(4 / 9)


def test_infeasible_exit_and_report(capsys):
    status = main([str(DATA / "infeasible.json"), "--maxiter", "6", "--restarts", "1"])
    assert status == 3
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "infeasible"
    assert report["exact"]["best"] is None
    assert report["quantum"]["status"] == "no_feasible_sample"
    assert report["absolute_gap"] is None


def test_missing_or_invalid_input_exit(tmp_path, capsys):
    for path in [tmp_path / "missing.json", tmp_path / "bad.json"]:
        if path.name == "bad.json":
            path.write_text('{"name": "only"}', encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            main([str(path)])
        assert exc.value.code == 2
        assert "error:" in capsys.readouterr().err


def test_finite_shots_miss_is_distinct_from_infeasibility(capsys):
    status = main(
        [
            str(DATA / "simple.json"),
            "--shots",
            "1",
            "--mixer",
            "x",
            "--maxiter",
            "6",
            "--restarts",
            "1",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert status == 4
    assert report["status"] == "no_feasible_sample"
    assert report["exact"]["status"] == "optimal"
    assert report["quantum"]["best"]["objective"] is None
    assert report["absolute_gap"] is None
