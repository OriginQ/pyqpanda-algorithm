import json

import pytest

from pyqpanda_alg.QSEncode.cli import main


PROBABILITIES = [
    0.0006917643261373052,
    0.015724004731018214,
    0.1261730210273901,
    0.3574112099154543,
    0.3574112099154544,
    0.1261730210273902,
    0.01572400473101823,
    0.0006917643261373052,
]


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "QSEncode-Insight" in capsys.readouterr().out


def test_cli_valid_standard_analysis_outputs_json(capsys):
    exit_code = main([
        "analyze",
        "--basis", "fourier",
        "--input-json", json.dumps(PROBABILITIES),
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection"]["decision"] == "compress"
    assert payload["selection"]["selected_candidate_id"] == "compressed__k4__sparse_isometry"
    assert payload["semantic_verification"]["status"] == "not_run_by_standard"


def test_cli_invalid_basis_is_rejected(capsys):
    with pytest.raises(SystemExit) as error:
        main([
            "analyze",
            "--basis", "auto",
            "--input-json", "[0.5, 0.5]",
        ])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["not-json", '{"not": "a list"}'])
def test_cli_json_input_failure_is_structured(value, capsys):
    with pytest.raises(SystemExit) as error:
        main([
            "analyze",
            "--basis", "walsh",
            "--input-json", value,
        ])
    assert error.value.code == 2
    assert "probability input" in capsys.readouterr().err


def test_cli_reads_json_file(tmp_path, capsys):
    path = tmp_path / "probabilities.json"
    path.write_text(json.dumps(PROBABILITIES), encoding="utf-8")

    assert main([
        "analyze",
        "--basis", "walsh",
        "--input-file", str(path),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection"]["decision"] == "do_not_compress"
