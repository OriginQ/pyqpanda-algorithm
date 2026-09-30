# Licensed under the Apache License, Version 2.0.
"""End-to-end checks for the reproducible CPUQVM benchmark driver."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_bfdcqo_cpuqvm_benchmarks.py"
BF_EXECUTION_MODEL = (
    "PyQPanda3 CPUQVM ideal probabilities + seeded NumPy multinomial "
    "feedback sampling"
)
QAOA_EXECUTION_MODEL = "pyqpanda_alg.QAOA.CPUQVM shot execution"


def _clean_environment():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def test_cli_help_works_without_pythonpath_from_an_arbitrary_directory(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--output" in completed.stdout


def test_quick_cli_writes_real_cpuqvm_evidence(tmp_path):
    output = tmp_path / "evidence"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--output", str(output)],
        cwd=tmp_path,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary_names = (
        "maxcut_summary.csv",
        "max3sat_summary.csv",
        "family_summary.csv",
        "mwis_summary.csv",
    )
    for name in summary_names:
        path = output / name
        assert path.is_file(), name
        with path.open(newline="", encoding="utf-8") as handle:
            assert next(csv.DictReader(handle), None) is not None

    metadata = json.loads(
        (output / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["mode"] == "quick"
    assert metadata["seeds"] == [0]
    assert metadata["backends"] == {
        "bfdcqo": "pyqpanda3.CPUQVM",
        "qaoa": "pyqpanda_alg.QAOA.CPUQVM",
    }
    assert isinstance(metadata["git_dirty"], bool)
    assert metadata["driver_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()
    assert metadata["dependency_versions"].keys() == {
        "numpy", "scipy", "sympy", "matplotlib"
    }
    assert metadata["execution_models"] == {
        "bfdcqo": BF_EXECUTION_MODEL,
        "qaoa": QAOA_EXECUTION_MODEL,
    }
    assert "NumPy multinomial" in metadata["seed_scope"]["bfdcqo"]
    assert "optimizer initialization only" in metadata["seed_scope"]["qaoa"]
    assert "resource proxy" in metadata["budget_semantics"]["bfdcqo"]
    assert metadata["budget_semantics"]["qaoa"] == (
        "observed circuit_evaluations * shots"
    )

    for raw_name in (
        "maxcut_raw.csv",
        "max3sat_raw.csv",
        "family_raw.csv",
        "mwis_raw.csv",
    ):
        assert (output / raw_name).is_file()

    expected_summary_rows = {
        "maxcut": 6,
        "max3sat": 2,
        "family": 6,
        "mwis": 4,
    }
    for name, expected_rows in expected_summary_rows.items():
        with (output / f"{name}_summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected_rows
        assert {row["seed_count"] for row in rows} == {"1"}
        assert {row["execution_model"] for row in rows} <= {
            BF_EXECUTION_MODEL,
            QAOA_EXECUTION_MODEL,
        }
        assert all(float(row["mean_budget_delta"]) == 0.0 for row in rows)

    with (output / "maxcut_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        maxcut_rows = list(csv.DictReader(handle))
    assert {
        (row["instance"], row["solver"]) for row in maxcut_rows
    } == {
        (instance, solver)
        for instance in ("complete5", "cycle6", "weighted6")
        for solver in ("BF-DCQO", "QAOA-p1-upstream")
    }


def test_quick_mode_rejects_documentation_output(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--output",
            str(tmp_path / "raw"),
            "--docs-output",
            str(tmp_path / "docs"),
        ],
        cwd=tmp_path,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode != 0
    assert "official mode" in completed.stderr
    assert not (tmp_path / "raw").exists()
