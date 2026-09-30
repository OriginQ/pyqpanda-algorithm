"""Tests for the public :mod:`pyqpanda_alg.BFDCQO` API."""

import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "pyqpanda-algorithm"


def test_bfdcqo_is_available_from_top_level_package_in_fresh_process():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pyqpanda_alg; assert hasattr(pyqpanda_alg, 'BFDCQO')",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_all_public_classes_and_functions_have_docstrings():
    module = importlib.import_module("pyqpanda_alg.BFDCQO")
    missing = [
        name
        for name in module.__all__
        if (inspect.isclass(getattr(module, name)) or inspect.isfunction(getattr(module, name)))
        and not inspect.getdoc(getattr(module, name))
    ]

    assert missing == []
