"""Fixtures for the Plan 3 algorithm runtime-migration suites."""

import os

import pytest

from pyqpanda_alg import QARM
from pyqpanda_alg.QARM import QuantumAssociationRulesMining


def _read_transactions(file_path):
    """Read the comma-separated transaction file used by the QARM tests."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist!")
    with open(file_path, "r", encoding="utf8") as handle:
        data_line = handle.readlines()
    if not data_line:
        raise ValueError(f"The file {file_path} has no any data!")
    return [
        [data.strip() for data in line.strip().split(",")]
        for line in data_line
        if line
    ]


@pytest.fixture
def qarm_fixture():
    """QARM miner over the package's ``dataset/data2.txt`` fixture.

    Uses the same support/confidence values as the active QARM test
    (``test/QARM/Test_qarm.py``): support 0.2, confidence 0.5.  The
    fixture is credential-free and deterministic.
    """
    data_file = os.path.join(QARM.__path__[0], "dataset/data2.txt")
    transactions = _read_transactions(data_file)
    return QuantumAssociationRulesMining(transactions, 0.2, 0.5)
