"""Runtime-contract tests for QARM sampling execution.

Plan 3 Task 3: the association-rule search flows through
``backend.submit_sample`` and consumes only the public batch-result
surface, never legacy QCloudResult-style attributes such as
``get_prob_dict``.  The QCloud entry point survives one release as a
deprecated shim that requires an explicit backend instead of an
``api_key``.

The shared ``recording_backend`` fixture scripts the exact per-round
counts of the legacy search on ``dataset/data2.txt``, so the mined
rules are deterministic and pin the legacy result exactly.
"""

import pytest

from pyqpanda_alg.execution import AlgorithmInputError


def test_qarm_runtime_uses_execution_backend(qarm_fixture, recording_backend):
    result = qarm_fixture.run(backend=recording_backend)
    assert result
    assert recording_backend.sample_call_count > 0
    assert "get_prob_dict" not in recording_backend.accessed_result_attributes


def test_qarm_cloud_entry_warns_and_requires_explicit_backend(qarm_fixture):
    with pytest.warns(DeprecationWarning):
        with pytest.raises(AlgorithmInputError):
            qarm_fixture.run(machine_type="QCloud")


def test_qarm_cloud_entry_runs_on_explicit_backend(qarm_fixture, recording_backend):
    with pytest.warns(DeprecationWarning):
        result = qarm_fixture.run(machine_type="QCloud", backend=recording_backend)
    assert result
    assert recording_backend.sample_call_count > 0


def test_qarm_unknown_machine_type_keeps_raising(qarm_fixture):
    with pytest.raises(TypeError):
        qarm_fixture.run(machine_type="GPU")


def test_qarm_result_pins_legacy_data2_rules(qarm_fixture, recording_backend):
    result = qarm_fixture.run(backend=recording_backend)
    assert result == {
        "牛奶->面包": 1.0,
        "奶酪->面包": 1.0,
        "黄油->面包": 1.0,
        "奶酪->黄油": 0.5,
        "黄油->奶酪": 0.5,
        "奶酪,面包->黄油": 0.5,
        "面包,黄油->奶酪": 0.5,
        "奶酪,黄油->面包": 1.0,
    }
