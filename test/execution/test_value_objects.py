import pytest
from pyqpanda_alg.execution import ExecutionOptions, PreflightMode


def test_execution_options_reject_nonpositive_shots():
    with pytest.raises(ValueError, match="shots"):
        ExecutionOptions(shots=0)


def test_default_options_are_safe_for_runtime():
    options = ExecutionOptions()
    assert options.shots == 1000
    assert options.timeout == 1800.0
    assert options.preflight is PreflightMode.TRANSPILE_ONLY
    assert options.is_mapping is True
