import numpy as np
import pytest

from pyqpanda_alg.execution import (
    AlgorithmInputError,
    EstimateBatchResult,
    ExecutionOptions,
    PreflightMode,
    SampleBatchResult,
    StatevectorBatchResult,
)


def test_execution_options_reject_nonpositive_shots():
    with pytest.raises(ValueError, match="shots"):
        ExecutionOptions(shots=0)


def test_default_options_are_safe_for_runtime():
    options = ExecutionOptions()
    assert options.shots == 1000
    assert options.timeout == 1800.0
    assert options.preflight is PreflightMode.TRANSPILE_ONLY
    assert options.is_mapping is True


def test_single_counts_requires_exactly_one_result():
    result = SampleBatchResult(counts=({"00": 1}, {"11": 1}), shots=1000)
    with pytest.raises(AlgorithmInputError, match="exactly one"):
        result.single_counts()
    empty = SampleBatchResult(counts=(), shots=1000)
    with pytest.raises(AlgorithmInputError, match="exactly one"):
        empty.single_counts()
    assert SampleBatchResult(counts=({"00": 1000},), shots=1000).single_counts() == {
        "00": 1000
    }


def test_single_value_requires_exactly_one_result():
    result = EstimateBatchResult(values=(0.1, 0.2))
    with pytest.raises(AlgorithmInputError, match="exactly one"):
        result.single_value()
    empty = EstimateBatchResult(values=())
    with pytest.raises(AlgorithmInputError, match="exactly one"):
        empty.single_value()
    assert EstimateBatchResult(values=(0.5,)).single_value() == 0.5


def test_single_statevector_requires_exactly_one_result():
    result = StatevectorBatchResult(statevectors=([1.0, 0.0], [0.0, 1.0]))
    with pytest.raises(AlgorithmInputError, match="exactly one"):
        result.single_statevector()
    empty = StatevectorBatchResult(statevectors=())
    with pytest.raises(AlgorithmInputError, match="exactly one"):
        empty.single_statevector()
    single = StatevectorBatchResult(statevectors=([1.0, 0.0],)).single_statevector()
    assert np.allclose(single, [1.0, 0.0])


def test_raw_metadata_is_deep_copied():
    nested = {"inner": {"key": 1}}
    result = SampleBatchResult(
        counts=({"00": 1},), shots=1, raw_metadata={"nested": nested}
    )
    result.raw_metadata["nested"]["inner"]["key"] = 99
    assert nested["inner"]["key"] == 1  # caller's dict is not aliased
