"""Local CPU backend tests: sampling, estimation, and statevector execution.

LocalBackend is the default CPU execution path: work is synchronous and
every submission returns an already-finished task.  These tests pin the
sampling, estimation, and state-vector surfaces plus the
``resolve_backend()`` default rule.
"""

import numpy as np
import pytest
from pyqpanda3.core import CNOT, H, QProg, measure
from pyqpanda3.hamiltonian import Hamiltonian

from pyqpanda_alg.execution import (
    AlgorithmInputError,
    ExecutionOptions,
    LocalBackend,
    resolve_backend,
)
from pyqpanda_alg.execution.result_normalization import probability_to_counts


def bell_program():
    prog = QProg()
    prog << H(0) << CNOT(0, 1) << measure([0, 1], [0, 1])
    return prog


def test_local_sample_returns_counts():
    task = LocalBackend().submit_sample(
        bell_program(), options=ExecutionOptions(shots=200)
    )
    counts = task.result().single_counts()
    assert sum(counts.values()) == 200
    assert set(counts).issubset({"00", "11"})


def test_local_estimate_returns_float():
    prog = QProg()
    prog << H(0)
    task = LocalBackend().submit_estimate(
        (prog, Hamiltonian({"X0": 1.0})),
        options=ExecutionOptions(shots=1),
    )
    assert abs(task.result().single_value() - 1.0) < 1e-9


def test_local_statevector_returns_normalized_state():
    prog = QProg()
    prog << H(0)
    state = LocalBackend().submit_statevector(prog, options=ExecutionOptions()).result().single_statevector()
    assert np.allclose(np.abs(state) ** 2, [0.5, 0.5])


def test_resolve_backend_defaults_only_when_none():
    local = resolve_backend(None)
    supplied = LocalBackend()
    assert isinstance(local, LocalBackend)
    assert resolve_backend(supplied) is supplied


def test_local_sample_result_preserves_shots():
    result = LocalBackend().submit_sample(
        bell_program(), options=ExecutionOptions(shots=123)
    ).result()
    assert result.shots == 123


def test_local_estimate_rejects_measured_circuit():
    with pytest.raises(AlgorithmInputError):
        LocalBackend().submit_estimate(
            (bell_program(), Hamiltonian({"X0": 1.0})),
            options=ExecutionOptions(),
        )


def test_local_statevector_rejects_measured_circuit():
    with pytest.raises(AlgorithmInputError):
        LocalBackend().submit_statevector(bell_program(), options=ExecutionOptions())


def test_local_statevector_is_readonly_copy():
    prog = QProg()
    prog << H(0)
    state = LocalBackend().submit_statevector(prog, options=ExecutionOptions()).result().single_statevector()
    assert not state.flags.writeable
    with pytest.raises(ValueError):
        state[0] = 1.0


def test_probability_to_counts_rejects_nonpositive_shots():
    with pytest.raises(ValueError, match="shots"):
        probability_to_counts({"0": 1.0}, shots=0)


def test_probability_to_counts_apportions_rounding_remainder():
    counts = probability_to_counts({"0": 0.5, "1": 0.5}, shots=3)
    assert counts == {"0": 2, "1": 1}
    assert sum(counts.values()) == 3


def test_probability_to_counts_rejects_malformed_probability_sum():
    with pytest.raises(ValueError, match="at most 1.0"):
        probability_to_counts({"0": 0.6, "1": 0.6}, shots=10)
