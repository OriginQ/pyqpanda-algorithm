"""Runtime-contract tests for GroverAdaptiveSearch (GAS) sampling tasks.

Plan 3 Task 3: the adaptive search is a checkpointable sampling state
machine.  ``submit()`` returns before any round is sampled, each
``poll()`` runs exactly one search round, ``checkpoint()`` preserves
the current round, and ``resume()`` continues from there without
re-submitting completed rounds.  QUBO_GAS delegates to the migrated
searcher.

Scripted single-outcome counts keep every round deterministic: with
``m < 2`` in the first rounds the rotation count is always 1, and the
outcome is simply the scripted key.
"""

import sympy as sp

from pyqpanda_alg.Grover.Grover_core import GroverAdaptiveSearch
from pyqpanda_alg.QUBO import QUBO
from pyqpanda_alg.execution import AlgorithmTask, TaskStatus
from test.execution.fakes import RecordingBackend


def test_gas_submit_returns_before_any_round():
    backend = RecordingBackend(sample_counts=[{"00": 1}])
    searcher = GroverAdaptiveSearch(init_value=0, n_index=2)
    task = searcher.submit(
        continue_times=3,
        n_value_function=lambda current_min: 1,
        value_function=lambda key: 0,
        backend=backend,
    )
    assert task.status() is TaskStatus.PENDING
    assert backend.sample_call_count == 0
    assert task.poll() is TaskStatus.RUNNING
    assert backend.sample_call_count == 1
    assert task.backend_task_ids == ["recorded-sample-1"]
    assert task.poll() is TaskStatus.RUNNING
    assert task.poll() is TaskStatus.SUCCEEDED
    assert backend.sample_call_count == 3
    assert task.backend_task_ids == [
        "recorded-sample-1",
        "recorded-sample-2",
        "recorded-sample-3",
    ]
    # No improvement ever found: empty solution set at the init value.
    assert task.result() == ([], 0)


def test_gas_run_uses_sampling_without_legacy_result_api():
    backend = RecordingBackend(sample_counts=[{"10": 1}])
    searcher = GroverAdaptiveSearch(init_value=0, n_index=2)
    minimum_indexes, minimum_res = searcher.run(
        continue_times=2,
        n_value_function=lambda current_min: 1,
        value_function=lambda key: -1 if key == "10" else 0,
        backend=backend,
    )
    assert minimum_indexes == [[0, 1]]
    assert minimum_res == -1
    assert backend.sample_call_count == 3
    assert "get_prob_dict" not in backend.accessed_result_attributes


def test_gas_resume_continues_without_replaying_completed_rounds(tmp_path):
    backend = RecordingBackend(sample_counts=[{"00": 1}])
    searcher = GroverAdaptiveSearch(init_value=0, n_index=2)
    task = searcher.submit(
        continue_times=3,
        n_value_function=lambda current_min: 1,
        value_function=lambda key: 0,
        backend=backend,
    )
    assert task.poll() is TaskStatus.RUNNING
    assert task.poll() is TaskStatus.RUNNING
    assert task.backend_task_ids == ["recorded-sample-1", "recorded-sample-2"]
    checkpoint = tmp_path / "gas-checkpoint.json"
    task.checkpoint(checkpoint)

    resumed = AlgorithmTask.resume(checkpoint, backend=backend)
    # Completed rounds are carried over, not replayed.
    assert resumed.backend_task_ids == task.backend_task_ids
    assert backend.sample_call_count == 2
    assert resumed.poll() is TaskStatus.SUCCEEDED
    assert resumed.backend_task_ids == [
        "recorded-sample-1",
        "recorded-sample-2",
        "recorded-sample-3",
    ]
    assert len(resumed.backend_task_ids) == len(set(resumed.backend_task_ids))
    assert backend.sample_call_count == len(resumed.backend_task_ids)


def test_qubo_gas_delegates_to_migrated_searcher():
    x0, x1, x2 = sp.symbols("x0 x1 x2")
    function = (
        -0.5 * x0 * x1
        - 0.7 * x0 * x1
        + 0.9 * x1 * x2
        + 1.3 * x0
        - x1
        - 0.5 * x2
    )
    backend = RecordingBackend(sample_counts=[{"010": 1}])
    model = QUBO.QUBO_GAS_origin(function)
    gas_solution, gas_value = model.run(init_value=0, continue_times=2, backend=backend)
    assert gas_solution == [[0, 1, 0]]
    assert gas_value == -1.0
    assert backend.sample_call_count == 3
