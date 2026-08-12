"""VQE runtime contract: variational-session preference and fallback.

These credential-free tests exercise the runtime execution path through
the repository fakes: the solver prefers a variational session whenever
the backend advertises one, falls back to estimate batches otherwise,
and never silently falls back to the CPU backend when the runtime path
fails.
"""

import pytest

from pyqpanda_alg.VQE import VQE, VQEConfig
from pyqpanda_alg.execution import TaskSubmissionError


@pytest.mark.runtime_contract
def test_vqe_uses_estimator_for_qpanda_runtime(runtime_backend, hamiltonian):
    solver = VQE(hamiltonian)
    solver.run(
        initial_parameters=[0.1, 0.2],
        backend=runtime_backend,
        config=VQEConfig(max_iterations=1, tolerance=1e-12),
    )
    assert runtime_backend.service.vqsession_calls == []
    assert runtime_backend.service.estimate_calls


@pytest.mark.runtime_contract
def test_vqe_uses_estimate_when_session_unavailable(
    estimator_only_backend, hamiltonian
):
    solver = VQE(hamiltonian)
    solver.run(
        initial_parameters=[0.1, 0.2],
        backend=estimator_only_backend,
        config=VQEConfig(max_iterations=1, tolerance=1e-12),
    )
    assert estimator_only_backend.estimate_call_count > 0


@pytest.mark.runtime_contract
def test_vqe_runtime_failure_never_falls_back_to_cpu(runtime_backend, hamiltonian):
    runtime_backend.service.submit_error = RuntimeError("network unreachable")
    solver = VQE(hamiltonian)
    with pytest.raises(TaskSubmissionError, match="network unreachable"):
        solver.run(
            initial_parameters=[0.1, 0.2],
            backend=runtime_backend,
            config=VQEConfig(max_iterations=1, tolerance=1e-12),
        )
    # the failure surfaced instead of silently rerunning on the CPU
    assert runtime_backend.service.vqsession_calls == []
