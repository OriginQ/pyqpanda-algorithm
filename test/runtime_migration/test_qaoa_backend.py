"""Runtime-contract tests for QAOA estimator execution.

Plan 3 Task 2: QAOA builds a measurement-free QProg and Hamiltonian and
submits estimation instead of running a CPUQVM.  The recording backend
replays canned expectations so the whole optimization pipeline runs
deterministically without any QVM, and the call mix proves the
objective is estimated rather than sampled.
"""

import sympy as sp

from pyqpanda_alg.QAOA.qaoa import QAOA
from test.execution.fakes import RecordingBackend


def test_qaoa_runtime_uses_estimator_without_cpu_fallback():
    x0 = sp.Symbol("x0")
    backend = RecordingBackend(expectations=[0.5, -0.25, -0.5])
    model = QAOA(x0)
    model.run(layer=1, optimizer_option={"options": {"maxiter": 1}}, backend=backend)
    assert backend.estimate_call_count > 0
    assert backend.sample_call_count == 0
