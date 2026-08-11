"""Runtime-contract tests for QSVD observable-based optimization.

Plan 3 Task 5: the QSVD overlap cost is expressed as an observable and
submitted through ``backend.submit_estimate``, so optimizing never
constructs a CPUQVM or StateVector internally.  The recording backend
replays canned expectations so the whole SLSQP pipeline runs
deterministically without any QVM, and the call mix proves the cost is
estimated rather than sampled or statevector-evolved.  Exact
singular-vector requests (``return_diag``/``max_eig``) require a backend
with state-vector capability and raise ``DeviceCapabilityError`` before
submission otherwise.
"""

import numpy as np
import pytest

from pyqpanda_alg.QSVD import SVD
from pyqpanda_alg.execution import (
    BackendCapabilities,
    DeviceCapabilityError,
    LocalBackend,
)

from test.execution.fakes import RecordingBackend


def test_qsvd_runtime_cost_uses_observable_estimates(recording_backend):
    solver = SVD([[1.0, 0.0], [0.0, 0.5]])
    solver.QSVD_min(backend=recording_backend, maxiter=1)
    assert recording_backend.estimate_call_count > 0
    assert recording_backend.statevector_call_count == 0


def test_qsvd_local_backend_recovers_singular_values():
    matrix = np.array([[1.0, 0.0], [0.0, 0.5]])
    solver = SVD(matrix_in=matrix)
    para = solver.QSVD_min(backend=LocalBackend())
    qeig = solver.return_diag(para)
    q_singular_values = np.sort(np.diag(qeig))[::-1]
    np_singular_values = np.sort(np.linalg.svd(matrix)[1])[::-1]
    assert np.all(q_singular_values >= 0)
    assert np.allclose(q_singular_values, np_singular_values, atol=0.05)


def test_qsvd_rectangular_overlap_measures_exact_diagonal():
    # Rectangular non-diagonal matrices: the overlap observable must
    # count only the exact diagonal entries P(row=i, col=i) over
    # i < 2**min(q0, q1), i.e. the projector onto zero on the extra
    # high bits of the larger register.  Zero variational parameters
    # leave the circuit the identity, so the cost is the overlap of the
    # amplitude-encoded matrix itself, and LocalBackend's expectation
    # is exact (no sampling).
    for matrix in (
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),  # q0 > q1
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),                # q1 > q0
    ):
        solver = SVD(matrix_in=matrix)
        para = np.zeros((solver.q0 + solver.q1) * solver.iter_depth)
        overlap = 1.0 - solver.loss(para, return_type=True)
        classical = np.sum(np.diag(solver.q_matrix) ** 2) / solver.normal_value**2
        assert overlap == pytest.approx(classical, abs=1e-6)


def test_qsvd_records_loss_value_and_uncertainty(recording_backend):
    solver = SVD([[1.0, 0.0], [0.0, 0.5]])
    solver.QSVD_min(backend=recording_backend, maxiter=1)
    # The recording backend replays a 0.5 expectation for every cost
    # estimate, so the last recorded loss is 1 - 0.5 with the Bernoulli
    # shot-count standard error at the default 1000 shots.
    assert solver.loss_value == pytest.approx(0.5)
    assert solver.loss_uncertainty == pytest.approx((0.5 * 0.5 / 1000) ** 0.5)


def test_qsvd_singular_vectors_raise_capability_error_before_submission():
    backend = RecordingBackend()
    backend.capabilities = BackendCapabilities(statevector=False, tomography=False)
    solver = SVD([[1.0, 0.0], [0.0, 0.5]])
    para = solver.QSVD_min(backend=backend, maxiter=1)
    with pytest.raises(DeviceCapabilityError):
        solver.return_diag(para)
    with pytest.raises(DeviceCapabilityError):
        solver.max_eig("0", para, 0)
    assert backend.statevector_call_count == 0
