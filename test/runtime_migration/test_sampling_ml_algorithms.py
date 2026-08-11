"""Runtime-contract tests for the Plan 3 Task 4 sampling entry points.

Plan 3 Task 4: QKmeans, QPCA, QSVM, QSVR, and QSEncode flow through
``backend.submit_sample`` (QSEncode uses ``submit_statevector`` only on
backends that advertise that capability) and consume only the public
batch-result surface, never legacy QCloudResult-style attributes such as
``get_prob_dict``.

The module-local ``recording_backend`` fixture scripts per-round counts
every entry point tolerates: the ancilla/overlap ``|1>`` weight
alternates so the QKMeans distance loop stays non-degenerate, and every
dict keeps ``0``/``1`` keys so QPCA's phase-register readout and
QSEncode's probability list stay well-defined.  The root fixture's QARM
scripts (17-bit index-register keys) would starve these algorithms'
single-bit and overlap reads, so this suite pins its own backend.
"""

import numpy as np
import pytest

from pyqpanda_alg.QKmeans import QuantumKmeans
from pyqpanda_alg.QPCA import qpca
from pyqpanda_alg.QSEncode import QSpare_Code
from pyqpanda_alg.QSVM import QuantumKernel_vqnet
from pyqpanda_alg.QSVR import Quantum_SVR
from pyqpanda_alg.execution import BackendCapabilities, DeviceCapabilityError, LocalBackend

from test.execution.fakes import RecordingBackend


def _sampling_counts_pattern():
    """Per-round counts every sampling entry point can parse.

    QKMeans measures the swap-test ancilla (one qubit) and needs the
    ``|1>`` weight to alternate per point/centroid call so its distance
    argmin stays non-degenerate and the loop converges; the other entry
    points tolerate any of these dicts (QPCA reads the trailing bit,
    QSVM falls back when the ``00`` key is absent, QSVR and QSEncode
    default to zero mass on keys they do not see).
    """
    high = {"1": 20, "0": 10}
    low = {"1": 10, "0": 10}
    return [high, low, low, high] * 7


class SamplingOnlyRecordingBackend(RecordingBackend):
    """Recording backend without the state-vector capability.

    QSEncode must take its sampling path on this backend so one
    scripted fixture can drive all five sampling entry points.
    """

    capabilities = BackendCapabilities(statevector=False)


@pytest.fixture
def recording_backend():
    """Backend scripting the five sampling entry points."""
    return SamplingOnlyRecordingBackend(sample_counts=_sampling_counts_pattern())


def assert_sampled(callable_, recording_backend):
    before = recording_backend.sample_call_count
    callable_()
    assert recording_backend.sample_call_count > before


def test_public_sampling_entry_points_use_supplied_backend(recording_backend):
    points = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]])
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])

    assert_sampled(lambda: QuantumKmeans(k=2).fit(points, backend=recording_backend), recording_backend)
    assert_sampled(lambda: qpca(points, 1, backend=recording_backend), recording_backend)
    assert_sampled(
        lambda: QuantumKernel_vqnet(n_qbits=2).evaluate(x, backend=recording_backend),
        recording_backend,
    )
    assert_sampled(lambda: Quantum_SVR(x, y).get_res(backend=recording_backend), recording_backend)
    assert_sampled(
        lambda: QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(backend=recording_backend),
        recording_backend,
    )


def test_sampling_entry_points_avoid_legacy_result_attributes(recording_backend):
    points = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]])
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])

    QuantumKmeans(k=2).fit(points, backend=recording_backend)
    qpca(points, 1, backend=recording_backend)
    QuantumKernel_vqnet(n_qbits=2).evaluate(x, backend=recording_backend)
    Quantum_SVR(x, y).get_res(backend=recording_backend)
    QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(backend=recording_backend)
    assert "get_prob_dict" not in recording_backend.accessed_result_attributes
    assert "get_prob_list" not in recording_backend.accessed_result_attributes


def test_qkmeans_local_backend_clusters_the_points():
    points = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]])
    centers, clusters = QuantumKmeans(k=2).fit(points, backend=LocalBackend())
    assert centers.shape == (2, 2)
    assert clusters.shape == (4,)


def test_qpca_local_backend_reduces_dimensions():
    points = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]])
    data = qpca(points, 1, backend=LocalBackend())
    assert data.shape == (4, 1)
    assert isinstance(data, np.ndarray)


def test_qsvm_local_backend_builds_kernel_matrix():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    kernel = QuantumKernel_vqnet(n_qbits=2).evaluate(x, backend=LocalBackend())
    assert kernel.shape == (2, 2)


def test_qsvr_local_backend_predicts_training_set():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])
    y_pred, y_true = Quantum_SVR(x, y).get_res(backend=LocalBackend())
    assert y_pred.shape == (2,)
    assert np.array_equal(y_true, y)


def test_qsen_code_local_backend_returns_probability_distribution():
    res = QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(backend=LocalBackend())
    assert res == pytest.approx([0.5, 0.5])


def test_qsen_code_uses_statevector_when_backend_advertises_it():
    backend = RecordingBackend(statevector=[[0.5 ** 0.5, 0.5 ** 0.5]])
    res = QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(backend=backend)
    assert backend.statevector_call_count == 1
    assert backend.sample_call_count == 0
    assert res == pytest.approx([0.5, 0.5])


def test_qsen_code_observables_are_estimated():
    backend = RecordingBackend(expectations=[0.5])
    res = QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(
        observables=["Z0"], backend=backend
    )
    assert backend.estimate_call_count == 1
    assert res == [0.5]


def test_qsen_code_raises_capability_error_before_submission():
    backend = RecordingBackend()
    backend.capabilities = BackendCapabilities(
        sampling=False, statevector=False, estimation=False
    )
    with pytest.raises(DeviceCapabilityError):
        QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(backend=backend)
    assert backend.sample_call_count == 0
    assert backend.statevector_call_count == 0


def test_qsen_code_observables_raise_capability_error_before_submission():
    backend = RecordingBackend()
    backend.capabilities = BackendCapabilities(estimation=False)
    with pytest.raises(DeviceCapabilityError):
        QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(
            observables=["Z0"], backend=backend
        )
    assert backend.estimate_call_count == 0
