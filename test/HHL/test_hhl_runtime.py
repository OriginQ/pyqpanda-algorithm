"""Runtime sampling contract tests for the HHL solver.

Covers the default runtime result (success probability without
reconstruction), the tomography cost contract — ``reconstruct=True``
submits exactly the basis-measurement circuits declared by
:class:`~pyqpanda_alg.HHL.resources.HHLResourceEstimate` — the
capability gate that rejects backends without sampling support before
any submission, requested data-observable expectations, and the
checkpoint-after-batch recovery of the reconstruction state machine.
All runtime-contract tests run credential-free against
:class:`~test.execution.fakes.RecordingBackend`.
"""

import numpy as np
import pytest

from pyqpanda_alg.HHL import HHL, HHLConfig
from pyqpanda_alg.HHL.resources import estimate_hhl_resources
from pyqpanda_alg.execution import (
    AlgorithmInputError,
    AlgorithmTask,
    BackendCapabilities,
    DeviceCapabilityError,
    TaskStatus,
)
from test.execution.fakes import RecordingBackend


@pytest.mark.runtime_contract
def test_runtime_hhl_does_not_reconstruct_by_default(recording_backend):
    result = HHL(np.eye(2), np.array([1.0, 0.0])).run(backend=recording_backend)
    assert result.classical_vector is None
    assert result.statevector is None
    assert result.success_probability is not None


@pytest.mark.runtime_contract
def test_runtime_reconstruct_submits_exact_tomography_circuits():
    # Scripted counts of the |0> state: the X and Y bases split evenly
    # and the Z basis returns 0 with certainty; every key postselects on
    # success (first measured bit).
    backend = RecordingBackend(
        sample_counts=[
            {"10": 500, "11": 500},  # X basis
            {"10": 500, "11": 500},  # Y basis
            {"10": 1000, "11": 0},   # Z basis
        ]
    )
    matrix = np.eye(2)
    vector = np.array([1.0, 0.0])
    result = HHL(matrix, vector, precision=1e-2).run(
        backend=backend, reconstruct=True
    )
    estimate = estimate_hhl_resources(matrix, vector, HHLConfig())
    assert backend.sample_call_count == estimate.tomography_circuits
    assert result.metadata["tomography"]["circuits"] == estimate.tomography_circuits
    assert result.metadata["tomography"]["shots"] == 1000
    assert result.success_probability == pytest.approx(1.0)
    assert result.classical_vector is not None
    assert abs(result.classical_vector[0]) > 0.99


@pytest.mark.runtime_contract
def test_runtime_tomography_metadata_clamps_nonphysical_fidelity():
    # Heavy noise in every basis: each X/Y/Z circuit reads the data
    # qubit 0 in 100 shots and 1 in 900, so every Pauli expectation is
    # -0.8 and the linear-inversion density matrix has
    # lambda_max = (1 + 0.8*sqrt(3)) / 2 ~ 1.19 > 1.  The reported
    # fidelity must be clamped into [0, 1] and the uncertainty stay
    # non-negative.
    backend = RecordingBackend(
        sample_counts=[
            {"10": 100, "11": 900},  # X basis
            {"10": 100, "11": 900},  # Y basis
            {"10": 100, "11": 900},  # Z basis
        ]
    )
    result = HHL(np.eye(2), np.array([1.0, 0.0]), precision=1e-2).run(
        backend=backend, reconstruct=True
    )
    tomography = result.metadata["tomography"]
    assert 0.0 <= tomography["fidelity"] <= 1.0
    assert tomography["uncertainty"] >= 0.0
    assert tomography["uncertainty"] == pytest.approx(
        1.0 - tomography["fidelity"]
    )
    assert tomography["fidelity"] == 1.0
    assert tomography["uncertainty"] == 0.0


@pytest.mark.runtime_contract
def test_runtime_backend_without_sampling_capability_fails_before_submission():
    backend = RecordingBackend()
    backend.capabilities = BackendCapabilities(sampling=False)
    with pytest.raises(DeviceCapabilityError, match="sampling"):
        HHL(np.eye(2), np.array([1.0, 0.0]), precision=1e-2).run(backend=backend)
    assert backend.sample_call_count == 0


@pytest.mark.runtime_contract
def test_runtime_backend_without_capabilities_fails_before_submission():
    class _CapabilitylessBackend:
        def submit_sample(self, circuit, *, options):
            raise AssertionError("sampling must not be submitted")

    with pytest.raises(DeviceCapabilityError, match="sampling"):
        HHL(np.eye(2), np.array([1.0, 0.0]), precision=1e-2).run(
            backend=_CapabilitylessBackend()
        )


@pytest.mark.runtime_contract
def test_runtime_reports_requested_data_observables():
    backend = RecordingBackend(sample_counts=[{"10": 700, "11": 300}])
    result = HHL(np.eye(2), np.array([1.0, 0.0]), precision=1e-2).run(
        backend=backend, observables=["Z0"]
    )
    assert result.classical_vector is None
    assert result.statevector is None
    assert result.metadata["observables"]["Z0"] == pytest.approx(0.4)
    assert result.success_probability == pytest.approx(1.0)


@pytest.mark.runtime_contract
def test_runtime_rejects_observables_combined_with_reconstruction():
    backend = RecordingBackend()
    with pytest.raises(AlgorithmInputError, match="reconstruct"):
        HHL(np.eye(2), np.array([1.0, 0.0]), precision=1e-2).run(
            backend=backend, reconstruct=True, observables=["Z0"]
        )
    assert backend.sample_call_count == 0


@pytest.mark.runtime_contract
def test_runtime_rejects_observables_outside_the_data_register():
    backend = RecordingBackend()
    with pytest.raises(AlgorithmInputError, match="data qubit"):
        HHL(np.eye(2), np.array([1.0, 0.0]), precision=1e-2).run(
            backend=backend, observables=["X1"]
        )
    assert backend.sample_call_count == 0


@pytest.mark.runtime_contract
def test_runtime_reconstruction_checkpoints_after_each_basis_batch(tmp_path):
    backend = RecordingBackend(
        sample_counts=[
            {"10": 500, "11": 500},
            {"10": 500, "11": 500},
            {"10": 1000, "11": 0},
        ]
    )
    checkpoint = tmp_path / "hhl-tomography.json"
    result = HHL(np.eye(2), np.array([1.0, 0.0]), precision=1e-2).run(
        backend=backend, reconstruct=True, checkpoint_path=checkpoint
    )
    assert checkpoint.exists()
    resumed = AlgorithmTask.resume(checkpoint, backend=backend)
    assert resumed.status() is TaskStatus.SUCCEEDED
    assert resumed.backend_task_ids == [
        "recorded-sample-1",
        "recorded-sample-2",
        "recorded-sample-3",
    ]
    assert result.classical_vector is not None


def test_local_backend_rejects_runtime_observables():
    with pytest.raises(AlgorithmInputError, match="sampling"):
        HHL(np.eye(2), np.array([1.0, 0.0])).run(observables=["Z0"])
