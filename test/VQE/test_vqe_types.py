import numpy as np
import pytest

from pyqpanda_alg.VQE import VQEConfig, VQEResult


def test_vqe_result_records_execution_provenance():
    result = VQEResult(
        energy=-1.0,
        optimal_parameters=np.array([0.1]),
        converged=True,
        iterations=2,
        energy_history=(-0.5, -1.0),
        optimal_circuit=None,
        task_ids=("task-1",),
        metadata={"backend": "runtime"},
    )
    assert result.energy == -1.0
    assert result.task_ids == ("task-1",)


def test_vqe_result_rejects_non_finite_energy_history():
    with pytest.raises(ValueError, match="finite"):
        VQEResult(
            energy=-1.0,
            optimal_parameters=np.array([0.1]),
            converged=True,
            iterations=1,
            energy_history=(-0.5, float("nan")),
        )
    with pytest.raises(ValueError, match="finite"):
        VQEResult(
            energy=-1.0,
            optimal_parameters=np.array([0.1]),
            converged=True,
            iterations=1,
            energy_history=(-0.5, float("inf")),
        )


def test_vqe_result_rejects_non_1d_parameter_arrays():
    with pytest.raises(ValueError, match="one-dimensional"):
        VQEResult(
            energy=-1.0,
            optimal_parameters=np.array([[0.1, 0.2]]),
            converged=True,
            iterations=1,
            energy_history=(-0.5,),
        )


def test_vqe_result_copies_inputs_immutably():
    params = np.array([0.1, 0.2])
    metadata = {"backend": "runtime"}
    result = VQEResult(
        energy=-1.0,
        optimal_parameters=params,
        converged=True,
        iterations=2,
        energy_history=(-0.5, -1.0),
        task_ids=("task-1",),
        metadata=metadata,
    )
    params[0] = 9.9
    metadata["backend"] = "mutated"
    assert result.optimal_parameters[0] == 0.1
    assert result.metadata["backend"] == "runtime"
    with pytest.raises(ValueError):
        result.optimal_parameters[0] = 0.0  # array is read-only


def test_vqe_config_defaults_match_execution_shots():
    config = VQEConfig()
    assert config.max_iterations == 100
    assert config.tolerance == 1e-6
    assert config.shots == 1000
    assert config.optimizer == "SLSQP"


def test_vqe_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="max_iterations"):
        VQEConfig(max_iterations=0)
    with pytest.raises(ValueError, match="shots"):
        VQEConfig(shots=0)
    with pytest.raises(ValueError, match="tolerance"):
        VQEConfig(tolerance=-1.0)
    with pytest.raises(ValueError, match="optimizer"):
        VQEConfig(optimizer="")
