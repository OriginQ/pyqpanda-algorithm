"""Local execution backends for the case5 QAOA demonstration.

The public package deliberately separates local state-vector sampling from the
optional Runtime FakeBackend adapter.  No function in this module contacts a
cloud service or submits a quantum task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .qubo import QuboModel

BackendMode = Literal["statevector", "local_fake", "local_noisy"]


@dataclass(frozen=True)
class NoiseConfig:
    """Synthetic, reproducible gate/readout noise for an offline rehearsal."""

    name: str = "light"
    single_qubit_depolarizing: float = 0.002
    two_qubit_depolarizing: float = 0.010
    readout_bit_flip: float = 0.010

    def __post_init__(self) -> None:
        for field_name in (
            "single_qubit_depolarizing",
            "two_qubit_depolarizing",
            "readout_bit_flip",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
            object.__setattr__(self, field_name, value)
        if not str(self.name).strip():
            raise ValueError("noise profile name must be non-empty")

    @classmethod
    def from_profile(cls, name: str) -> "NoiseConfig":
        profiles = {
            "light": cls("light", 0.002, 0.010, 0.010),
            "moderate": cls("moderate", 0.010, 0.030, 0.020),
            "heavy": cls("heavy", 0.020, 0.060, 0.050),
        }
        key = str(name).strip().lower()
        if key not in profiles:
            raise ValueError(f"unknown noise profile {name!r}; choose {sorted(profiles)}")
        return profiles[key]

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "single_qubit_depolarizing": self.single_qubit_depolarizing,
            "two_qubit_depolarizing": self.two_qubit_depolarizing,
            "readout_bit_flip": self.readout_bit_flip,
        }


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for an offline state-vector or sampled simulation."""

    mode: BackendMode = "statevector"
    shots: int = -1
    noise: NoiseConfig | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower().replace("-", "_")
        aliases = {"state_vector": "statevector", "sampled": "local_fake", "noisy": "local_noisy"}
        mode = aliases.get(mode, mode)
        if mode not in {"statevector", "local_fake", "local_noisy"}:
            raise ValueError("mode must be statevector, local_fake, or local_noisy")
        object.__setattr__(self, "mode", mode)
        if mode == "statevector":
            if self.shots not in {-1, 0}:
                raise ValueError("statevector mode uses shots=-1")
            object.__setattr__(self, "shots", -1)
        elif int(self.shots) <= 0:
            object.__setattr__(self, "shots", 1024)
        if mode == "local_noisy" and self.noise is None:
            object.__setattr__(self, "noise", NoiseConfig.from_profile("light"))

    @classmethod
    def from_environment(
        cls,
        mode: BackendMode = "statevector",
        *,
        shots: int | None = None,
        noise_profile: str = "light",
    ) -> "BackendConfig":
        normalized = str(mode).strip().lower().replace("-", "_")
        chosen_shots = -1 if normalized in {"statevector", "state_vector"} else (1024 if shots is None else int(shots))
        return cls(
            mode=mode,
            shots=chosen_shots,
            noise=NoiseConfig.from_profile(noise_profile) if normalized == "local_noisy" else None,
        )


@dataclass(frozen=True)
class BackendRun:
    probabilities: dict[str, float]
    parameters: tuple[float, ...]
    loss: float
    backend: str
    execution_environment: str
    noise_stage: str
    circuit_evaluations: int
    noise_profile: dict[str, float | str] | None = None
    algorithm_stage: str = "local optimizer"
    closure_scope: str = "one small case5 instance"
    cloud_backend_name: str | None = None
    cloud_job_id: str | None = None
    cloud_status_history: tuple[str, ...] = ()
    cloud_backend_catalog: dict[str, bool] | None = None
    cloud_result_payload: dict[str, object] | None = None


def build_noise_model(config: NoiseConfig):
    """Build a pyqpanda3 noise model without contacting a remote service."""

    from pyqpanda3.core import GateType, NoiseModel, depolarizing_error

    model = NoiseModel()
    one_qubit = depolarizing_error(config.single_qubit_depolarizing)
    two_qubit = depolarizing_error(config.two_qubit_depolarizing)
    for gate_type in (GateType.H, GateType.RX, GateType.RZ):
        model.add_all_qubit_quantum_error(one_qubit, gate_type)
    model.add_all_qubit_quantum_error(two_qubit, GateType.CNOT)
    readout = float(config.readout_bit_flip)
    model.add_all_qubit_read_out_error([[1.0 - readout, readout], [readout, 1.0 - readout]])
    return model


def run_qaoa(
    model: QuboModel,
    config: BackendConfig,
    *,
    layers: int = 1,
    optimizer: str = "SLSQP",
    optimizer_options: dict | None = None,
    initial_parameters: Sequence[float] | None = None,
    seed: int = 2026,
) -> BackendRun:
    """Optimise and sample the QAOA circuit on a local CPUQVM."""

    if int(layers) < 1:
        raise ValueError("layers must be positive")
    np.random.seed(int(seed))
    # Reuse the repository's public QAOA implementation.  The small adapter
    # below only supplies the case-specific QUBO and backend configuration.
    from ..QAOA.qaoa import QAOA

    noise_model = build_noise_model(config.noise) if config.mode == "local_noisy" and config.noise else None
    qaoa = QAOA(model.expression, noise_model=noise_model)
    options = optimizer_options or {"options": {"maxiter": 25, "disp": False}}
    shots = -1 if config.mode == "statevector" else int(config.shots)
    probabilities, parameters, loss = qaoa.run(
        layer=int(layers),
        initial_para=None if initial_parameters is None else np.asarray(initial_parameters, dtype=float),
        shots=shots,
        optimizer=optimizer,
        optimizer_option=options,
    )
    return BackendRun(
        probabilities={str(key): float(value) for key, value in probabilities.items()},
        parameters=tuple(float(value) for value in np.asarray(parameters).reshape(-1)),
        loss=float(loss),
        backend=config.mode,
        execution_environment={
            "statevector": "local ideal CPUQVM",
            "local_fake": "local sampled CPUQVM",
            "local_noisy": "local noisy CPUQVM",
        }[config.mode],
        noise_stage={
            "statevector": "none",
            "local_fake": "finite shots only",
            "local_noisy": "synthetic gate and readout noise plus finite shots",
        }[config.mode],
        circuit_evaluations=int(getattr(qaoa, "circuit_iter", 0)),
        noise_profile=None if config.noise is None else config.noise.as_dict(),
    )


__all__ = ["BackendConfig", "BackendMode", "BackendRun", "NoiseConfig", "build_noise_model", "run_qaoa"]

