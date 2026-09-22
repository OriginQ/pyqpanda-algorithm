"""Hybrid QAOA + economic-dispatch solver for MATPOWER case5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .backend import BackendConfig, run_qaoa
from .case5 import Case5UCInstance
from .case5_classical import ScheduleEvaluation, evaluate_schedule
from .case5_qubo import Case5CommitmentQuboBuilder, commitment_capacity, decode_commitment
from .qubo import QuboModel


@dataclass(frozen=True)
class Case5Candidate:
    """One measured commitment state and its optional dispatch evaluation."""

    bitstring: str
    probability: float
    commitments: tuple[tuple[int, ...], ...]
    committed_capacity_mw: tuple[float, ...]
    capacity_feasible: bool
    qubo_energy: float
    schedule: ScheduleEvaluation | None = None

    @property
    def physically_feasible(self) -> bool:
        return bool(self.capacity_feasible and self.schedule is not None and self.schedule.success)

    def as_dict(self) -> dict[str, object]:
        return {
            "bitstring": self.bitstring,
            "probability": self.probability,
            "commitments": [list(row) for row in self.commitments],
            "committed_capacity_mw": list(self.committed_capacity_mw),
            "capacity_feasible": self.capacity_feasible,
            "qubo_energy": self.qubo_energy,
            "physically_feasible": self.physically_feasible,
            "schedule": None if self.schedule is None else self.schedule.as_dict(),
        }


@dataclass
class Case5QAOAResult:
    """Result bundle for the network-scale hybrid experiment."""

    model: QuboModel
    backend: str
    probabilities: dict[str, float]
    parameters: tuple[float, ...]
    loss: float
    candidates: list[Case5Candidate]
    feasible_probability: float
    qaoa_layers: int
    logical_qubits: int
    shots: int
    circuit_evaluations: int
    optimizer: str
    best_candidate: Case5Candidate | None = None
    noise_profile: dict[str, float | str] | None = None
    execution_environment: str = "local ideal simulator"
    noise_stage: str = "none"
    closure_scope: str = "one small problem instance"
    cloud_backend_name: str | None = None
    cloud_job_id: str | None = None
    cloud_status_history: tuple[str, ...] = ()
    cloud_backend_catalog: dict[str, bool] | None = None
    cloud_result_payload: dict[str, object] | None = None

    @property
    def capacity_feasible_probability(self) -> float:
        """Probability mass with committed capacity above each demand target."""

        return float(self.feasible_probability)

    @property
    def best_cost(self) -> float | None:
        if self.best_candidate is None or self.best_candidate.schedule is None:
            return None
        return float(self.best_candidate.schedule.total_cost)

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "logical_qubits": self.logical_qubits,
            "qaoa_layers": self.qaoa_layers,
            "shots": self.shots,
            "circuit_evaluations": self.circuit_evaluations,
            "optimizer": self.optimizer,
            "noise_profile": self.noise_profile,
            "execution_environment": self.execution_environment,
            "noise_stage": self.noise_stage,
            "closure_scope": self.closure_scope,
            "cloud_backend_name": self.cloud_backend_name,
            "cloud_job_id": self.cloud_job_id,
            "cloud_status_history": list(self.cloud_status_history),
            "cloud_backend_catalog": self.cloud_backend_catalog,
            "cloud_result_payload": self.cloud_result_payload,
            "loss": self.loss,
            "parameters": list(self.parameters),
            "feasible_probability": self.feasible_probability,
            "capacity_feasible_probability": self.capacity_feasible_probability,
            "best_cost": self.best_cost,
            "best_candidate": None if self.best_candidate is None else self.best_candidate.as_dict(),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass
class Case5QAOA:
    """Public facade for a case5 commitment QUBO and hybrid post-processing."""

    instance: Case5UCInstance | None = None
    backend: BackendConfig | str = "statevector"
    layers: int = 1
    optimizer: str = "SLSQP"
    shots: int | None = None
    seed: int = 2026
    optimizer_options: dict | None = None
    qubo_builder: Case5CommitmentQuboBuilder | None = None

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            self.backend = BackendConfig.from_environment(self.backend, shots=self.shots)
        elif self.shots is not None and self.backend.mode != "statevector":
            self.backend = BackendConfig(
                mode=self.backend.mode,
                shots=int(self.shots),
                noise=self.backend.noise,
            )
        if self.layers < 1:
            raise ValueError("layers must be positive")

    def build_model(self) -> QuboModel:
        if self.instance is None:
            raise ValueError("instance is required when building a case5 QUBO")
        return (self.qubo_builder or Case5CommitmentQuboBuilder()).build(self.instance)

    def solve(
        self,
        model: QuboModel | None = None,
        *,
        top_k: int = 16,
        evaluate_method: str = "slsqp",
        enforce_network: bool = True,
    ) -> Case5QAOAResult:
        """Run QAOA and dispatch only the most relevant measured states.

        No classical state enumeration is used.  State-vector mode returns the
        full probability distribution generated by QAOA; the post-processing
        evaluates the highest-probability capacity-feasible states and the
        lowest-energy feasible state.
        """

        if top_k < 1:
            raise ValueError("top_k must be positive")
        instance = self.instance
        model = self.build_model() if model is None else model
        if instance is None:
            instance = model.instance
        if not isinstance(instance, Case5UCInstance):
            raise ValueError("model must be built from a Case5UCInstance")
        run = run_qaoa(
            model,
            self.backend,
            layers=self.layers,
            optimizer=self.optimizer,
            optimizer_options=self.optimizer_options,
            seed=self.seed,
        )
        base_candidates: list[Case5Candidate] = []
        for bitstring, probability in run.probabilities.items():
            commitments = decode_commitment(instance, model.registry, bitstring)
            capacities = tuple(
                commitment_capacity(instance, commitments, period)
                for period in instance.periods
            )
            capacity_feasible = all(
                capacity + 1.0e-7 >= instance.demand_mw[period] + instance.reserve_mw[period]
                for period, capacity in enumerate(capacities)
            )
            base_candidates.append(
                Case5Candidate(
                    bitstring=str(bitstring),
                    probability=float(probability),
                    commitments=commitments,
                    committed_capacity_mw=capacities,
                    capacity_feasible=capacity_feasible,
                    qubo_energy=float(model.energy(model.registry.bits_from_bitstring(str(bitstring)))),
                )
            )
        base_candidates.sort(key=lambda candidate: (-candidate.probability, candidate.qubo_energy, candidate.bitstring))
        feasible = [candidate for candidate in base_candidates if candidate.capacity_feasible]
        evaluate_keys = {candidate.bitstring for candidate in feasible[:top_k]}
        if feasible:
            evaluate_keys.add(min(feasible, key=lambda candidate: (candidate.qubo_energy, candidate.bitstring)).bitstring)
        evaluated: list[Case5Candidate] = []
        for candidate in base_candidates:
            if candidate.bitstring in evaluate_keys:
                schedule = evaluate_schedule(
                    instance,
                    candidate.commitments,
                    method=evaluate_method,
                    enforce_network=enforce_network,
                )
                candidate = Case5Candidate(
                    bitstring=candidate.bitstring,
                    probability=candidate.probability,
                    commitments=candidate.commitments,
                    committed_capacity_mw=candidate.committed_capacity_mw,
                    capacity_feasible=candidate.capacity_feasible,
                    qubo_energy=candidate.qubo_energy,
                    schedule=schedule,
                )
            evaluated.append(candidate)
        evaluated.sort(
            key=lambda candidate: (
                not candidate.physically_feasible,
                float("inf") if candidate.schedule is None else candidate.schedule.total_cost,
                -candidate.probability,
                candidate.bitstring,
            )
        )
        best = next((candidate for candidate in evaluated if candidate.physically_feasible), None)
        return Case5QAOAResult(
            model=model,
            backend=run.backend,
            probabilities=run.probabilities,
            parameters=run.parameters,
            loss=run.loss,
            candidates=evaluated[:top_k],
            feasible_probability=float(sum(candidate.probability for candidate in feasible)),
            qaoa_layers=self.layers,
            logical_qubits=model.n_variables,
            shots=self.backend.shots,
            circuit_evaluations=run.circuit_evaluations,
            optimizer=self.optimizer,
            best_candidate=best,
            noise_profile=run.noise_profile,
            execution_environment=run.execution_environment,
            noise_stage=run.noise_stage,
            closure_scope=run.closure_scope,
            cloud_backend_name=run.cloud_backend_name,
            cloud_job_id=run.cloud_job_id,
            cloud_status_history=run.cloud_status_history,
            cloud_backend_catalog=run.cloud_backend_catalog,
            cloud_result_payload=run.cloud_result_payload,
        )


__all__ = ["Case5Candidate", "Case5QAOA", "Case5QAOAResult"]

