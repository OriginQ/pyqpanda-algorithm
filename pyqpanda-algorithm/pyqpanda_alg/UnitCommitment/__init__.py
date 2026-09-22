"""Day-ahead unit commitment and dispatch on a five-bus benchmark.

The module keeps the power-system model and the scheduling workflow separate
from the repository-wide :mod:`pyqpanda_alg.QAOA` implementation.  Local
state-vector and finite-shot noisy modes are available without a cloud
credential; the optional Runtime adapter reads a caller-owned API key from an
environment variable and never records its value.
"""

from .backend import BackendConfig, BackendMode, BackendRun, NoiseConfig, build_noise_model, run_qaoa
from .case5 import Case5UCInstance, MATPOWER_CASE5_REFERENCE, MATPOWER_CASE5_URL, load_case5_uc, load_matpower_case5
from .case5_classical import economic_dispatch, evaluate_schedule, solve_milp_uc
from .case5_qubo import Case5CommitmentQuboBuilder, audit_case5_qubo, commitment_capacity, decode_commitment
from .case5_solver import Case5Candidate, Case5QAOA, Case5QAOAResult
from .day_ahead import (
    DEFAULT_DAY_AHEAD_LOAD_SCALES,
    DayAheadQuantumResult,
    HourlyQAOAResult,
    default_day_ahead_load_scales,
    load_case5_day_ahead,
    solve_hourly_qaoa,
)
from .runtime_fakebackend import RuntimeFakeBackendResult, RuntimeFakeBackendRunner

__all__ = [
    "BackendConfig", "BackendMode", "BackendRun", "NoiseConfig", "build_noise_model", "run_qaoa",
    "Case5UCInstance", "MATPOWER_CASE5_REFERENCE", "MATPOWER_CASE5_URL", "load_case5_uc", "load_matpower_case5",
    "economic_dispatch", "evaluate_schedule", "solve_milp_uc",
    "Case5CommitmentQuboBuilder", "audit_case5_qubo", "commitment_capacity", "decode_commitment",
    "Case5Candidate", "Case5QAOA", "Case5QAOAResult",
    "DEFAULT_DAY_AHEAD_LOAD_SCALES", "DayAheadQuantumResult", "HourlyQAOAResult",
    "default_day_ahead_load_scales", "load_case5_day_ahead", "solve_hourly_qaoa",
    "RuntimeFakeBackendResult", "RuntimeFakeBackendRunner",
]

