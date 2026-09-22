"""Seeded CPU statevector QAOA using upstream phase and XY circuit components."""

from __future__ import annotations

import itertools
from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np
from pyqpanda3.core import CPUQVM, RX, H, QCircuit, QProg
from pyqpanda3.hamiltonian import PauliOperator
from scipy.optimize import minimize

from pyqpanda_alg.QAOA.default_circuits import iswap, linear_w_state
from pyqpanda_alg.QAOA.qaoa import pauli_z_operator_to_circuit

from .model import Model, evaluate

MAX_QUBITS = 16


def to_ising(model: Model, scale: float = 1.0) -> PauliOperator:
    """Map x_i=(I-Z_i)/2 with explicit indices, preserving the constant offset."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    terms = {"": float(model.offset)}
    for i, coefficient in enumerate(model.linear):
        terms[""] += coefficient / 2
        terms[f"Z{i}"] = terms.get(f"Z{i}", 0.0) - coefficient / 2
    for i, j, coefficient in model.quadratic:
        terms[""] += coefficient / 4
        terms[f"Z{i}"] = terms.get(f"Z{i}", 0.0) - coefficient / 4
        terms[f"Z{j}"] = terms.get(f"Z{j}", 0.0) - coefficient / 4
        terms[f"Z{i} Z{j}"] = coefficient / 4
    return PauliOperator({key: value / scale for key, value in terms.items()})


def build_circuit(
    model: Model,
    parameters: list[float],
    mixer: str = "xy",
    *,
    phase_mode: str = "auto",
) -> QProg:
    """Build alternating cost/mixer layers; parameter order is gamma,beta per layer.

    XY starts each task in a W state and applies adjacent exchange rotations.
    Singleton domains receive no mixer, preserving their required bit value.
    X starts in |+> and applies RX(2*beta) to every qubit as an ablation.
    auto omits the identically zero exact-one penalty ONLY for XY. full keeps
    the original phase for equivalence/resource audits. Scaling stays unchanged.
    """
    _check_model(model)
    if mixer not in ("xy", "x"):
        raise ValueError("mixer must be xy or x")
    if not parameters or len(parameters) % 2 or not np.all(np.isfinite(parameters)):
        raise ValueError("parameters must be finite gamma,beta pairs")
    qubits = list(range(len(model.variables)))
    circuit = QCircuit()
    if mixer == "xy":
        for domain in model.domains:
            circuit << linear_w_state(list(domain))
    else:
        for qubit in qubits:
            circuit << H(qubit)
    if phase_mode not in ("auto", "full"):
        raise ValueError("phase_mode must be auto or full")
    phase_model = model
    if mixer == "xy" and phase_mode == "auto":
        phase_model = replace(
            model,
            offset=0,
            linear=tuple(v.cost for v in model.variables),
            quadratic=tuple(
                (i, j, c)
                for i, j, c in model.quadratic
                if model.variables[i].task != model.variables[j].task
            ),
        )
    operator = to_ising(phase_model, float(model.penalty))
    for gamma, beta in zip(parameters[::2], parameters[1::2], strict=True):
        phase, _ = pauli_z_operator_to_circuit(operator, qubits, gamma)
        circuit << phase
        if mixer == "xy":
            for domain in model.domains:
                for left, right in itertools.pairwise(domain):
                    circuit << iswap(left, right, 2 * beta)
        else:
            for qubit in qubits:
                circuit << RX(qubit, 2 * beta)
    program = QProg(len(qubits))
    program << circuit
    return program


def circuit_probabilities(
    model: Model,
    parameters: list[float],
    mixer: str = "xy",
    *,
    phase_mode: str = "auto",
) -> np.ndarray:
    """Simulate without measurement; qubit 0 is the least-significant bit."""
    machine = CPUQVM()
    machine.run(build_circuit(model, parameters, mixer, phase_mode=phase_mode), shots=1)
    probabilities = np.abs(np.asarray(machine.result().get_state_vector())) ** 2
    if len(probabilities) != 1 << len(model.variables):
        raise RuntimeError("unexpected simulator statevector dimension")
    return probabilities / probabilities.sum()


def solve_qaoa(
    model: Model,
    seed: int = 7,
    layers: int = 1,
    shots: int = 512,
    maxiter: int = 60,
    restarts: int = 2,
    mixer: str = "xy",
    *,
    phase_mode: str = "auto",
) -> dict[str, Any]:
    """Optimize exact expected QUBO energy, then sample the final quantum state.

    Uses independent NumPy generators for parameter initialization and final
    Born-distribution sampling because CPUQVM does not expose a portable seed.
    No exact optimum, feasibility-filtered state space or classical repair enters
    training. Only sampled bitstrings compete for the returned schedule.
    """
    started = perf_counter()
    for name, value, lower, upper in (
        ("seed", seed, 0, 2**32 - 1),
        ("layers", layers, 1, 8),
        ("shots", shots, 1, 1_000_000),
        ("maxiter", maxiter, 6, 100_000),
        ("restarts", restarts, 1, 100),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    if maxiter < 2 * layers + 2:
        raise ValueError("maxiter must be at least 2 * layers + 2 for COBYLA")
    if mixer not in ("xy", "x"):
        raise ValueError("mixer must be xy or x")
    if phase_mode not in ("auto", "full"):
        raise ValueError("phase_mode must be auto or full")
    if any(not domain for domain in model.domains):
        return {
            "status": "infeasible_domain",
            "best": None,
            "seed": seed,
            "mixer": mixer,
            "runtime_seconds": perf_counter() - started,
            "circuit_evaluations": 0,
        }
    if not model.variables:
        return {
            "status": "trivial",
            "best": evaluate(model, []),
            "seed": seed,
            "mixer": mixer,
            "runtime_seconds": perf_counter() - started,
            "circuit_evaluations": 0,
        }
    _check_model(model)
    init_seed, sample_seed = np.random.SeedSequence(seed).spawn(2)
    initializer = np.random.default_rng(init_seed)
    sampler = np.random.default_rng(sample_seed)
    indices = np.arange(1 << len(model.variables), dtype=np.int64)
    energies = np.full(len(indices), model.offset, dtype=float)
    for i, coefficient in enumerate(model.linear):
        energies += coefficient * ((indices >> i) & 1)
    for i, j, coefficient in model.quadratic:
        energies += coefficient * ((indices >> i) & 1) * ((indices >> j) & 1)
    best_loss = float("inf")
    best_parameters: list[float] = []
    best_probabilities = np.empty(0)
    history = []

    def loss(parameters: np.ndarray) -> float:
        """Record a real simulator evaluation and retain the best parameters."""
        nonlocal best_loss, best_parameters, best_probabilities
        probabilities = circuit_probabilities(
            model, parameters.tolist(), mixer, phase_mode=phase_mode
        )
        expected = float(probabilities @ energies)
        history.append(expected)
        if expected < best_loss:
            best_loss = expected
            best_parameters = parameters.tolist()
            best_probabilities = probabilities
        return expected / model.penalty

    optimizer_runs = []
    for _ in range(restarts):
        initial = initializer.uniform(0.0, np.pi, size=2 * layers)
        result = minimize(
            loss,
            initial,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.5, "tol": 1e-5},
        )
        optimizer_runs.append(
            {
                "success": bool(result.success),
                "message": str(result.message),
                "evaluations": int(result.nfev),
            }
        )
    counts = sampler.multinomial(shots, best_probabilities)
    observed = [
        (_bits(int(i), len(model.variables)), int(counts[i]))
        for i in np.flatnonzero(counts)
    ]
    candidates = [evaluate(model, bits) for bits, _ in observed]
    best = min(
        candidates,
        key=lambda row: (not row["feasible"], row["qubo_energy"], row["bits"]),
    )
    # These diagnostics use the input validator, not a classical optimal solution.
    feasible_probability = 0.0
    one_hot_probability = 0.0
    for state_index in np.flatnonzero(best_probabilities > 1e-15):
        i = int(state_index)
        bits = _bits(i, len(model.variables))
        if all(sum(bits[j] for j in domain) == 1 for domain in model.domains):
            one_hot_probability += float(best_probabilities[i])
            if evaluate(model, bits)["feasible"]:
                feasible_probability += float(best_probabilities[i])
    sampled_feasible = sum(
        count
        for row, (_, count) in zip(candidates, observed, strict=True)
        if row["feasible"]
    )
    return {
        "status": "feasible" if best["feasible"] else "no_feasible_sample",
        "best": best,
        "seed": seed,
        "mixer": mixer,
        "phase_mode": phase_mode,
        "layers": layers,
        "shots": shots,
        "maxiter": maxiter,
        "restarts": restarts,
        "parameters": best_parameters,
        "expected_energy": best_loss,
        "initial_expected_energy": history[0],
        "history": history,
        "feasible_probability": feasible_probability,
        "one_hot_probability": one_hot_probability,
        "sampled_feasible_fraction": sampled_feasible / shots,
        "counts": {"".join(map(str, bits[::-1])): count for bits, count in observed},
        "circuit_evaluations": len(history),
        "optimizer_runs": optimizer_runs,
        "sampling": "seeded NumPy multinomial from CPUQVM Born probabilities",
        "runtime_seconds": perf_counter() - started,
    }


def _check_model(model: Model) -> None:
    if not 1 <= len(model.variables) <= MAX_QUBITS:
        raise ValueError(
            f"CPU simulation requires 1..{MAX_QUBITS} surviving candidates"
        )
    if any(not domain for domain in model.domains):
        raise ValueError("cannot prepare an empty task domain")


def _bits(index: int, size: int) -> list[int]:
    return [(index >> i) & 1 for i in range(size)]
