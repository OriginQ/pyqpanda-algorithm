"""State-vector solvers for constrained QAOA QmRMR feature selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
from pyqpanda3.core import CPUQVM, QProg
from scipy.optimize import minimize

from .circuits import constrained_qaoa_circuit, exchange_ansatz_circuit
from .problem import (
    QmRMRProblem,
    cvar_from_probabilities,
    expectation_from_probabilities,
)


OptimizerName = Literal["slsqp", "spsa"]
BitOrder = Literal["qubit", "feature"]
ObjectiveMode = Literal["expectation", "cvar"]


@dataclass
class OptimizationResult:
    """Common result object returned by the variational solvers."""

    method: str
    parameters: np.ndarray
    history: list[float]
    probabilities: dict[str, float]
    selected_state: tuple[int, ...]
    selected_value: float
    selected_probability: float
    expected_value: float
    feasible_probability: float
    optimum_probability: float
    optimum_value: float
    evaluations: int
    gate_count: dict[str, int]
    depth: int
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def selected_key(self) -> str:
        return "".join(str(bit) for bit in self.selected_state)

    @property
    def optimum_gap(self) -> float:
        return float(self.selected_value - self.optimum_value)

    @property
    def expected_gap(self) -> float:
        """Gap of the optimized distribution expectation to the oracle."""

        return float(self.expected_value - self.optimum_value)

    def summary(self) -> dict[str, object]:
        return {
            "method": self.method,
            "selected_state": self.selected_key,
            "selected_value": float(self.selected_value),
            "selected_probability": float(self.selected_probability),
            "expected_value": float(self.expected_value),
            "optimization_value": float(
                self.metadata.get("optimization_value", self.expected_value)
            ),
            "optimization_objective": self.metadata.get(
                "optimization_objective", "expectation"
            ),
            "cvar_alpha": self.metadata.get("cvar_alpha", ""),
            "expected_gap": float(self.expected_gap),
            "feasible_probability": float(self.feasible_probability),
            "optimum_probability": float(self.optimum_probability),
            "optimum_value": float(self.optimum_value),
            "optimum_gap": float(self.optimum_gap),
            "evaluations": int(self.evaluations),
            "parameter_count": int(self.parameters.size),
            "gate_count": dict(self.gate_count),
            "depth": int(self.depth),
            "metadata": dict(self.metadata),
        }


def _run_exact_probabilities(
    circuit,
    n_qubits: int,
) -> tuple[dict[str, float], QProg]:
    """Run a circuit without measurement and return exact probabilities."""

    program = QProg(int(n_qubits))
    program.append(circuit)
    machine = CPUQVM()
    # In pyQPanda3 0.3.x a positive shot argument is required even when no
    # measurement is appended. The returned probability dictionary remains the
    # exact simulator distribution.
    machine.run(program, 1)
    result = machine.result()
    probabilities = {
        str(key): float(value)
        for key, value in result.get_prob_dict(program.qubits()).items()
    }
    return probabilities, program


def _feature_key(raw_key: str, bit_order: BitOrder) -> str:
    return raw_key[::-1] if bit_order == "qubit" else raw_key


def _distribution_metrics(
    problem: QmRMRProblem,
    probabilities: dict[str, float],
    *,
    bit_order: BitOrder,
    optimum_value: float,
    optimum_states: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], float, float, float, float]:
    feasible: list[tuple[tuple[int, ...], float]] = []
    feasible_mass = 0.0
    optimum_mass = 0.0
    optimum_set = set(optimum_states)
    for raw_key, probability in probabilities.items():
        key = _feature_key(raw_key, bit_order)
        bits = problem.bits_from_key(key)
        if len(bits) != problem.n_features:
            raise ValueError("probability key length does not match problem size")
        if sum(bits) != problem.select_num:
            continue
        p = float(probability)
        feasible.append((bits, p))
        feasible_mass += p
        if bits in optimum_set or abs(problem.objective(bits) - optimum_value) <= 1e-10:
            optimum_mass += p
    if not feasible:
        raise RuntimeError("the circuit produced no feasible probability mass")

    # Highest probability wins; the lexicographically smallest state breaks a
    # tie so benchmark output is reproducible.
    selected_state, selected_probability = max(
        feasible,
        key=lambda item: (item[1], tuple(-bit for bit in item[0])),
    )
    selected_value = problem.objective(selected_state)
    return (
        selected_state,
        float(selected_value),
        float(selected_probability),
        float(feasible_mass),
        float(optimum_mass),
    )


def _spsa(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    *,
    maxiter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[float], int]:
    """Small deterministic SPSA implementation for an optimizer comparison."""

    if int(maxiter) < 1:
        raise ValueError("maxiter must be positive")
    params = np.asarray(initial, dtype=float).copy()
    history = [float(objective(params.copy()))]
    evaluations = 1
    for iteration in range(1, int(maxiter) + 1):
        a_k = 0.1 / (iteration + 1) ** 0.602
        c_k = 0.1 / (iteration + 1) ** 0.101
        delta = rng.choice(np.array([-1.0, 1.0]), size=params.size)
        plus = params + c_k * delta
        minus = params - c_k * delta
        f_plus = float(objective(plus))
        f_minus = float(objective(minus))
        evaluations += 2
        gradient = (f_plus - f_minus) / (2.0 * c_k) * delta
        params -= a_k * gradient
        history.append(float(objective(params.copy())))
        evaluations += 1
    return params, history, evaluations


def _minimize(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    *,
    optimizer: OptimizerName,
    maxiter: int,
    seed: int,
    bounds: Sequence[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[float], int]:
    if int(maxiter) < 1:
        raise ValueError("maxiter must be positive")
    params0 = np.asarray(initial, dtype=float).copy()
    if optimizer == "spsa":
        return _spsa(
            objective,
            params0,
            maxiter=maxiter,
            rng=np.random.default_rng(seed),
        )
    if optimizer != "slsqp":
        raise ValueError("optimizer must be 'slsqp' or 'spsa'")

    history: list[float] = []

    def tracked(value: np.ndarray) -> float:
        result = float(objective(np.asarray(value, dtype=float)))
        history.append(result)
        return result

    result = minimize(
        tracked,
        params0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-9, "disp": False},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("SLSQP returned non-finite parameters")
    return np.asarray(result.x, dtype=float), history, len(history)


def _result_priority(result: OptimizationResult) -> tuple[float, float, float]:
    """Compare restarts by the quantity that the optimizer actually minimizes."""

    return (
        float(result.metadata.get("optimization_value", result.expected_value)),
        float(result.selected_value),
        -float(result.optimum_probability),
    )


class ExchangeAnsatzSelector:
    """Exchange-ansatz baseline with explicit bit-order handling."""

    def __init__(
        self,
        problem: QmRMRProblem,
        *,
        bit_order: BitOrder = "qubit",
    ) -> None:
        if bit_order not in {"qubit", "feature"}:
            raise ValueError("bit_order must be 'qubit' or 'feature'")
        self.problem = problem
        self.bit_order = bit_order
        self.effective_parameter_count = (
            problem.n_features // 2
        ) * (problem.n_features - 1)
        self.supplied_parameter_count = (
            problem.n_features // 2
        ) * problem.n_features

    def _evaluate(self, parameters: np.ndarray) -> tuple[float, dict[str, float]]:
        circuit, _ = exchange_ansatz_circuit(self.problem, parameters)
        probabilities, _ = _run_exact_probabilities(
            circuit,
            self.problem.n_features,
        )
        value = expectation_from_probabilities(
            self.problem,
            probabilities,
            reverse_key=self.bit_order == "qubit",
        )
        return float(value), probabilities

    def _optimize_once(
        self,
        initial: np.ndarray,
        *,
        optimizer: OptimizerName,
        maxiter: int,
        seed: int,
        optimize_unused: bool,
    ) -> OptimizationResult:
        active = self.effective_parameter_count
        supplied = np.asarray(initial, dtype=float).reshape(-1)
        if supplied.size < active:
            raise ValueError(
                f"exchange ansatz needs at least {active} parameters, got {supplied.size}"
            )
        parameters0 = supplied if optimize_unused else supplied[:active]

        def objective(parameters: np.ndarray) -> float:
            return self._evaluate(parameters)[0]

        bounds = [(-np.pi, np.pi)] * parameters0.size
        parameters, history, evaluations = _minimize(
            objective,
            parameters0,
            optimizer=optimizer,
            maxiter=maxiter,
            seed=seed,
            bounds=bounds if optimizer == "slsqp" else None,
        )
        value, probabilities = self._evaluate(parameters)
        exact = self.problem.exact_solution()
        selected, selected_value, selected_probability, feasible_mass, optimum_mass = (
            _distribution_metrics(
                self.problem,
                probabilities,
                bit_order=self.bit_order,
                optimum_value=exact.optimal_value,
                optimum_states=exact.optimal_states,
            )
        )
        circuit, _ = exchange_ansatz_circuit(self.problem, parameters)
        program = _run_exact_probabilities(circuit, self.problem.n_features)[1]
        return OptimizationResult(
            method="exchange_baseline",
            parameters=parameters,
            history=history,
            probabilities=probabilities,
            selected_state=selected,
            selected_value=selected_value,
            selected_probability=selected_probability,
            expected_value=float(value),
            feasible_probability=feasible_mass,
            optimum_probability=optimum_mass,
            optimum_value=exact.optimal_value,
            evaluations=evaluations,
            gate_count=program.count_ops(),
            depth=program.depth(),
            metadata={
                "optimizer": optimizer,
                "bit_order": self.bit_order,
                "objective_convention": self.problem.objective_convention,
                "effective_parameter_count": active,
                "supplied_parameter_count": self.supplied_parameter_count,
                "optimize_unused": bool(optimize_unused),
            },
        )

    def optimize(
        self,
        initial_parameters: Sequence[float] | None = None,
        *,
        optimizer: OptimizerName = "slsqp",
        maxiter: int = 60,
        seed: int = 0,
        optimize_unused: bool = False,
        restarts: int = 1,
    ) -> OptimizationResult:
        """Optimize the baseline, optionally retaining the best of restarts."""

        if not isinstance(restarts, (int, np.integer)) or int(restarts) < 1:
            raise ValueError("restarts must be a positive integer")
        restart_count = int(restarts)
        supplied = (
            None
            if initial_parameters is None
            else np.asarray(initial_parameters, dtype=float).reshape(-1)
        )
        random_size = self.effective_parameter_count
        if supplied is not None:
            if supplied.size < self.effective_parameter_count:
                raise ValueError(
                    f"exchange ansatz needs at least {self.effective_parameter_count} "
                    f"parameters, got {supplied.size}"
                )
            random_size = (
                supplied.size
                if optimize_unused
                else self.effective_parameter_count
            )

        rng = np.random.default_rng(seed)
        results: list[OptimizationResult] = []
        for restart in range(restart_count):
            if restart == 0 and supplied is not None:
                initial = supplied.copy()
            else:
                initial = rng.uniform(-np.pi, np.pi, size=random_size)
            results.append(
                self._optimize_once(
                    initial,
                    optimizer=optimizer,
                    maxiter=maxiter,
                    seed=seed + restart,
                    optimize_unused=optimize_unused,
                )
            )
        best = min(results, key=_result_priority)
        metadata = dict(best.metadata)
        selected_index = min(
            range(len(results)),
            key=lambda index: _result_priority(results[index]),
        )
        metadata.update(
            {
                "restart_count": restart_count,
                "restart_expected_values": [
                    float(result.expected_value) for result in results
                ],
                "selected_restart": int(selected_index),
            }
        )
        best.metadata = metadata
        return best


class ConstrainedQAOAQmRMR:
    """QAOA with a Dicke initial state and a number-preserving XY mixer."""

    def __init__(
        self,
        problem: QmRMRProblem,
        *,
        layers: int = 1,
        initializer: str = "auto",
        initializer_strength: float = 1.0,
        ring: bool = False,
        objective_mode: ObjectiveMode = "expectation",
        cvar_alpha: float = 0.25,
    ) -> None:
        if not isinstance(layers, (int, np.integer)) or int(layers) < 1:
            raise ValueError("layers must be a positive integer")
        self.problem = problem
        self.layers = int(layers)
        self.initializer = initializer
        if not np.isfinite(initializer_strength) or float(initializer_strength) < 0.0:
            raise ValueError("initializer_strength must be non-negative and finite")
        self.initializer_strength = float(initializer_strength)
        self.ring = bool(ring)
        if objective_mode not in {"expectation", "cvar"}:
            raise ValueError("objective_mode must be 'expectation' or 'cvar'")
        if not np.isfinite(cvar_alpha) or not 0.0 < float(cvar_alpha) <= 1.0:
            raise ValueError("cvar_alpha must be finite and satisfy 0 < alpha <= 1")
        self.objective_mode = objective_mode
        self.cvar_alpha = float(cvar_alpha)
        self.parameter_count = 2 * self.layers

    @staticmethod
    def lift_parameters(
        parameters: Sequence[float],
        target_layers: int,
    ) -> np.ndarray:
        """Extend a lower-depth schedule by repeating its final layer.

        This deterministic continuation is a simple warm-start policy: the
        lower-depth circuit is embedded unchanged and the newly appended layer
        begins with the same angles as the previous final layer.
        """

        target = int(target_layers)
        if target < 1:
            raise ValueError("target_layers must be positive")
        params = np.asarray(parameters, dtype=float).reshape(-1)
        if params.size == 0 or params.size % 2:
            raise ValueError("parameters must contain gamma/beta pairs")
        source = params.size // 2
        if source > target:
            raise ValueError("target_layers cannot be smaller than source layers")
        if source == target:
            return params.copy()
        return np.concatenate([params, np.tile(params[-2:], target - source)])

    def _evaluate(self, parameters: np.ndarray) -> tuple[float, dict[str, float]]:
        circuit = constrained_qaoa_circuit(
            self.problem,
            parameters,
            initializer=self.initializer,
            initializer_strength=self.initializer_strength,
            ring=self.ring,
        )
        probabilities, _ = _run_exact_probabilities(
            circuit,
            self.problem.n_features,
        )
        if self.objective_mode == "cvar":
            value = cvar_from_probabilities(
                self.problem,
                probabilities,
                alpha=self.cvar_alpha,
                reverse_key=True,
            )
        else:
            value = expectation_from_probabilities(
                self.problem,
                probabilities,
                reverse_key=True,
            )
        return float(value), probabilities

    def _optimize_once(
        self,
        initial: np.ndarray,
        *,
        optimizer: OptimizerName,
        maxiter: int,
        seed: int,
    ) -> OptimizationResult:
        parameters0 = np.asarray(initial, dtype=float).reshape(-1)
        if parameters0.size != self.parameter_count:
            raise ValueError(
                f"QAOA needs {self.parameter_count} parameters, got {parameters0.size}"
            )

        def objective(parameters: np.ndarray) -> float:
            return self._evaluate(parameters)[0]

        bounds = [(-np.pi, np.pi)] * self.parameter_count
        parameters, history, evaluations = _minimize(
            objective,
            parameters0,
            optimizer=optimizer,
            maxiter=maxiter,
            seed=seed,
            bounds=bounds if optimizer == "slsqp" else None,
        )
        optimization_value, probabilities = self._evaluate(parameters)
        expected_value = expectation_from_probabilities(
            self.problem,
            probabilities,
            reverse_key=True,
        )
        exact = self.problem.exact_solution()
        selected, selected_value, selected_probability, feasible_mass, optimum_mass = (
            _distribution_metrics(
                self.problem,
                probabilities,
                bit_order="qubit",
                optimum_value=exact.optimal_value,
                optimum_states=exact.optimal_states,
            )
        )
        circuit = constrained_qaoa_circuit(
            self.problem,
            parameters,
            initializer=self.initializer,
            initializer_strength=self.initializer_strength,
            ring=self.ring,
        )
        program = _run_exact_probabilities(circuit, self.problem.n_features)[1]
        return OptimizationResult(
            method="constrained_qaoa_xy",
            parameters=parameters,
            history=history,
            probabilities=probabilities,
            selected_state=selected,
            selected_value=selected_value,
            selected_probability=selected_probability,
            expected_value=float(expected_value),
            feasible_probability=feasible_mass,
            optimum_probability=optimum_mass,
            optimum_value=exact.optimal_value,
            evaluations=evaluations,
            gate_count=program.count_ops(),
            depth=program.depth(),
            metadata={
                "optimizer": optimizer,
                "layers": self.layers,
                "initializer": self.initializer,
                "initializer_strength": self.initializer_strength,
                "ring": self.ring,
                "optimization_objective": self.objective_mode,
                "optimization_value": float(optimization_value),
                "cvar_alpha": self.cvar_alpha
                if self.objective_mode == "cvar"
                else None,
                "objective_convention": self.problem.objective_convention,
                "parameter_count": self.parameter_count,
            },
        )

    def optimize(
        self,
        initial_parameters: Sequence[float] | None = None,
        *,
        optimizer: OptimizerName = "slsqp",
        maxiter: int = 60,
        seed: int = 0,
        restarts: int = 1,
        initial_scale: float = 0.5,
    ) -> OptimizationResult:
        """Optimize QAOA and optionally keep the best independent restart."""

        if not isinstance(restarts, (int, np.integer)) or int(restarts) < 1:
            raise ValueError("restarts must be a positive integer")
        if not np.isfinite(initial_scale) or float(initial_scale) <= 0.0:
            raise ValueError("initial_scale must be a positive finite number")
        restart_count = int(restarts)
        supplied = (
            None
            if initial_parameters is None
            else np.asarray(initial_parameters, dtype=float).reshape(-1)
        )
        if supplied is not None and supplied.size != self.parameter_count:
            raise ValueError(
                f"QAOA needs {self.parameter_count} parameters, got {supplied.size}"
            )
        rng = np.random.default_rng(seed)
        results: list[OptimizationResult] = []
        for restart in range(restart_count):
            if restart == 0 and supplied is not None:
                initial = supplied.copy()
            else:
                initial = rng.uniform(
                    -float(initial_scale),
                    float(initial_scale),
                    size=self.parameter_count,
                )
            results.append(
                self._optimize_once(
                    initial,
                    optimizer=optimizer,
                    maxiter=maxiter,
                    seed=seed + restart,
                )
            )
        best = min(results, key=_result_priority)
        metadata = dict(best.metadata)
        selected_index = min(
            range(len(results)),
            key=lambda index: _result_priority(results[index]),
        )
        metadata.update(
            {
                "restart_count": restart_count,
                "restart_expected_values": [
                    float(result.expected_value) for result in results
                ],
                "restart_optimization_values": [
                    float(
                        result.metadata.get(
                            "optimization_value", result.expected_value
                        )
                    )
                    for result in results
                ],
                "selected_restart": int(selected_index),
                "initial_scale": float(initial_scale),
            }
        )
        best.metadata = metadata
        return best

    def optimize_progressive(
        self,
        *,
        optimizer: OptimizerName = "slsqp",
        maxiter: int = 60,
        seed: int = 0,
        restarts: int = 1,
        initial_scale: float = 0.5,
    ) -> list[OptimizationResult]:
        """Optimize depths 1..``layers`` with deterministic warm starts.

        The returned list contains one result per depth.  The first depth uses
        the normal initializer; every later depth receives the previous depth's
        optimized angles as its first restart, while retaining any requested
        additional random restarts.
        """

        previous: np.ndarray | None = None
        stages: list[OptimizationResult] = []
        for layer in range(1, self.layers + 1):
            solver = ConstrainedQAOAQmRMR(
                self.problem,
                layers=layer,
                initializer=self.initializer,
                initializer_strength=self.initializer_strength,
                ring=self.ring,
                objective_mode=self.objective_mode,
                cvar_alpha=self.cvar_alpha,
            )
            initial = (
                None
                if previous is None
                else self.lift_parameters(previous, layer)
            )
            result = solver.optimize(
                initial_parameters=initial,
                optimizer=optimizer,
                maxiter=maxiter,
                seed=seed + layer,
                restarts=restarts,
                initial_scale=initial_scale,
            )
            metadata = dict(result.metadata)
            metadata.update(
                {
                    "warm_start_used": bool(previous is not None),
                    "warm_start_source_layers": layer - 1 if previous is not None else None,
                    "progressive_target_layers": self.layers,
                }
            )
            result.metadata = metadata
            stages.append(result)
            previous = result.parameters
        return stages

