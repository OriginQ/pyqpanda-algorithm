# Licensed under the Apache License, Version 2.0.
"""Reference QAOA baseline used by the BF-DCQO benchmark suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from ..backends.numpy_statevector import apply_pauli_rotation, plus_state, probabilities
from ..core.pauli import pair, single
from ..problems.ising import IsingProblem


@dataclass
class QAOAResult:
    """Result returned by :class:`QAOASolver`."""

    best_solution: str
    best_energy: float
    exact_ground_energy: float
    exact_ground_states: list[str]
    expected_energy: float
    ground_state_probability: float
    parameters: list[float]
    objective_evaluations: int
    circuit_evaluations: int
    training_shots: int
    final_shots: int
    total_shots: int
    backend: str
    optimizer_success: bool
    optimizer_message: str

    @property
    def success(self) -> bool:
        """Whether the best sampled state reaches the exact reference energy."""
        return bool(np.isclose(self.best_energy, self.exact_ground_energy, atol=1e-9))


class QAOASolver:
    """Standard X-mixer QAOA baseline for quadratic Ising problems.

    This small, transparent implementation exists so the benchmark can account
    for every objective evaluation and every shot.  It supports the same NumPy
    reference backend and PyQPanda3 CPUQVM backend as BF-DCQO.

    The parameter vector is ordered as ``[beta_0, ..., gamma_0, ...]``.  When
    ``shots_per_eval`` is not ``None``, each classical optimizer call consumes
    that many shots.  If ``total_shot_budget`` is supplied, the remaining shots
    after training are used for the final sample.
    """

    def __init__(
        self,
        problem: IsingProblem,
        *,
        p: int = 1,
        backend: Literal["numpy", "pyqpanda"] = "numpy",
        shots_per_eval: int | None = 256,
        max_evals: int = 20,
        final_shots: int = 1024,
        total_shot_budget: int | None = None,
        seed: int = 7,
        method: str = "COBYLA",
    ) -> None:
        if p < 1:
            raise ValueError("p must be >= 1")
        if not hasattr(problem, "h") or not hasattr(problem, "J"):
            raise TypeError(
                "QAOA baseline accepts quadratic Ising/QUBO problems only; "
                "quadratize a HUBO before using QAOA."
            )
        if shots_per_eval is not None and shots_per_eval < 1:
            raise ValueError("shots_per_eval must be None or >= 1")
        if max_evals < 1 or final_shots < 1:
            raise ValueError("max_evals and final_shots must be >= 1")

        self.problem = problem
        self.p = int(p)
        self.backend = backend
        self.shots_per_eval = shots_per_eval
        self.max_evals = int(max_evals)
        self.final_shots = int(final_shots)
        self.total_shot_budget = total_shot_budget
        self.seed = int(seed)
        self.method = method
        self._energies = problem.energies()
        self._rng = np.random.default_rng(seed)
        self._evals = 0

    def _numpy_prob(self, params: np.ndarray) -> np.ndarray:
        betas = np.asarray(params[: self.p], dtype=float)
        gammas = np.asarray(params[self.p :], dtype=float)
        state = plus_state(self.problem.n)
        for beta, gamma in zip(betas, gammas):
            for i, hi in enumerate(self.problem.h):
                if abs(hi) > 1e-15:
                    state = apply_pauli_rotation(
                        state,
                        single(self.problem.n, i, "Z"),
                        gamma * hi,
                    )
            for i in range(self.problem.n):
                for j in range(i + 1, self.problem.n):
                    coupling = self.problem.J[i, j]
                    if abs(coupling) > 1e-15:
                        state = apply_pauli_rotation(
                            state,
                            pair(self.problem.n, i, "Z", j, "Z"),
                            gamma * coupling,
                        )
            for q in range(self.problem.n):
                state = apply_pauli_rotation(
                    state,
                    single(self.problem.n, q, "X"),
                    beta,
                )
        return probabilities(state)

    def _exact_prob(self, params: np.ndarray) -> np.ndarray:
        if self.backend == "numpy":
            return self._numpy_prob(params)
        if self.backend == "pyqpanda":
            from ..backends.pyqpanda3_backend import qaoa_probabilities

            return qaoa_probabilities(
                self.problem,
                np.asarray(params[: self.p]),
                np.asarray(params[self.p :]),
            )
        raise ValueError(f"unknown backend {self.backend}")

    def probabilities(self, parameters) -> np.ndarray:
        """Return the exact simulator distribution for supplied parameters.

        This helper does not consume the finite-shot budget and is used by
        application benchmarks that need to marginalize auxiliary variables.
        """
        return self._exact_prob(np.asarray(parameters, dtype=float))

    def _sample_prob(self, params: np.ndarray, shots: int) -> np.ndarray:
        if self.backend == "numpy":
            exact = self._exact_prob(params)
            return self._rng.multinomial(shots, exact) / shots
        if self.backend == "pyqpanda":
            from ..backends.pyqpanda3_backend import qaoa_sample_counts

            counts = qaoa_sample_counts(
                self.problem,
                np.asarray(params[: self.p]),
                np.asarray(params[self.p :]),
                shots,
            )
            prob = np.zeros(1 << self.problem.n)
            total = sum(counts.values())
            for bitstring, count in counts.items():
                prob[int(bitstring, 2)] += count / total
            return prob
        raise ValueError(f"unknown backend {self.backend}")

    def _objective(self, params: np.ndarray) -> float:
        self._evals += 1
        if self.shots_per_eval is None:
            prob = self._exact_prob(params)
        else:
            prob = self._sample_prob(params, self.shots_per_eval)
        return float(prob @ self._energies)

    def solve(self) -> QAOAResult:
        """Optimize QAOA parameters and sample the final circuit."""
        self._evals = 0
        self._rng = np.random.default_rng(self.seed)
        init_rng = np.random.default_rng(self.seed + 991)
        x0 = np.concatenate(
            [
                init_rng.uniform(0.05, 0.45, self.p),
                init_rng.uniform(0.05, 1.2, self.p),
            ]
        )
        bounds = [(0.0, np.pi / 2)] * self.p + [(0.0, 2 * np.pi)] * self.p
        result = minimize(
            self._objective,
            x0,
            method=self.method,
            bounds=bounds,
            options={"maxiter": self.max_evals},
        )
        params = np.asarray(result.x if result.x is not None else x0, dtype=float)
        exact_prob = self._exact_prob(params)

        training_shots = (
            0 if self.shots_per_eval is None else self._evals * self.shots_per_eval
        )
        final_shots = self.final_shots
        if self.total_shot_budget is not None and self.shots_per_eval is not None:
            remaining = int(self.total_shot_budget) - training_shots
            if remaining < 1:
                raise RuntimeError(
                    "QAOA training consumed "
                    f"{training_shots} shots, exceeding total_shot_budget="
                    f"{self.total_shot_budget}."
                )
            final_shots = remaining

        final_prob = self._sample_prob(params, final_shots)
        supported = np.flatnonzero(final_prob > 0)
        index = int(supported[np.argmin(self._energies[supported])])
        exact_energy, states = self.problem.exact_ground_states()
        exact_indices = {int(state, 2) for state in states}

        return QAOAResult(
            best_solution=format(index, f"0{self.problem.n}b"),
            best_energy=float(self._energies[index]),
            exact_ground_energy=float(exact_energy),
            exact_ground_states=states,
            expected_energy=float(exact_prob @ self._energies),
            ground_state_probability=float(sum(exact_prob[i] for i in exact_indices)),
            parameters=[float(value) for value in params],
            objective_evaluations=self._evals,
            circuit_evaluations=self._evals + 1,
            training_shots=training_shots,
            final_shots=final_shots,
            total_shots=training_shots + final_shots,
            backend=self.backend,
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        )
