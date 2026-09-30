# Licensed under the Apache License, Version 2.0.
"""BF-DCQO family: basic, CVaR, adaptive-shot, and adaptive-CVaR variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from ..problems.ising import IsingProblem
from ..problems.polynomial import PolynomialIsingProblem
from ..core.hamiltonian import build_initial_hamiltonian, build_problem_hamiltonian, interpolate
from ..core.agp import first_order_gauge_potential
from ..core.measurement import initial_ground_angles, z_expectation_from_probabilities
from ..core.schedule import schedule_lambda, schedule_lambda_dot
from ..core.feedback import cvar_feedback_from_counts, cvar_feedback_from_probabilities
from ..backends.numpy_statevector import apply_pauli_rotation, probabilities, product_state_from_ry


@dataclass
class IterationRecord:
    """Diagnostics recorded after one BF-DCQO feedback iteration."""

    iteration: int
    best_energy: float
    best_bitstring: str
    ground_state_probability: float
    expected_energy: float
    bias: np.ndarray
    raw_bias: np.ndarray | None = None
    alpha1: list[float] = field(default_factory=list)
    shots: int = 0
    effective_feedback_shots: float = 0.0
    feedback_alpha: float = 1.0
    feedback_energy_threshold: float = float("inf")
    uncertainty: float = 1.0
    bias_delta: float = float("nan")


@dataclass
class BFDCQOResult:
    """Result and resource accounting for a BF-DCQO-family solve."""

    best_solution: str
    best_energy: float
    exact_ground_energy: float
    exact_ground_states: list[str]
    history: list[IterationRecord]
    total_shots: int
    circuit_evaluations: int
    backend: str
    variant: str = "basic"
    shot_budget: int | None = None
    stopped_early: bool = False
    stop_reason: str | None = None

    @property
    def success(self) -> bool:
        return bool(np.isclose(self.best_energy, self.exact_ground_energy, atol=1e-9))

    @property
    def shot_history(self) -> list[int]:
        return [int(r.shots) for r in self.history]

    @property
    def uncertainty_history(self) -> list[float]:
        return [float(r.uncertainty) for r in self.history]


class BFDCQO:
    """Basic first-order BF-DCQO supporting native QUBO and HUBO Hamiltonians."""

    variant = "basic"

    def __init__(
        self,
        problem: IsingProblem | PolynomialIsingProblem,
        *,
        n_trotter: int = 3,
        dt: float = 0.1,
        iterations: int = 10,
        shots: int | None = 2000,
        hx: float = -1.0,
        backend: Literal["numpy", "pyqpanda"] = "numpy",
        seed: int = 7,
        impulse_only: bool = False,
    ):
        if n_trotter < 1 or iterations < 1:
            raise ValueError("n_trotter and iterations must be >=1")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if shots is not None and shots < 1:
            raise ValueError("shots must be None or >=1")
        self.problem = problem
        self.n_trotter = int(n_trotter)
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.shots = shots
        self.hx = float(hx)
        self.backend = backend
        self.seed = int(seed)
        self.impulse_only = bool(impulse_only)
        self._hf = build_problem_hamiltonian(problem)

    @property
    def T(self):
        return self.n_trotter * self.dt

    def _layers(self, bias):
        hi = build_initial_hamiltonian(self.problem.n, bias, self.hx)
        layers = []
        alphas = []
        for k in range(1, self.n_trotter + 1):
            t = k * self.dt
            lam = schedule_lambda(t, self.T)
            dlam = schedule_lambda_dot(t, self.T)
            g = first_order_gauge_potential(hi, self._hf, lam)
            alphas.append(g.alpha1)
            terms = []
            if not self.impulse_only:
                for p, c in interpolate(hi, self._hf, lam).items():
                    terms.append((p, self.dt * float(np.real(c))))
            for p, c in g.A1.items():
                terms.append((p, self.dt * dlam * float(c)))
            layers.append(terms)
        return layers, alphas

    def _probabilities(self, bias):
        angles = initial_ground_angles(bias, self.hx)
        layers, alphas = self._layers(bias)
        if self.backend == "numpy":
            state = product_state_from_ry(angles)
            for layer in layers:
                for p, theta in layer:
                    state = apply_pauli_rotation(state, p, theta)
            return probabilities(state), alphas
        if self.backend == "pyqpanda":
            from ..backends.pyqpanda3_backend import run_bfdcqo_probabilities
            return run_bfdcqo_probabilities(self.problem.n, angles, layers), alphas
        raise ValueError(f"unknown backend {self.backend}")

    def _feedback(self, *, exact_prob, counts, energies, n, shots):
        if counts is None:
            bias = z_expectation_from_probabilities(exact_prob, n)
            return bias, 1.0, 1.0, float("inf")
        feedback = counts.astype(float) / float(shots)
        bias = z_expectation_from_probabilities(feedback, n)
        return bias, float(shots), 1.0, float("inf")

    def _solve_with_shot_schedule(self, shot_schedule: list[int | None]) -> BFDCQOResult:
        n = self.problem.n
        exact_energy, exact_states = self.problem.exact_ground_states()
        exact_indices = {int(s, 2) for s in exact_states}
        energies = self.problem.energies()
        rng = np.random.default_rng(self.seed)
        bias = np.zeros(n)
        history: list[IterationRecord] = []
        best = float("inf")
        best_state = None
        total_shots = 0

        for it, iteration_shots in enumerate(shot_schedule):
            exact_prob, alphas = self._probabilities(bias)
            counts = None
            if iteration_shots is not None:
                counts = rng.multinomial(int(iteration_shots), exact_prob)
                total_shots += int(iteration_shots)
                supported = np.flatnonzero(counts > 0)
            else:
                supported = np.flatnonzero(exact_prob > 1e-15)
            if supported.size == 0:
                supported = np.array([int(exact_prob.argmax())])
            idx = int(supported[np.argmin(energies[supported])])
            e = float(energies[idx])
            if e < best:
                best = e
                best_state = format(idx, f"0{n}b")

            pgs = float(sum(exact_prob[i] for i in exact_indices))
            eexp = float(exact_prob @ energies)
            raw_feedback_prob = exact_prob if counts is None else counts.astype(float) / float(iteration_shots)
            raw_bias = z_expectation_from_probabilities(raw_feedback_prob, n)
            new_bias, eff, alpha, threshold = self._feedback(
                exact_prob=exact_prob,
                counts=counts,
                energies=energies,
                n=n,
                shots=iteration_shots,
            )
            uncertainty = float(np.mean(1.0 - np.abs(new_bias)))
            bias_delta = float(np.mean(np.abs(new_bias - bias)))
            history.append(
                IterationRecord(
                    iteration=it,
                    best_energy=e,
                    best_bitstring=format(idx, f"0{n}b"),
                    ground_state_probability=pgs,
                    expected_energy=eexp,
                    bias=new_bias.copy(),
                    raw_bias=raw_bias.copy(),
                    alpha1=list(alphas),
                    shots=0 if iteration_shots is None else int(iteration_shots),
                    effective_feedback_shots=float(eff),
                    feedback_alpha=float(alpha),
                    feedback_energy_threshold=float(threshold),
                    uncertainty=uncertainty,
                    bias_delta=bias_delta,
                )
            )
            bias = new_bias

        assert best_state is not None
        return BFDCQOResult(
            best_solution=best_state,
            best_energy=best,
            exact_ground_energy=float(exact_energy),
            exact_ground_states=exact_states,
            history=history,
            total_shots=total_shots,
            circuit_evaluations=len(history),
            backend=self.backend,
            variant=self.variant,
            shot_budget=total_shots if self.shots is not None else None,
        )

    def solve(self) -> BFDCQOResult:
        return self._solve_with_shot_schedule([self.shots] * self.iterations)


class CVaRBFDCQO(BFDCQO):
    """BF-DCQO using a low-energy CVaR subset to update the longitudinal bias.

    ``cvar_alpha`` is the retained fraction of lowest-energy samples.  Setting
    ``cvar_alpha=1`` exactly recovers the basic BF feedback for a fixed seed.
    """

    variant = "cvar"

    def __init__(self, *args, cvar_alpha: float = 0.1, **kwargs):
        if not (0.0 < float(cvar_alpha) <= 1.0):
            raise ValueError("cvar_alpha must lie in (0,1]")
        self.cvar_alpha = float(cvar_alpha)
        super().__init__(*args, **kwargs)

    def _feedback(self, *, exact_prob, counts, energies, n, shots):
        if counts is None:
            info = cvar_feedback_from_probabilities(exact_prob, energies, n, self.cvar_alpha)
            return info.bias, info.effective_samples, info.alpha, info.energy_threshold
        info = cvar_feedback_from_counts(counts, energies, n, self.cvar_alpha)
        return info.bias, info.effective_samples, info.alpha, info.energy_threshold


class AdaptiveShotBFDCQO(BFDCQO):
    """Shot-budget-aware BF-DCQO.

    The scheduler uses the current confidence c=mean_i(|<Z_i>|)=1-u, where
    u=mean_i(1-|<Z_i>|). Early uncertain rounds are sampled cheaply to preserve
    exploration, while later, more confident refinement receives progressively
    more precision. The scheduler reserves ``min_shots`` for every remaining
    iteration, never exceeds the budget, and lets the final iteration consume
    the exact remainder when all ``max_iterations`` are used.

    Optionally, ``cvar_alpha`` combines adaptive shot allocation with CVaR
    feedback. Set it to ``None`` to isolate adaptive shot allocation from CVaR.
    """

    variant = "adaptive"

    def __init__(
        self,
        problem: IsingProblem | PolynomialIsingProblem,
        *,
        total_shot_budget: int,
        max_iterations: int = 6,
        min_shots: int = 128,
        max_shots: int | None = None,
        initial_shots: int | None = None,
        cvar_alpha: float | None = None,
        early_stop: bool = False,
        min_iterations: int = 3,
        uncertainty_tol: float = 0.03,
        bias_delta_tol: float = 0.02,
        **kwargs,
    ):
        total_shot_budget = int(total_shot_budget)
        max_iterations = int(max_iterations)
        min_shots = int(min_shots)
        if total_shot_budget < 1 or max_iterations < 1 or min_shots < 1:
            raise ValueError("budget/iterations/min_shots must be positive")
        if total_shot_budget < min_shots * max_iterations:
            raise ValueError("total_shot_budget must be >= min_shots * max_iterations")
        base = total_shot_budget // max_iterations
        if max_shots is None:
            max_shots = max(min_shots, 2 * base)
        if initial_shots is None:
            initial_shots = base
        self.total_shot_budget = total_shot_budget
        self.max_iterations = max_iterations
        self.min_shots = min_shots
        self.max_shots = int(max_shots)
        self.initial_shots = int(initial_shots)
        self.cvar_alpha = None if cvar_alpha is None else float(cvar_alpha)
        if self.cvar_alpha is not None and not (0.0 < self.cvar_alpha <= 1.0):
            raise ValueError("cvar_alpha must lie in (0,1]")
        self.early_stop = bool(early_stop)
        self.min_iterations = int(min_iterations)
        self.uncertainty_tol = float(uncertainty_tol)
        self.bias_delta_tol = float(bias_delta_tol)
        # ``shots``/``iterations`` in BFDCQO are not used by the adaptive loop,
        # but retaining sensible values keeps resource helpers compatible.
        kwargs.pop("shots", None)
        kwargs.pop("iterations", None)
        super().__init__(problem, shots=min_shots, iterations=max_iterations, **kwargs)

    def _feedback(self, *, exact_prob, counts, energies, n, shots):
        if self.cvar_alpha is None:
            return super()._feedback(
                exact_prob=exact_prob, counts=counts, energies=energies, n=n, shots=shots
            )
        info = cvar_feedback_from_counts(counts, energies, n, self.cvar_alpha)
        return info.bias, info.effective_samples, info.alpha, info.energy_threshold

    def _next_shots(self, iteration: int, spent: int, uncertainty: float | None) -> int:
        remaining = self.total_shot_budget - spent
        remaining_iterations = self.max_iterations - iteration
        if remaining_iterations <= 1:
            return remaining
        reserve = self.min_shots * (remaining_iterations - 1)
        available_now = remaining - reserve
        if iteration == 0 or uncertainty is None:
            target = self.initial_shots
        else:
            # Confidence-progressive allocation. At maximum uncertainty the
            # target is roughly half of the uniform per-iteration budget; as
            # the feedback becomes confident, more precision is allocated.
            base = self.total_shot_budget / self.max_iterations
            u = float(np.clip(uncertainty, 0.0, 1.0))
            confidence = 1.0 - u
            target = int(round(base * (0.5 + 0.5 * confidence)))
            target = min(target, self.max_shots)
        return int(np.clip(target, self.min_shots, max(self.min_shots, available_now)))

    def solve(self) -> BFDCQOResult:
        n = self.problem.n
        exact_energy, exact_states = self.problem.exact_ground_states()
        exact_indices = {int(s, 2) for s in exact_states}
        energies = self.problem.energies()
        rng = np.random.default_rng(self.seed)
        bias = np.zeros(n)
        history: list[IterationRecord] = []
        best = float("inf")
        best_state = None
        spent = 0
        previous_uncertainty: float | None = None
        stopped_early = False
        stop_reason = None

        for it in range(self.max_iterations):
            shots = self._next_shots(it, spent, previous_uncertainty)
            if shots <= 0:
                break
            exact_prob, alphas = self._probabilities(bias)
            counts = rng.multinomial(shots, exact_prob)
            spent += shots
            supported = np.flatnonzero(counts > 0)
            if supported.size == 0:
                supported = np.array([int(exact_prob.argmax())])
            idx = int(supported[np.argmin(energies[supported])])
            e = float(energies[idx])
            if e < best:
                best = e
                best_state = format(idx, f"0{n}b")
            pgs = float(sum(exact_prob[i] for i in exact_indices))
            eexp = float(exact_prob @ energies)
            raw_feedback_prob = counts.astype(float) / float(shots)
            raw_bias = z_expectation_from_probabilities(raw_feedback_prob, n)
            new_bias, eff, alpha, threshold = self._feedback(
                exact_prob=exact_prob, counts=counts, energies=energies, n=n, shots=shots
            )
            uncertainty = float(np.mean(1.0 - np.abs(new_bias)))
            bias_delta = float(np.mean(np.abs(new_bias - bias)))
            history.append(
                IterationRecord(
                    iteration=it,
                    best_energy=e,
                    best_bitstring=format(idx, f"0{n}b"),
                    ground_state_probability=pgs,
                    expected_energy=eexp,
                    bias=new_bias.copy(),
                    raw_bias=raw_bias.copy(),
                    alpha1=list(alphas),
                    shots=shots,
                    effective_feedback_shots=float(eff),
                    feedback_alpha=float(alpha),
                    feedback_energy_threshold=float(threshold),
                    uncertainty=uncertainty,
                    bias_delta=bias_delta,
                )
            )
            bias = new_bias
            previous_uncertainty = uncertainty

            if (
                self.early_stop
                and len(history) >= self.min_iterations
                and uncertainty <= self.uncertainty_tol
                and bias_delta <= self.bias_delta_tol
            ):
                stopped_early = True
                stop_reason = "bias_converged"
                break
            if spent >= self.total_shot_budget:
                break

        assert best_state is not None
        return BFDCQOResult(
            best_solution=best_state,
            best_energy=best,
            exact_ground_energy=float(exact_energy),
            exact_ground_states=exact_states,
            history=history,
            total_shots=spent,
            circuit_evaluations=len(history),
            backend=self.backend,
            variant="adaptive+cvar" if self.cvar_alpha is not None else self.variant,
            shot_budget=self.total_shot_budget,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )


# Short alias useful in user-facing examples.
AdaptiveBFDCQO = AdaptiveShotBFDCQO
