"""Classical optimizer adapters for the VQE package.

The built-in adapter wraps :func:`scipy.optimize.minimize` and owns an
explicit, serializable state: the optimizer method, the convergence
criteria (tolerance and iteration budget), the initial and current
parameters, the per-evaluation energy and parameter history, the number
of energy evaluations performed (iterations), the task IDs of every
submitted estimation, and the convergence flag.  ``state_dict`` /
``load_state_dict`` expose the state as JSON-primitive dicts so the
checkpoint layer can snapshot and resume a run exactly where it
stopped.

Two driving modes exist.  ``minimize(evaluate, initial_parameters)``
runs the whole optimization in one call; ``step(evaluate)`` runs one
classical iteration from the current parameters, which is how the
resumable VQE state machine advances one poll per iteration.  Custom
optimizer adapters implement the same surface: the one-shot
``minimize`` contract always works, and a resumable adapter additionally
implements ``step``, the state attributes below, and the
``state_dict`` / ``load_state_dict`` protocol.
"""

from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize


class SciPyOptimizer:
    """Built-in optimizer adapter over :func:`scipy.optimize.minimize`.

    ``minimize(evaluate, initial_parameters)`` minimizes the supplied
    objective with the configured method, recording every energy
    evaluation (parameters, energy, iteration count) and the task ID of
    the underlying estimation into the adapter's explicit state.  After
    the optimizer returns, one final evaluation at the returned optimum
    pins ``parameters``/``energy`` to the same point, and that value
    closes the energy history.

    ``step(evaluate)`` is the resumable counterpart: it runs exactly one
    classical iteration from the current ``parameters`` (seeded by the
    caller or left by a previous step), records one energy-history entry
    at the returned point, and reports convergence.  Probe evaluations
    inside the step submit to the backend and record their task IDs, but
    only the recorded evaluation closes an iteration, so the history
    length always equals the iteration count.
    """

    resume_supported = True

    def __init__(self, method: str, *, max_iterations: int, tolerance: float) -> None:
        """Configure the adapter; the state is reset by :meth:`minimize`."""
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.initial_parameters: Optional[np.ndarray] = None
        self.parameters: Optional[np.ndarray] = None
        self.energy: Optional[float] = None
        self.energy_history: list[float] = []
        self.parameter_history: list[np.ndarray] = []
        self.task_ids: list[str] = []
        self.iterations: int = 0
        self.converged: bool = False

    def minimize(self, evaluate: Callable[[np.ndarray], float], initial_parameters):
        """Minimize ``evaluate`` and record every evaluation in the state.

        ``evaluate`` returns the energy of one parameter point; each
        call counts as one iteration.  The returned scipy
        :class:`~scipy.optimize.OptimizeResult` carries the optimizer's
        own diagnostics, while the adapter state holds the run history.
        """
        initial = np.asarray(initial_parameters, dtype=float)
        if initial.ndim != 1:
            raise ValueError(
                f"initial_parameters must be one-dimensional, got shape {initial.shape}"
            )
        self.initial_parameters = initial.copy()
        self.parameters = initial.copy()
        self.energy = None
        self.energy_history = []
        self.parameter_history = []
        self.task_ids = []
        self.iterations = 0
        self.converged = False

        def objective(params: np.ndarray) -> float:
            self.parameters = np.asarray(params, dtype=float).copy()
            energy = float(evaluate(self.parameters))
            self.energy = energy
            self.energy_history.append(energy)
            self.parameter_history.append(self.parameters.copy())
            self.iterations += 1
            return energy

        result = minimize(
            objective,
            initial,
            method=self.method,
            tol=self.tolerance,
            options={"maxiter": self.max_iterations},
        )

        # Pin parameters and energy to the exact returned optimum; the
        # evaluation is recorded in the state and closes the history.
        self.parameters = np.asarray(result.x, dtype=float).copy()
        objective(self.parameters)
        self.converged = bool(result.success)
        return result

    def step(self, evaluate: Callable[[np.ndarray], float]) -> bool:
        """Run one classical optimization iteration from the current point.

        One iteration evaluates the objective at the current point and
        its finite-difference neighbors, lets the optimizer take one
        step, and records the energy at the returned point in the state
        (energy and parameter history, iteration counter).  The internal
        probe evaluations do not enter the history, so the history
        length equals the iteration count.  Returns True once the
        optimizer reports convergence; a run that never converges keeps
        returning False until the iteration budget is exhausted.

        Raises:
            ValueError: If ``parameters`` was never seeded — call
                :meth:`minimize` first or set ``parameters`` directly.
        """
        if self.parameters is None:
            raise ValueError(
                "step() requires current parameters; seed the adapter's "
                "parameters attribute or run minimize() first"
            )
        current = np.asarray(self.parameters, dtype=float).copy()
        result = minimize(
            evaluate,
            current,
            method=self.method,
            tol=self.tolerance,
            options={"maxiter": 1},
        )
        # Pin parameters and energy to the exact point the step returned;
        # the recorded evaluation closes this iteration.
        self.parameters = np.asarray(result.x, dtype=float).copy()
        energy = float(evaluate(self.parameters))
        self.energy = energy
        self.energy_history.append(energy)
        self.parameter_history.append(self.parameters.copy())
        self.iterations += 1
        if result.success:
            self.converged = True
        return self.converged

    def state_dict(self) -> dict:
        """Return the explicit state as JSON-primitive dict."""
        parameters = self.parameters.tolist() if self.parameters is not None else None
        initial = (
            self.initial_parameters.tolist()
            if self.initial_parameters is not None
            else None
        )
        return {
            "method": self.method,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "initial_parameters": initial,
            "parameters": parameters,
            "energy": self.energy,
            "energy_history": [float(e) for e in self.energy_history],
            "parameter_history": [p.tolist() for p in self.parameter_history],
            "task_ids": list(self.task_ids),
            "iterations": self.iterations,
            "converged": self.converged,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the explicit state from a ``state_dict`` snapshot."""
        self.method = state["method"]
        self.max_iterations = state["max_iterations"]
        self.tolerance = state["tolerance"]
        self.initial_parameters = _as_parameters(state["initial_parameters"])
        self.parameters = _as_parameters(state["parameters"])
        self.energy = state["energy"]
        self.energy_history = [float(e) for e in state["energy_history"]]
        self.parameter_history = [_as_parameters(p) for p in state["parameter_history"]]
        self.task_ids = [str(t) for t in state["task_ids"]]
        self.iterations = state["iterations"]
        self.converged = bool(state["converged"])


def _as_parameters(value) -> Optional[np.ndarray]:
    """Return a 1-D float array copy of ``value``, or None for None."""
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"parameters must be one-dimensional, got shape {array.shape}")
    array.setflags(write=False)
    return array
