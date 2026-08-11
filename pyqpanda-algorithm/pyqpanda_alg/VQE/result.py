"""VQE outcome wrapper: one canonical snapshot per solver run.

:class:`VQEResult` is a frozen record of a completed or interrupted
optimization: the best energy found and its parameters, convergence
bookkeeping, the per-iteration energy history, the concrete circuit at
the optimum, and transport provenance (task IDs and free-form
metadata).  Arrays and containers are copied on construction so the
result can never be mutated through caller-owned objects.
"""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VQEResult:
    """Immutable outcome of a VQE optimization run.

    ``energy`` and ``optimal_parameters`` describe the best point
    found; ``energy_history`` records every evaluated energy in
    iteration order.  ``optimal_circuit`` is the concrete circuit at
    the optimum when the solver materialized one, and None otherwise.
    ``task_ids`` and ``metadata`` keep execution provenance without
    exposing any live service or credential.
    """

    energy: float
    optimal_parameters: np.ndarray
    converged: bool
    iterations: int
    energy_history: tuple[float, ...]
    optimal_circuit: Any | None = None
    task_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        history = tuple(float(v) for v in self.energy_history)
        if not all(np.isfinite(v) for v in history):
            raise ValueError("energy_history must contain only finite values")
        object.__setattr__(self, "energy", float(self.energy))
        object.__setattr__(self, "energy_history", history)

        parameters = np.array(self.optimal_parameters, dtype=float, copy=True)
        if parameters.ndim != 1:
            raise ValueError(
                f"optimal_parameters must be one-dimensional, got shape {parameters.shape}"
            )
        parameters.setflags(write=False)
        object.__setattr__(self, "optimal_parameters", parameters)

        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "task_ids", tuple(str(t) for t in self.task_ids))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
