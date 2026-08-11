"""VQE configuration: the classical knobs of a solver run.

:class:`VQEConfig` carries the optimization controls (iteration
budget, convergence tolerance, optimizer method) and the sampling
budget.  ``shots`` follows :class:`ExecutionOptions` semantics — the
same positive sampling count, defaulting to 1000 — and is merged into
the execution options at run time.  Device selection and the remaining
execution knobs stay in the execution layer and are not part of the
config.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VQEConfig:
    """Immutable VQE run configuration.

    ``max_iterations`` bounds the classical optimization loop,
    ``tolerance`` is the convergence criterion on the energy change,
    and ``shots`` is the sampling budget per energy evaluation,
    matching the :class:`ExecutionOptions` default of 1000.
    ``optimizer`` names the classical optimizer method.
    """

    max_iterations: int = 100
    tolerance: float = 1e-6
    shots: int = 1000
    optimizer: str = "SLSQP"

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")
        if self.shots <= 0:
            raise ValueError(f"shots must be positive, got {self.shots}")
        if not self.optimizer:
            raise ValueError("optimizer must name a classical optimizer method")
