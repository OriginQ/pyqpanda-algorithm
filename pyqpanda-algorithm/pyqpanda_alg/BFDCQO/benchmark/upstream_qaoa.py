# Licensed under the Apache License, Version 2.0.
"""Adapter for the repository's public PyQPanda QAOA implementation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional
import warnings

import numpy as np
import sympy as sp

from pyqpanda_alg.QAOA import qaoa as upstream_qaoa

from ..problems import IsingProblem


def ising_to_sympy(model: IsingProblem) -> sp.Expr:
    """Convert ``model`` to the binary expression expected by upstream QAOA.

    The conversion substitutes :math:`z_i = 1 - 2x_i` directly into the
    quadratic Ising objective.  The upstream implementation infers its qubit
    count from free symbols, so a model containing an unconstrained variable
    is rejected instead of silently running a lower-dimensional problem.
    """

    if not isinstance(model, IsingProblem):
        raise TypeError("model must be an IsingProblem")

    if model.n == 0:
        raise ValueError("upstream QAOA requires at least one variable")

    # Upstream sorts symbols lexicographically.  Fixed-width names preserve
    # numeric qubit order for dimensions where names such as x2 and x10 would
    # otherwise be interleaved.
    symbol_width = max(1, len(str(model.n - 1)))
    variables = tuple(
        sp.Symbol(f"x{index:0{symbol_width}d}")
        for index in range(model.n)
    )
    spins = [1 - 2 * variable for variable in variables]
    expression = sp.Float(model.offset)

    for i in range(model.n):
        expression += sp.Float(model.h[i]) * spins[i]
        for j in range(i + 1, model.n):
            expression += sp.Float(model.J[i, j]) * spins[i] * spins[j]

    expression = sp.expand(expression)
    if expression.free_symbols != set(variables):
        missing = sorted(
            set(variables) - expression.free_symbols,
            key=lambda symbol: symbol.name,
        )
        missing_names = ", ".join(str(symbol) for symbol in missing)
        raise ValueError(
            "upstream QAOA cannot represent unconstrained variables; "
            f"missing from the simplified objective: {missing_names}"
        )
    return expression


@dataclass(frozen=True)
class UpstreamQAOAResult:
    """Observed output and execution budget of one upstream QAOA run.

    ``initial_parameter_seed`` controls only the locally generated optimizer
    starting point.  CPUQVM shot sampling has its own randomness and is not
    claimed to be reproducible from this value.
    """

    distribution: dict[str, float]
    initial_parameters: np.ndarray
    initial_parameter_seed: Optional[int]
    parameters: np.ndarray
    loss: float
    objective_evaluations: int
    circuit_evaluations: int
    shots_per_circuit: int
    total_shots: int
    backend: str = "pyqpanda3.CPUQVM"
    implementation: str = "pyqpanda_alg.QAOA.qaoa.QAOA"


class UpstreamQAOASolver:
    """Run the repository's public QAOA solver on its real CPUQVM path.

    Upstream CPUQVM distributions already use the convention that q0 is the
    right-most (least-significant) bit.  Keys are therefore validated and
    preserved unchanged to match :meth:`IsingProblem.energy_bitstring`.

    ``initial_parameter_seed`` seeds only a local NumPy generator used to
    create the explicit optimizer starting point.  It does not seed CPUQVM
    measurement sampling.  ``seed`` remains as a deprecated compatibility
    alias and has the same deliberately narrow meaning.
    """

    def __init__(
        self,
        model: IsingProblem,
        *,
        p: int = 1,
        shots: int = 1024,
        max_evals: int = 20,
        initial_parameter_seed: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(model, IsingProblem):
            raise TypeError("model must be an IsingProblem")
        if model.n == 0:
            raise ValueError("model must contain at least one variable")
        if isinstance(p, bool) or not isinstance(p, Integral) or p <= 0:
            raise ValueError("p must be a positive integer")
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if (
            isinstance(max_evals, bool)
            or not isinstance(max_evals, Integral)
            or max_evals <= 0
        ):
            raise ValueError("max_evals must be a positive integer")
        if initial_parameter_seed is not None and seed is not None:
            raise ValueError(
                "pass initial_parameter_seed or deprecated seed, not both"
            )
        if seed is not None:
            warnings.warn(
                "seed is deprecated; use initial_parameter_seed (it controls "
                "only optimizer initial parameters, not CPUQVM sampling)",
                DeprecationWarning,
                stacklevel=2,
            )
            initial_parameter_seed = seed
        if initial_parameter_seed is not None and (
            isinstance(initial_parameter_seed, bool)
            or not isinstance(initial_parameter_seed, Integral)
            or not 0 <= initial_parameter_seed < 2**32
        ):
            raise ValueError(
                "initial_parameter_seed must be an integer in [0, 2**32)"
            )

        self.model: IsingProblem = model
        self.p: int = int(p)
        self.shots: int = int(shots)
        self.max_evals: int = int(max_evals)
        self.initial_parameter_seed: Optional[int] = (
            None
            if initial_parameter_seed is None
            else int(initial_parameter_seed)
        )
        self._qaoa_instance: Optional[Any] = None

    @property
    def qaoa_instance(self) -> Any:
        """Return the concrete upstream instance after :meth:`run`."""

        if self._qaoa_instance is None:
            raise RuntimeError("QAOA has not been run yet")
        return self._qaoa_instance

    def run(self) -> UpstreamQAOAResult:
        """Optimize and sample using upstream QAOA and a PyQPanda3 CPUQVM."""

        expression = ising_to_sympy(self.model)
        engine: Any = upstream_qaoa.QAOA(expression)
        self._qaoa_instance = engine

        initial_parameters: np.ndarray = np.asarray(
            np.random.default_rng(self.initial_parameter_seed).random(self.p * 2)
            * np.pi,
            dtype=float,
        )
        distribution, parameters, loss = engine.run(
            layer=self.p,
            initial_para=initial_parameters.copy(),
            shots=self.shots,
            optimizer="COBYLA",
            optimizer_option={"options": {"maxiter": self.max_evals}},
        )

        if not distribution:
            raise RuntimeError("upstream QAOA must return a non-empty distribution")
        normalized_distribution = {
            str(bitstring): float(probability)
            for bitstring, probability in distribution.items()
        }
        invalid_keys = [
            bitstring
            for bitstring in normalized_distribution
            if len(bitstring) != self.model.n
            or any(bit not in "01" for bit in bitstring)
        ]
        if invalid_keys:
            raise RuntimeError(
                "upstream QAOA returned invalid computational-basis keys: "
                f"{invalid_keys!r}"
            )
        probabilities = np.asarray(
            list(normalized_distribution.values()), dtype=float
        )
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("upstream QAOA probabilities must be finite")
        if np.any(probabilities < 0.0):
            raise RuntimeError("upstream QAOA probabilities must be non-negative")
        probability_total = float(probabilities.sum())
        if not np.isfinite(probability_total) or probability_total <= 0.0:
            raise RuntimeError(
                "upstream QAOA probabilities must have a positive total"
            )
        normalized_distribution = {
            bitstring: probability / probability_total
            for bitstring, probability in normalized_distribution.items()
        }

        final_parameters = np.asarray(parameters, dtype=float).copy()
        if not np.all(np.isfinite(final_parameters)) or not np.isfinite(loss):
            raise RuntimeError(
                "upstream QAOA parameters and loss must be finite"
            )

        objective_evaluations = max(int(engine.iter) - 1, 0)
        circuit_evaluations = int(engine.circuit_iter)
        return UpstreamQAOAResult(
            distribution=normalized_distribution,
            initial_parameters=initial_parameters.copy(),
            initial_parameter_seed=self.initial_parameter_seed,
            parameters=final_parameters,
            loss=float(loss),
            objective_evaluations=objective_evaluations,
            circuit_evaluations=circuit_evaluations,
            shots_per_circuit=self.shots,
            total_shots=circuit_evaluations * self.shots,
        )
