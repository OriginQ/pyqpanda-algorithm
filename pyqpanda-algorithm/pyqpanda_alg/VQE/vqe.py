"""Hamiltonian-level variational quantum eigensolver with local execution.

:class:`VQE` validates the observable (non-empty, Hermitian, finite
coefficients), infers the qubit count from its Pauli operator, and
builds the built-in hardware-efficient ansatz — or accepts a custom one
following the same callable contract.  :meth:`VQE.run` minimizes the
energy expectation with the built-in SciPy optimizer adapter; every
energy evaluation materializes the concrete circuit and submits
``(circuit, Hamiltonian)`` to the execution backend's estimator, so the
default CPU path needs no runtime installation.  The completed run is
snapshotted into the immutable :class:`VQEResult`.
"""

import dataclasses
from typing import Any, Optional

import numpy as np
from pyqpanda3.core import QProg, RY, RZ
from pyqpanda3.hamiltonian import Hamiltonian
from pyqpanda3.vqcircuit import VQCircuit

from ..execution import ExecutionOptions, resolve_backend
from .ansatz import hardware_efficient_ansatz
from .config import VQEConfig
from .optimizer import SciPyOptimizer
from .result import VQEResult


class VQE:
    """Variational quantum eigensolver for a pyqpanda3 :class:`Hamiltonian`."""

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        ansatz: Optional[Any] = None,
        layers: int = 1,
    ) -> None:
        """Validate the observable and prepare the ansatz.

        Args:
            hamiltonian: The observable whose ground state is sought.
                Must be non-empty, Hermitian (real coefficients in the
                Pauli basis) and have finite coefficients; the qubit
                count is inferred from the Pauli operator.
            ansatz: Optional variational circuit following the callable
                contract ``ansatz(parameters) -> .circuits()[0] ->
                QProg``.  Defaults to the built-in hardware-efficient
                ansatz, which for a single qubit reduces to the RY/RZ
                layers without the CNOT entanglement chain.
            layers: Number of rotation-and-entanglement repetitions of
                the default ansatz.

        Raises:
            TypeError: If ``hamiltonian`` is not a pyqpanda3
                :class:`Hamiltonian`.
            ValueError: If the observable is empty, acts on no qubits,
                is non-Hermitian or has non-finite coefficients, or if
                ``layers`` is smaller than 1.
        """
        num_qubits, hamiltonian = _validate_hamiltonian(hamiltonian)
        self._hamiltonian = hamiltonian
        self._num_qubits = num_qubits
        self._ansatz = ansatz if ansatz is not None else _default_ansatz(num_qubits, layers)

    def run(
        self,
        initial_parameters,
        *,
        backend: Optional[Any] = None,
        execution_options: Optional[ExecutionOptions] = None,
        config: Optional[VQEConfig] = None,
    ) -> VQEResult:
        """Minimize the energy expectation and return the immutable result.

        Args:
            initial_parameters: One-dimensional starting parameters,
                matching the ansatz parameter count.
            backend: The execution backend to submit estimations to.
                Keyword-only.  When None, the local CPU backend is used.
            execution_options: Submission options for the backend.
                Keyword-only.  When None, defaults are used with the
                sampling budget taken from ``config.shots``.
            config: Run configuration.  Keyword-only.  When None,
                :class:`VQEConfig` defaults apply.

        Returns:
            The frozen :class:`VQEResult` snapshot of the run: energy
            and optimal parameters at the found optimum, convergence
            bookkeeping, the per-evaluation energy history, the
            materialized circuit at the optimum, and the task IDs and
            metadata of the execution.

        Raises:
            ValueError: If ``initial_parameters`` is not one-dimensional
                or its length does not match the ansatz parameter count.
        """
        cfg = config if config is not None else VQEConfig()
        execution_backend = resolve_backend(backend)
        options = dataclasses.replace(
            execution_options if execution_options is not None else ExecutionOptions(),
            shots=cfg.shots,
        )
        initial = np.asarray(initial_parameters, dtype=float)
        if initial.ndim != 1:
            raise ValueError(
                f"initial_parameters must be one-dimensional, got shape {initial.shape}"
            )
        expected = self._ansatz.mutable_parameter_total()
        if initial.shape[0] != expected:
            raise ValueError(
                f"initial_parameters has {initial.shape[0]} entries, "
                f"the ansatz expects {expected}"
            )

        optimizer = SciPyOptimizer(
            cfg.optimizer, max_iterations=cfg.max_iterations, tolerance=cfg.tolerance
        )

        def evaluate(parameters: np.ndarray) -> float:
            prog = QProg() << self._ansatz(parameters).circuits()[0]
            task = execution_backend.submit_estimate(
                (prog, self._hamiltonian), options=options
            )
            optimizer.task_ids.append(task.id)
            return float(task.result().single_value())

        optimizer.minimize(evaluate, initial)

        return VQEResult(
            energy=optimizer.energy,
            optimal_parameters=optimizer.parameters,
            converged=optimizer.converged,
            iterations=optimizer.iterations,
            energy_history=tuple(optimizer.energy_history),
            optimal_circuit=QProg() << self._ansatz(optimizer.parameters).circuits()[0],
            task_ids=tuple(optimizer.task_ids),
            metadata={
                "optimizer": optimizer.method,
                "backend": type(execution_backend).__name__,
            },
        )


def _validate_hamiltonian(hamiltonian: Hamiltonian) -> tuple[int, Hamiltonian]:
    """Return ``(num_qubits, hamiltonian)``, rejecting invalid observables.

    The qubit count is inferred from ``pauli_operator().qubits()`` as
    the highest qubit index plus one, so the ansatz spans the full
    Hilbert space of the observable.
    """
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError(
            f"hamiltonian must be a pyqpanda3 Hamiltonian, got {type(hamiltonian).__name__}"
        )
    pauli = hamiltonian.pauli_operator()
    terms = pauli.terms()
    if not terms:
        raise ValueError("hamiltonian must contain at least one term")
    identity_qubits, active_qubits = pauli.qubits()
    qubits = identity_qubits + active_qubits
    if not qubits:
        raise ValueError("hamiltonian must act on at least one qubit")
    for term in terms:
        coef = term.coef()
        if not (np.isfinite(coef.real) and np.isfinite(coef.imag)):
            raise ValueError("hamiltonian coefficients must be finite")
        if coef.imag != 0:
            raise ValueError(
                "hamiltonian must be Hermitian: every coefficient in the "
                "Pauli basis must be real"
            )
    return max(qubits) + 1, hamiltonian


def _default_ansatz(num_qubits: int, layers: int) -> VQCircuit:
    """Built-in hardware-efficient ansatz, CNOT-free for a single qubit.

    Two or more qubits delegate to
    :func:`~pyqpanda_alg.VQE.ansatz.hardware_efficient_ansatz`.  A
    single qubit has no entanglement partner, so the default reduces to
    the same layer-major, qubit-major RY-then-RZ parameter ordering
    without the CNOT chain.
    """
    if num_qubits >= 2:
        return hardware_efficient_ansatz(num_qubits, layers)
    if layers < 1:
        raise ValueError(f"layers must be at least 1, got {layers}")
    ansatz = VQCircuit(1)
    ansatz.set_Param([2 * layers])
    index = 0
    for _ in range(layers):
        ansatz << RY(0, ansatz.Param([index]))
        index += 1
        ansatz << RZ(0, ansatz.Param([index]))
        index += 1
    return ansatz
