"""Backend-independent resource estimation for the HHL solver.

:func:`estimate_hhl_resources` validates the raw linear system (exactly
as :func:`normalize_linear_system` does) and then reports the qubit
counts, controlled evolutions, gate budget, tomography circuits, and
shot request the solver will need — before any circuit is synthesized.
The estimates are independent of a backend: they describe the circuit
and submission plan, not any transpilation a particular machine would
perform.

Counting rules
--------------
``data_qubits``
    ``log2(padded_dimension)`` — the register that holds the solution.
``phase_qubits``
    The phase-estimation register size from ``HHLConfig.phase_qubits``.
``ancilla_qubits``
    Always 1: the success qubit post-selected during measurement.
``total_qubits``
    ``data_qubits + phase_qubits + ancilla_qubits``.
``controlled_evolutions``
    One controlled ``exp(iAt)`` application per phase qubit (evolution
    powers ``2**0 .. 2**(phase_qubits - 1)``), so exactly
    ``phase_qubits``.  Exact.
``synthesized_gates``
    An approximate pre-transpilation gate count, computed with the
    order-of-magnitude heuristic under "Gate-count heuristic" below.
    This is the only approximate field; see ``approximate_labels``.
``tomography_circuits``
    ``3 ** data_qubits`` — one basis circuit per Pauli string over the
    data register, measuring each data qubit in X, Y, or Z.  Full
    density-matrix reconstruction of a ``data_qubits``-qubit register
    needs all ``3**data_qubits`` X/Y/Z strings; the identity component
    is fixed by unit trace, so these strings determine every remaining
    Pauli coefficient.  **Cross-task contract:** the solver's
    ``reconstruct=True`` mode must submit exactly
    ``tomography_circuits`` basis-measurement circuits, one per Pauli
    string, and reconstruct the density matrix from their outcomes.
``shots``
    The requested sampling shots per submitted circuit.  Defaults to
    1000, matching the package-wide ``ExecutionOptions.shots`` default;
    a run may override it, so the estimate records the documented
    default.
``approximate_labels``
    The set of field names that are estimates rather than exact
    counts — currently exactly ``{"synthesized_gates"}``.  Everything
    else in the estimate is exact by construction.

Gate-count heuristic
--------------------
A dense ``2**data_qubits``-square unitary synthesis needs
O(4**data_qubits) elementary gates, and each of the ``2 * phase_qubits``
forward and inverse controlled evolutions costs one such synthesis.
State preparation is approximated by ``2**data_qubits``
amplitude-encoding gates, and the phase-register Hadamards, inverse
QFT, and reciprocal rotations by ``3 * phase_qubits`` gates::

    synthesized_gates = 2 * phase_qubits * 2**(2 * data_qubits)
                        + 2**data_qubits
                        + 3 * phase_qubits
"""

from dataclasses import dataclass
from typing import Any

from pyqpanda_alg.execution import AlgorithmInputError

from .model import HHLConfig
from .validation import normalize_linear_system

#: Success ancilla qubits used by the HHL circuit.
_ANCILLA_QUBITS = 1

#: Field names of :class:`HHLResourceEstimate` that are estimates.
_APPROXIMATE_FIELDS = frozenset({"synthesized_gates"})

#: Requested shots per submitted circuit, matching ExecutionOptions.
_DEFAULT_SHOTS = 1000


@dataclass(frozen=True)
class HHLResourceEstimate:
    """Immutable backend-independent estimate of HHL circuit resources.

    Qubit counts, the controlled-evolution count, the tomography
    basis-circuit count, and the shot request are exact by
    construction; ``synthesized_gates`` is an order-of-magnitude
    estimate, as recorded by ``approximate_labels``.  See the module
    docstring for the counting rules and the gate-count heuristic.
    """

    data_qubits: int
    phase_qubits: int
    total_qubits: int
    controlled_evolutions: int
    synthesized_gates: int
    tomography_circuits: int
    ancilla_qubits: int = 1
    shots: int = _DEFAULT_SHOTS
    approximate_labels: frozenset[str] = _APPROXIMATE_FIELDS

    def __post_init__(self) -> None:
        _require_positive(self.data_qubits, "data_qubits")
        _require_positive(self.phase_qubits, "phase_qubits")
        _require_positive(self.ancilla_qubits, "ancilla_qubits")
        _require_positive(self.controlled_evolutions, "controlled_evolutions")
        _require_positive(self.synthesized_gates, "synthesized_gates")
        _require_positive(self.tomography_circuits, "tomography_circuits")
        _require_positive(self.shots, "shots")
        expected = self.data_qubits + self.phase_qubits + self.ancilla_qubits
        if self.total_qubits != expected:
            raise ValueError(
                f"total_qubits {self.total_qubits} does not match "
                f"data + phase + ancilla = {expected}"
            )
        object.__setattr__(
            self, "approximate_labels", frozenset(self.approximate_labels)
        )


def estimate_hhl_resources(
    matrix, vector, config: HHLConfig
) -> HHLResourceEstimate:
    """Estimate the HHL circuit resources for a linear system.

    ``matrix`` and ``vector`` are validated and padded exactly as in
    :func:`normalize_linear_system`; a system whose padded dimension is
    one (a single row) leaves no data qubit and is rejected.  The
    returned estimate is backend-independent and reports the qubit
    counts, the number of controlled evolutions, the approximate
    synthesized gate count, the tomography basis-circuit count
    (``3 ** data_qubits``), and the default shot request.
    """
    system = normalize_linear_system(matrix, vector, config)
    data_qubits = system.padded_dimension.bit_length() - 1
    if data_qubits < 1:
        raise AlgorithmInputError(
            "a 1-by-1 system leaves no data qubit; "
            "HHL needs at least a 2-by-2 system"
        )
    phase_qubits = config.phase_qubits
    return HHLResourceEstimate(
        data_qubits=data_qubits,
        phase_qubits=phase_qubits,
        total_qubits=data_qubits + phase_qubits + _ANCILLA_QUBITS,
        controlled_evolutions=phase_qubits,
        synthesized_gates=_synthesized_gates(data_qubits, phase_qubits),
        tomography_circuits=3 ** data_qubits,
        ancilla_qubits=_ANCILLA_QUBITS,
        shots=_DEFAULT_SHOTS,
        approximate_labels=_APPROXIMATE_FIELDS,
    )


def _synthesized_gates(data_qubits: int, phase_qubits: int) -> int:
    """Approximate elementary gate count of the HHL circuit."""
    controlled_evolution_cost = 1 << (2 * data_qubits)
    return (
        2 * phase_qubits * controlled_evolution_cost
        + (1 << data_qubits)
        + 3 * phase_qubits
    )


def _require_positive(value: Any, name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
