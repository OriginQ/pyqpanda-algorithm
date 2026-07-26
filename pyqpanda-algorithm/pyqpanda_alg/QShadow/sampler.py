"""Reference simulation and PyQPanda-compatible measurement execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import pi, sqrt
from typing import Any, Sequence

import numpy as np

from .dataset import ShadowDataset
from .planner import AXES, MeasurementPlan

Counts = Mapping[str, int]
BasisRunner = Callable[[str, int], Counts]
ProgramExecutor = Callable[[Any, int], Counts]

_HADAMARD = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / sqrt(2.0)
_RX_PI_OVER_2 = np.asarray(
    [[1.0, -1.0j], [-1.0j, 1.0]], dtype=complex
) / sqrt(2.0)
_IDENTITY: np.ndarray = np.eye(2, dtype=complex)


def _apply_single_qubit_gate(
    state: np.ndarray, gate: np.ndarray, qubit: int
) -> np.ndarray:
    result = state.copy()
    stride = 1 << qubit
    block = stride << 1
    for start in range(0, state.size, block):
        for offset in range(stride):
            zero = start + offset
            one = zero + stride
            amplitude_zero = state[zero]
            amplitude_one = state[one]
            result[zero] = gate[0, 0] * amplitude_zero + gate[0, 1] * amplitude_one
            result[one] = gate[1, 0] * amplitude_zero + gate[1, 1] * amplitude_one
    return result


def _basis_probabilities(state: np.ndarray, basis: str) -> np.ndarray:
    rotated = state
    for qubit, axis in enumerate(basis):
        gate = _HADAMARD if axis == "X" else _RX_PI_OVER_2 if axis == "Y" else _IDENTITY
        if axis != "Z":
            rotated = _apply_single_qubit_gate(rotated, gate, qubit)
    probabilities = np.abs(rotated) ** 2
    probabilities /= probabilities.sum()
    return probabilities.real


def simulate_pauli_measurements(
    statevector: Sequence[complex],
    plan: MeasurementPlan,
    *,
    seed: int | None = None,
) -> ShadowDataset:
    """Sample ``plan`` from an exact statevector using only NumPy.

    This independent reference path is useful for testing the estimator before
    involving a simulator or QPU backend.  It is exponential in qubit count and
    intentionally intended only for small validation cases.
    """

    state: np.ndarray = np.asarray(statevector, dtype=np.complex128)
    if state.ndim != 1 or state.size != 1 << plan.n_qubits:
        raise ValueError(
            f"statevector must have length {1 << plan.n_qubits} for this plan"
        )
    norm = float(np.vdot(state, state).real)
    if not np.all(np.isfinite(state)) or not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError("statevector must be finite and normalised")

    rng = np.random.default_rng(seed)
    probability_cache: dict[str, np.ndarray] = {}
    bitstrings: list[str] = []
    for basis in plan.bases:
        probabilities = probability_cache.get(basis)
        if probabilities is None:
            probabilities = _basis_probabilities(state, basis)
            probability_cache[basis] = probabilities
        outcome = int(rng.choice(state.size, p=probabilities))
        bitstrings.append(
            "".join(str((outcome >> qubit) & 1) for qubit in range(plan.n_qubits))
        )
    return ShadowDataset.from_strings(plan.bases, bitstrings, plan.probabilities)


def _normalise_counts(
    counts: Counts,
    *,
    n_qubits: int,
    expected_shots: int,
    bit_order: str,
) -> dict[str, int]:
    if not isinstance(counts, Mapping):
        raise TypeError("runner must return a mapping of bitstrings to integer counts")
    if bit_order not in {"q0_first", "msb_first"}:
        raise ValueError("bit_order must be 'q0_first' or 'msb_first'")
    normalised: dict[str, int] = {}
    for raw_bits, raw_count in counts.items():
        bits = str(raw_bits).replace(" ", "")
        if len(bits) != n_qubits or set(bits) - {"0", "1"}:
            raise ValueError(f"invalid {n_qubits}-qubit result key: {raw_bits!r}")
        if not isinstance(raw_count, (int, np.integer)) or isinstance(raw_count, bool):
            raise TypeError("measurement counts must be integers, not probabilities")
        count = int(raw_count)
        if count < 0:
            raise ValueError("measurement counts cannot be negative")
        if bit_order == "msb_first":
            bits = bits[::-1]
        normalised[bits] = normalised.get(bits, 0) + count
    if sum(normalised.values()) != expected_shots:
        raise ValueError(
            f"runner returned {sum(normalised.values())} counts; "
            f"expected {expected_shots}"
        )
    return normalised


def run_measurement_plan(
    plan: MeasurementPlan,
    runner: BasisRunner,
    *,
    bit_order: str = "q0_first",
    max_jobs: int | None = None,
) -> ShadowDataset:
    """Execute each unique basis and convert counts into a QShadow dataset.

    Basis grouping may reorder shots within this call, which is safe because a
    :class:`MeasurementPlan` has one static probability matrix shared by all of
    its shots.  Execute plans with different probability matrices separately
    and combine their datasets with :meth:`ShadowDataset.concatenate`; this
    preserves the appropriate propensity record for each round.

    ``max_jobs`` is a hard safety guard for remote or paid backends.  No job is
    submitted if the compressed plan exceeds the limit.
    """

    if not callable(runner):
        raise TypeError("runner must be callable")
    grouped = plan.grouped_shots
    if max_jobs is not None:
        if not isinstance(max_jobs, int) or isinstance(max_jobs, bool) or max_jobs <= 0:
            raise ValueError("max_jobs must be a positive integer")
        if len(grouped) > max_jobs:
            raise RuntimeError(
                f"plan needs {len(grouped)} backend jobs, exceeding max_jobs={max_jobs}"
            )

    bases: list[str] = []
    bitstrings: list[str] = []
    for basis, shots in grouped.items():
        counts = _normalise_counts(
            runner(basis, shots),
            n_qubits=plan.n_qubits,
            expected_shots=shots,
            bit_order=bit_order,
        )
        for bits, count in counts.items():
            bases.extend([basis] * count)
            bitstrings.extend([bits] * count)
    return ShadowDataset.from_strings(bases, bitstrings, plan.probabilities)


def build_pyqpanda_program(
    preparation: Callable[[Sequence[int]], Any], n_qubits: int, basis: str
) -> Any:
    """Build a measured PyQPanda3 program for one q0-first basis string."""

    if not callable(preparation):
        raise TypeError("preparation must be callable")
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer")
    if len(basis) != n_qubits or set(basis) - set(AXES):
        raise ValueError("basis must be an n_qubits-long X/Y/Z string")

    # Lazy import keeps pure statistical post-processing independently testable.
    from pyqpanda3.core import H, RX, QProg, measure

    program = QProg(n_qubits)
    qubits = program.qubits()
    prepared = preparation(qubits)
    if prepared is not None:
        program << prepared
    for qubit, axis in zip(qubits, basis):
        if axis == "X":
            program << H(qubit)
        elif axis == "Y":
            # RX(+pi/2)^dagger Z RX(+pi/2) = Y for pre-measurement rotation.
            program << RX(qubit, pi / 2.0)
    for qubit in qubits:
        program << measure(qubit, qubit)
    return program


class PyQPandaRunner:
    """Create basis programs and execute them locally or through a callback.

    Parameters
    ----------
    preparation:
        Callable receiving the program's qubit indices and returning a QCircuit,
        QProg, gate, or ``None``.  Returning ``None`` prepares ``|0...0>``.
    n_qubits:
        Program width.
    executor:
        Optional ``executor(program, shots) -> counts`` callback.  If omitted,
        a local :class:`pyqpanda3.core.CPUQVM` is used.  A cloud adapter should
        capture an already-authenticated backend here; QShadow never accepts or
        stores API keys.
    backend_bit_order:
        PyQPanda count dictionaries are normally MSB-first.  Results are
        converted to QShadow's q0-first convention before return.
    """

    def __init__(
        self,
        preparation: Callable[[Sequence[int]], Any],
        n_qubits: int,
        *,
        executor: ProgramExecutor | None = None,
        backend_bit_order: str = "msb_first",
    ) -> None:
        if not callable(preparation):
            raise TypeError("preparation must be callable")
        if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")
        if backend_bit_order not in {"q0_first", "msb_first"}:
            raise ValueError("backend_bit_order must be 'q0_first' or 'msb_first'")
        if executor is not None and not callable(executor):
            raise TypeError("executor must be callable")
        self.preparation = preparation
        self.n_qubits = n_qubits
        self.backend_bit_order = backend_bit_order
        self._machine: Any | None = None
        if executor is None:
            from pyqpanda3.core import CPUQVM

            self._machine = CPUQVM()
            self.executor: ProgramExecutor = self._execute_local
        else:
            self.executor = executor

    def _execute_local(self, program: Any, shots: int) -> Counts:
        assert self._machine is not None
        self._machine.run(program, shots=shots)
        return self._machine.result().get_counts()

    def __call__(self, basis: str, shots: int) -> dict[str, int]:
        if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        program = build_pyqpanda_program(self.preparation, self.n_qubits, basis)
        counts = self.executor(program, shots)
        return _normalise_counts(
            counts,
            n_qubits=self.n_qubits,
            expected_shots=shots,
            bit_order=self.backend_bit_order,
        )
