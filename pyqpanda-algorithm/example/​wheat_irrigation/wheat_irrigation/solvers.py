"""Small-instance QAOA through pyqpanda_alg, with a NumPy reference backend.

Array index ``k`` denotes the basis state with ``x[i] = (k >> i) & 1``.
Both backends implement ``exp(-i gamma H)`` followed by ``RX(2 beta)``.
The NumPy backend simulates a quantum state; it is not quantum hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np

from .model import IrrigationQUBO


MAX_STATEVECTOR_QUBITS = 18


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _check_size(model: IrrigationQUBO, max_qubits: int) -> int:
    limit = _positive_integer(max_qubits, "max_qubits")
    if limit > MAX_STATEVECTOR_QUBITS:
        raise ValueError(
            f"max_qubits cannot exceed the safety limit {MAX_STATEVECTOR_QUBITS}; "
            "state-vector memory grows exponentially"
        )
    n = _positive_integer(model.n_variables, "model.n_variables")
    if n > limit:
        raise ValueError(
            f"This instance needs {n} qubits, exceeding max_qubits={limit}; "
            "reduce the number of fields/time slots or use a smaller instance"
        )
    return n


def enumerate_spectrum(
    model: IrrigationQUBO, max_qubits: int = MAX_STATEVECTOR_QUBITS
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all binary assignments and their original QUBO energies.

    Rows follow little-endian state-vector order, including infeasible
    schedules and all slack-register encodings. This is an exact classical
    reference with exponential cost, not part of a quantum speedup claim.
    """
    n = _check_size(model, max_qubits)
    indices = np.arange(1 << n, dtype=np.uint32)
    bits = np.empty((len(indices), n), dtype=np.uint8)
    # Fill columns to avoid allocating a large uint32 broadcasting temporary.
    for i in range(n):
        bits[:, i] = (indices >> i) & 1
    energies = np.asarray(model.energy(bits), dtype=float)
    if energies.shape != (len(indices),) or not np.all(np.isfinite(energies)):
        raise ValueError("model.energy must return one finite energy per assignment")
    return bits, energies


def _angles(gammas: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if gammas.ndim != 1 or betas.ndim != 1 or len(gammas) != len(betas):
        raise ValueError("gammas and betas must be one-dimensional arrays of equal length")
    if not len(gammas):
        raise ValueError("QAOA needs at least one layer")
    if not np.all(np.isfinite(gammas)) or not np.all(np.isfinite(betas)):
        raise ValueError("All QAOA angles must be finite")
    return gammas, betas


def _numpy_probabilities(
    phase_energies: np.ndarray, n: int, gammas: np.ndarray, betas: np.ndarray
) -> np.ndarray:
    state = np.full(len(phase_energies), 1 / np.sqrt(len(phase_energies)), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * phase_energies)
        c, s = np.cos(beta), -1j * np.sin(beta)
        for i in range(n):
            half = 1 << i
            blocks = state.reshape(-1, 2 * half)
            zero = blocks[:, :half].copy()
            one = blocks[:, half:].copy()
            blocks[:, :half] = c * zero + s * one
            blocks[:, half:] = s * zero + c * one
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def _pyqpanda_alg_runner(
    n: int, h: np.ndarray, j: np.ndarray, scale: float
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    try:
        from pyqpanda_alg.QAOA.qaoa import QAOA
        from pyqpanda3.hamiltonian import PauliOperator
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The pyqpanda_alg backend requires pyqpanda_alg and pyqpanda3. "
            "From this example directory, install the repository algorithm package "
            "with pip install -e ../.. and the quantum dependencies with "
            "pip install '.[pyqpanda]' in a supported Python/platform environment. "
            "For a separately copied project, install its '.[pyqpanda]' extra. "
            "You can explicitly select backend='numpy' or CLI --backend classical. "
            f"Original import error: {exc}"
        ) from exc

    # Explicit Pauli indices avoid lexicographic symbol ordering (x10 < x2).
    # The constant contributes only a global phase; reporting restores it.
    terms = {f"Z{i}": float(h[i] / scale) for i in range(n) if h[i] != 0}
    terms.update({
        f"Z{i} Z{k}": float(j[i, k] / scale)
        for i in range(n) for k in range(i + 1, n) if j[i, k] != 0
    })
    algorithm = QAOA(PauliOperator(terms or {"I": 0.0}), norm=False)
    # QAOA infers dimension from *active* indices. Keep the full register,
    # including middle/high inactive variables and a constant Hamiltonian.
    algorithm.problem_dimension = n

    def run(gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        # The library builds the circuit and executes CPUQVM on every call.
        # Its mixer is RX(-2 beta); negate beta to retain our RX(+2 beta).
        # shots=-1 requests exact probabilities. Seeded final sampling is
        # performed by solve_qaoa, just as for the NumPy reference backend.
        distribution = algorithm.run_qaoa_circuit(gammas, -betas, shots=-1)
        probabilities = np.zeros(1 << n, dtype=float)
        for bitstring, probability in distribution.items():
            if (not isinstance(bitstring, str) or len(bitstring) != n
                    or any(bit not in "01" for bit in bitstring)):
                raise RuntimeError(f"pyqpanda_alg returned invalid {n}-qubit key: {bitstring!r}")
            # Library keys are |q[n-1] ... q[0]>; q0 is the least significant bit.
            probabilities[int(bitstring, 2)] = float(probability)
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise RuntimeError("pyqpanda_alg returned invalid state probabilities")
        total = float(probabilities.sum())
        if not np.isclose(total, 1.0, atol=1e-8, rtol=1e-8):
            raise RuntimeError(f"pyqpanda_alg probability sum is {total}, expected 1")
        return probabilities / total

    return run


@dataclass
class _PreparedProblem:
    energies: np.ndarray
    phase_energies: np.ndarray
    scale: float
    offset: float
    run: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _prepare(model: IrrigationQUBO, backend: str, max_qubits: int) -> _PreparedProblem:
    if backend not in {"numpy", "pyqpanda_alg", "pyqpanda"}:
        raise ValueError("backend must be 'numpy', 'pyqpanda_alg' or 'pyqpanda' (alias)")
    n = _check_size(model, max_qubits)
    h, j, offset = model.to_ising()
    h, j, offset = np.asarray(h, dtype=float), np.asarray(j, dtype=float), float(offset)
    if h.shape != (n,) or j.shape != (n, n):
        raise ValueError("to_ising must return h[n], upper-triangular J[n,n], and offset")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(j)) or not np.isfinite(offset):
        raise ValueError("Ising coefficients must be finite")
    if np.any(np.tril(j) != 0):
        raise ValueError("Ising J must be strictly upper triangular")
    scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(j))))
    if scale == 0:
        scale = 1.0
    # Enumerate once per optimization run and release the binary matrix.
    bits, energies = enumerate_spectrum(model, max_qubits)
    del bits
    phase_energies = (energies - offset) / scale
    if not np.all(np.isfinite(phase_energies)):
        raise ValueError("Normalized phase energies must be finite")
    if backend == "numpy":
        def run(gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
            return _numpy_probabilities(phase_energies, n, gammas, betas)
    else:
        run = _pyqpanda_alg_runner(n, h, j, scale)
    return _PreparedProblem(energies, phase_energies, scale, offset, run)


def qaoa_probabilities(
    model: IrrigationQUBO,
    gammas: np.ndarray,
    betas: np.ndarray,
    backend: str = "pyqpanda_alg",
    max_qubits: int = MAX_STATEVECTOR_QUBITS,
) -> np.ndarray:
    """Return exact basis probabilities for the normalized Ising Hamiltonian.

    Normalize by the maximum absolute nonconstant Ising coefficient, or 1
    for a constant problem. Angles use this normalized Hamiltonian in both
    backends. Returned probabilities are indexed by little-endian integers.
    """
    gammas, betas = _angles(gammas, betas)
    prepared = _prepare(model, backend, max_qubits)
    return prepared.run(gammas, betas)


def solve_qaoa(
    model: IrrigationQUBO,
    depth: int = 2,
    maxiter: int = 100,
    restarts: int = 2,
    shots: int = 4096,
    seed: int = 7,
    backend: str = "pyqpanda_alg",
    max_qubits: int = MAX_STATEVECTOR_QUBITS,
) -> dict:
    """Optimize exact expectations with COBYLA, then draw finite-shot counts.

    Optimization uses an ideal state-vector expectation in both backends;
    shots apply only to the final reproducible NumPy multinomial sampling.
    Feasible candidate selection must use ``counts``, not the full spectrum.
    ``normalization`` is the divisor applied to nonconstant Ising terms;
    ``expectation`` always includes the original QUBO constant offset.
    """
    from scipy.optimize import minimize

    depth = _positive_integer(depth, "depth")
    maxiter = _positive_integer(maxiter, "maxiter")
    restarts = _positive_integer(restarts, "restarts")
    shots = _positive_integer(shots, "shots")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    prepared = _prepare(model, backend, max_qubits)
    best_loss = float("inf")
    best_parameters: np.ndarray | None = None
    evaluations = 0
    optimizer_runs = []

    def objective(parameters: np.ndarray) -> float:
        nonlocal best_loss, best_parameters, evaluations
        probabilities = prepared.run(parameters[:depth], parameters[depth:])
        loss = float(probabilities @ prepared.phase_energies)
        evaluations += 1
        # Keep the best actually evaluated point, even if COBYLA reaches its
        # evaluation budget and its final point is slightly worse.
        if loss < best_loss:
            best_loss = loss
            best_parameters = parameters.copy()
        return loss

    for run_index in range(restarts):
        if run_index == 0:
            initial = np.concatenate((np.linspace(0.1, 0.6, depth), np.linspace(0.6, 0.1, depth)))
        else:
            initial = np.concatenate((rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi, depth)))
        result = minimize(
            objective,
            initial,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.3, "tol": 1e-5},
        )
        optimizer_runs.append(
            {"success": bool(result.success), "message": str(result.message), "nfev": int(result.nfev)}
        )

    if best_parameters is None:
        raise RuntimeError("QAOA optimization did not produce a finite candidate")
    gammas, betas = best_parameters[:depth], best_parameters[depth:]
    probabilities = prepared.run(gammas, betas)
    sampled = rng.multinomial(shots, probabilities)
    observed = np.flatnonzero(sampled)
    counts = {int(index): int(sampled[index]) for index in observed}
    return {
        "counts": counts,
        "probabilities": probabilities,
        "parameters": {"gammas": gammas.tolist(), "betas": betas.tolist()},
        "expectation": float(probabilities @ prepared.energies),
        "evaluations": evaluations,
        "backend": backend,
        "engine": ("numpy.statevector" if backend == "numpy"
                   else "pyqpanda_alg.QAOA.qaoa.QAOA.run_qaoa_circuit"),
        "circuit_evaluations": evaluations + 1,
        "normalization": prepared.scale,
        "ising_offset": prepared.offset,
        "shots": shots,
        "seed": int(seed),
        "optimizer_runs": optimizer_runs,
    }
