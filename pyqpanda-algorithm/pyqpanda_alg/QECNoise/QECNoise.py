# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantum error correction coherent-noise scaling analysis.

This module implements the theta^4 loss scaling law for small quantum error
correcting codes under coherent single-qubit rotation noise.  It is a
numerical state-vector simulator that:

1. builds a logical zero state from the stabilizer group,
2. applies coherent rotations U(theta) = cos(theta/2) I + i sin(theta/2) E,
3. applies a lookup-table recovery,
4. estimates the post-recovery loss as a function of theta_max,
5. fits the log-log slope.

The predicted slope is 4 for coherent rotation noise, while an incoherent
random-Pauli control gives slope ~2.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations, product
from typing import Callable, Iterable, Sequence

try:
    from pyqpanda3.core import QCircuit, RX, RY, RZ  # type: ignore
    _HAS_PYQPANDA = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PYQPANDA = False


# --------------------------------------------------------------------------
# Pauli helpers
# --------------------------------------------------------------------------

def _pauli_matrix(n: int, t: Sequence[int]) -> np.ndarray:
    """Return the n-qubit Pauli matrix for t (0=I, 1=X, 2=Z, 3=Y)."""
    eye = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    y = 1j * x @ z
    mats = [eye, x, z, y]
    out = np.array([1.0])
    for ty in t:
        out = np.kron(out, mats[ty])
    return out


def _commutes(p: Sequence[int], q: Sequence[int]) -> bool:
    """Return True if two Pauli strings commute."""
    mp = _pauli_matrix(len(p), p)
    mq = _pauli_matrix(len(q), q)
    return np.allclose(mp @ mq, mq @ mp)


def syndrome_of(error: Sequence[int], gens: Sequence[Sequence[int]]) -> tuple:
    """Syndrome bits: 1 if the error anticommutes with the generator."""
    return tuple(int(not _commutes(error, g)) for g in gens)


# --------------------------------------------------------------------------
# Code definitions
# --------------------------------------------------------------------------

def five_qubit_code():
    """[[5,1,3]] perfect code."""
    n = 5
    gens = [
        [0, 1, 2, 2, 1],
        [1, 0, 1, 2, 2],
        [2, 1, 0, 1, 2],
        [2, 2, 1, 0, 1],
    ]
    lx = [1, 1, 1, 1, 1]
    lz = [2, 2, 2, 2, 2]
    return n, gens, lx, lz


def steane_code():
    """[[7,1,3]] Steane code."""
    n = 7
    h = [
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ]
    gens = []
    for row in h:
        gens.append([1 if b else 0 for b in row])
    for row in h:
        gens.append([2 if b else 0 for b in row])
    lx = [1] * n
    lz = [2] * n
    return n, gens, lx, lz


def shor_code():
    """[[9,1,3]] Shor code."""
    n = 9
    gens = [
        [2, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 2],
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
    ]
    lx = [1, 1, 1, 0, 0, 0, 0, 0, 0]
    lz = [2, 0, 0, 2, 0, 0, 2, 0, 0]
    return n, gens, lx, lz


# --------------------------------------------------------------------------
# Logical state / recovery
# --------------------------------------------------------------------------

def _stabilizer_group_mats(gens: Sequence[Sequence[int]]) -> list[np.ndarray]:
    n = len(gens[0])
    dim = 2 ** n
    mats = [_pauli_matrix(n, g) for g in gens]
    group = [np.eye(dim, dtype=complex)]
    for mat in mats:
        group = group + [g @ mat for g in group]
    return group


def logical_zero(n: int, gens: Sequence[Sequence[int]], seed: int = 0) -> np.ndarray:
    """Construct the logical |0_L> by projecting a random state onto the code space."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    w = sum(g @ v for g in _stabilizer_group_mats(gens))
    w /= np.linalg.norm(w)
    return w


def code_projector(n: int, gens: Sequence[Sequence[int]], lx: Sequence[int], psi0: np.ndarray):
    """Projector onto the k=1 code space {|0_L>, |1_L>}."""
    psi1 = _pauli_matrix(n, lx) @ psi0
    p = np.outer(psi0, psi0.conj()) + np.outer(psi1, psi1.conj())
    return p, psi1


def recovery_table(n: int, gens: Sequence[Sequence[int]]) -> dict:
    """Lookup table: syndrome -> one representative recovery Pauli."""
    table = {}
    zero = (0,) * len(gens)
    for w in (1, 2, 3):
        for idxs in combinations(range(n), w):
            for types in product((1, 2, 3), repeat=w):
                t = [0] * n
                for idx, ty in zip(idxs, types):
                    t[idx] = ty
                s = syndrome_of(t, gens)
                if s != zero and s not in table:
                    table[s] = t
    assert len(table) == 2 ** len(gens) - 1, f"syndrome coverage incomplete: {len(table)}"
    return table


def optimal_recovery_fidelity(
    psi_ideal: np.ndarray,
    psi_noisy: np.ndarray,
    projector: np.ndarray,
    table: dict,
) -> float:
    """Optimal recovery fidelity after syndrome lookup and correction."""
    n = int(np.log2(len(psi_ideal)))
    fid = abs(np.vdot(psi_ideal, psi_noisy)) ** 2
    for rec in table.values():
        amp = np.vdot(psi_ideal, _pauli_matrix(n, rec) @ psi_noisy)
        fid += abs(amp) ** 2
    return float(fid)


def rotation_state(psi: np.ndarray, error: Sequence[int], theta: float) -> np.ndarray:
    """Apply U(theta) = cos(theta/2) I + i sin(theta/2) E."""
    n = int(np.log2(len(psi)))
    mat = _pauli_matrix(n, error)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    out = c * psi + 1j * s * mat @ psi
    return out / np.linalg.norm(out)


# --------------------------------------------------------------------------
# Noise scans
# --------------------------------------------------------------------------

def run_theta4_scan(
    code_name: str = "[[7,1,3]]",
    thetas: Iterable[float] = (0.05, 0.1, 0.2, 0.4),
    trials: int = 10,
    seed: int = 0,
):
    """Coherent rotation scan. Returns (losses, slope)."""
    if code_name == "[[5,1,3]]":
        n, gens, lx, _ = five_qubit_code()
    elif code_name == "[[7,1,3]]":
        n, gens, lx, _ = steane_code()
    else:
        n, gens, lx, _ = shor_code()

    psi0 = logical_zero(n, gens, seed=seed)
    projector, _ = code_projector(n, gens, lx, psi0)
    table = recovery_table(n, gens)
    rng = np.random.default_rng(seed)

    thetas = list(thetas)
    losses = []
    for theta_max in thetas:
        values = []
        for _ in range(trials):
            psi = psi0.copy()
            for i in range(n):
                ty = int(rng.integers(1, 4))
                th = float(rng.uniform(0, theta_max))
                error = [0] * n
                error[i] = ty
                psi = rotation_state(psi, error, th)
            fid = optimal_recovery_fidelity(psi0, psi, projector, table)
            values.append(max(0.0, 1.0 - fid))
        losses.append(float(np.mean(values)))

    slope, _ = fit_loglog_slope(thetas, losses)
    return losses, slope


def run_pauli_control(
    code_name: str = "[[7,1,3]]",
    probs: Iterable[float] = (0.01, 0.02, 0.04, 0.08),
    trials: int = 300,
    seed: int = 5,
):
    """Incoherent random-Pauli control. Returns (losses, slope)."""
    if code_name == "[[5,1,3]]":
        n, gens, lx, _ = five_qubit_code()
    elif code_name == "[[7,1,3]]":
        n, gens, lx, _ = steane_code()
    else:
        n, gens, lx, _ = shor_code()

    psi0 = logical_zero(n, gens, seed=seed)
    projector, _ = code_projector(n, gens, lx, psi0)
    table = recovery_table(n, gens)
    rng = np.random.default_rng(seed)

    probs = list(probs)
    losses = []
    for p in probs:
        values = []
        for _ in range(trials):
            psi = psi0.copy()
            for i in range(n):
                if rng.random() < p:
                    ty = int(rng.integers(1, 4))
                    error = [0] * n
                    error[i] = ty
                    psi = _pauli_matrix(n, error) @ psi
            fid = optimal_recovery_fidelity(psi0, psi, projector, table)
            values.append(max(0.0, 1.0 - fid))
        losses.append(float(np.mean(values)))

    slope, _ = fit_loglog_slope(probs, losses)
    return losses, slope


def fit_loglog_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Fit slope/intercept in log-log space."""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.maximum(np.asarray(y, dtype=float), 1e-16))
    return tuple(np.polyfit(lx, ly, 1))


# --------------------------------------------------------------------------
# Optional pyQPanda circuit preview
# --------------------------------------------------------------------------

def rotation_circuit_originir(n: int, theta: float) -> str:
    """Return an OriginIR preview for a single-qubit coherent rotation.

    This helper uses pyQPanda to show how the same rotation can be expressed
    as a quantum circuit.  The numerical scaling analysis above is performed
    with exact state-vector algebra.
    """
    if not _HAS_PYQPANDA:
        return "pyqpanda3 not installed; circuit preview unavailable."
    cir = QCircuit()
    qubits = list(range(n))
    cir << RX(qubits[0], theta)
    return cir.originir()


if __name__ == "__main__":
    for code in ["[[5,1,3]]", "[[7,1,3]]"]:
        losses, slope = run_theta4_scan(code, trials=10, seed=42)
        print(f"{code}: coherent loss={['%.2e' % x for x in losses]}, slope={slope:.2f}")
    losses, slope = run_pauli_control("[[7,1,3]]", trials=100, seed=7)
    print(f"[[7,1,3]]: pauli control loss={['%.2e' % x for x in losses]}, slope={slope:.2f}")
