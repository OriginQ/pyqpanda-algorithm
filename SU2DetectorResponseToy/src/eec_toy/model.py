from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import X, Y, Z, expectation, normalize, pair_operator, site_operator


@dataclass(frozen=True)
class ToyConfig:
    num_qubits: int = 6
    exchange_j: float = 1.0
    electric_kappa: float = 0.12
    axial_decay: float = 2.5
    periodic: bool = False


def spin_component(num_qubits: int, site: int, pauli: np.ndarray) -> np.ndarray:
    return 0.5 * site_operator(num_qubits, site, pauli)


def color_dot_operator(num_qubits: int, left: int, right: int) -> np.ndarray:
    return 0.25 * (
        pair_operator(num_qubits, left, X, right, X)
        + pair_operator(num_qubits, left, Y, right, Y)
        + pair_operator(num_qubits, left, Z, right, Z)
    )


def total_spin_squared_operator(config: ToyConfig) -> np.ndarray:
    n = config.num_qubits
    s2 = 0.75 * n * np.eye(2**n, dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            s2 += 2.0 * color_dot_operator(n, i, j)
    return s2


def hamiltonian(config: ToyConfig, dimer_delta: float) -> np.ndarray:
    n = config.num_qubits
    dim = 2**n
    h = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        bond = config.exchange_j * (1.0 + dimer_delta * ((-1.0) ** i))
        h += bond * color_dot_operator(n, i, i + 1)
    if config.periodic and n > 2:
        bond = config.exchange_j * (1.0 + dimer_delta * ((-1.0) ** (n - 1)))
        h += bond * color_dot_operator(n, n - 1, 0)
    if config.electric_kappa != 0.0:
        for i in range(n):
            for j in range(i + 1, n):
                distance = j - i
                kernel = np.exp(-distance / max(config.axial_decay, 1e-9))
                h += config.electric_kappa * kernel * color_dot_operator(n, i, j)
    return h


def ground_state(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(h)
    order = np.argsort(values.real)
    return values[order].real, vectors[:, order[0]]


def apply_singlet_quench(state: np.ndarray, config: ToyConfig, theta: float) -> np.ndarray:
    n = config.num_qubits
    left = max(0, n // 2 - 1)
    right = min(n - 1, n // 2)
    generator = color_dot_operator(n, left, right)
    values, vectors = np.linalg.eigh(generator)
    phases = np.exp(-1j * theta * values)
    return normalize(vectors @ (phases * (vectors.conjugate().T @ state)))


def central_quench(state: np.ndarray, config: ToyConfig, theta: float) -> np.ndarray:
    return apply_singlet_quench(state, config, theta)


def detector_sites(config: ToyConfig) -> tuple[int, int]:
    n = config.num_qubits
    if n < 4:
        return 0, n - 1
    return max(0, n // 2 - 2), min(n - 1, n // 2 + 1)


def connected_detector_correlation(state: np.ndarray, config: ToyConfig) -> float:
    n = config.num_qubits
    left, right = detector_sites(config)
    conn = 0.0
    for pauli in (X, Y, Z):
        sl = spin_component(n, left, pauli)
        sr = spin_component(n, right, pauli)
        conn += (expectation(state, sl @ sr) - expectation(state, sl) * expectation(state, sr)).real
    return float(conn)


def total_spin_expectation(state: np.ndarray, config: ToyConfig) -> float:
    return float(expectation(state, total_spin_squared_operator(config)).real)


def spin_from_s2(s2_value: float) -> float:
    return float(max(0.0, (-1.0 + np.sqrt(max(0.0, 1.0 + 4.0 * s2_value))) / 2.0))


def energy(state: np.ndarray, h: np.ndarray) -> float:
    return float(expectation(state, h).real)


def spectral_weights(state: np.ndarray, h: np.ndarray, levels: int) -> list[float]:
    values, vectors = np.linalg.eigh(h)
    order = np.argsort(values.real)
    vectors = vectors[:, order[:levels]]
    return [float(abs(np.vdot(vectors[:, i], state)) ** 2) for i in range(vectors.shape[1])]


def spectrum_spin_table(h: np.ndarray, config: ToyConfig, levels: int) -> list[dict[str, float]]:
    values, vectors = np.linalg.eigh(h)
    order = np.argsort(values.real)
    values = values[order].real
    vectors = vectors[:, order]
    s2_op = total_spin_squared_operator(config)
    table = []
    for i in range(min(levels, len(values))):
        s2_value = float(expectation(vectors[:, i], s2_op).real)
        table.append({"level": i, "energy": float(values[i]), "s2": s2_value, "spin": spin_from_s2(s2_value)})
    return table


def lowest_sector_energies(h: np.ndarray, config: ToyConfig) -> dict[str, float]:
    values, vectors = np.linalg.eigh(h)
    order = np.argsort(values.real)
    values = values[order].real
    vectors = vectors[:, order]
    s2_op = total_spin_squared_operator(config)
    lowest_singlet = float("nan")
    lowest_triplet = float("nan")
    for i, value in enumerate(values):
        spin = spin_from_s2(float(expectation(vectors[:, i], s2_op).real))
        if np.isnan(lowest_singlet) and abs(spin - 0.0) < 0.35:
            lowest_singlet = float(value)
        if np.isnan(lowest_triplet) and abs(spin - 1.0) < 0.35:
            lowest_triplet = float(value)
        if not np.isnan(lowest_singlet) and not np.isnan(lowest_triplet):
            break
    return {
        "lowest_singlet_e": lowest_singlet,
        "lowest_triplet_e": lowest_triplet,
        "singlet_triplet_gap": lowest_triplet - lowest_singlet,
    }
