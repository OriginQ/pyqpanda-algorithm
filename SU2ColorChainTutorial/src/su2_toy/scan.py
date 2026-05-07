from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np

from .model import (
    ToyConfig,
    central_quench,
    connected_probe_correlation,
    energy,
    hamiltonian,
    spin_from_s2,
    total_spin_squared_operator,
)
from .operators import expectation


def _eigensystem(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(h)
    order = np.argsort(values.real)
    return values[order].real, vectors[:, order]


def _spin_table(vectors: np.ndarray, values: np.ndarray, s2_op: np.ndarray, levels: int) -> list[dict[str, float]]:
    table = []
    for level in range(min(levels, len(values))):
        s2_value = float(expectation(vectors[:, level], s2_op).real)
        table.append({"level": level, "energy": float(values[level]), "s2": s2_value, "spin": spin_from_s2(s2_value)})
    return table


def _lowest_sector_energies(vectors: np.ndarray, values: np.ndarray, s2_op: np.ndarray) -> dict[str, float]:
    lowest_singlet = float("nan")
    lowest_triplet = float("nan")
    for level, value in enumerate(values):
        spin = spin_from_s2(float(expectation(vectors[:, level], s2_op).real))
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


def _spectral_weights(state: np.ndarray, vectors: np.ndarray, levels: int) -> list[float]:
    kept = min(levels, vectors.shape[1])
    return [float(abs(np.vdot(vectors[:, level], state)) ** 2) for level in range(kept)]


def scan_parameter_grid(config: ToyConfig, delta_values: Iterable[float], quench_theta: float = 0.35, levels: int = 8) -> list[dict[str, float | int | list[float]]]:
    rows = []
    states = []
    s2_op = total_spin_squared_operator(config)
    for delta in delta_values:
        h = hamiltonian(config, float(delta))
        spectrum, vectors = _eigensystem(h)
        psi0 = vectors[:, 0]
        psiq = central_quench(psi0, config, quench_theta)
        kept = min(levels, len(spectrum))
        spin_table = _spin_table(vectors, spectrum, s2_op, kept)
        sector = _lowest_sector_energies(vectors, spectrum, s2_op)
        s2_ground = float(expectation(psi0, s2_op).real)
        states.append(psi0)
        rows.append(
            {
                "num_qubits": config.num_qubits,
                "delta": float(delta),
                "e0": float(spectrum[0]),
                "e1": float(spectrum[1]),
                "e2": float(spectrum[2]) if kept > 2 else float("nan"),
                "e3": float(spectrum[3]) if kept > 3 else float("nan"),
                "ground_s2": s2_ground,
                "ground_spin": spin_from_s2(s2_ground),
                "level_spins": [item["spin"] for item in spin_table],
                "level_s2": [item["s2"] for item in spin_table],
                "lowest_singlet_e": sector["lowest_singlet_e"],
                "lowest_triplet_e": sector["lowest_triplet_e"],
                "singlet_triplet_gap": sector["singlet_triplet_gap"],
                "probe_color_conn": connected_probe_correlation(psi0, config),
                "quench_delta_e": energy(psiq, h) - float(spectrum[0]),
                "quench_low_energy_weight": _spectral_weights(psiq, vectors, kept),
                "fidelity_to_previous": 1.0,
                "fidelity_loss_to_previous": 0.0,
                "probe_corr_abs_slope": 0.0,
            }
        )

    for i in range(1, len(rows)):
        fidelity = float(abs(states[i - 1].conjugate() @ states[i]) ** 2)
        rows[i]["fidelity_to_previous"] = fidelity
        rows[i]["fidelity_loss_to_previous"] = 1.0 - fidelity
    _fill_correlation_slopes(rows)
    return rows


def _fill_correlation_slopes(rows: list[dict[str, float | int | list[float]]]) -> None:
    if len(rows) < 3:
        return
    for i in range(1, len(rows) - 1):
        ddelta = float(rows[i + 1]["delta"]) - float(rows[i - 1]["delta"])
        if ddelta == 0:
            continue
        rows[i]["probe_corr_abs_slope"] = abs(
            (float(rows[i + 1]["probe_color_conn"]) - float(rows[i - 1]["probe_color_conn"])) / ddelta
        )


def summarize_scan(config: ToyConfig, rows: list[dict[str, float | int | list[float]]]) -> dict[str, object]:
    min_sector_gap_row = min(rows, key=lambda item: float(item["singlet_triplet_gap"]))
    correlation_slope_row = max(rows, key=lambda item: float(item["probe_corr_abs_slope"]))
    fidelity_dip_row = max(rows[1:] or rows, key=lambda item: float(item["fidelity_loss_to_previous"]))
    max_quench_row = max(rows, key=lambda item: float(item["quench_delta_e"]))
    return {
        "config": asdict(config),
        "minimum_singlet_triplet_gap": {
            "delta": min_sector_gap_row["delta"],
            "singlet_triplet_gap": min_sector_gap_row["singlet_triplet_gap"],
            "lowest_singlet_e": min_sector_gap_row["lowest_singlet_e"],
            "lowest_triplet_e": min_sector_gap_row["lowest_triplet_e"],
        },
        "crossover_by_correlation_slope": {
            "delta": correlation_slope_row["delta"],
            "probe_color_conn": correlation_slope_row["probe_color_conn"],
            "probe_corr_abs_slope": correlation_slope_row["probe_corr_abs_slope"],
            "singlet_triplet_gap": correlation_slope_row["singlet_triplet_gap"],
        },
        "crossover_by_fidelity_dip": {
            "delta": fidelity_dip_row["delta"],
            "fidelity_to_previous": fidelity_dip_row["fidelity_to_previous"],
            "fidelity_loss_to_previous": fidelity_dip_row["fidelity_loss_to_previous"],
            "singlet_triplet_gap": fidelity_dip_row["singlet_triplet_gap"],
        },
        "largest_singlet_quench_response": {
            "delta": max_quench_row["delta"],
            "quench_delta_e": max_quench_row["quench_delta_e"],
            "probe_color_conn": max_quench_row["probe_color_conn"],
        },
        "interpretation": "SU(2)-symmetric finite-chain demo: singlet/triplet sector energies, probe-site color correlation, fidelity loss, and a central color-singlet quench show a small-system precursor of finite-size spectral change without claiming a thermodynamic phase transition",
    }
