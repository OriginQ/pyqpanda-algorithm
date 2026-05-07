from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from .model import (
    ToyConfig,
    central_quench,
    connected_detector_correlation,
    energy,
    ground_state,
    hamiltonian,
    lowest_sector_energies,
    spectral_weights,
    spectrum_spin_table,
    spin_from_s2,
    total_spin_expectation,
)


def scan_parameter_grid(config: ToyConfig, delta_values: Iterable[float], quench_theta: float = 0.35, levels: int = 8) -> list[dict[str, float | int | list[float]]]:
    rows = []
    states = []
    for delta in delta_values:
        h = hamiltonian(config, float(delta))
        spectrum, psi0 = ground_state(h)
        psiq = central_quench(psi0, config, quench_theta)
        kept = min(levels, len(spectrum))
        spin_table = spectrum_spin_table(h, config, kept)
        sector = lowest_sector_energies(h, config)
        s2_ground = total_spin_expectation(psi0, config)
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
                "detector_color_conn": connected_detector_correlation(psi0, config),
                "quench_delta_e": energy(psiq, h) - float(spectrum[0]),
                "quench_low_level_weight": spectral_weights(psiq, h, kept),
                "fidelity_to_previous": 1.0,
                "fidelity_loss_to_previous": 0.0,
                "detector_conn_abs_slope": 0.0,
            }
        )

    for i in range(1, len(rows)):
        fidelity = float(abs(states[i - 1].conjugate() @ states[i]) ** 2)
        rows[i]["fidelity_to_previous"] = fidelity
        rows[i]["fidelity_loss_to_previous"] = 1.0 - fidelity
    _fill_detector_slopes(rows)
    return rows


def _fill_detector_slopes(rows: list[dict[str, float | int | list[float]]]) -> None:
    if len(rows) < 3:
        return
    for i in range(1, len(rows) - 1):
        ddelta = float(rows[i + 1]["delta"]) - float(rows[i - 1]["delta"])
        if ddelta == 0:
            continue
        rows[i]["detector_conn_abs_slope"] = abs(
            (float(rows[i + 1]["detector_color_conn"]) - float(rows[i - 1]["detector_color_conn"])) / ddelta
        )


def summarize_scan(config: ToyConfig, rows: list[dict[str, float | int | list[float]]]) -> dict[str, object]:
    min_sector_gap_row = min(rows, key=lambda item: float(item["singlet_triplet_gap"]))
    detector_slope_row = max(rows, key=lambda item: float(item["detector_conn_abs_slope"]))
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
        "crossover_by_detector_slope": {
            "delta": detector_slope_row["delta"],
            "detector_color_conn": detector_slope_row["detector_color_conn"],
            "detector_conn_abs_slope": detector_slope_row["detector_conn_abs_slope"],
            "singlet_triplet_gap": detector_slope_row["singlet_triplet_gap"],
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
            "detector_color_conn": max_quench_row["detector_color_conn"],
        },
        "interpretation": "SU(2)-symmetric finite-chain demo: singlet/triplet sector energies, detector color correlation, fidelity loss, and a central color-singlet quench show a small-system precursor of spectral rearrangement without claiming a thermodynamic phase transition",
    }
