from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from su2_toy.model import ToyConfig, central_quench, ground_state, hamiltonian, total_spin_expectation
from su2_toy.qasm_export import singlet_pair_state_prep_qasm
from su2_toy.scan import scan_parameter_grid, summarize_scan


def test_su2_hamiltonian_is_hermitian() -> None:
    config = ToyConfig(num_qubits=6)
    h = hamiltonian(config, dimer_delta=0.2)
    assert np.allclose(h, h.conjugate().T)


def test_ground_state_is_normalized_and_near_singlet() -> None:
    config = ToyConfig(num_qubits=6)
    values, psi0 = ground_state(hamiltonian(config, dimer_delta=0.2))
    assert values[1] >= values[0]
    assert np.isclose(np.linalg.norm(psi0), 1.0)
    assert total_spin_expectation(psi0, config) < 1e-8


def test_color_singlet_quench_preserves_norm() -> None:
    config = ToyConfig(num_qubits=6)
    _, psi0 = ground_state(hamiltonian(config, dimer_delta=0.2))
    psiq = central_quench(psi0, config, theta=0.35)
    assert np.isclose(np.linalg.norm(psiq), 1.0)


def test_scan_summary_has_su2_sector_diagnostics() -> None:
    config = ToyConfig(num_qubits=6)
    rows = scan_parameter_grid(config, [-0.4, 0.0, 0.4], quench_theta=0.2)
    summary = summarize_scan(config, rows)
    assert "minimum_singlet_triplet_gap" in summary
    assert "crossover_by_correlation_slope" in summary
    assert summary["minimum_singlet_triplet_gap"]["singlet_triplet_gap"] >= 0.0


def test_singlet_pair_qasm_shape() -> None:
    qasm = singlet_pair_state_prep_qasm(4)
    assert "OPENQASM 2.0" in qasm
    assert "cx q[0],q[1];" in qasm
    assert "cx q[2],q[3];" in qasm
