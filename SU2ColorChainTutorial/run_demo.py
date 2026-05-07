from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from su2_toy.io_utils import ascii_gap_chart, ascii_metric_chart, write_csv, write_json, write_text
from su2_toy.model import ToyConfig
from su2_toy.qasm_export import singlet_pair_state_prep_qasm, su2_color_quench_probe_qasm
from su2_toy.scan import scan_parameter_grid, summarize_scan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--delta-min", type=float, default=-0.75)
    parser.add_argument("--delta-max", type=float, default=0.75)
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--theta", type=float, default=0.35)
    parser.add_argument("--electric-kappa", type=float, default=0.12)
    parser.add_argument("--axial-decay", type=float, default=2.5)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    if args.num_qubits < 2 or args.num_qubits > 10 or args.num_qubits % 2 != 0:
        raise SystemExit("建议 num-qubits 取 2、4、6、8、10；更大系统请先改用稀疏矩阵。")
    if args.points < 3:
        raise SystemExit("points 至少为 3。")
    if abs(args.delta_min) >= 1.0 or abs(args.delta_max) >= 1.0:
        raise SystemExit("为保持反铁磁 SU(2) bond，建议 delta 范围在 (-1, 1) 内。")

    out_dir = Path(args.out)
    config = ToyConfig(num_qubits=args.num_qubits, electric_kappa=args.electric_kappa, axial_decay=args.axial_decay, periodic=args.periodic)
    delta_values = np.linspace(args.delta_min, args.delta_max, args.points)
    rows = scan_parameter_grid(config, delta_values, quench_theta=args.theta)
    summary = summarize_scan(config, rows)

    write_csv(out_dir / "su2_spectrum_scan.csv", rows)
    write_json(out_dir / "summary.json", {"summary": summary, "rows": rows})
    write_text(out_dir / "singlet_triplet_gap_chart.txt", ascii_gap_chart(rows))
    write_text(out_dir / "correlation_slope_chart.txt", ascii_metric_chart(rows, "probe_corr_abs_slope", "SU(2) probe-site color-correlation slope"))
    write_text(out_dir / "fidelity_loss_chart.txt", ascii_metric_chart(rows, "fidelity_loss_to_previous", "SU(2) ground-state fidelity loss between adjacent delta values"))
    write_text(out_dir / "state_prep_singlet_pairs.qasm", singlet_pair_state_prep_qasm(args.num_qubits))
    write_text(out_dir / "color_quench_probe.qasm", su2_color_quench_probe_qasm(args.num_qubits))

    print("SU(2) color-chain quantum simulation 完成")
    print(f"输出目录：{out_dir}")
    print(f"singlet-triplet gap 最小窗口：delta = {float(summary['minimum_singlet_triplet_gap']['delta']):.6f}")
    print(f"correlation slope 诊断窗口：delta = {float(summary['crossover_by_correlation_slope']['delta']):.6f}")
    print(f"fidelity dip 诊断窗口：delta = {float(summary['crossover_by_fidelity_dip']['delta']):.6f}")
    print(f"probe-site color correlation = {float(summary['crossover_by_correlation_slope']['probe_color_conn']):.6f}")


if __name__ == "__main__":
    main()
