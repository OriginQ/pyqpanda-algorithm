from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit(f"没有在 {path} 中找到 rows 数据。")
    return rows


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def plot_line(rows: list[dict[str, Any]], y_key: str, ylabel: str, title: str, out_path: Path) -> None:
    x = values(rows, "delta")
    y = values(rows, y_key)
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
    ax.plot(x, y, marker="o", linewidth=1.8, markersize=4)
    ax.set_xlabel("dimerization delta")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_summary_panels(rows: list[dict[str, Any]], out_path: Path) -> None:
    x = values(rows, "delta")
    panels = [
        ("singlet_triplet_gap", "singlet-triplet gap"),
        ("detector_color_conn", "detector color correlation"),
        ("fidelity_loss_to_previous", "ground-state fidelity loss"),
        ("quench_delta_e", "quench energy injection"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), dpi=150, sharex=True)
    for ax, (key, title) in zip(axes.flatten(), panels):
        ax.plot(x, values(rows, key), marker="o", linewidth=1.6, markersize=3.5)
        ax.set_title(title)
        ax.set_xlabel("delta")
        ax.grid(True, alpha=0.28)
    fig.suptitle("SU(2) detector-response toy diagnostics", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="originq_eec_toy_demo/outputs_su2_n6/summary.json")
    parser.add_argument("--out", default="originq_eec_toy_demo/outputs_su2_n6/figures")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_path)

    plot_line(rows, "singlet_triplet_gap", "E_triplet - E_singlet", "SU(2) singlet-triplet gap", out_dir / "singlet_triplet_gap.png")
    plot_line(rows, "detector_color_conn", "connected color correlation", "left-right detector color correlation", out_dir / "detector_color_correlation.png")
    plot_line(rows, "fidelity_loss_to_previous", "1 - fidelity", "ground-state fidelity loss", out_dir / "fidelity_loss.png")
    plot_line(rows, "quench_delta_e", "Delta E after quench", "central color-singlet quench response", out_dir / "quench_delta_energy.png")
    plot_summary_panels(rows, out_dir / "summary_panels.png")

    print("SU(2) demo 图表已生成")
    print(f"输出目录：{out_dir}")


if __name__ == "__main__":
    main()
