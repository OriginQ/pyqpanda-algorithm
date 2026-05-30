from pathlib import Path
from typing import Dict, Iterable, List

from .config import ExperimentConfig


def write_report(path: Path, config: ExperimentConfig, metrics, resources, robustness, split_summary) -> None:
    lines = [
        "# CCF2026 Professional Track: MNIST Binary Classification",
        "",
        "## Constraint Check",
        "",
        f"- Qubits: `{config.n_qubits}`",
        f"- Ansatz layers: `{config.ansatz_layers}`",
        "- Circuit parameters: `32`",
        "- Classical head parameters: `9`",
        "- Total trainable parameters: `41 <= 100`",
        f"- Evaluation noise probability: `{config.eval_noise_prob}`",
        "",
        "## Data",
        "",
        "A deterministic 8x8 MNIST-like 3-vs-8 binary dataset is generated locally. "
        "The full pipeline can be swapped to official MNIST arrays by replacing "
        "`make_mnist_like_binary_dataset` while keeping the same compressed 8-feature contract.",
        "",
        markdown_table(["split", "samples", "class_0_digit3", "class_1_digit8"], [
            [name, item["samples"], item["class_0"], item["class_1"]]
            for name, item in split_summary.items()
        ]),
        "",
        "## Test Metrics",
        "",
        markdown_table(["model", "accuracy", "f1", "auc", "log_loss"], [
            [row["model"], row["accuracy"], row["f1"], row["auc"], row["log_loss"]]
            for row in metrics if row["split"] == "test"
        ]),
        "",
        "## Noise Robustness",
        "",
        markdown_table(["model", "noise_prob", "accuracy", "f1", "auc"], [
            [row["model"], row["noise_prob"], row["accuracy"], row["f1"], row["auc"]]
            for row in robustness
        ]),
        "",
        "## Quantum Resources",
        "",
        markdown_table(
            ["model", "qubits", "depth", "total_params", "1q", "2q", "shots", "backend"],
            [
                [
                    row["model"],
                    row["qubits"],
                    row["depth"],
                    row["total_params"],
                    row["one_qubit_gates"],
                    row["two_qubit_gates"],
                    row["shots"],
                    row["backend"],
                ]
                for row in resources
            ],
        ),
        "",
        "## Why Noise-Aware Training",
        "",
        "The clean VQC is optimized for ideal statevector features. The noise-aware VQC "
        "injects angle jitter plus readout/gate attenuation during training, so its head "
        "and variational parameters see the same distribution shift that appears during "
        "noisy evaluation. This is intentionally small enough for the 8-qubit/100-parameter "
        "professional-track constraint.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(headers: Iterable[object], rows: Iterable[Iterable[object]]) -> str:
    headers = [str(item) for item in headers]
    rows = [[str(value) for value in row] for row in rows]
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(output)


def write_bar_svg(path: Path, title: str, values: Dict[str, float]) -> None:
    width, height, margin = 760, 340, 58
    max_value = max([0.01] + list(values.values()))
    bar_width = 120
    gap = 48
    bars = []
    for idx, (name, value) in enumerate(values.items()):
        x = margin + idx * (bar_width + gap)
        h = int((height - 2 * margin) * value / max_value)
        y = height - margin - h
        bars.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" fill="#2458d3"/>'
            f'<text x="{x + bar_width / 2}" y="{y - 8}" text-anchor="middle" font-size="12">{value:.3f}</text>'
            f'<text x="{x + bar_width / 2}" y="{height - 28}" text-anchor="middle" font-size="11">{_esc(name)}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20">{_esc(title)}</text>'
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>'
        + "".join(bars)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def _esc(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

