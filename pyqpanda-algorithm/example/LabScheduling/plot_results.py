"""Render a shareable SVG from saved measurements, without rerunning a solver."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Plot all seeds' feasibility and the medium XY seed-7 schedule."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    with (args.results / "summary.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    report = json.loads((args.results / "medium-xy-7.json").read_text(encoding="utf-8"))
    source = Path(__file__).parent / "data" / "medium.json"
    if hashlib.sha256(source.read_bytes()).hexdigest() != report["input_sha256"]:
        raise ValueError("medium input changed since these results were recorded")
    problem = json.loads(source.read_text(encoding="utf-8"))
    seeds = ", ".join(dict.fromkeys(row["seed"] for row in rows))
    best = report["quantum"]["best"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for i, dataset in enumerate(("simple", "medium")):
        for j, mixer in enumerate(("xy", "x")):
            values = [
                float(r["feasible_probability"])
                for r in rows
                if r["dataset"] == dataset and r["mixer"] == mixer
            ]
            axes[0].bar(
                i + (j - 0.5) * 0.3,
                np.mean(values),
                width=0.27,
                color=("#167d9a", "#c67944")[j],
                label=("XY", "X")[j] if i == 0 else None,
            )
            axes[0].scatter(
                np.full(len(values), i + (j - 0.5) * 0.3),
                values,
                color="#152536",
                s=18,
                zorder=3,
            )
        uniform = float(
            next(r for r in rows if r["dataset"] == dataset)[
                "uniform_feasible_probability"
            ]
        )
        axes[0].hlines(
            uniform,
            i - 0.4,
            i + 0.4,
            color="#586363",
            linestyles="--",
            label="Uniform one-hot" if i == 0 else None,
        )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Simple (6 qubits)", "Medium (12 qubits)"],
        ylabel="Feasible Born probability",
        ylim=(0, 1),
        title=f"Seeds: {seeds}; p={report['quantum']['layers']}, "
        f"{report['quantum']['shots']} shots",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.15)
    resource_y = {"spectrometer": 1, "microscope": 0}
    for row in report["quantum"]["best"]["schedule"]:
        y = resource_y[row["resource"]]
        color = "#167d9a" if row["group"] == "group-A" else "#c67944"
        axes[1].barh(
            y, row["end"] - row["start"], left=row["start"], height=0.5, color=color
        )
        axes[1].barh(
            y,
            row["reserved_end"] - row["end"],
            left=row["end"],
            height=0.5,
            color=color,
            alpha=0.25,
            hatch="///",
        )
        axes[1].text(
            (row["start"] + row["end"]) / 2,
            y,
            row["task"],
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
    for resource in problem["resources"]:
        for start, end in resource.get("downtime", []):
            axes[1].barh(
                resource_y[resource["id"]],
                end - start,
                left=start,
                height=0.65,
                color="#cccccc",
                hatch="xx",
            )
    axes[1].set(
        yticks=[0, 1],
        yticklabels=["Microscope", "Spectrometer"],
        xticks=range(13),
        xlim=(0, 12),
        xlabel="Time slot (hatched tail = cleanup)",
        title=f"Medium / XY / seed 7: cost {best['objective']}, "
        f"violations {best['violation_count']}",
    )
    axes[1].grid(axis="x", alpha=0.15)
    matplotlib.rcParams["svg.hashsalt"] = "lab-reschedule"
    matplotlib.rcParams["svg.fonttype"] = "none"
    output = args.results / "overview.svg"
    fig.savefig(output, metadata={"Date": None})
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
