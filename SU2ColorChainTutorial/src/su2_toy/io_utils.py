from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("write_csv 需要至少一行数据")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in out.items():
                if isinstance(value, (list, dict)):
                    out[key] = json.dumps(value)
            writer.writerow(out)


def ascii_metric_chart(rows: list[dict[str, Any]], metric: str, title: str, x_key: str = "delta", width: int = 42) -> str:
    values = [abs(float(row[metric])) for row in rows]
    max_value = max(values) if values else 1.0
    lines = [title, f"{x_key:<9} {metric:<24} chart"]
    for row, value in zip(rows, values):
        raw = float(row[metric])
        bar_len = int(width * value / max_value) if max_value > 0 else 0
        lines.append(f"{float(row[x_key]):9.4f} {raw:24.9f}  " + "█" * bar_len)
    return "\n".join(lines) + "\n"


def ascii_gap_chart(rows: list[dict[str, Any]], width: int = 42) -> str:
    return ascii_metric_chart(rows, "singlet_triplet_gap", "SU(2) singlet-triplet spectral gap", width=width)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
