"""Transparent CSV booking windows to explicit scheduling JSON candidates."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .problem import Problem

COLUMNS = (
    "id",
    "group",
    "duration",
    "original_resource",
    "original_start",
    "allowed_resources",
    "earliest",
    "latest",
    "step",
    "change_cost",
    "shift_cost",
)


def import_bookings(path: str | Path, calendar: dict[str, Any]) -> dict[str, Any]:
    """Expand every allowed resource/start in inclusive CSV booking windows.

    calendar uses the regular problem schema without tasks. Resources are pipe
    separated; cost is shift_cost*abs(start-original_start). No candidate is
    silently dropped: horizon/downtime pruning belongs to the QUBO audit.
    At most 256 raw candidates are expanded; costs and references use the same
    strict Problem validation as manually authored JSON.
    """
    if not isinstance(calendar, dict) or set(calendar) - {
        "name",
        "horizon",
        "resources",
        "precedence",
    }:
        raise ValueError("calendar must contain problem fields without tasks")
    tasks = []
    total = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError(f"CSV headers must be: {','.join(COLUMNS)}")
        for row in reader:
            if set(row) != set(COLUMNS) or any(v is None for v in row.values()):
                raise ValueError("CSV row has missing or extra cells")
            numbers = {}
            for key in (
                "duration",
                "original_start",
                "earliest",
                "latest",
                "step",
                "change_cost",
                "shift_cost",
            ):
                value = row[key]
                if not value.isascii() or not value.isdecimal():
                    raise ValueError(f"CSV {key} must be a nonnegative integer")
                numbers[key] = int(value)
                if numbers[key] > 1_000_000:
                    raise ValueError(f"CSV {key} exceeds 1000000")
            if numbers["step"] < 1 or numbers["earliest"] > numbers["latest"]:
                raise ValueError("CSV requires step >= 1 and earliest <= latest")
            resources = row["allowed_resources"].split("|")
            starts = range(numbers["earliest"], numbers["latest"] + 1, numbers["step"])
            total += len(resources) * len(starts)
            if total > 256:
                raise ValueError("CSV expansion supports at most 256 raw candidates")
            tasks.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "duration": numbers["duration"],
                    "original": {
                        "resource": row["original_resource"],
                        "start": numbers["original_start"],
                    },
                    "change_cost": numbers["change_cost"],
                    "options": [
                        {
                            "resource": resource,
                            "start": start,
                            "cost": numbers["shift_cost"]
                            * abs(start - numbers["original_start"]),
                        }
                        for resource in resources
                        for start in starts
                    ],
                }
            )
    data = {**calendar, "tasks": tasks}
    Problem.from_dict(data)
    return data
