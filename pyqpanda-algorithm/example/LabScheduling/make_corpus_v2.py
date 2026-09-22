"""Freeze heterogeneous synthetic bookings before evaluating any solver."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def generate(tasks: int, scenario: int, seed: int) -> dict[str, Any]:
    """Create valid original bookings, then introduce an unfiltered outage.

    Original reservations are sequential even across resources. Alternatives
    vary in start, resource and cost; no feasibility or optimizer filtering is
    used. Scenarios select one short outage, two extended outages, or a chain.
    """
    rng = np.random.default_rng(seed)
    resources = [
        {"id": f"r{i}", "cleanup": int(rng.integers(0, 3)), "downtime": []}
        for i in range(3)
    ]
    rows = []
    start = 0
    for ti in range(tasks):
        resource = resources[ti % 3]
        duration = int(rng.integers(1, 4))
        count = (2, 3, 4)[ti % 3]
        original = {"resource": resource["id"], "start": start}
        universe = [
            (r["id"], s)
            for r in resources
            for s in range(max(0, start - 3), start + 7)
            if (r["id"], s) != (resource["id"], start)
        ]
        choices = rng.choice(len(universe), count - 1, replace=False)
        options = [{**original, "cost": 0}] + [
            {
                "resource": universe[i][0],
                "start": universe[i][1],
                "cost": abs(universe[i][1] - start) + int(rng.integers(0, 4)),
            }
            for i in choices
        ]
        rows.append(
            {
                "id": f"t{ti}",
                "group": f"g{ti % 2}",
                "duration": duration,
                "original": original,
                "change_cost": int(rng.integers(1, 5)),
                "options": options,
            }
        )
        start += duration + resource["cleanup"] + int(rng.integers(0, 2))
    for ti in (0, 1) if scenario == 1 else (0,):
        task = rows[ti]
        left = task["original"]["start"]
        length = task["duration"] + resources[ti % 3]["cleanup"]
        resources[ti % 3]["downtime"] = [
            [left, left + (length + 2 if scenario == 1 else 1)]
        ]
    return {
        "name": f"heterogeneous-n{tasks}-s{scenario}-seed{seed}",
        "horizon": start + 10,
        "resources": resources,
        "tasks": rows,
        "precedence": [[f"t{i}", f"t{i + 1}"] for i in range(tasks - 1)]
        if scenario == 2
        else [],
    }


def write_corpus(output: Path) -> None:
    """Write all 36 registered instances, hashes and unchanged training budgets."""
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for tasks in (3, 4, 5):
        for scenario in range(3):
            for replicate in range(4):
                seed = 2026092300 + tasks * 100 + scenario * 10 + replicate
                raw = generate(tasks, scenario, seed)
                name = f"n{tasks}-s{scenario}-r{replicate}.json"
                content = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
                (output / name).write_text(content, encoding="utf-8")
                records.append(
                    {
                        "file": name,
                        "tasks": tasks,
                        "scenario": scenario,
                        "replicate": replicate,
                        "data_seed": seed,
                        "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    }
                )
    manifest = {
        "version": 2,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "instances": records,
        "optimization_seeds": list(range(10)),
        "training": {"layers": 1, "shots": 512, "maxiter": 60, "restarts": 2},
        "shots": [8, 16, 32, 64, 128, 512],
        "selection": "all instances retained; no solver-based selection or tuning",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Generate the separately versioned corpus without changing v1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "data/corpus-v2"
    )
    write_corpus(parser.parse_args().output)


if __name__ == "__main__":
    main()
