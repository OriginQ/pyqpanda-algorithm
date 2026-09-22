"""Deterministic synthetic corpus; no filtering by feasibility or solver outcome."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def generate(tasks: int, density: int, seed: int) -> dict[str, Any]:
    """Generate two surviving choices per task plus one outage-pruned booking.

    Density labels control resource count and slot range, not measured density.
    All generated instances are retained, including infeasible instances.
    """
    rng = np.random.default_rng(seed)
    resource_count = (3, 2, 1)[density]
    span = (2 * tasks, tasks + 1, max(3, tasks // 2 + 1))[density]
    resources = [
        {"id": f"r{i}", "cleanup": 0, "downtime": [[0, 1]] if i == 0 else []}
        for i in range(resource_count)
    ]
    rows = []
    for ti in range(tasks):
        universe = [
            (f"r{r}", s) for r in range(resource_count) for s in range(1, span + 1)
        ]
        indices = rng.choice(len(universe), size=2, replace=False)
        options = [
            {
                "resource": universe[i][0],
                "start": universe[i][1],
                "cost": int(rng.integers(0, 5)),
            }
            for i in indices
        ]
        original = {"resource": options[0]["resource"], "start": options[0]["start"]}
        if ti == 0:
            options.insert(0, {"resource": "r0", "start": 0, "cost": 0})
            original = {"resource": "r0", "start": 0}
        rows.append(
            {
                "id": f"t{ti}",
                "group": f"g{ti // 2}",
                "duration": 1,
                "options": options,
                "original": original,
                "change_cost": int(rng.integers(1, 5)),
            }
        )
    return {
        "name": f"synthetic-n{tasks}-d{density}-s{seed}",
        "horizon": span + 1,
        "resources": resources,
        "tasks": rows,
        "precedence": [["t0", "t1"]] if density == 1 else [],
    }


def write_corpus(output: Path) -> None:
    """Freeze 12 evaluation inputs, one development input and content hashes."""
    output.mkdir(parents=True, exist_ok=True)
    records = []
    specs = [("development", 3, 0, 20260921)] + [
        ("evaluation", n, density, 20260922 + 10 * n + density)
        for n in (3, 5, 6, 8)
        for density in range(3)
    ]
    for split, tasks, density, seed in specs:
        raw = generate(tasks, density, seed)
        filename = f"{split}-n{tasks}-d{density}.json"
        content = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
        (output / filename).write_text(content, encoding="utf-8")
        records.append(
            {
                "file": filename,
                "split": split,
                "tasks": tasks,
                "density_parameter": density,
                "data_seed": seed,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "synthetic": True,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "optimization_seeds": list(range(10)),
        "training": {"layers": 1, "maxiter": 60, "restarts": 2},
        "shots": [8, 16, 32, 64, 128, 512],
        "instances": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Write or identically regenerate the fixed corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "data/corpus"
    )
    write_corpus(parser.parse_args().output)


if __name__ == "__main__":
    main()
