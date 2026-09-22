"""Classical-only 20/35/50-task cascade stress cases; no quantum scaling claim."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pyqpanda_alg.LabScheduling import Problem, compile_qubo, reduce_model, solve_milp


def generate_chain(tasks: int) -> dict[str, Any]:
    """Create a connected shared-oven cascade with three choices per booking."""
    return {
        "name": f"synthetic-classical-chain-{tasks}",
        "horizon": 2 * tasks + 4,
        "resources": [{"id": "oven", "cleanup": 1, "downtime": [[0, 2]]}],
        "tasks": [
            {
                "id": f"t{i}",
                "group": f"g{i}",
                "duration": 1,
                "original": {"resource": "oven", "start": 2 * i},
                "change_cost": 2,
                "options": [
                    {"resource": "oven", "start": 2 * (i + j), "cost": j}
                    for j in range(3)
                ],
            }
            for i in range(tasks)
        ],
    }


def main() -> None:
    """Save full synthetic inputs and independent MILP certificates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/classical-scaling")
    )
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    for count in (20, 35, 50):
        data = generate_chain(count)
        content = json.dumps(data, indent=2) + "\n"
        (output / f"chain-{count}.json").write_text(content)
        problem = Problem.from_dict(data)
        certificate = solve_milp(problem)
        reduction = reduce_model(compile_qubo(problem))
        report = {
            "tasks": count,
            "input_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "milp": certificate,
            "reduction": reduction.audit(),
            "quantum_executed": False,
            "scope": "classical-only; connected component exceeds CPU quantum cap",
        }
        (output / f"chain-{count}-result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            count,
            certificate["status"],
            certificate["best"]["objective"] if certificate["best"] else None,
            reduction.audit()["max_component_qubits"],
        )


if __name__ == "__main__":
    main()
