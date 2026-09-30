from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from pyqpanda_alg.BFDCQO import BFDCQO, IsingProblem


def make_instance() -> IsingProblem:
    # Fixed 6-spin frustrated Ising instance used by the validation suite.
    h = np.array([0.42, -0.31, 0.18, -0.57, 0.23, -0.14])
    J = np.array([
        [0.00,  0.71, -0.64,  0.00,  0.32, -0.28],
        [0.71,  0.00,  0.51, -0.77,  0.00,  0.24],
        [-0.64, 0.51,  0.00,  0.46, -0.39,  0.00],
        [0.00, -0.77, 0.46,  0.00,  0.62, -0.55],
        [0.32,  0.00, -0.39,  0.62,  0.00,  0.67],
        [-0.28, 0.24,  0.00, -0.55,  0.67,  0.00],
    ])
    return IsingProblem(h=h, J=J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["numpy", "pyqpanda"], default="numpy")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    problem = make_instance()
    solver = BFDCQO(
        problem,
        n_trotter=3,
        dt=0.1,
        iterations=args.iterations,
        shots=args.shots,
        backend=args.backend,
        seed=args.seed,
    )
    result = solver.solve()

    rows = []
    for r in result.history:
        row = {
            "iteration": r.iteration,
            "best_energy": r.best_energy,
            "best_bitstring": r.best_bitstring,
            "ground_state_probability": r.ground_state_probability,
            "expected_energy": r.expected_energy,
            "bias": [float(x) for x in r.bias],
            "alpha1": [float(x) for x in r.alpha1],
        }
        rows.append(row)
        print(
            f"iter={r.iteration:02d} best={r.best_energy: .6f} "
            f"p_gs={r.ground_state_probability:.6f} "
            f"<E>={r.expected_energy: .6f} state={r.best_bitstring}"
        )

    summary = {
        "backend": result.backend,
        "success": bool(result.success),
        "best_solution": result.best_solution,
        "best_energy": result.best_energy,
        "exact_ground_energy": result.exact_ground_energy,
        "exact_ground_states": result.exact_ground_states,
        "total_shots": result.total_shots,
        "history": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
