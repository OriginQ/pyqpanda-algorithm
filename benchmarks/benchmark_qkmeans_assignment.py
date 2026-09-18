"""Compare uniform measurements with budgeted adaptive centroid assignment.

Run with --runner module:function; the callback executes the supplied measured
QProg and returns integer counts. No backend is chosen implicitly. Each case
has a separate alpha budget; these runs are not an empirical coverage theorem.
"""
import argparse
from dataclasses import asdict
import importlib
import json
from time import perf_counter

from pyqpanda_alg.QKmeans import adaptive_assign, build_assignment_program


CASES = [
    ("separated", [-1, -1], [[-1, 1], [-1, -1], [0, 1]]),
    ("interior", [-.2, .1], [[.8, -.1], [-.2, .1], [.8, .5]]),
    ("close", [-1, 0], [[-1, .1], [-1, .12], [1, 0]]),
    ("tie", [-.2, .1], [[-.2, .1], [-.2, .1]]),
    ("off_center", [-.2, .1], [[.8, -.1], [-.15, .2], [.8, .5]]),
]


def benchmark(run_counts, shots=512, batch=64, alpha=.05):
    rows = []
    for name, point, centers in CASES:
        start = perf_counter()
        frequencies = []
        for center in centers:
            program = build_assignment_program(point, center)
            counts = run_counts(program, shots)
            # Use the same strict contract for the uniform baseline.
            from pyqpanda_alg.QKmeans.adaptive_assignment import _ones_from_counts
            frequencies.append(_ones_from_counts(counts, shots) / shots)
        uniform_time = perf_counter() - start
        start = perf_counter()
        result = adaptive_assign(point, centers, run_counts, batch_shots=batch,
                                 max_shots=shots * len(centers), alpha=alpha)
        adaptive_time = perf_counter() - start
        rows.append({
            "case": name, "point": point, "centroids": centers,
            "uniform_means": frequencies,
            "uniform_argmin": min(range(len(centers)), key=frequencies.__getitem__),
            "uniform_shots": shots * len(centers),
            "uniform_seconds": uniform_time,
            "adaptive": asdict(result), "adaptive_seconds": adaptive_time,
        })
    return {"cases": rows, "alpha_per_assignment": alpha,
            "scope": "Uniform argmin has no separation guarantee. Adaptive may decline a label. "
                     "Both timers include program building, runner calls and aggregation; "
                     "backend startup depends on the supplied runner. No universal speed or shot saving."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="MODULE:FUNCTION returning counts")
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    module, sep, name = args.runner.partition(":")
    if not sep or not module or not name or args.shots <= 0 or args.batch <= 0:
        parser.error("valid MODULE:FUNCTION and positive shots/batch are required")
    if not 0 < args.alpha < 1:
        parser.error("alpha must be between zero and one")
    callback = getattr(importlib.import_module(module), name)
    print(json.dumps(benchmark(callback, args.shots, args.batch, args.alpha), indent=2))


if __name__ == "__main__":
    main()
