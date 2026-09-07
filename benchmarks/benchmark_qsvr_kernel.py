"""Compare the original full kernel loop with symmetry reuse on CPUQVM.

Run from the repository root with PYTHONPATH=pyqpanda-algorithm.
This measures local simulator runtime, not quantum hardware speedup.
"""

import argparse
import importlib.metadata
import json
import platform
import statistics
from time import perf_counter

import numpy as np

from pyqpanda_alg.QSVR import Quantum_SVR


def dense_kernel(model, x):
    matrix = np.empty((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            matrix[i, j] = model.dist(x[i], x[j])
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 2 for size in args.sizes):
        parser.error("repeats must be positive and sample sizes must be at least 2")

    results = []
    for size in args.sizes:
        rng = np.random.default_rng(20260907 + size)
        model = Quantum_SVR(rng.normal(size=(size, 2)), rng.normal(size=size))
        x = model.x
        model.dist(x[0], x[1])  # Warm the simulator before either timed path.
        durations = {"dense": [], "symmetric": []}
        max_error = 0.0
        for repeat in range(args.repeats):
            order = ["dense", "symmetric"]
            if repeat % 2:
                order.reverse()
            matrices = {}
            for method in order:
                start = perf_counter()
                matrices[method] = (dense_kernel(model, x) if method == "dense"
                                    else model.k_kernel(x, x.copy()))
                durations[method].append(perf_counter() - start)
            np.testing.assert_allclose(matrices["dense"], matrices["symmetric"],
                                       atol=1e-12, rtol=1e-12)
            max_error = max(max_error, float(np.max(np.abs(
                matrices["dense"] - matrices["symmetric"]))))
        dense = statistics.median(durations["dense"])
        symmetric = statistics.median(durations["symmetric"])
        results.append({
            "samples": size,
            "dense_calls": size * size,
            "symmetric_calls": size * (size + 1) // 2,
            "dense_median_seconds": dense,
            "symmetric_median_seconds": symmetric,
            "speedup": dense / symmetric,
            "max_absolute_error": max_error,
        })
    print(json.dumps({
        "python": platform.python_version(),
        "pyqpanda3": importlib.metadata.version("pyqpanda3"),
        "numpy": np.__version__,
        "repeats": args.repeats,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
