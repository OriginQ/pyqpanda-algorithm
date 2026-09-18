"""Run the fixed ideal-map estimator with an explicitly supplied counts adapter.

Usage: python docs/examples/qsvm_factorized_counts.py \
    --runner my_backend:run_counts --shots 1024

No simulator or analytic fallback is selected here. Local simulator results
must not be labelled hardware data or a full noisy-device kernel.
"""

import argparse
from importlib import import_module
import json
from time import perf_counter

import numpy as np

from pyqpanda_alg.QSVM import FactorizedQuantumKernel


def run_example(run_counts, shots=1024):
    """Return measured matrices and resource records for two fresh evaluations."""
    calls = []

    def tracked_counts(program, requested_shots):
        counts = run_counts(program, requested_shots)
        calls.append({
            "width": program.qubits_num(),
            "unitary_gates": sum(program.count_ops().values()),
            "measurement_bits": len(program.get_measure_nodes()),
            "requested_shots": requested_shots,
            "observed_shots": sum(counts.values()),
        })
        return counts

    # Repeated component values can be reused at the SAME feature position.
    train = np.array([
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.2, 0.3, 0.8, 0.5, 0.9, 0.7],
        [0.9, 0.1, 0.4, 0.2, 0.6, 0.3],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # exact duplicate
    ])
    test = np.array([
        [0.2, 0.3, 0.4, 0.2, 0.9, 0.3],
        [0.9, 0.1, 0.8, 0.5, 0.6, 0.7],
    ])
    kernel = FactorizedQuantumKernel(tracked_counts, shots=shots, batch_size=2)
    result = {"feature_dimension": train.shape[1], "shots_per_component": shots}
    for name, x, y in [("train", train, None), ("test_train", test, train)]:
        calls.clear()
        start = perf_counter()
        matrix = kernel.evaluate(x, y)
        elapsed = perf_counter() - start
        result[name] = {
            "matrix": matrix.tolist(),
            "calls": list(calls),
            "programs_submitted": len(calls),
            "max_width": max((c["width"] for c in calls), default=0),
            "total_shots": sum(c["requested_shots"] for c in calls),
            "shot_weighted_unitary_gates": sum(
                c["unitary_gates"] * c["requested_shots"] for c in calls
            ),
            "elapsed_seconds_including_construction": elapsed,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="External module:function counts adapter")
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()
    module_name, separator, function_name = args.runner.partition(":")
    if not separator or not module_name or not function_name:
        parser.error("--runner must be module:function")
    if args.shots < 1:
        parser.error("--shots must be positive")
    runner = getattr(import_module(module_name), function_name)
    record = run_example(runner, args.shots)
    record["adapter"] = args.runner
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
