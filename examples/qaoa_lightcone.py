"""Small sampling-only example. No backend is selected implicitly.

Run without --runner to inspect plans only. To sample, supply a configured
module:function implementing (QProg, shots) -> integer counts; see
../docs/qaoa_lightcone.md for the cbit convention and statistical limitations.
"""

import argparse
import importlib

from pyqpanda_alg.QAOA import QAOALightCone


def make_example():
    """A depth-one chain: small cones in an explicit 12-qubit register."""
    terms: dict[tuple[int, ...], float] = {(q, q + 1): 0.5 for q in range(11)}
    terms.update({(): 0.5, (0,): -0.3})
    return QAOALightCone(n_qubits=12, terms=terms, depth=1)


def run_example(run_counts, shots=256):
    """Evaluate two settings with fresh samples but the same structural plans."""
    executor = make_example()
    return tuple(executor.estimate(gammas, betas, shots, run_counts)
                 for gammas, betas in [([0.27], [-0.18]), ([0.31], [0.22])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", metavar="MODULE:FUNCTION",
                        help="explicit counts callback; omitted = plan preview only")
    parser.add_argument("--shots", type=int, default=256,
                        help="positive shots per local program, per evaluation")
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("--shots must be positive")
    executor = make_example()
    jobs = executor.build_programs([0.27], [-0.18])
    print("global width:", executor.n_qubits, "local programs:", len(jobs))
    for job in jobs:
        print("target:", job.plan.target.qubits,
              "cone:", job.plan.global_qubits,
              "width:", job.program.qubits_num(),
              "gates:", len(job.program.gate_operations()),
              "two-qubit gates:", len(job.program.gate_operations(only_q2=True)))
    if args.runner is None:
        print("Preview only: no backend called. Supply --runner MODULE:FUNCTION to sample.")
        return
    module_name, separator, function_name = args.runner.partition(":")
    if not separator or not module_name or not function_name:
        parser.error("--runner must be MODULE:FUNCTION")
    run_counts = getattr(importlib.import_module(module_name), function_name)
    for index, (gammas, betas) in enumerate([([0.27], [-0.18]), ([0.31], [0.22])]):
        result = executor.estimate(gammas, betas, args.shots, run_counts)
        print("evaluation:", index, "mean:", result.mean,
              "shots per program:", result.shots_per_program,
              "total shots:", result.total_shots)


if __name__ == "__main__":
    main()
