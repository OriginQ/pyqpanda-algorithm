"""Compare full and local QAOA sampling with an explicit counts runner.

Example: PYTHONPATH=pyqpanda-algorithm python benchmarks/benchmark_qaoa_lightcone.py
    --runner my_device:run_counts --shots 256
No default simulator, exact probability or state inspection is used.
"""
import argparse
import importlib
import json
from time import perf_counter

from pyqpanda3.core import QProg, measure
from pyqpanda3.hamiltonian import PauliOperator
from pyqpanda_alg.QAOA import QAOALightCone
from pyqpanda_alg.QAOA.qaoa import QAOA
from pyqpanda_alg.QAOA.lightcone import _parity_mean


def benchmark(run_counts, shots=256):
    n = 8
    terms = {(): .3, **{tuple(sorted((q, (q + 1) % n))): .5 + .1 * (q % 3)
                       for q in range(n)}}
    gamma, beta = [.31], [.23]
    start = perf_counter()
    full = QAOA(PauliOperator({"I" if not qs else " ".join(f"Z{q}" for q in qs): w
                              for qs, w in terms.items()}))
    program = QProg(n)
    program << full._qaoa_circuit(list(range(n)), gamma, beta)
    program << measure(list(range(n)), list(range(n)))
    counts = run_counts(program, shots)
    supports = sorted(qs for qs, weight in terms.items() if qs and weight != 0)
    means = [_parity_mean(counts, n, support, shots) for support in supports]
    full_time = perf_counter() - start
    start = perf_counter()
    local = QAOALightCone(n, terms, 1)
    estimate = local.estimate(gamma, beta, shots, run_counts)
    local_time = perf_counter() - start
    return {"qubits": n, "depth": 1, "gammas": gamma, "betas": beta,
            "full_term_means": means, "local_term_means": estimate.term_means,
            "full_preparations": shots, "local_preparations": estimate.total_shots,
            "full_seconds": full_time, "local_seconds": local_time,
            "local_widths": [len(p.global_qubits) for p in local.plans],
            "note": "Independent finite samples: not identical results or equal total shot budgets. "
                    "Compare repeated trials; timings include each side's cold construction, execution and aggregation. "
                    "Only ideal target marginals, not the global distribution, are preserved."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="MODULE:FUNCTION counts callback")
    parser.add_argument("--shots", type=int, default=256)
    args = parser.parse_args()
    module, sep, name = args.runner.partition(":")
    if not sep or not module or not name or args.shots <= 0:
        parser.error("supply MODULE:FUNCTION and positive --shots")
    print(json.dumps(benchmark(getattr(importlib.import_module(module), name), args.shots), indent=2))


if __name__ == "__main__":
    main()
