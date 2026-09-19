"""通过显式测量计数接口，对比完整QSVM线路与分量抽样。

此重复分量样例展示有条件的复用收益，测量次数的节省并不覆盖所有输入。
不隐式选择模拟器，不取精确概率，也不进行半正定修复。
"""
import argparse
import importlib
import json
from time import perf_counter

import numpy as np
from pyqpanda3.core import QProg, measure
from pyqpanda_alg.QSVM import FactorizedQuantumKernel
from pyqpanda_alg.QSVM.quantum_kernel_svm import _build_circuit
from pyqpanda_alg.QSVM.factorized_kernel import _zero_frequency


def benchmark(run_counts, shots=256):
    x = np.array([[.2, .4, .1, .3], [.6, .8, .1, .3],
                  [.2, .4, .7, .3], [.6, .8, .7, .3]])
    full = np.eye(len(x))
    start = perf_counter()
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            program = QProg(4)
            program << _build_circuit(list(range(4)), 4, x[i], x[j])
            program << measure(list(range(4)), list(range(4)))
            counts = run_counts(program, shots)
            full[i, j] = full[j, i] = _zero_frequency(counts, 4, shots)
    full_time = perf_counter() - start
    widths = []

    def counted(program, requested):
        widths.append(program.qubits_num())
        return run_counts(program, requested)

    start = perf_counter()
    result = FactorizedQuantumKernel(counted, shots=shots).evaluate(x)
    factor_time = perf_counter() - start
    return {"data": x.tolist(), "full_kernel": full.tolist(),
            "factorized_kernel": result.tolist(), "shots_per_program": shots,
            "full_calls": len(x) * (len(x) - 1) // 2, "factorized_calls": len(widths),
            "factorized_widths": widths, "full_seconds": full_time,
            "factorized_seconds": factor_time,
            "note": "Ideal fixed map; independent component experiments are required. "
                    "Reusing a sample correlates matrix entries and does not create extra evidence. "
                    "Different datasets can require MORE preparations than the full circuit."}


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
