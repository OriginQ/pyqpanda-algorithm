"""Measure current checkout's QAE construction and simulation independently."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import sys
import subprocess
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pyqpanda-algorithm'))

from pyqpanda3.core import CPUQVM, QCircuit, RY, RZ, CNOT
from pyqpanda_alg.QAE import QAE


class ConstructionComplete(Exception):
    pass


class TimedMachine:
    def __init__(self, build_only=False):
        self.backend = CPUQVM()
        self.build_only = build_only

    def run(self, program, shots):
        entered = time.perf_counter()
        self.build_seconds = entered - self.started
        if self.build_only:
            raise ConstructionComplete
        self.backend.run(program, shots)
        self.simulation_seconds = time.perf_counter() - entered

    def result(self):
        return self.backend.result()


def trial(ancillas, build_only=False):
    calls = 0

    def prepare(qubits):
        nonlocal calls
        calls += 1
        circuit = QCircuit()
        for layer in range(5):
            for i, q in enumerate(qubits):
                circuit << RY(q, 0.13 * (i + 1) + 0.07 * layer)
                circuit << RZ(q, 0.11 * (layer + 1))
            for i in range(len(qubits) - 1):
                circuit << CNOT(qubits[i], qubits[i + 1])
        return circuit

    estimator = QAE(operator_in=prepare, qnumber=3, epsilon=2.0 ** -ancillas)
    machine = TimedMachine(build_only)
    estimator.machine = machine
    machine.started = time.perf_counter()
    try:
        estimate = estimator.run()
    except ConstructionComplete:
        return dict(build_seconds=machine.build_seconds, preparation_calls=calls)
    elapsed = time.perf_counter() - machine.started
    probabilities = machine.result().get_prob_list(list(range(3, 3 + ancillas)))
    return dict(build_seconds=machine.build_seconds,
                simulation_seconds=machine.simulation_seconds,
                total_seconds=elapsed, preparation_calls=calls,
                estimate=float(estimate), probabilities=probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--ancillas', type=int, nargs='+', default=[4, 6, 7])
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--baseline', action='store_true',
                        help='Load QAE from the recorded upstream commit.')
    args = parser.parse_args()
    if args.baseline:
        global QAE
        repo = ROOT
        source = subprocess.check_output([
            'git', '-c', f'safe.directory={repo.as_posix()}', '-C', str(repo),
            'show', '5f973efccb84bc193157d1ccebe32137e307293b:'
            'pyqpanda-algorithm/pyqpanda_alg/QAE/QAE.py'], text=True, encoding='utf-8')
        module = types.ModuleType('pyqpanda_alg.QAE._benchmark_baseline')
        module.__package__ = 'pyqpanda_alg.QAE'
        exec(compile(source, '<upstream-QAE>', 'exec'), module.__dict__)
        QAE = module.QAE
    trial(3, args.build_only)  # Warm up imports/backend, excluded from timing.
    rows = []
    for ancillas in args.ancillas:
        runs = [trial(ancillas, args.build_only) for _ in range(args.repeats)]
        row = dict(ancillas=ancillas, epsilon=2.0 ** -ancillas,
                   preparation_calls=runs[0]['preparation_calls'])
        keys = ['build_seconds']
        if not args.build_only:
            row.update(estimate=runs[0]['estimate'],
                       probabilities=runs[0]['probabilities'])
            keys.extend(['simulation_seconds', 'total_seconds'])
        for key in keys:
            row[key + '_median'] = statistics.median(r[key] for r in runs)
            row[key + '_samples'] = [r[key] for r in runs]
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items()
                          if k != 'probabilities'}, ensure_ascii=False), flush=True)
    result = dict(label=args.label, python=platform.python_version(),
                  platform=platform.platform(), repeats=args.repeats,
                  build_only=args.build_only,
                  versions={name: importlib.metadata.version(name)
                            for name in ('pyqpanda3', 'numpy', 'scipy')}, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
