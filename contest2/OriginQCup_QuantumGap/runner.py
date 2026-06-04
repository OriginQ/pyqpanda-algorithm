# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Backend adapters and a small CLI for the NISQ Benchmarking Suite.

Two adapters are provided:

- ``cpuqvm_runner``  : runs on the local pyqpanda3 CPU simulator (CPUQVM).
- ``qcloud_runner``  : runs on an Origin QCloud backend (Wukong WK_C180,
                       full_amplitude simulator, etc).

Both expose the same minimal signature::

    runner(prog: QProg, shots: int) -> dict[str, int]

so the benchmark functions in this package are backend-agnostic.

Run from the command line::

    python -m contest2.OriginQCup_QuantumGap.runner --backend cpu
    python -m contest2.OriginQCup_QuantumGap.runner --backend full_amplitude
    ORIGINQC_API_KEY=... python -m contest2.OriginQCup_QuantumGap.runner \\
        --backend WK_C180

A short JSON report is printed to stdout.
"""

import argparse
import json
import os
import sys


def cpuqvm_runner(prog, shots: int):
    """Run ``prog`` on a fresh local CPUQVM and return histogram counts."""
    from pyqpanda3.core import CPUQVM
    qvm = CPUQVM()
    qvm.run(prog, shots)
    return qvm.result().get_counts()


def _origin_state_vector_to_counts(state_vec_obj, n_qubits: int, shots: int):
    """Convert Origin's amplitude payload into a sampled counts dict.

    Origin's ``full_amplitude`` returns a probability list keyed by
    hex-string outcomes, not a counts histogram. We sample from the
    distribution so all benchmarks can consume one unified shape.
    """
    import numpy as np
    keys = state_vec_obj.get("key", [])
    values = state_vec_obj.get("value", [])
    if not keys or not values:
        return {}
    probs = np.asarray([float(v) for v in values], dtype=float)
    s = probs.sum()
    if s <= 0:
        return {}
    probs = probs / s
    indices = list(range(len(keys)))
    rng = np.random.default_rng()
    draws = rng.choice(indices, size=shots, p=probs)
    counts = {}
    for d in draws:
        key = keys[int(d)]
        idx = int(key, 16) if isinstance(key, str) and key.startswith("0x") else int(key)
        bitstring = format(idx, f"0{n_qubits}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def qcloud_runner(backend_name: str, api_key: str | None = None):
    """Build a ``runner(prog, shots)`` bound to an Origin QCloud backend.

    Parameters
    ----------
    backend_name : str
        e.g. ``"WK_C180"`` (real Wukong QPU), ``"full_amplitude"`` (cloud
        simulator), ``"partial_amplitude"``, ``"single_amplitude"``.
    api_key : str, optional
        Origin API key. Falls back to the ``ORIGINQC_API_KEY`` env var.
    """
    from pyqpanda3.qcloud.qcloud import QCloudService

    key = api_key or os.environ.get("ORIGINQC_API_KEY")
    if not key:
        raise RuntimeError(
            "No Origin API key. Set ORIGINQC_API_KEY or pass api_key=..."
        )
    svc = QCloudService(api_key=key)
    backend = svc.backend(backend_name)

    def runner(prog, shots: int):
        # Count measurements in the program to find qubit width.
        # We trust the caller to have included exactly one measure per qubit.
        n_qubits = _measured_qubit_count(prog)
        qubits = list(range(n_qubits))

        if backend_name in ("full_amplitude", "partial_amplitude", "single_amplitude"):
            job = backend.run(prog, qubits)
            result = job.result()
            data = result.origin_data() or {}
            obj = data.get("obj") or {}
            task_result = obj.get("taskResult") or []
            if task_result:
                state_obj = json.loads(task_result[0])
                return _origin_state_vector_to_counts(state_obj, n_qubits, shots)
            return {}
        else:
            job = backend.run(prog, shots)
            result = job.result()
            return result.get_counts()

    return runner


def _measured_qubit_count(prog) -> int:
    """Best-effort measured-qubit count by inspecting the program text."""
    text = str(prog)
    if "M" not in text:
        return 0
    # Count distinct row indices containing an 'M' marker.
    # Falls back to counting 'M' characters if structure isn't recognised.
    count = 0
    for line in text.splitlines():
        if "M" in line and ("q_" in line or line.strip().startswith("q")):
            count += 1
    return max(count, text.count("M"))


def run_suite(runner, n: int = 3, shots: int = 1024, seed: int = 42) -> dict:
    """Run the full suite against ``runner`` and return a JSON-safe report."""
    from .bell import bell_benchmark
    from .ghz import ghz_benchmark
    from .quantum_volume import quantum_volume_probe
    from .mirror_circuit import mirror_circuit_probe

    return {
        "bell": bell_benchmark(runner, shots=shots),
        "ghz": ghz_benchmark(runner, n=n, shots=shots),
        "quantum_volume": quantum_volume_probe(runner, n=n, shots=shots, seed=seed),
        "mirror_circuit": mirror_circuit_probe(
            runner, n=n, depth=n, shots=shots, seed=seed
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NISQ Benchmarking Suite")
    parser.add_argument(
        "--backend",
        default="cpu",
        help=(
            "Backend: 'cpu' (local CPUQVM), 'full_amplitude' / "
            "'partial_amplitude' / 'single_amplitude' (Origin cloud sims), "
            "or a real chip name like 'WK_C180'."
        ),
    )
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.backend == "cpu":
        runner = cpuqvm_runner
    else:
        runner = qcloud_runner(args.backend)

    report = run_suite(runner, n=args.qubits, shots=args.shots, seed=args.seed)
    # Trim verbose 'counts' for human-readable summary.
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "counts"}
        for k, v in report.items()
    }
    print(json.dumps({"backend": args.backend, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
