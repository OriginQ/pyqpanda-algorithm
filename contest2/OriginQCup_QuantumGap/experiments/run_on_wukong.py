# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run the four NISQ benchmarks against real Origin Wukong WK_C180 hardware.

This is the verification script that captures the real-hardware task IDs
cited in the PR README. Each benchmark is run with conservative shots so
the total QPU spend stays within the sign-up free-tier budget.

Output: ``wukong_results.json`` with task IDs, raw counts, and decoded
benchmark numbers for every run. The PR README's "Origin QCloud
end-to-end" section quotes these task IDs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make the suite importable when this script is run directly.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from contest2.OriginQCup_QuantumGap.bell import _build_bell_prog
from contest2.OriginQCup_QuantumGap.ghz import _build_ghz_prog
from contest2.OriginQCup_QuantumGap.mirror_circuit import mirror_circuit_probe
from contest2.OriginQCup_QuantumGap.quantum_volume import (
    _ideal_distribution,
    _heavy_set,
    _random_su4,
)
from pyqpanda3.core import QCircuit, QProg, measure
from pyqpanda3.qcloud.qcloud import QCloudService, QCloudOptions
import numpy as np


BACKEND_NAME = "WK_C180"
SHOTS = 1024


def _submit(svc, prog, label: str, *, shots: int = SHOTS) -> dict:
    """Submit ``prog`` to WK_C180, wait for completion, return a record."""
    backend = svc.backend(BACKEND_NAME)
    opts = QCloudOptions()
    opts.set_mapping(True)
    opts.set_optimization(True)
    opts.set_amend(True)
    opts.set_is_prob_counts(True)
    t0 = time.time()
    job = backend.run(prog, shots, opts)
    job_id = job.job_id()
    print(f"  [{label}] submitted, job_id={job_id}")
    result = job.result()
    elapsed = time.time() - t0
    counts = result.get_counts()
    raw = result.origin_data() or {}
    print(f"  [{label}] finished in {elapsed:.1f}s  counts={dict(counts)}")
    return {
        "label": label,
        "backend": BACKEND_NAME,
        "shots": shots,
        "job_id": job_id,
        "wall_seconds": elapsed,
        "counts": dict(counts),
        "raw_obj": raw.get("obj") if isinstance(raw, dict) else None,
    }


def main() -> int:
    api_key = os.environ.get("ORIGINQC_API_KEY")
    if not api_key:
        print("ERROR: ORIGINQC_API_KEY not set in environment.", file=sys.stderr)
        return 1

    svc = QCloudService(api_key=api_key)
    print(f"Available backends: {svc.backends()}")
    print()

    results: list[dict] = []

    # 1. Bell ------------------------------------------------------------------
    print("== Bell ==")
    results.append(_submit(svc, _build_bell_prog(), "bell"))

    # 2. GHZ (n=3) -------------------------------------------------------------
    print("== GHZ (n=3) ==")
    results.append(_submit(svc, _build_ghz_prog(3), "ghz3"))

    # 3. Mirror probe (n=3, depth=3) -------------------------------------------
    # The mirror_circuit_probe builds its own program; we re-implement here so
    # we capture the WK_C180 job_id alongside the decoded survival probability.
    print("== Mirror (n=3, depth=3) ==")
    from contest2.OriginQCup_QuantumGap.mirror_circuit import (
        _random_layer, _apply_layer, _apply_inverse_layer,
    )
    rng = np.random.default_rng(42)
    forward = [_random_layer(rng, 3) for _ in range(3)]
    circuit = QCircuit()
    for layer in forward:
        _apply_layer(circuit, layer)
    for layer in reversed(forward):
        _apply_inverse_layer(circuit, layer)
    prog = QProg()
    prog << circuit
    for q in range(3):
        prog << measure(q, q)
    results.append(_submit(svc, prog, "mirror_n3_d3"))

    # 4. QV probe (n=3) --------------------------------------------------------
    print("== QV probe (n=3) ==")
    rng = np.random.default_rng(42)
    circuit = QCircuit()
    n = 3
    for _ in range(n):
        perm = rng.permutation(n)
        for j in range(0, n - 1, 2):
            q0, q1 = int(perm[j]), int(perm[j + 1])
            circuit << _random_su4(rng, q0, q1)
    probs = _ideal_distribution(circuit, n)
    heavy = _heavy_set(probs)
    prog = QProg()
    prog << circuit
    for q in range(n):
        prog << measure(q, q)
    qv_record = _submit(svc, prog, "qv_n3")
    qv_record["ideal_heavy_output_frequency"] = float(sum(probs[i] for i in heavy))
    qv_record["heavy_set_indices"] = sorted(heavy)
    results.append(qv_record)

    # Decode each benchmark's headline number from the captured counts.
    decoded = []
    for r in results:
        counts = r["counts"]
        total = sum(counts.values()) or 1
        if r["label"] == "bell":
            d = {
                "metric": "bell_fidelity_proxy",
                "value": (counts.get("00", 0) + counts.get("11", 0)) / total,
                "noiseless_target": 1.0,
            }
        elif r["label"] == "ghz3":
            d = {
                "metric": "ghz_fidelity_proxy",
                "value": (counts.get("000", 0) + counts.get("111", 0)) / total,
                "noiseless_target": 1.0,
            }
        elif r["label"] == "mirror_n3_d3":
            d = {
                "metric": "survival_probability",
                "value": counts.get("000", 0) / total,
                "noiseless_target": 1.0,
            }
        elif r["label"] == "qv_n3":
            heavy_set = set(r["heavy_set_indices"])
            hits = 0
            for bs, c in counts.items():
                try:
                    if int(bs, 2) in heavy_set:
                        hits += c
                except ValueError:
                    continue
            d = {
                "metric": "heavy_output_frequency",
                "value": hits / total,
                "ideal_heavy_output_frequency": r["ideal_heavy_output_frequency"],
                "pass_threshold": 2.0 / 3.0,
            }
        else:
            d = {"metric": "unknown"}
        d["label"] = r["label"]
        d["job_id"] = r["job_id"]
        decoded.append(d)
        print(f"  {r['label']:<14} {d['metric']:<24} = {d['value']:.4f}")

    out_path = _HERE / "wukong_results.json"
    out_path.write_text(json.dumps(
        {"backend": BACKEND_NAME, "shots": SHOTS, "results": results, "decoded": decoded},
        indent=2,
        default=str,
    ))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
