# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bell-state entanglement benchmark.

Prepares the maximally-entangled Bell state |Phi+> = (|00>+|11>)/sqrt(2)
and reports the combined probability P(|00>) + P(|11>).

A noiseless backend returns 1.0. The gap to 1.0 measures combined
single-qubit prep error, CX error, and readout error on a 2-qubit subgraph
of the device.
"""

from pyqpanda3.core import QCircuit, QProg, H, CNOT, measure


def _build_bell_prog() -> QProg:
    circuit = QCircuit()
    circuit << H(0) << CNOT(0, 1)
    prog = QProg()
    prog << circuit << measure(0, 0) << measure(1, 1)
    return prog


def bell_benchmark(runner, shots: int = 1024) -> dict:
    """Run a Bell-state benchmark on ``runner``.

    Parameters
    ----------
    runner : callable
        A callable with signature ``runner(prog, shots) -> dict[str, int]``
        that executes ``prog`` and returns a histogram of bitstring counts.
        See ``runner.py`` for ready-made adapters for CPUQVM and QCloud.
    shots : int
        Number of shots. Default 1024.

    Returns
    -------
    dict
        ``{"counts": {...}, "p00": float, "p11": float,
        "bell_fidelity_proxy": float, "shots": int}``.
        ``bell_fidelity_proxy = (P(|00>) + P(|11>)) / 1.0`` and is bounded in
        [0, 1]. A perfect Bell state would yield 1.0; uniform random would
        yield 0.5.
    """
    counts = runner(_build_bell_prog(), shots)
    total = sum(counts.values()) or 1
    p00 = counts.get("00", 0) / total
    p11 = counts.get("11", 0) / total
    return {
        "counts": dict(counts),
        "p00": p00,
        "p11": p11,
        "bell_fidelity_proxy": p00 + p11,
        "shots": total,
    }
