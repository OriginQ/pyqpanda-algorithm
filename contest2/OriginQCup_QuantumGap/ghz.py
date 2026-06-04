# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""GHZ-state multi-qubit benchmark.

Builds an n-qubit GHZ state |GHZ_n> = (|0...0> + |1...1>) / sqrt(2)
via Hadamard on q0 followed by a CX ladder, then measures all qubits.

The reported "ghz_fidelity_proxy" = P(|0...0>) + P(|1...1>) bounds the
true GHZ-state fidelity from below (it ignores coherence terms but is the
canonical population check used in literature). A noiseless backend returns
1.0; uniform random would give 2 / 2^n.

References
----------
- D. M. Greenberger, M. A. Horne, A. Zeilinger, "Going beyond Bell's theorem",
  in: Bell's Theorem, Quantum Theory and Conceptions of the Universe, 1989.
- IBM Quantum's GHZ-state population test (`ghz_state` examples).
"""

from pyqpanda3.core import QCircuit, QProg, H, CNOT, measure


def _build_ghz_prog(n: int) -> QProg:
    if n < 2:
        raise ValueError("GHZ requires at least 2 qubits")
    circuit = QCircuit()
    circuit << H(0)
    for q in range(n - 1):
        circuit << CNOT(q, q + 1)
    prog = QProg()
    prog << circuit
    for q in range(n):
        prog << measure(q, q)
    return prog


def ghz_benchmark(runner, n: int = 3, shots: int = 1024) -> dict:
    """Run an n-qubit GHZ population benchmark on ``runner``.

    Parameters
    ----------
    runner : callable
        ``runner(prog, shots) -> dict[str, int]`` histogram callback.
    n : int
        Number of qubits, n >= 2. Default 3.
    shots : int
        Number of shots. Default 1024.

    Returns
    -------
    dict
        ``{"n": int, "counts": {...}, "p_all_zero": float, "p_all_one": float,
        "ghz_fidelity_proxy": float, "shots": int}``.
    """
    if n < 2:
        raise ValueError("GHZ requires at least 2 qubits")
    counts = runner(_build_ghz_prog(n), shots)
    total = sum(counts.values()) or 1
    all_zero = "0" * n
    all_one = "1" * n
    p_all_zero = counts.get(all_zero, 0) / total
    p_all_one = counts.get(all_one, 0) / total
    return {
        "n": n,
        "counts": dict(counts),
        "p_all_zero": p_all_zero,
        "p_all_one": p_all_one,
        "ghz_fidelity_proxy": p_all_zero + p_all_one,
        "shots": total,
    }
