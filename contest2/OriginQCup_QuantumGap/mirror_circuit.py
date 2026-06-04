# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mirror-circuit benchmark.

Runs a small random Clifford-like layer forwards then backwards. On a
noise-free backend the inverse cancels the forward and the all-zero
bitstring survives with probability 1. The measured P(|0...0>) is the
"survival probability" of a depth-2d mirror circuit - a holistic, scalable
estimate of mid-depth circuit fidelity.

This mirrors the construction used by Proctor et al. (Sandia) to benchmark
NISQ devices. We keep the per-layer gateset simple (random single-qubit
{H, S, X} and nearest-neighbour CNOTs) so the inverse is straightforward
to construct.

References
----------
- T. Proctor, K. Rudinger, K. Young, E. Nielsen, R. Blume-Kohout,
  "Measuring the capabilities of quantum computers", Nat. Phys. 18,
  75-79 (2022).
"""

import numpy as np

from pyqpanda3.core import QCircuit, QProg, H, S, X, CNOT, measure


def _random_layer(rng: np.random.Generator, n: int) -> list:
    """One layer as a list of (gate_name, qubit(s)) tuples."""
    layer = []
    used = set()
    for q in range(n):
        gate = rng.choice(["H", "S", "X"])
        layer.append((gate, q))
        used.add(q)
    # add a few nearest-neighbour CNOTs (every other pair)
    for q in range(0, n - 1, 2):
        layer.append(("CNOT", q, q + 1))
    return layer


def _apply_layer(circuit: QCircuit, layer: list) -> None:
    for op in layer:
        if op[0] == "H":
            circuit << H(op[1])
        elif op[0] == "S":
            circuit << S(op[1])
        elif op[0] == "X":
            circuit << X(op[1])
        elif op[0] == "CNOT":
            circuit << CNOT(op[1], op[2])


def _apply_inverse_layer(circuit: QCircuit, layer: list) -> None:
    # Apply in reverse order with each gate's inverse.
    # H, X are self-inverse. S^{-1} = S^3.
    for op in reversed(layer):
        if op[0] == "H":
            circuit << H(op[1])
        elif op[0] == "S":
            circuit << S(op[1]) << S(op[1]) << S(op[1])
        elif op[0] == "X":
            circuit << X(op[1])
        elif op[0] == "CNOT":
            circuit << CNOT(op[1], op[2])


def mirror_circuit_probe(
    runner,
    n: int = 3,
    depth: int = 3,
    shots: int = 1024,
    seed: int | None = None,
) -> dict:
    """Run a width-n, depth-2d mirror circuit benchmark on ``runner``.

    Parameters
    ----------
    runner : callable
        ``runner(prog, shots) -> dict[str, int]`` histogram callback.
    n : int
        Number of qubits.
    depth : int
        Number of forward layers (and equally many inverse layers).
    shots : int
        Number of shots. Default 1024.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``{"n": n, "depth": depth, "counts": {...},
        "survival_probability": float, "noiseless_target": 1.0,
        "shots": int, "seed": int | None}``
    """
    rng = np.random.default_rng(seed)

    forward = [_random_layer(rng, n) for _ in range(depth)]

    circuit = QCircuit()
    for layer in forward:
        _apply_layer(circuit, layer)
    for layer in reversed(forward):
        _apply_inverse_layer(circuit, layer)

    prog = QProg()
    prog << circuit
    for q in range(n):
        prog << measure(q, q)

    counts = runner(prog, shots)
    total = sum(counts.values()) or 1
    survival = counts.get("0" * n, 0) / total

    return {
        "n": n,
        "depth": depth,
        "counts": dict(counts),
        "survival_probability": survival,
        "noiseless_target": 1.0,
        "shots": total,
        "seed": seed,
    }
