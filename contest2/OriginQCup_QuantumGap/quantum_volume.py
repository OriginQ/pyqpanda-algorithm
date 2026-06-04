# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Quantum Volume single-circuit probe.

Implements one trial of the Quantum Volume (QV) protocol of Cross et al.
2019: a width = depth = n square model circuit built from random SU(4)
pairs on a random qubit permutation per layer. After running, the
"heavy-output frequency" is the fraction of shots whose measured bitstring
is in the upper half of the noiseless ideal distribution.

A noiseless ideal QV circuit gives heavy_output_frequency ~= (1 + ln 2) / 2
~= 0.847; the published QV pass threshold is 2/3.

This is a *probe* implementation: it runs ONE trial, not the statistically
significant >= 100 trials needed to claim a QV score. It is intended as a
fast smoke test of whether a backend handles a width-n square circuit at
all - useful for picking between Wukong backends before committing to a
multi-trial campaign.

References
----------
- A. W. Cross, L. S. Bishop, S. Sheldon, P. D. Nation, J. M. Gambetta,
  "Validating quantum computers using randomized model circuits",
  Phys. Rev. A 100, 032328 (2019). arXiv:1811.12926
"""

import numpy as np

from pyqpanda3.core import QCircuit, QProg, U3, CNOT, measure


def _random_su4(rng: np.random.Generator, q0: int, q1: int) -> QCircuit:
    """Build a 2-qubit random SU(4) gate via Cartan-like KAK decomposition.

    We use a pragmatic decomposition that is dense in SU(4):
    U3 x U3 -> CNOT -> U3 x U3 -> CNOT -> U3 x U3 -> CNOT -> U3 x U3
    with uniformly random Euler angles.
    """
    block = QCircuit()
    for _ in range(4):
        a = rng.uniform(0, 2 * np.pi, size=6)
        block << U3(q0, a[0], a[1], a[2]) << U3(q1, a[3], a[4], a[5])
        if _ < 3:
            block << CNOT(q0, q1)
    return block


def _ideal_distribution(prog_circuit: QCircuit, n: int) -> np.ndarray:
    """Compute the noiseless probability vector of ``prog_circuit`` on n qubits.

    Uses pyqpanda3's CPUQVM with full-state probability extraction.
    """
    from pyqpanda3.core import CPUQVM, QProg
    qvm = CPUQVM()
    prog = QProg()
    prog << prog_circuit
    qvm.run(prog, 1)
    state = qvm.result().get_state_vector()
    state = np.asarray(state, dtype=np.complex128)
    probs = (state.conj() * state).real
    if probs.size != 2 ** n:
        probs = np.resize(probs, 2 ** n)
    s = probs.sum()
    return probs / s if s > 0 else probs


def _heavy_set(probs: np.ndarray) -> set:
    """Return the set of bitstring indices in the upper half of the distribution."""
    median = float(np.median(probs))
    return {i for i, p in enumerate(probs) if p > median}


def quantum_volume_probe(
    runner,
    n: int = 3,
    shots: int = 1024,
    seed: int | None = None,
) -> dict:
    """Run a single-trial Quantum Volume probe of width = depth = n.

    Parameters
    ----------
    runner : callable
        ``runner(prog, shots) -> dict[str, int]`` histogram callback.
    n : int
        Width and depth. Default 3. Keep small on real hardware (n<=5).
    shots : int
        Number of shots. Default 1024.
    seed : int, optional
        RNG seed for reproducibility of the random model circuit.

    Returns
    -------
    dict
        Dictionary with the trial's heavy-output frequency and metadata::

            {
                "n": n,
                "counts": {...},
                "heavy_output_frequency": float,
                "ideal_heavy_output_frequency": float,
                "pass_threshold": 2 / 3,
                "noiseless_target": (1 + ln 2) / 2,
                "shots": int,
                "seed": int | None,
            }
    """
    rng = np.random.default_rng(seed)
    circuit = QCircuit()
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
    counts = runner(prog, shots)

    total = sum(counts.values()) or 1
    heavy_hits = 0
    for bitstring, c in counts.items():
        try:
            idx = int(bitstring, 2)
        except ValueError:
            continue
        if idx in heavy:
            heavy_hits += c

    return {
        "n": n,
        "counts": dict(counts),
        "heavy_output_frequency": heavy_hits / total,
        "ideal_heavy_output_frequency": float(sum(probs[i] for i in heavy)),
        "pass_threshold": 2.0 / 3.0,
        "noiseless_target": (1.0 + np.log(2)) / 2.0,
        "shots": total,
        "seed": seed,
    }
