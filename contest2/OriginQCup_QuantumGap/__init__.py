# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""NISQ Hardware Benchmarking Suite for pyqpanda3.

A small, dependency-light suite of textbook hardware benchmarks that run on
both pyqpanda3 simulators (CPUQVM) and Origin QCloud backends (Wukong WK_C180
family, full_amplitude simulator). Each benchmark reports a single, comparable
number so a user can characterise a backend in one run.

Benchmarks
----------
- bell_benchmark        : 2q entanglement fidelity proxy (|00>+|11> survival).
- ghz_benchmark         : n-q GHZ state fidelity proxy.
- quantum_volume_probe  : Heavy-output frequency at width=depth=n. Pass=2/3.
- mirror_circuit_probe  : Forward-then-inverse Clifford layers; ideal survival
                          P(|0...0>) = 1.0 on a noise-free backend.

Why this is here
----------------
pyqpanda-algorithm ships strong application algorithms (Grover, QAOA, QSVM,
QAE, ...) but no hardware-characterisation tooling. A user wanting to pick
between Wukong WK_C180 and WK_C180_2 for an experiment currently has no
in-library way to compare them. This suite closes that gap with the
standard primitives used across IBM / Sandia / Google benchmarking work,
implemented natively in pyqpanda3.

Submitted for the 2026 CCF Quantum Computing Programming Challenge
"OriginQ Cup" Open-Source Innovation Track by team Quantum Gap.
"""

from .bell import bell_benchmark
from .ghz import ghz_benchmark
from .quantum_volume import quantum_volume_probe
from .mirror_circuit import mirror_circuit_probe

__all__ = [
    "bell_benchmark",
    "ghz_benchmark",
    "quantum_volume_probe",
    "mirror_circuit_probe",
]
