# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
QLattice: Quantum Algorithms for Lattice-Based Cryptography

This module provides quantum algorithms for solving lattice-based cryptographic
problems, including Shortest Vector Problem (SVP) and Module Learning With
Errors (MLWE) problem.

Key algorithms included:

QuantumLLL : Quantum-accelerated LLL lattice reduction algorithm using QFT.
    Achieves O(n^2) polynomial speedup over classical LLL algorithm.

VQLS : Variational Quantum Lattice Solver using VQE approach.
    Encodes SVP as Hamiltonian minimization problem.

QAOA_MLWE : QAOA-based MLWE solver.
    Transforms MLWE into QUBO problem and solves with QAOA.

QuantumKernel : Quantum kernel method for MLWE.
    Maps MLWE samples to high-dimensional Hilbert space using SWAP test.

These algorithms provide polynomial-level speedup over classical algorithms
for solving lattice problems, which are the foundation of post-quantum
cryptography standards (ML-KEM, ML-DSA).

References
    [1] Grover, L.K. "A fast quantum mechanical algorithm for database search".
    [2] Farhi, E., et al. "A Quantum Approximate Optimization Algorithm".
    [3] Peruzzo, A., et al. "A variational eigenvalue solver on a photonic quantum processor".
    [4] Harrow, A.W., et al. "Quantum algorithm for solving linear systems of equations".

'''

from .quantum_lll import QuantumLLL
from .vqls import VQLS
from .qaoa_mlwe import QAOA_MLWE
from .quantum_kernel import QuantumKernel

__all__ = ['QuantumLLL', 'VQLS', 'QAOA_MLWE', 'QuantumKernel']
