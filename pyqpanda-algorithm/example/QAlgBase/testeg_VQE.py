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

r"""
VQE -- 量子化学能级计算示例 (Quantum Chemistry Energy-Level Calculation)
=========================================================================

本示例演示如何使用 ``pyqpanda_alg.VQE`` 计算分子的基态能量以及凝聚态物理
模型的能谱，对应赛题"创新应用 -- 新增量子化学能级计算示例"。

This example demonstrates the Variational Quantum Eigensolver (VQE) on three
benchmark Hamiltonians shipped with the library:

1. ``H2``  molecule   -- 2-qubit reduced molecular Hamiltonian (quantum chemistry)
2. Transverse-field Ising model -- 3-site spin chain (condensed matter)
3. Heisenberg XXX model         -- 3-site spin chain (condensed matter)

For every system we compare the VQE estimate against the exact ground-state
energy obtained by dense diagonalization.
"""

import numpy as np

from pyqpanda_alg.VQE import vqe, hamiltonian, ansatz


def demo_h2_ground_state():
    """计算 H2 分子基态能量 (Compute the H2 molecular ground-state energy)."""
    print("=" * 64)
    print(" Example 1: H2 ground-state energy (quantum chemistry)")
    print("=" * 64)

    # 1. 构建 H2 分子哈密顿量 (build the 2-qubit H2 Hamiltonian)
    h2 = hamiltonian.h2_hamiltonian()
    exact = hamiltonian.exact_ground_energy(h2)
    print("Hamiltonian terms:", len(h2.terms()))
    print("Exact ground-state energy      : %.6f Hartree" % exact)

    # 2. 构造 VQE 求解器 (default hardware-efficient ansatz)
    np.random.seed(0)
    solver = vqe.VQE(h2)
    print("Ansatz params                  :", solver.n_params)

    # 3. 运行变分优化 (run the variational optimization)
    energy, params, history = solver.run(optimizer="L-BFGS-B",
                                         gradient=True, max_iter=200)
    print("VQE ground-state energy        : %.6f Hartree" % energy)
    print("Error                          : %.2e" % abs(energy - exact))
    print("Circuit evaluations            :", solver.circuit_evals)
    return energy, exact


def demo_h2_excited_states():
    """计算 H2 分子低激发态能级 (Compute low-lying excited-state energies via QSE)."""
    print("\n" + "=" * 64)
    print(" Example 2: H2 excited states (Quantum Subspace Expansion)")
    print("=" * 64)

    h2 = hamiltonian.h2_hamiltonian()
    spectrum = np.sort(np.linalg.eigvalsh(np.array(h2.matrix())).real)
    print("Exact spectrum                 :", np.round(spectrum, 5))

    np.random.seed(0)
    solver = vqe.VQE(h2)
    e0, params, _ = solver.run(optimizer="COBYLA", max_iter=300)
    qse = solver.excited_state_energies(params, n_states=4)
    print("QSE energies                   :", np.round(qse, 5))


def demo_transverse_field_ising():
    """计算横场 Ising 模型基态能量 (Transverse-field Ising ground state)."""
    print("\n" + "=" * 64)
    print(" Example 3: Transverse-field Ising model (3 sites, J=1, h=1)")
    print("=" * 64)

    h_tfi = hamiltonian.transverse_field_ising(3, 1.0, 1.0)
    exact = hamiltonian.exact_ground_energy(h_tfi)
    print("Exact ground-state energy      : %.6f" % exact)

    # 使用 3 层 hardware-efficient ansatz 提供足够表达能力
    def ansatz_3layer(n, params):
        return ansatz.hardware_efficient_ansatz(n, params, layers=3)

    np.random.seed(1)
    solver = vqe.VQE(h_tfi, ansatz=ansatz_3layer,
                     n_params=ansatz.hardware_efficient_n_params(3, layers=3))
    energy, params, _ = solver.run(optimizer="COBYLA", max_iter=600)
    print("VQE ground-state energy        : %.6f" % energy)
    print("Error                          : %.2e" % abs(energy - exact))


def demo_heisenberg():
    """计算 Heisenberg XXX 模型基态能量 (Heisenberg XXX ground state)."""
    print("\n" + "=" * 64)
    print(" Example 4: Heisenberg XXX model (3 sites)")
    print("=" * 64)

    h_heis = hamiltonian.heisenberg_model(3)
    exact = hamiltonian.exact_ground_energy(h_heis)
    print("Exact ground-state energy      : %.6f" % exact)

    def ansatz_2layer(n, params):
        return ansatz.hardware_efficient_ansatz(n, params, layers=2)

    np.random.seed(2)
    solver = vqe.VQE(h_heis, ansatz=ansatz_2layer,
                     n_params=ansatz.hardware_efficient_n_params(3, layers=2))
    energy, params, _ = solver.run(optimizer="COBYLA", max_iter=500)
    print("VQE ground-state energy        : %.6f" % energy)
    print("Error                          : %.2e" % abs(energy - exact))


def demo_molecular_hamiltonian_from_integrals():
    """从积分构造分子哈密顿量 (Build a molecular Hamiltonian from integrals)."""
    print("\n" + "=" * 64)
    print(" Example 5: molecular_hamiltonian from one-/two-body integrals")
    print("=" * 64)

    # 对角 one-body 积分 eps=[1, 2] 的无相互作用体系，
    # 占据数能量应为 0, 1, 2, 3 (occupations of two spin-orbitals)
    h1 = np.diag([1.0, 2.0])
    h2 = np.zeros((2, 2, 2, 2))
    h_mol = hamiltonian.molecular_hamiltonian(0.0, h1, h2)
    ev = np.sort(np.linalg.eigvalsh(np.array(h_mol.matrix())).real)
    print("Eigenvalues:", np.round(ev, 4), "(expected 0,1,2,3)")


if __name__ == "__main__":
    demo_h2_ground_state()
    demo_h2_excited_states()
    demo_transverse_field_ising()
    demo_heisenberg()
    demo_molecular_hamiltonian_from_integrals()
    print("\nAll VQE examples finished.")
