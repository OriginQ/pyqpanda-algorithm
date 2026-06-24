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

import numpy as np
import pytest

from pyqpanda_alg.VQE import vqe, hamiltonian, ansatz


"""
接口功能概述: VQE.VQE 是变分量子特征值求解器 (Variational Quantum Eigensolver)，
通过参数化量子线路 (ansatz) 与经典优化器的混合迭代，求解给定哈密顿量的基态能量。

核心功能
1. 分子基态能量计算: 给定量子化学哈密顿量 (PauliOperator)，返回基态能量近似值
2. 参数移位 (parameter-shift) 梯度: 对任意 ansatz 提供精确的、可在硬件上测量的梯度
3. 激发态能级估计: 通过量子子空间展开 (QSE) 估计低激发态能量
4. 多种 ansatz: hardware-efficient / UCC / symmetry-preserving
"""


class Test_VQE_Hamiltonian:
    """测试 hamiltonian 模块的哈密顿量构造功能"""

    def test_h2_ground_energy(self):
        """H2 约化哈密顿量基态能量应为 -1.857275 Hartree"""
        h2 = hamiltonian.h2_hamiltonian()
        e = hamiltonian.exact_ground_energy(h2)
        assert abs(e - (-1.857275)) < 1e-4

    def test_jordan_wigner_number_operator(self):
        """Jordan-Wigner: a†_0 a_0 应等于 (I - Z_0) / 2"""
        n0 = hamiltonian.jw_create(2, 0) * hamiltonian.jw_annihilate(2, 0)
        mat = np.array(n0.matrix(), dtype=complex)
        expected = 0.5 * np.eye(2) - 0.5 * np.diag([1.0, -1.0])
        assert np.allclose(mat, expected)

    def test_transverse_field_ising_qubits(self):
        """TFI 模型应作用在 n 个量子比特上"""
        h = hamiltonian.transverse_field_ising(4, 1.0, 0.5)
        assert h.max_qbit_idx() + 1 == 4

    def test_heisenberg_ground_state_sign(self):
        """Heisenberg XXX 链基态能量应为负 (反铁磁关联)"""
        h = hamiltonian.heisenberg_model(4)
        assert hamiltonian.exact_ground_energy(h) < 0

    def test_molecular_hamiltonian_occupations(self):
        """对角 one-body 积分 eps=[1,2] 的占据数能量应为 0,1,2,3"""
        h1 = np.diag([1.0, 2.0])
        h2 = np.zeros((2, 2, 2, 2))
        h_mol = hamiltonian.molecular_hamiltonian(0.0, h1, h2)
        ev = np.sort(np.linalg.eigvalsh(np.array(h_mol.matrix())).real)
        assert np.allclose(ev, [0.0, 1.0, 2.0, 3.0])


class Test_VQE_Ansatz:
    """测试 ansatz 模块"""

    def test_hardware_efficient_n_params(self):
        n = ansatz.hardware_efficient_n_params(3, layers=2)
        assert n == 3 * 2 * 2  # n_qubits * n_rot * layers

    def test_hardware_efficient_returns_circuit(self):
        from pyqpanda3.core import QCircuit
        n_p = ansatz.hardware_efficient_n_params(2, layers=1)
        circ = ansatz.hardware_efficient_ansatz(2, np.zeros(n_p), layers=1)
        assert isinstance(circ, QCircuit)

    def test_hardware_efficient_param_count_mismatch(self):
        with pytest.raises(ValueError):
            ansatz.hardware_efficient_ansatz(3, [0.1, 0.2], layers=1)

    def test_hardware_efficient_invalid_entangler_single_qubit(self):
        """invalid entangler must be caught even when n_qubits=1 (no ladder)"""
        with pytest.raises(ValueError):
            ansatz.hardware_efficient_ansatz(1, [0.1, 0.2], entangler="BOGUS")

    def test_hardware_efficient_invalid_rotation(self):
        with pytest.raises(ValueError):
            ansatz.hardware_efficient_ansatz(2, [0.1] * 4, rotations=("RX", "ZZ"))

    def test_ucc_ansatz_returns_circuit(self):
        from pyqpanda3.core import QCircuit
        circ = ansatz.ucc_ansatz(2, [0.5], excitations=[(0, 1)])
        assert isinstance(circ, QCircuit)

    def test_symmetry_preserving_n_params(self):
        n = ansatz.symmetry_preserving_n_params(3, layers=2)
        assert n == 2 * ((3 - 1) + 3)


class Test_VQE_Solver:
    """测试 VQE 核心求解器"""

    def test_vqe_type_error(self):
        """非 PauliOperator 输入应抛出 TypeError"""
        with pytest.raises(TypeError):
            vqe.VQE("not a hamiltonian")

    def test_vqe_custom_ansatz_requires_n_params(self):
        """自定义 ansatz 必须显式给出 n_params"""
        def noop(n, params):
            return None
        with pytest.raises(ValueError):
            vqe.VQE(hamiltonian.h2_hamiltonian(), ansatz=noop)

    def test_vqe_initial_para_length_checked(self):
        """initial_para 长度不匹配应给出明确错误"""
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        with pytest.raises(ValueError):
            solver.run(initial_para=[0.1, 0.2], max_iter=5)

    def test_vqe_gradient_free_optimizer_warns(self):
        """gradient=True 配合无梯度优化器应发出 warning"""
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        with pytest.warns(UserWarning):
            solver.run(optimizer="COBYLA", gradient=True, max_iter=3)

    def test_vqe_circuit_evals_not_doubled(self):
        """circuit_evals 不应被 callback 重复计数 (每次迭代只算 1 次)"""
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        _, _, history = solver.run(optimizer="COBYLA", max_iter=30)
        # evals should be on the order of the iteration count, not 2x
        assert solver.circuit_evals < 2 * len(history)

    def test_vqe_h2_ground_state(self):
        """VQE 应在化学精度 (1e-3 Hartree) 内求出 H2 基态能量"""
        np.random.seed(0)
        h2 = hamiltonian.h2_hamiltonian()
        exact = hamiltonian.exact_ground_energy(h2)
        solver = vqe.VQE(h2)
        energy, params, history = solver.run(optimizer="COBYLA", max_iter=300)
        assert abs(energy - exact) < 1e-3
        assert len(history) >= 1

    def test_vqe_l_bfgs_gradient(self):
        """使用参数移位梯度的 L-BFGS-B 应收敛到精确基态"""
        np.random.seed(0)
        h2 = hamiltonian.h2_hamiltonian()
        exact = hamiltonian.exact_ground_energy(h2)
        solver = vqe.VQE(h2)
        energy, _, _ = solver.run(optimizer="L-BFGS-B",
                                  gradient=True, max_iter=200)
        assert abs(energy - exact) < 1e-4

    def test_param_shift_gradient_exactness(self):
        """参数移位梯度应与有限差分梯度一致"""
        h2 = hamiltonian.h2_hamiltonian()
        solver = vqe.VQE(h2)
        theta = np.array([0.3, -0.2, 0.5, 0.1])
        g_ps = solver.param_shift_gradient(theta)
        # finite difference
        g_fd = np.zeros_like(theta)
        eps = 1e-6
        for k in range(len(theta)):
            tp = theta.copy(); tp[k] += eps
            tm = theta.copy(); tm[k] -= eps
            g_fd[k] = (solver.expectation(tp) - solver.expectation(tm)) / (2 * eps)
        assert np.allclose(g_ps, g_fd, atol=1e-6)

    def test_vqe_heisenberg(self):
        """VQE 应求出 3-site Heisenberg 基态能量 (约 -4.0)"""
        np.random.seed(2)
        h_heis = hamiltonian.heisenberg_model(3)
        exact = hamiltonian.exact_ground_energy(h_heis)

        def ansatz_2layer(n, params):
            return ansatz.hardware_efficient_ansatz(n, params, layers=2)
        solver = vqe.VQE(h_heis, ansatz=ansatz_2layer,
                         n_params=ansatz.hardware_efficient_n_params(3, layers=2))
        energy, _, _ = solver.run(optimizer="COBYLA", max_iter=500)
        assert abs(energy - exact) < 1e-2

    def test_get_statevector_normalized(self):
        """ansatz 生成的态矢量应归一化"""
        h2 = hamiltonian.h2_hamiltonian()
        solver = vqe.VQE(h2)
        sv = solver.get_statevector([0.1, 0.2, 0.3, 0.4])
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-9

    def test_excited_state_energies(self):
        """QSE 激发态能量应包含基态能量"""
        np.random.seed(0)
        h2 = hamiltonian.h2_hamiltonian()
        solver = vqe.VQE(h2)
        e0, params, _ = solver.run(optimizer="COBYLA", max_iter=300)
        qse = solver.excited_state_energies(params, n_states=2)
        assert qse[0] < qse[1]
        assert abs(qse[0] - e0) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
