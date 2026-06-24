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

from pyqpanda3.hamiltonian import PauliOperator
from pyqpanda3.core import QCircuit, CPUQVM, QProg, expval_pauli_operator

from pyqpanda_alg.VQE import vqe, hamiltonian, ansatz
from pyqpanda_alg.VQE.vqe import _full_matrix, _GRADIENT_METHODS


"""
接口功能概述: VQE 模块 (Variational Quantum Eigensolver) 测试套件。

测试组织:
  Test_JordanWigner          -- Jordan-Wigner 变换的物理正确性 (反对易关系等)
  Test_Hamiltonian_Builders  -- 各哈密顿量构造器 (分子/Ising/Heisenberg/H2)
  Test_Hamiltonian_Properties-- 哈密顿量通用性质 (厄米性、能级)
  Test_Ansatz_Construction   -- 三类变分线路构造与参数计数
  Test_Ansatz_Properties     -- 线路性质 (粒子数守恒、归一化)
  Test_VQE_Constructor       -- VQE 构造与输入校验
  Test_VQE_Energy            -- 能量期望与态矢量
  Test_VQE_Gradient          -- 参数移位梯度
  Test_VQE_Optimization      -- 优化过程 (多优化器、收敛、确定性)
  Test_VQE_Physics           -- 变分原理等物理不变量
  Test_VQE_ExcitedStates     -- QSE 激发态
"""


def _mat(op, n):
    """把 PauliOperator 填充到 n 比特的稠密矩阵 (用于物理断言)。"""
    return _full_matrix(op, n)


# ===========================================================================
#  Jordan-Wigner 变换
# ===========================================================================
class Test_JordanWigner:

    def test_number_operator(self):
        """a†_p a_p = (I - Z_p) / 2  (粒子数算符)"""
        n0 = hamiltonian.jw_create(2, 0) * hamiltonian.jw_annihilate(2, 0)
        expected = 0.5 * np.eye(2) - 0.5 * np.diag([1.0, -1.0])
        assert np.allclose(np.array(n0.matrix()), expected)

    def test_number_operator_idempotent(self):
        """n_p² = n_p  (投影算符幂等)"""
        n = 3
        for p in range(n):
            nop = hamiltonian.jw_create(n, p) * hamiltonian.jw_annihilate(n, p)
            assert np.allclose(_mat(nop * nop, n), _mat(nop, n))

    def test_anticommutation_aa(self):
        """费米子反对易关系 {a_p, a_q} = a_p a_q + a_q a_p = 0"""
        n = 3
        for p in range(n):
            for q in range(n):
                ap = hamiltonian.jw_annihilate(n, p)
                aq = hamiltonian.jw_annihilate(n, q)
                assert np.allclose(_mat(ap * aq + aq * ap, n), 0.0, atol=1e-9)

    def test_anticommutation_creation_annihilation(self):
        """{a_p, a†_q} = a_p a†_q + a†_q a_p = δ_pq I"""
        n = 3
        for p in range(n):
            for q in range(n):
                ap = hamiltonian.jw_annihilate(n, p)
                aqd = hamiltonian.jw_create(n, q)
                anti = _mat(ap * aqd + aqd * ap, n)
                expected = np.eye(2 ** n) if p == q else np.zeros((2 ** n, 2 ** n))
                assert np.allclose(anti, expected, atol=1e-9)

    def test_creation_is_dagger_of_annihilation(self):
        """a†_p = (a_p)†"""
        n = 3
        for p in range(n):
            a = _mat(hamiltonian.jw_annihilate(n, p), n)
            ad = _mat(hamiltonian.jw_create(n, p), n)
            assert np.allclose(ad, a.conj().T)


# ===========================================================================
#  哈密顿量构造器
# ===========================================================================
class Test_Hamiltonian_Builders:

    def test_h2_ground_energy(self):
        """H2 约化哈密顿量基态能量 = -1.857275 Hartree"""
        h2 = hamiltonian.h2_hamiltonian()
        assert abs(hamiltonian.exact_ground_energy(h2) - (-1.857275)) < 1e-4

    def test_h2_is_two_qubit(self):
        h2 = hamiltonian.h2_hamiltonian()
        assert h2.max_qbit_idx() + 1 == 2

    def test_h2_nonzero_terms(self):
        """H2 哈密顿量应含 5 个 Pauli 项"""
        h2 = hamiltonian.h2_hamiltonian()
        assert len(h2.terms()) == 5

    def test_molecular_hamiltonian_occupations(self):
        """对角 one-body 积分 eps=[1,2] 的占据数能量 = 0,1,2,3"""
        h1 = np.diag([1.0, 2.0])
        h2 = np.zeros((2, 2, 2, 2))
        h_mol = hamiltonian.molecular_hamiltonian(0.0, h1, h2)
        ev = np.sort(np.linalg.eigvalsh(np.array(h_mol.matrix())).real)
        assert np.allclose(ev, [0.0, 1.0, 2.0, 3.0])

    def test_molecular_hamiltonian_nuclear_repulsion(self):
        """核排斥能应整体平移所有能级"""
        h1 = np.diag([1.0, 2.0])
        h2 = np.zeros((2, 2, 2, 2))
        e0 = np.sort(np.linalg.eigvalsh(
            np.array(hamiltonian.molecular_hamiltonian(0.0, h1, h2).matrix())).real)
        e1 = np.sort(np.linalg.eigvalsh(
            np.array(hamiltonian.molecular_hamiltonian(5.0, h1, h2).matrix())).real)
        assert np.allclose(e1 - e0, 5.0)

    def test_jordan_wigner_function_directly(self):
        """直接调用 jordan_wigner() 与 molecular_hamiltonian() 一致 (无核排斥)"""
        h1 = np.diag([1.0, 2.0])
        h2 = np.zeros((2, 2, 2, 2))
        a = np.array(hamiltonian.jordan_wigner(h1, h2).matrix())
        b = np.array(hamiltonian.molecular_hamiltonian(0.0, h1, h2).matrix())
        assert np.allclose(a, b)

    def test_transverse_field_ising_qubits(self):
        h = hamiltonian.transverse_field_ising(4, 1.0, 0.5)
        assert h.max_qbit_idx() + 1 == 4

    def test_transverse_field_ising_open_vs_periodic(self):
        """周期/开放边界应给出不同能量 (n>2)"""
        h_open = hamiltonian.transverse_field_ising(4, 1.0, 1.0)
        h_per = hamiltonian.transverse_field_ising(4, 1.0, 1.0, periodic=True)
        assert abs(hamiltonian.exact_ground_energy(h_open)
                   - hamiltonian.exact_ground_energy(h_per)) > 1e-6

    def test_transverse_field_ising_single_qubit(self):
        """n=1 边界: 仅剩横场项 h*X0"""
        h = hamiltonian.transverse_field_ising(1, 1.0, 0.7)
        assert hamiltonian.exact_ground_energy(h) == pytest.approx(-0.7)

    def test_heisenberg_ground_state_sign(self):
        """Heisenberg XXX 反铁磁基态能量为负"""
        assert hamiltonian.exact_ground_energy(hamiltonian.heisenberg_model(4)) < 0

    def test_heisenberg_anisotropic(self):
        """各向异性 (Ising 极限 Jx=Jy=0) 应退化为 ZZ 模型"""
        n = 4
        h_aniso = hamiltonian.heisenberg_model(n, jx=0.0, jy=0.0, jz=1.0)
        h_zz = sum(
            (hamiltonian.PauliOperator({"Z%d Z%d" % (i, i + 1): 1.0})
             for i in range(n - 1)),
            hamiltonian.PauliOperator({"": 0.0}))
        assert np.allclose(np.array(h_aniso.matrix()), np.array(h_zz.matrix()))

    def test_heisenberg_periodic_adds_wrapping_coupling(self):
        """periodic=True 应在 n>2 时额外增加首尾耦合项"""
        h_open = hamiltonian.heisenberg_model(4, periodic=False)
        h_per = hamiltonian.heisenberg_model(4, periodic=True)
        assert len(h_per.terms()) > len(h_open.terms())

    def test_exact_ground_energy_on_identity(self):
        """单位算符的基态能量 = 1"""
        I = hamiltonian.PauliOperator({"": 1.0})
        assert hamiltonian.exact_ground_energy(I) == pytest.approx(1.0)


# ===========================================================================
#  哈密顿量通用性质
# ===========================================================================
class Test_Hamiltonian_Properties:

    @pytest.mark.parametrize("builder", [
        lambda: hamiltonian.h2_hamiltonian(),
        lambda: hamiltonian.transverse_field_ising(4, 1.0, 0.8),
        lambda: hamiltonian.transverse_field_ising(4, 1.0, 0.8, periodic=True),
        lambda: hamiltonian.heisenberg_model(4),
        lambda: hamiltonian.heisenberg_model(4, jx=0.5, jy=1.0, jz=1.5),
        lambda: hamiltonian.molecular_hamiltonian(0.3, np.diag([1.0, 2.0]),
                                                   np.zeros((2, 2, 2, 2))),
    ])
    def test_hermiticity(self, builder):
        """所有构造器生成的 H 都应厄米: H = H†"""
        h = builder()
        mat = np.array(h.matrix())
        assert np.allclose(mat, mat.conj().T)


# ===========================================================================
#  Ansatz 构造
# ===========================================================================
class Test_Ansatz_Construction:

    def test_hardware_efficient_returns_circuit(self):
        n_p = ansatz.hardware_efficient_n_params(2, layers=1)
        circ = ansatz.hardware_efficient_ansatz(2, np.zeros(n_p), layers=1)
        assert isinstance(circ, QCircuit)

    def test_hardware_efficient_n_params_cnot(self):
        assert ansatz.hardware_efficient_n_params(3, layers=2) == 3 * 2 * 2

    def test_hardware_efficient_n_params_rxx(self):
        """RXX 纠缠门每对额外消耗 1 个参数"""
        assert ansatz.hardware_efficient_n_params(3, layers=1, entangler="RXX") \
            == 3 * 2 + 2  # n_rot*n + entangle pairs

    def test_hardware_efficient_param_count_mismatch(self):
        with pytest.raises(ValueError):
            ansatz.hardware_efficient_ansatz(3, [0.1, 0.2], layers=1)

    def test_hardware_efficient_invalid_entangler_single_qubit(self):
        """n_qubits=1 时 (无纠缠梯) 非法 entangler 仍应报错"""
        with pytest.raises(ValueError):
            ansatz.hardware_efficient_ansatz(1, [0.1, 0.2], entangler="BOGUS")

    def test_hardware_efficient_invalid_rotation(self):
        with pytest.raises(ValueError):
            ansatz.hardware_efficient_ansatz(2, [0.1] * 4, rotations=("RX", "ZZ"))

    def test_hardware_efficient_invalid_entangler_value(self):
        with pytest.raises(ValueError):
            ansatz.hardware_efficient_ansatz(2, [0.1] * 4, entangler="XXX")

    def test_hardware_efficient_single_qubit(self):
        """n=1 不应有纠缠门, 仍能正常构造"""
        circ = ansatz.hardware_efficient_ansatz(1, [0.1, 0.2], layers=1)
        assert isinstance(circ, QCircuit)

    def test_hardware_efficient_rxx_entangler(self):
        """RXX 参数化纠缠门构造成功"""
        n_p = ansatz.hardware_efficient_n_params(3, layers=1, entangler="RXX")
        circ = ansatz.hardware_efficient_ansatz(3, [0.1] * n_p,
                                                layers=1, entangler="RXX")
        assert isinstance(circ, QCircuit)

    def test_hardware_efficient_custom_rotations(self):
        circ = ansatz.hardware_efficient_ansatz(
            2, [0.1, 0.2], layers=1, rotations=("RX",))
        assert isinstance(circ, QCircuit)

    def test_ucc_ansatz_returns_circuit(self):
        circ = ansatz.ucc_ansatz(2, [0.5], excitations=[(0, 1)])
        assert isinstance(circ, QCircuit)

    def test_ucc_ansatz_default_excitations(self):
        """默认激发对 = 所有相邻对"""
        circ = ansitz_helper_default_ucc(3, [0.1, 0.2])
        assert isinstance(circ, QCircuit)

    def test_ucc_ansatz_param_count_mismatch(self):
        with pytest.raises(ValueError):
            ansatz.ucc_ansatz(3, [0.1], excitations=[(0, 1), (1, 2)])

    def test_ucc_n_params(self):
        assert ansatz.ucc_n_params([(0, 1), (1, 2), (0, 2)]) == 3

    def test_symmetry_preserving_returns_circuit(self):
        n_p = ansatz.symmetry_preserving_n_params(2, layers=1)
        circ = ansatz.symmetry_preserving_ansatz(2, [0.1] * n_p, layers=1)
        assert isinstance(circ, QCircuit)

    def test_symmetry_preserving_n_params(self):
        assert ansatz.symmetry_preserving_n_params(3, layers=2) == 2 * ((3 - 1) + 3)

    def test_symmetry_preserving_param_count_mismatch(self):
        with pytest.raises(ValueError):
            ansatz.symmetry_preserving_ansatz(3, [0.1, 0.2], layers=1)

    def test_symmetry_preserving_requires_two_qubits(self):
        with pytest.raises(ValueError):
            ansatz.symmetry_preserving_ansatz(1, [], layers=1)


def ansitz_helper_default_ucc(n, params):
    """用默认激发对构造 UCC ansatz 的辅助函数。"""
    return ansatz.ucc_ansatz(n, params)


# ===========================================================================
#  Ansatz 性质
# ===========================================================================
class Test_Ansatz_Properties:

    @staticmethod
    def _state(circuit, n):
        qvm = CPUQVM()
        prog = QProg(n)
        prog << circuit
        qvm.run(prog, n)
        return np.array(qvm.result().get_state_vector(), dtype=complex)

    def test_ansatz_output_normalized(self):
        """所有 ansatz 生成的态矢量都应归一化"""
        cases = [
            ansatz.hardware_efficient_ansatz(3, [0.3] * 6, layers=1),
            ansatz.ucc_ansatz(2, [0.7], excitations=[(0, 1)]),
            ansatz.symmetry_preserving_ansatz(3, [0.3, 0.4, 0.5, 0.6, 0.7], layers=1),
        ]
        for i, circ in enumerate(cases):
            n = [3, 2, 3][i]
            sv = self._state(circ, n)
            assert abs(np.linalg.norm(sv) - 1.0) < 1e-9

    def test_symmetry_preserving_conserves_particle_number(self):
        """symmetry-preserving ansatz 从 |0> 出发应保持粒子数 = 0"""
        n = 3
        numop = sum((hamiltonian.jw_create(n, i) * hamiltonian.jw_annihilate(n, i)
                     for i in range(n)), hamiltonian.PauliOperator({"": 0.0}))
        qvm = CPUQVM()
        prog = QProg(n)
        prog << ansatz.symmetry_preserving_ansatz(n, [0.3, 0.4, 0.5, 0.6, 0.7], layers=1)
        qvm.run(prog, n)
        nexp = float(expval_pauli_operator(prog, numop).real)
        assert abs(nexp) < 1e-9

    def test_hardware_efficient_does_not_conserve_particle_number(self):
        """对照: hardware-efficient ansatz 一般不守恒粒子数"""
        n = 3
        numop = sum((hamiltonian.jw_create(n, i) * hamiltonian.jw_annihilate(n, i)
                     for i in range(n)), hamiltonian.PauliOperator({"": 0.0}))
        qvm = CPUQVM()
        prog = QProg(n)
        prog << ansatz.hardware_efficient_ansatz(n, [0.5] * 6, layers=1)
        qvm.run(prog, n)
        nexp = float(expval_pauli_operator(prog, numop).real)
        assert abs(nexp) > 1e-3

    def test_zero_params_produce_known_state(self):
        """零参数 hardware-efficient (RY(0)=RZ(0)=I, 无 CNOT 改变) -> |0...0>"""
        circ = ansatz.hardware_efficient_ansatz(2, [0.0, 0.0, 0.0, 0.0], layers=1)
        sv = self._state(circ, 2)
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        assert np.allclose(sv, expected)


# ===========================================================================
#  VQE 构造与校验
# ===========================================================================
class Test_VQE_Constructor:

    def test_type_error_non_pauli(self):
        with pytest.raises(TypeError):
            vqe.VQE("not a hamiltonian")

    def test_custom_ansatz_requires_n_params(self):
        def noop(n, params):
            return None
        with pytest.raises(ValueError):
            vqe.VQE(hamiltonian.h2_hamiltonian(), ansatz=noop)

    def test_n_qubits_inferred_from_hamiltonian(self):
        """n_qubits 缺省时由 hamiltonian 推断"""
        solver = vqe.VQE(hamiltonian.transverse_field_ising(4, 1.0, 1.0))
        assert solver.n_qubits == 4

    def test_n_qubits_override(self):
        """允许显式指定 n_qubits (大于哈密顿量所需)"""
        h2 = hamiltonian.h2_hamiltonian()  # 2 qubits
        solver = vqe.VQE(h2, n_qubits=3)
        assert solver.n_qubits == 3

    def test_default_ansatz_param_count(self):
        solver = vqe.VQE(hamiltonian.transverse_field_ising(3, 1.0, 1.0))
        # 默认单层 (RY,RZ)+CNOT: n_qubits*2 = 6
        assert solver.n_params == 6

    def test_custom_ansatz_with_n_params(self):
        """提供 n_params 时自定义 ansatz 正常构造"""
        def he2(n, params):
            return ansatz.hardware_efficient_ansatz(n, params, layers=2)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian(), ansatz=he2, n_params=8)
        assert solver.n_params == 8

    def test_initial_attributes(self):
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        assert solver.energy_history == []
        assert solver.optimal_params is None
        assert solver.optimal_energy is None
        assert solver.circuit_evals == 0


# ===========================================================================
#  能量期望与态矢量
# ===========================================================================
class Test_VQE_Energy:

    def test_expectation_is_real(self):
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        e = solver.expectation([0.1, 0.2, 0.3, 0.4])
        assert isinstance(e, float)

    def test_expectation_zero_params(self):
        """零参数 (|00>) 的 H2 能量 = -1.0637"""
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        assert solver.expectation([0.0, 0.0, 0.0, 0.0]) == pytest.approx(-1.0637, abs=1e-3)

    def test_expectation_matches_statevector(self):
        """expectation 应与 <sv|H|sv> 一致"""
        h2 = hamiltonian.h2_hamiltonian()
        solver = vqe.VQE(h2)
        params = [0.3, -0.2, 0.5, 0.1]
        e_direct = solver.expectation(params)
        sv = solver.get_statevector(params)
        h_mat = np.array(h2.matrix())
        e_sv = float(np.vdot(sv, h_mat @ sv).real)
        assert e_direct == pytest.approx(e_sv, abs=1e-8)

    def test_expectation_increments_circuit_evals(self):
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        solver.expectation([0.1] * 4)
        solver.expectation([0.2] * 4)
        assert solver.circuit_evals == 2

    def test_get_statevector_dimension_and_norm(self):
        solver = vqe.VQE(hamiltonian.transverse_field_ising(3, 1.0, 1.0))
        sv = solver.get_statevector([0.1] * 6)
        assert sv.shape == (8,)
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-9


# ===========================================================================
#  参数移位梯度
# ===========================================================================
class Test_VQE_Gradient:

    def test_gradient_matches_finite_difference(self):
        """参数移位梯度 == 中心有限差分"""
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        theta = np.array([0.3, -0.2, 0.5, 0.1])
        g_ps = solver.param_shift_gradient(theta)
        g_fd = np.zeros_like(theta)
        eps = 1e-6
        for k in range(len(theta)):
            tp = theta.copy(); tp[k] += eps
            tm = theta.copy(); tm[k] -= eps
            g_fd[k] = (solver.expectation(tp) - solver.expectation(tm)) / (2 * eps)
        assert np.allclose(g_ps, g_fd, atol=1e-6)

    def test_gradient_length(self):
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        g = solver.param_shift_gradient([0.1, 0.2, 0.3, 0.4])
        assert len(g) == 4

    def test_gradient_near_zero_at_optimum(self):
        """收敛到极小值后梯度应接近 0"""
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        _, params, _ = solver.run(optimizer="L-BFGS-B", gradient=True, max_iter=200)
        g = solver.param_shift_gradient(params)
        assert np.allclose(g, 0.0, atol=1e-3)

    def test_gradient_methods_constant_complete(self):
        """_GRADIENT_METHODS 应包含主流梯度优化器"""
        for m in ("CG", "BFGS", "L-BFGS-B", "SLSQP"):
            assert m in _GRADIENT_METHODS


# ===========================================================================
#  优化过程
# ===========================================================================
class Test_VQE_Optimization:

    def test_run_returns_three_objects(self):
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        np.random.seed(0)
        out = solver.run(max_iter=50)
        assert len(out) == 3
        energy, params, history = out
        assert isinstance(energy, float)
        assert len(params) == solver.n_params
        assert isinstance(history, list)

    def test_run_h2_cobyla_chemical_accuracy(self):
        """COBYLA 在化学精度 (1e-3 Hartree) 内求出 H2 基态"""
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        exact = hamiltonian.exact_ground_energy(hamiltonian.h2_hamiltonian())
        energy, _, _ = solver.run(optimizer="COBYLA", max_iter=300)
        assert abs(energy - exact) < 1e-3

    def test_run_h2_lbfgsb_gradient(self):
        """L-BFGS-B + 参数移位梯度收敛到精确基态"""
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        exact = hamiltonian.exact_ground_energy(hamiltonian.h2_hamiltonian())
        energy, _, _ = solver.run(optimizer="L-BFGS-B", gradient=True, max_iter=200)
        assert abs(energy - exact) < 1e-4

    def test_run_h2_powell(self):
        """Powell (无梯度) 也能收敛 H2"""
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        exact = hamiltonian.exact_ground_energy(hamiltonian.h2_hamiltonian())
        energy, _, _ = solver.run(optimizer="Powell", max_iter=300)
        assert abs(energy - exact) < 1e-2

    def test_run_initial_para_length_checked(self):
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        with pytest.raises(ValueError):
            solver.run(initial_para=[0.1, 0.2], max_iter=5)

    def test_run_gradient_free_optimizer_warns(self):
        """gradient=True + COBYLA 应发出 warning"""
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        with pytest.warns(UserWarning):
            solver.run(optimizer="COBYLA", gradient=True, max_iter=3)

    def test_run_circuit_evals_not_doubled(self):
        """circuit_evals 不被 callback 重复计数"""
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        _, _, history = solver.run(optimizer="COBYLA", max_iter=30)
        assert solver.circuit_evals < 2 * len(history)

    def test_run_sets_optimal_attributes(self):
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        energy, params, _ = solver.run(max_iter=50)
        assert solver.optimal_energy == pytest.approx(energy)
        assert np.allclose(solver.optimal_params, params)

    def test_run_history_records_convergence(self):
        """history 应非空, 且最优能量 ≈ 记录中的最佳能量 (回调末值不一定等于 res.x)"""
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        energy, _, history = solver.run(optimizer="COBYLA", max_iter=80)
        assert len(history) >= 1
        # res.fun 可能比最后一次回调点略优, 因此与 history 的最小值比较
        assert energy <= min(history) + 1e-9
        assert energy == pytest.approx(min(history), abs=1e-3)

    def test_run_verbose_does_not_crash(self, capsys):
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        solver.run(max_iter=60, verbose=True)
        out = capsys.readouterr().out
        assert "iter" in out and "energy" in out

    def test_run_deterministic_with_fixed_seed(self):
        """固定随机种子下两次运行结果应一致"""
        h2 = hamiltonian.h2_hamiltonian()
        np.random.seed(123)
        e1, _, _ = vqe.VQE(h2).run(max_iter=80)
        np.random.seed(123)
        e2, _, _ = vqe.VQE(h2).run(max_iter=80)
        assert e1 == pytest.approx(e2)


# ===========================================================================
#  物理不变量
# ===========================================================================
class Test_VQE_Physics:

    @pytest.mark.parametrize("label", ["h2", "tfi", "heisenberg"])
    def test_variational_principle(self, label):
        """变分原理: VQE 能量 >= 精确基态能量"""
        np.random.seed(0)
        if label == "h2":
            h = hamiltonian.h2_hamiltonian()
            solver = vqe.VQE(h)
            iters = 200
        elif label == "tfi":
            h = hamiltonian.transverse_field_ising(3, 1.0, 1.0)
            def ans3(n, p): return ansatz.hardware_efficient_ansatz(n, p, layers=3)
            solver = vqe.VQE(h, ansatz=ans3,
                             n_params=ansatz.hardware_efficient_n_params(3, layers=3))
            iters = 500
        else:
            h = hamiltonian.heisenberg_model(3)
            def ans2(n, p): return ansatz.hardware_efficient_ansatz(n, p, layers=2)
            solver = vqe.VQE(h, ansatz=ans2,
                             n_params=ansatz.hardware_efficient_n_params(3, layers=2))
            iters = 400
        exact = hamiltonian.exact_ground_energy(h)
        energy, _, _ = solver.run(optimizer="COBYLA", max_iter=iters)
        assert energy >= exact - 1e-6  # 容许数值误差

    def test_ucc_ansatz_solves_h2(self):
        """UCC ansatz 应能近似求解 H2 基态"""
        np.random.seed(0)
        h2 = hamiltonian.h2_hamiltonian()
        exact = hamiltonian.exact_ground_energy(h2)

        def ucc1(n, params):
            return ansatz.ucc_ansatz(n, params, excitations=[(0, 1)])
        solver = vqe.VQE(h2, ansatz=ucc1, n_params=1)
        energy, _, _ = solver.run(optimizer="COBYLA", max_iter=200)
        # UCC 单激发有表达力上限, 但应在 ~0.1 Hartree 内
        assert abs(energy - exact) < 0.1

    def test_reoptimized_energy_matches_expectation(self):
        """run() 返回的能量应等于用最优参数重算的 expectation"""
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        energy, params, _ = solver.run(optimizer="COBYLA", max_iter=150)
        re = solver.expectation(params)
        assert energy == pytest.approx(re, abs=1e-5)


# ===========================================================================
#  激发态 (QSE)
# ===========================================================================
class Test_VQE_ExcitedStates:

    @staticmethod
    def _ground_solver():
        np.random.seed(0)
        solver = vqe.VQE(hamiltonian.h2_hamiltonian())
        e, p, _ = solver.run(optimizer="COBYLA", max_iter=200)
        return solver, e, p

    def test_qse_returns_ascending_energies(self):
        solver, _, p = self._ground_solver()
        qse = solver.excited_state_energies(p, n_states=2)
        assert len(qse) == 2
        assert qse[0] < qse[1]

    def test_qse_ground_matches_vqe(self):
        """QSE 最低能级应与 VQE 基态能量一致"""
        solver, e0, p = self._ground_solver()
        qse = solver.excited_state_energies(p, n_states=2)
        assert qse[0] == pytest.approx(e0, abs=1e-3)

    def test_qse_custom_excitations(self):
        solver, _, p = self._ground_solver()
        qse = solver.excited_state_energies(p, n_states=1, excitations=[(0, 1)])
        assert len(qse) == 1

    def test_qse_return_vectors(self):
        solver, _, p = self._ground_solver()
        energies, vectors = solver.excited_state_energies(
            p, n_states=2, return_vectors=True)
        assert energies.shape == (2,)
        assert vectors.shape[0] >= 2  # subspace dimension

    def test_qse_on_three_qubit_system(self):
        """QSE 在 3 比特系统上不应崩溃 (padding 正确性)"""
        np.random.seed(5)
        h = hamiltonian.transverse_field_ising(3, 1.0, 0.5)
        solver = vqe.VQE(h)
        e, p, _ = solver.run(optimizer="COBYLA", max_iter=100)
        qse = solver.excited_state_energies(p, n_states=2)
        assert len(qse) == 2
        assert np.all(np.isfinite(qse))


# ===========================================================================
#  内部辅助
# ===========================================================================
class Test_VQE_Internals:

    def test_full_matrix_pads_identity(self):
        """_full_matrix 把单位算子填充到 n 比特空间"""
        I = PauliOperator({"": 1.0})
        m = _full_matrix(I, 3)
        assert m.shape == (8, 8)
        assert np.allclose(m, np.eye(8))

    def test_full_matrix_pads_single_qubit_op(self):
        """_full_matrix 把单比特算子嵌入多比特空间"""
        X0 = PauliOperator({"X0": 1.0})
        m = _full_matrix(X0, 3)
        assert m.shape == (8, 8)
        # 嵌入后应仍为厄米
        assert np.allclose(m, m.conj().T)

    def test_full_matrix_no_pad_needed(self):
        """算子已覆盖全部比特时无需 padding"""
        H2 = hamiltonian.h2_hamiltonian()  # 2 qubits
        m = _full_matrix(H2, 2)
        assert m.shape == (4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
