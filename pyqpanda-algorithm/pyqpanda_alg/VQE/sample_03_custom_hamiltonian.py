"""
示例 3：自定义哈密顿量 —— 求解 2-qubit 反铁磁 Heisenberg 模型基态

场景：凝聚态物理中的经典问题。Heisenberg 模型描述量子自旋链的相互作用，
     其哈密顿量为：
         H = J * (XX + YY + ZZ)
     其中 J > 0 表示反铁磁耦合。

     基态是自旋单态 (singlet state): |ψ⟩ = (|01⟩ - |10⟩) / √2
     对应基态能量 E₀ = -3J（精确解可验证）。

     本示例展示如何用 VQE 处理任意自定义哈密顿量。

运行方式：
$ source .venv/bin/activate
$ python3 sample_03_custom_hamiltonian.py

Author: Bai
"""

import numpy as np
from pyqpanda_alg.VQE import VQESolver, PauliHamiltonian


def build_heisenberg_hamiltonian(J=1.0):
    """
    构建 2-qubit 反铁磁 Heisenberg 模型哈密顿量。

    H = J * (X₁X₂ + Y₁Y₂ + Z₁Z₂)

    参数：
        J : float — 耦合强度（J>0 反铁磁，J<0 铁磁）

    精确基态能量：E₀ = -3J
    """
    H = PauliHamiltonian()
    H.add_term(J, "XX")  # X₁X₂ 相互作用
    H.add_term(J, "YY")  # Y₁Y₂ 相互作用
    H.add_term(J, "ZZ")  # Z₁Z₂ 相互作用
    return H


def build_heisenberg_with_field(J=1.0, h=0.5):
    """
    带外磁场的 Heisenberg 模型（更复杂的情况）。

    H = J * (XX + YY + ZZ) + h * (ZI + IZ)

    外磁场 h 破坏了自旋旋转对称性，基态不再是简单的 singlet。
    此时精确解需要数值对角化，正是 VQE 发挥作用的场景。
    """
    H = PauliHamiltonian()
    H.add_term(J, "XX")
    H.add_term(J, "YY")
    H.add_term(J, "ZZ")
    H.add_term(h, "ZI")  # 外磁场作用在第 1 个自旋
    H.add_term(h, "IZ")  # 外磁场作用在第 2 个自旋
    return H


def main():
    print("=== 示例 3: Heisenberg 自旋链 VQE 求解 ===")
    print()

    # ----- 实验 A：纯 Heisenberg 模型 -----
    J = 1.0
    H1 = build_heisenberg_hamiltonian(J)
    exact1 = H1.exact_ground_energy()

    print(f"[实验 A] 纯 Heisenberg 模型 (J={J})")
    print(f"  哈密顿量: H = {J}*(XX + YY + ZZ)")
    print(f"  理论基态能量: E₀ = -3J = {-3*J:.4f}")
    print(f"  numpy 对角化: E₀ = {exact1:.4f}")

    solver1 = VQESolver(H1, ansatz='hea', num_layers=2, optimizer='COBYLA', shots=4096)
    result1 = solver1.run()

    print(f"  VQE 结果:     E  = {result1['energy']:.4f}")
    print(f"  误差:         {abs(result1['energy'] - exact1):.6f}")
    print()

    # ----- 实验 B：带外磁场 -----
    h = 0.5
    H2 = build_heisenberg_with_field(J, h)
    exact2 = H2.exact_ground_energy()

    print(f"[实验 B] Heisenberg + 外磁场 (J={J}, h={h})")
    print(f"  哈密顿量: H = {J}*(XX + YY + ZZ) + {h}*(ZI + IZ)")
    print(f"  numpy 对角化: E₀ = {exact2:.6f}")

    solver2 = VQESolver(H2, ansatz='hea', num_layers=3, optimizer='COBYLA', shots=4096)
    result2 = solver2.run()

    print(f"  VQE 结果:     E  = {result2['energy']:.6f}")
    print(f"  误差:         {abs(result2['energy'] - exact2):.6f}")
    print()

    # ----- 实验 C：扫描耦合强度 J -----
    print("[实验 C] 扫描耦合强度 J，观察基态能量变化")
    print(f"  {'J':<6} {'精确 E₀':<12} {'VQE E':<12} {'误差'}")
    print("  " + "-" * 44)

    for J_val in [0.2, 0.5, 1.0, 1.5, 2.0]:
        H_scan = build_heisenberg_hamiltonian(J_val)
        exact_scan = H_scan.exact_ground_energy()

        np.random.seed(0)  # 固定种子保证可复现
        solver_scan = VQESolver(
            H_scan, ansatz='ry_linear', num_layers=2,
            optimizer='COBYLA', shots=4096, maxiter=100
        )
        r = solver_scan.run()
        error = abs(r['energy'] - exact_scan)
        print(f"  {J_val:<6.1f} {exact_scan:<12.4f} {r['energy']:<12.4f} {error:.6f}")

    print()
    print("结论：VQE 能准确追踪不同参数下的基态能量，")
    print("      展示了其作为通用量子求解器的灵活性。")


if __name__ == "__main__":
    main()
