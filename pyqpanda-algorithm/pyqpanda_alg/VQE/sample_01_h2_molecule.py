"""
示例 1：用 VQE 求解 H₂ 分子基态能量

场景：量子化学中最经典的入门问题 —— 求氢分子 (H₂) 的基态能量。
     已知 H₂ 在 STO-3G 基组、键长 0.735 Å 时的精确基态能量为 -1.857 Ha。
     我们用 VQE 在量子模拟器上逼近这个值。

运行方式：
$ source .venv/bin/activate
$ python3 sample_01_h2_molecule.py

Author: Bai
"""

import numpy as np
from pyqpanda_alg.VQE import VQESolver, PauliHamiltonian


def build_h2_hamiltonian():
    """
    构建 H₂ 分子的哈密顿量（STO-3G 基组，键长 0.735 Å）。
    该哈密顿量已经过 Jordan-Wigner 变换，映射为 2 量子比特 Pauli 算符。
    """
    H = PauliHamiltonian()
    H.add_term(-1.0523, "II")  # 常数项（核排斥能 + 电子贡献）
    H.add_term(0.3979, "IZ")   # 单体项
    H.add_term(-0.3979, "ZI")  # 单体项
    H.add_term(-0.0112, "ZZ")  # 双体项（电子-电子相互作用）
    H.add_term(0.1809, "XX")   # 双体项（电子跃迁）
    return H


def main():
    # 第一步：构建哈密顿量
    H = build_h2_hamiltonian()
    print("=== H₂ 分子 VQE 求解 ===")
    print(f"哈密顿量: {H}")
    print(f"量子比特数: {H.num_qubits}")
    print(f"Pauli 项数: {H.num_terms}")
    print()

    # 第二步：精确对角化（用作对比基准）
    exact_energy = H.exact_ground_energy()
    print(f"精确基态能量（numpy对角化）: {exact_energy:.6f} Ha")
    print()

    # 第三步：VQE 求解
    solver = VQESolver(
        hamiltonian=H,
        ansatz='hea',         # Hardware Efficient Ansatz
        num_layers=2,         # 2 层参数化线路
        optimizer='COBYLA',   # 无梯度优化器
        shots=4096,           # 测量次数
        maxiter=200,          # 最大迭代次数
    )
    print(f"求解器配置: {solver}")
    print("正在优化...")

    result = solver.run()

    # 第四步：输出结果
    print()
    print("=== 结果 ===")
    print(f"VQE 基态能量:  {result['energy']:.6f} Ha")
    print(f"精确基态能量:  {exact_energy:.6f} Ha")
    print(f"绝对误差:      {abs(result['energy'] - exact_energy):.6f} Ha")
    print(f"迭代次数:      {result['num_iterations']}")
    print(f"优化器收敛:    {result['success']}")
    print()

    # 第五步：展示能量收敛趋势（取前20步）
    history = result['history']
    print("=== 能量收敛过程（前 20 步）===")
    for i, e in enumerate(history[:20]):
        bar = "█" * max(1, int((e - exact_energy) * 50))
        print(f"  Step {i+1:3d}: {e:+.4f}  {bar}")


if __name__ == "__main__":
    main()
