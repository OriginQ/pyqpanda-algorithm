"""
示例 2：对比不同优化器和 Ansatz 对 VQE 收敛速度的影响

场景：同一个哈密顿量，用不同配置运行 VQE，观察：
     - 不同优化器（COBYLA vs Powell vs Nelder-Mead）的收敛差异
     - 不同 Ansatz（HEA vs RY-linear）的表达能力差异
     帮助学员理解"选择什么优化策略"对结果质量的影响。

运行方式：
$ source .venv/bin/activate
$ python3 sample_02_optimizer_comparison.py

Author: Bai
"""

import numpy as np
from pyqpanda_alg.VQE import VQESolver, PauliHamiltonian


def build_hamiltonian():
    """构建一个 2-qubit 测试哈密顿量（H₂ 简化模型）"""
    H = PauliHamiltonian()
    H.add_term(-1.0523, "II")
    H.add_term(0.3979, "IZ")
    H.add_term(-0.3979, "ZI")
    H.add_term(-0.0112, "ZZ")
    H.add_term(0.1809, "XX")
    return H


def run_experiment(H, ansatz, optimizer, num_layers=2):
    """运行一次 VQE 实验，返回结果摘要"""
    # 固定随机种子，确保初始参数一致（公平对比）
    np.random.seed(42)

    solver = VQESolver(
        hamiltonian=H,
        ansatz=ansatz,
        num_layers=num_layers,
        optimizer=optimizer,
        shots=4096,
        maxiter=150,
    )
    result = solver.run()
    return {
        'ansatz': ansatz,
        'optimizer': optimizer,
        'energy': result['energy'],
        'iterations': result['num_iterations'],
        'success': result['success'],
    }


def main():
    H = build_hamiltonian()
    exact = H.exact_ground_energy()
    print("=== VQE 优化器 & Ansatz 对比实验 ===")
    print(f"目标: H₂ 基态能量 = {exact:.6f} Ha")
    print()

    # 实验配置
    experiments = [
        ('hea', 'COBYLA'),
        ('hea', 'Powell'),
        ('hea', 'Nelder-Mead'),
        ('ry_linear', 'COBYLA'),
        ('ry_linear', 'Powell'),
    ]

    # 运行所有实验
    results = []
    for ansatz, optimizer in experiments:
        print(f"  运行中... ansatz={ansatz:10s} optimizer={optimizer:12s}", end="")
        r = run_experiment(H, ansatz, optimizer)
        error = abs(r['energy'] - exact)
        print(f"  -> E={r['energy']:+.6f}  误差={error:.6f}  迭代={r['iterations']}")
        r['error'] = error
        results.append(r)

    # 打印汇总表
    print()
    print("=" * 72)
    print(f"{'Ansatz':<12} {'优化器':<14} {'能量 (Ha)':<12} {'误差':<10} {'迭代次数'}")
    print("-" * 72)
    for r in results:
        print(f"{r['ansatz']:<12} {r['optimizer']:<14} {r['energy']:+.6f}    "
              f"{r['error']:.6f}    {r['iterations']}")
    print("=" * 72)
    print()

    # 找出最佳组合
    best = min(results, key=lambda x: x['error'])
    print(f"最佳组合: ansatz={best['ansatz']}, optimizer={best['optimizer']}")
    print(f"最低误差: {best['error']:.8f} Ha")
    print()
    print("结论：HEA 比 RY-linear 表达能力更强（参数多一倍），")
    print("      但 RY-linear 线路更浅，在真实量子硬件上噪声更小。")
    print("      优化器选择取决于具体问题——无免费午餐定理。")


if __name__ == "__main__":
    main()
