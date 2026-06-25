# LindbladMagnus — 开量子系统的变分量子模拟

本模块实现了论文

> J.-C. Huang, H.-E. Li, Y.-C. Wang, G.-Z. Zhang, J. Li, H.-S. Hu,  
> *Towards Robust Variational Quantum Simulation of Lindblad Dynamics via
> Stochastic Magnus Expansion*, **PRX Quantum** 6, 040312 (2025).
> [arXiv:2503.22099](https://arxiv.org/abs/2503.22099)

中的算法，将 **Lindblad 主方程** 通过量子态扩散（QSD）随机轨迹
unravelling 后，用 **随机 Magnus 展开（Scheme I–IV）** 构造每一步的非厄米
有效哈密顿量 `H_eff`，再用 **McLachlan 变分原理** 在参数化量子线路上完成
一步变分时间演化。所有量子电路构造与态矢量模拟均基于 `pyqpanda3`，数值
计算仅依赖 `numpy` / `scipy`，**不引入 qiskit / qutip 等其它量子框架**。

## 快速上手

```python
import numpy as np
from pyqpanda_alg.LindbladMagnus import (
    LindbladMagnusSolver, HardwareEfficientAnsatz,
    tfim_model, mesolve,
)

# 1. 准备开量子系统模型（这里用阻尼 TFIM 作为示例）
H, c_ops, e_ops, psi0, labels = tfim_model()

# 2. 构造变分 ansatz（theta=0 时为恒等，自动保留初态）
ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=2, init_state=psi0)

# 3. 构造求解器并运行多条轨迹
solver = LindbladMagnusSolver(H, c_ops, ansatz,
                              qsd_type="nonlinear",
                              magnus_order=1,     # 0=EM, 1..4=Magnus 阶数
                              integrator="rk4")
times = np.linspace(0, 2.5, 51)
result = solver.solve(psi0, times, e_ops,
                      traj_num=30, seed=42, verbose="tqdm")

# 4. 与精确 Liouvillian 解对比
exact = mesolve(H, psi0, times, c_ops, e_ops)
print(f"最大误差: {np.abs(result.expect - exact).max():.4f}")
```

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `magnus.py` | 随机 Magnus 积分器（Scheme I–IV）与 Euler-Maruyama，生成 `H_eff` |
| `ansatz.py` | `VariationalAnsatz` 与 `HardwareEfficientAnsatz`（参数化 RX/RY/RZ + RZZ/RXX/RYY） |
| `variational.py` | 面向非厄米 `H_eff` 的 McLachlan 变分原理 + Euler / RK4 时间积分 |
| `lindblad.py` | `LindbladMagnusSolver` 顶层求解器、`LindbladResult` 结果容器、函数式 `solve` |
| `models.py` | FMO / TFIM / RPM 物理模型 + 自研 `mesolve` 精确解（Liouvillian） |

## 关键 API

- `LindbladMagnusSolver(...)` — 构造一次，多次调用 `.solve(...)` 复用配置
- `LindbladResult` — dataclass，包含 `expect` / `std` / `norms` / `solver_info` / `seeds`
- `solve(H, c_ops, ansatz, psi0, tlist, e_ops, ...)` — 一次性函数式接口
- `mesolve(H, psi0, tlist, c_ops, e_ops)` — 基于 Liouvillian 的精确解（替代 `qutip.mesolve`）
- `HardwareEfficientAnsatz(n_qubits, layers, init_state=...)` — 默认推荐 ansatz

## 可选依赖

- `tqdm` — 启用 `verbose="tqdm"` 进度条；未安装时自动回退到 `verbose=False`
- `matplotlib` — 示例脚本绘图

## 应用案例

以下路径均相对仓库根目录：

- `pyqpanda-algorithm/example/QAlgBase/testeg_Lindblad_TFIM.py` — TFIM 阻尼模型端到端 demo
- `pyqpanda-algorithm/example/QAlgBase/testeg_Lindblad_FMO.py` — FMO 光合复合体能量传输 demo
- `pyqpanda-algorithm/test/11-LindbladMagnus/demo01-LindbladMagnus-TFIM_FMO.ipynb` — notebook 教程
- `test/LindbladMagnus/` — 单元测试与回归测试（`pytest`，已在 `test/pytest.ini` 注册）

## 参数调优建议

- **时间步长 `dt`**：建议从 `dt ≈ 0.05–0.1 × 2π/‖H‖` 起步；过大会导致 McLachlan
  最小二乘方程病态，表现为 `Ground` / `Sink` 等通道人口过冲。
- **Ansatz 层数 `layers`**：2–3 层对 TFIM 类问题足够；FMO 等多体系统建议 3+ 层
  或换成 Hamiltonian Variational Ansatz。
- **轨迹数 `traj_num`**：nonlinear QSD 下 20–50 条通常已收敛；linear QSD 下
  因方差更大建议 ≥ 100 条。
- **正则化 `eps`**：若日志中出现 `LinAlgError`，将 `eps` 调大到 `1e-8` 通常即可。
