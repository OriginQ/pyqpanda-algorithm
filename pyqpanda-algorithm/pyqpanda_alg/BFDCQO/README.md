# PyQPanda 的 PyBF-DCQO 算法族

`pyqpanda_alg.BFDCQO` 提供偏置场数字化反绝热量子优化（Bias-Field Digitized Counterdiabatic Quantum Optimization，BF-DCQO）算法族的模块化参考实现。模块覆盖 QUBO、原生多项式 Ising/HUBO、有限采样反馈与受预算的经典局部 refinement，并以可复现的小规模实例用于算法验证和研究 benchmark。

## 功能概览

| 能力 | 主要 API | 说明 |
|---|---|---|
| 基础 BF-DCQO | `BFDCQOConfig(variant="basic")` | 基础偏置场反馈协议 |
| CVaR 反馈 | `variant="cvar"` | 使用低能样本分位更新偏置场 |
| 自适应采样 | `variant="adaptive"` | 在固定总 shot 预算内按轮次分配样本 |
| 自适应采样 + CVaR | `variant="adaptive_cvar"` | 组合预算分配与低能样本反馈 |
| 受预算混合求解 | `variant="hybrid"` | 量子阶段后执行置信度引导或随机块局部搜索 |
| QUBO / 二次 Ising | `IsingProblem`、`MaxCutProblem`、`MWISProblem` | 二次对角目标函数 |
| 原生 HUBO | `PolynomialIsingProblem` | 任意阶对角 Pauli-Z 目标函数 |
| MAX-3-SAT | `Max3SATProblem` | 原生三次 HUBO，也可作 Rosenberg QUBO 约化 |
| 小规模精确参照 | `ExactSolver` | 为回归测试和诊断提供精确基态 |
| QAOA 对照 | `QAOASolver` | BFDCQO 模块内透明、可检查的 p-layer reference 实现；不是仓库上游公开 QAOA |

## 统一 API

推荐通过 `BFDCQOSolver` 与 `BFDCQOConfig` 选择算法变体；研究实验仍可直接使用公开的底层求解器类。下面示例可直接运行，`backend="numpy"` 用于确定性的参考验证；安装并配置 PyQPanda3 后可改为 `backend="pyqpanda"`。

```python
from pyqpanda_alg.BFDCQO import BFDCQOConfig, BFDCQOSolver, MaxCutProblem

problem = MaxCutProblem.from_edges(
    5,
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
).to_ising()

config = BFDCQOConfig(
    variant="adaptive_cvar",
    iterations=6,
    shots=512,
    total_shot_budget=3072,
    cvar_alpha=0.10,
)
result = BFDCQOSolver(problem, config, backend="numpy", seed=7).solve()

print(result.best_solution, result.best_energy, result.total_shots)
```

## 原生 HUBO

高阶对角目标直接表示为

$$
H = c + \sum_S J_S \prod_{i\in S} Z_i.
$$

无需先引入辅助变量把高阶项降为二次项：

```python
from pyqpanda_alg.BFDCQO import (
    BFDCQOConfig,
    BFDCQOSolver,
    PolynomialIsingProblem,
)

problem = PolynomialIsingProblem(
    n=4,
    terms={
        (0,): 0.3,
        (0, 1): -0.8,
        (0, 1, 3): 1.2,
    },
    offset=0.1,
)
result = BFDCQOSolver(
    problem,
    BFDCQOConfig(variant="basic", iterations=4, shots=256),
    backend="numpy",
    seed=7,
).solve()

print(result.best_solution, result.best_energy)
```

一阶嵌套对易子绝热规范势（adiabatic gauge potential）由稀疏 Pauli 代数构造，因此同一路径同时支持二体与高阶项。PyQPanda3 后端通过基变换和 parity CNOT ladder 分解任意 Pauli rotation。

## 求解器变体

### 基础 BF-DCQO

每轮从带偏置的乘积态基态出发，执行数字化 counterdiabatic 演化并测量输出分布，再以所得 $\langle Z_i\rangle$ 更新下一轮纵向偏置。

### CVaR BF-DCQO

仅用最低能量的 `cvar_alpha` 比例样本质量更新偏置。若截断点穿过简并能级，实现会按比例纳入该能级，避免由 bitstring 排序引入人为偏差。

### 自适应采样 BF-DCQO

预先限定 `total_shot_budget`，根据平均偏置置信度为后续轮次分配采样精度，同时为剩余每轮保留最小配额。可选提前停止同时检查不确定度和偏置变化。

### Hybrid BF-DCQO

量子阶段结束后，以 $c_i=|\langle Z_i\rangle|$ 衡量变量置信度，并在最低置信度变量组成的小块邻域内进行精确局部搜索。结果对象显式记录经典局部评估次数，可在相同预算下与随机块消融对照。

## 应用与 benchmark

仓库示例和 benchmark 包括：

- 加权 MaxCut（QUBO）；
- 加权最大独立集 MWIS（QUBO）；
- 加权 MAX-3-SAT（原生三次 HUBO）；
- 原生 HUBO 与经 quadratization 后 QAOA 的资源及质量对照；
- 基础、CVaR、自适应采样与 Hybrid 变体在固定小规模实例上的对照。

这些内容将算法实现与可复现实验分开组织，benchmark 结果用于研究比较，不构成对大规模硬件表现的外推。

模块内公开的 `QAOASolver` 是用于逐步核对算法和预算的透明 reference 实现，并非仓库已有的上游 `pyqpanda_alg.QAOA`。正式 CPUQVM 证据通过以下 benchmark 适配器直接调用仓库公开 QAOA：

```python
from pyqpanda_alg.BFDCQO.benchmark import UpstreamQAOASolver
```

官方表格将该路径标为 `pyqpanda_alg.QAOA.CPUQVM`，并按实际观测的 `circuit_evaluations × shots` 记录总采样数；两条 QAOA 路径不会混作同一个实现。

## 正确性验证

测试覆盖以下性质：

- 对小规模 fixture 的每个计算基态核对 QUBO/HUBO 能量保持；
- 稀疏 Pauli 对易子与稠密矩阵计算一致；
- Pauli rotation 与 `scipy.linalg.expm` 一致；
- 带偏置乘积态制备与 shot 预算守恒；
- CVaR 简并截断处理；
- 原生 HUBO MAX-3-SAT 编码与 Rosenberg quadratization；
- 相同预算下置信度引导与随机局部搜索的比较；
- 在 PyQPanda3 可用时核对 NumPy 与 PyQPanda3 输出分布。

## 适用范围与限制

当前实现定位为**小规模算法验证、教学与 benchmark 的参考实现**。精确基态诊断以及部分指标会枚举完整计算基，NumPy 参考后端还会维护完整状态向量，因此其时间和内存开销随问题规模快速增长。

在完成 count-only 执行路径，并把精确基态诊断改为可选项之前，本项目不声称能够沿当前代码路径直接扩展到大规模量子硬件。硬件实验还应单独评估设备拓扑、编译、噪声、采样预算和执行服务约束。

## 参考文献与归属

本实现依据公开论文中的算法和公式独立编写，不包含第三方专有实现代码。归属边界如下：

- Basic BF-DCQO 的核心协议归于 A. Gomez Cadavid 等人的工作。
- native HUBO、CVaR BF-DCQO 与 MAX-3-SAT 构造归于 S. V. Romero 等人的工作。
- MWIS 已经是 A. Gomez Cadavid 等人原始 BF-DCQO 工作中的应用；本项目仅将其作为 QUBO 验证问题，不主张首次将 BF-DCQO 应用于 MWIS。
- 经典 refinement 的先行概念可参见 Hybrid Sequential Quantum Computing（HSQC）；本实现提供的是受预算、可记录评估次数的工程化对照路径。

参考文献：

1. A. Gomez Cadavid *et al.*, **Bias-field digitized counterdiabatic quantum optimization**, *Physical Review Research* 7, L022010 (2025), arXiv:2405.13898.
2. S. V. Romero *et al.*, **Bias-Field Digitized Counterdiabatic Quantum Algorithm for Higher-Order Binary Optimization**, *Communications Physics* 8, 348 (2025), arXiv:2409.04477.
3. P. Chandarana *et al.*, **Hybrid Sequential Quantum Computing**, arXiv:2510.05851 (2025).
