# VQE 模块 — 变分量子本征求解器

> Author: Bai

---

## 重要：请先安装缺失的依赖

项目 `requirements.txt` **未包含**所有运行时依赖，使用前请先执行：

```bash
# 系统库（pyqpanda3 的 C++ 后端需要 OpenMP 支持）
# sudo apt-get install -y libgomp1      # Ubuntu/Debian
# sudo yum install -y libgomp           # Amazon Linux / RHEL

# requirements.txt 中遗漏但其他模块需要的 Python 包
pip install pandas scikit-learn
```

---

## 算法原理

### 目标

VQE 的目标是：**求一个哈密顿量 H 的最小本征值（基态能量）**。

最经典的应用场景：求解分子的基态能量。

数学表述：E0 = minθ<ψ(θ)|H|ψ(θ)>


其中：
- H是系统哈密顿量（一个 Hermitian 矩阵）
- ∣ψ(θ)⟩是参数化量子态（由量子线路产生）
- θ是待优化的经典参数

### 核心思想：变分原理

**变分原理**保证：对于任意试探态∣ψ⟩，其能量期望值一定 ≥ 基态能量：⟨ψ∣H∣ψ⟩≥E0
​

所以我们只需要不断调整θ，使得能量期望值越来越低，最终逼近E0

### 算法整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                    VQE 混合量子-经典循环                    │
│                                                             │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │ 经典优化器 │────▶│ 参数化量子线路 │────▶│ 测量期望值     │
│  │ (更新 θ)  │◀────│  (Ansatz)    │     │ ⟨ψ(θ)|H|ψ(θ)⟩│     │
│  └──────────┘     └──────────────┘     └──────────────┘     │
│       ▲                                        │            │
│       │            能量值 E(θ)                  │           │
│       └────────────────────────────────────────┘            │
│                                                             │
│  收敛条件: |E(θ_new) - E(θ_old)| < ε                        │
└─────────────────────────────────────────────────────────────┘
```

**四大核心组件：**

| 组件 | 作用 | 对应文件 |
|------|------|----------|
| 哈密顿量   | 将物理系统表示为 Pauli 算符的加权和 | hamiltonian.py |
| 参数化线路 | 生成试探量子态                      | ansatz.py      |
| 期望值测量 | 在量子硬件上计算                    | measurement.py |
| 经典优化器 | 调整参数使能量最小                  | vqe_solver.py  |


### 参数化线路 (Ansatz)

本模块提供两种 Ansatz：

**Hardware Efficient Ansatz (HEA)**：通用、硬件友好
```
每层结构：RY旋转 → RZ旋转 → CNOT纠缠链
参数数量 = 量子比特数 × 2 × 层数
```

**RY-Linear Ansatz**：更简单，仅用 RY 门
```
每层结构：RY旋转 → CNOT纠缠链
参数数量 = 量子比特数 × 层数
```

## 环境要求

- Python 3.11 / 3.12 / 3.13
- pyqpanda3 >= 0.3.5
- numpy, scipy, matplotlib, sympy

### 哈密顿量功能测试（不依赖量子模拟器）

```bash
python3 -c "
from pyqpanda_alg.VQE import PauliHamiltonian
H = PauliHamiltonian()
H.add_term(-1.0523, 'II')
H.add_term(0.3979, 'IZ')
H.add_term(-0.3979, 'ZI')
H.add_term(-0.0112, 'ZZ')
H.add_term(0.1809, 'XX')
print(H)
print(f'量子比特数: {H.num_qubits}')
print(f'精确基态能量: {H.exact_ground_energy():.6f}')
"
```

期望输出：
```
-1.0523 * II +0.3979 * IZ -0.3979 * ZI -0.0112 * ZZ +0.1809 * XX
量子比特数: 2
精确基态能量: -1.857202
```

### 完整 VQE 求解测试（需要 pyqpanda3 量子模拟器）

```bash
python3 -c "
from pyqpanda_alg.VQE import VQESolver, PauliHamiltonian

H = PauliHamiltonian()
H.add_term(-1.0523, 'II')
H.add_term(0.3979, 'IZ')
H.add_term(-0.3979, 'ZI')
H.add_term(-0.0112, 'ZZ')
H.add_term(0.1809, 'XX')

solver = VQESolver(H, ansatz='hea', num_layers=2, optimizer='COBYLA', shots=4096)
print(solver)
result = solver.run()

print(f'VQE 能量:  {result[\"energy\"]:.6f}')
print(f'精确能量:  {H.exact_ground_energy():.6f}')
print(f'误差:      {abs(result[\"energy\"] - H.exact_ground_energy()):.6f}')
print(f'迭代次数:  {result[\"num_iterations\"]}')
print(f'是否收敛:  {result[\"success\"]}')
"
```

期望输出：
```
VQESolver(qubits=2, ansatz='hea', layers=2, optimizer='COBYLA', shots=4096)
VQE 能量:  -1.857202
精确能量:  -1.857202
误差:      0.000000
迭代次数:  200
是否收敛:  False
```

> 说明：`是否收敛: False` 表示 COBYLA 跑满了 200 次迭代上限后停止。实际能量已到达全局最优——优化器很早就找到了最优点，后续迭代是在微调已经足够好的参数。这是正常现象。

