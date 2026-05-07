# SU(2) Color-Chain Quantum Simulation Tutorial

## 项目简介

本项目是一个面向量子算法创新应用的 SU(2) 对称量子多体模拟示例。项目使用小规模 color-chain toy Hamiltonian，展示 SU(2) singlet 态制备、singlet/triplet sector 能谱分析、局域 color-singlet quench 以及左右探测点的 connected color correlation。

该示例默认使用 4 到 8 个 qubit，可在本地完成精确对角化扫描、图表生成和基础线路导出；同时提供 pyQPanda CPUQVM 示例，便于学习和迁移到 pyQPanda 生态。

本项目定位为一个自包含教程型贡献：NumPy 精确对角化部分提供可复现实验参考量，pyQPanda CPUQVM 脚本展示对应的 singlet 态制备与局域 probe 线路构造方式。

## 任务类型

```text
【创新应用】SU(2) color-chain quantum simulation tutorial
```

## 主要功能

- 构造 SU(2)-scalar color interaction：$S_i \cdot S_j$。
- 扫描 dimerization 参数下的低能谱结构。
- 计算 total spin sector，用于区分 singlet 与 triplet 能级。
- 对中心 bond 施加 color-singlet local quench。
- 计算 quench energy injection 与 low-energy spectral weights。
- 计算左右 probe sites 的 connected color correlation。
- 导出 singlet-pair state preparation 与 color-quench probe 的 OpenQASM 原型。
- 提供 pyQPanda CPUQVM 示例脚本。

## 目录结构

```text
SU2ColorChainTutorial/
├── README.md
├── SUBMISSION_CHECKLIST.md
├── contest_pitch.txt
├── requirements.txt
├── run_demo.py
├── make_plots.py
├── pyqpanda_demo.py
├── src/su2_toy/
├── tests/test_su2_toy.py
└── Tutorials/SU2ColorChainTutorial/
    ├── README.md
    ├── DERIVATION.md
    ├── CONTRIBUTION_NOTE.md
    ├── PR_DESCRIPTION.md
    ├── run_tutorial.py
    ├── make_tutorial_plots.py
    └── pyqpanda_cpuqvm_example.py
```

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：

```text
numpy
matplotlib
pyqpanda
```

其中 `pyqpanda` 用于 CPUQVM 示例；如果只运行数值扫描和画图，`numpy` 与 `matplotlib` 即可。

## 赛事指定工具使用说明

本项目使用 pyQPanda CPUQVM 构造并运行 singlet-pair preparation 与 central color-quench probe 线路。NumPy 精确对角化部分用于生成参考观测量，帮助解释和验证量子线路示例的物理含义；pyQPanda CPUQVM 部分用于展示可执行的量子编程入口；OpenQASM 导出用于连接线路原型与后续模拟器或硬件迁移。

因此，本项目的设计关系为：

```text
NumPy exact diagonalization -> reference observables
OpenQASM export             -> portable circuit prototype
pyQPanda CPUQVM             -> executable circuit realization
```

## 运行示例

### 1. 运行 SU(2) 扫描

```bash
python run_demo.py --num-qubits 6 --points 17 --out outputs_su2_n6
```

示例终端输出如下：

```text
SU(2) color-chain quantum simulation 完成
输出目录：outputs_su2_n6
singlet-triplet gap 最小窗口：delta = -0.750000
correlation slope 诊断窗口：delta = -0.562500
fidelity dip 诊断窗口：delta = -0.187500
probe-site color correlation = -0.073882
```

### 2. 生成图表

```bash
python make_plots.py --input outputs_su2_n6/summary.json --out outputs_su2_n6/figures
```

### 3. 运行 tutorial 入口

```bash
python Tutorials/SU2ColorChainTutorial/run_tutorial.py --num-qubits 6 --points 17
python Tutorials/SU2ColorChainTutorial/make_tutorial_plots.py
```

### 4. 运行 pyQPanda CPUQVM 示例

```bash
python pyqpanda_demo.py --num-qubits 6 --shots 1000
```

或：

```bash
python Tutorials/SU2ColorChainTutorial/pyqpanda_cpuqvm_example.py --num-qubits 6 --shots 1000
```

运行后会输出 bitstring counts，并保存为 JSON 文件。示例终端输出如下：

```text
pyQPanda CPUQVM SU(2) singlet-pair demo 完成
输出文件：outputs_pyqpanda/counts.json
非零 bitstring 数：...
```

## 输出文件

运行扫描后会生成：

```text
su2_spectrum_scan.csv
summary.json
singlet_triplet_gap_chart.txt
correlation_slope_chart.txt
fidelity_loss_chart.txt
state_prep_singlet_pairs.qasm
color_quench_probe.qasm
```

运行画图脚本后会生成：

```text
singlet_triplet_gap.png
probe_color_correlation.png
fidelity_loss.png
quench_delta_energy.png
summary_panels.png
```

图表含义：

- `singlet_triplet_gap.png`：展示最低 singlet 与 triplet sector 的能量差随参数变化的趋势。
- `probe_color_correlation.png`：展示左右 probe sites 的 connected color correlation。
- `fidelity_loss.png`：展示相邻参数点之间 ground-state overlap 的变化。
- `quench_delta_energy.png`：展示 central color-singlet quench 后的能量注入。
- `summary_panels.png`：汇总主要诊断量，便于快速检查教程输出。

## 数值诊断量与线路对应关系

数值扫描用于生成参考观测量，pyQPanda / OpenQASM 部分用于展示可迁移的线路构造入口。

| 数值或线路对象 | 物理含义 | 主要输出文件 | pyQPanda / OpenQASM 对应关系 |
|---|---|---|---|
| singlet-pair state preparation | SU(2) singlet 初态构造 | `state_prep_singlet_pairs.qasm` | pyQPanda 示例中的 pair preparation 线路 |
| singlet/triplet sector energies | sector-resolved spectroscopy | `su2_spectrum_scan.csv`, `singlet_triplet_gap_chart.txt` | 作为后续 VQE 或谱测量线路的 reference observable |
| central color-singlet quench energy injection | 中心 bond 上的 SU(2)-scalar 局域扰动响应 | `summary.json`, `quench_delta_energy.png` | `color_quench_probe.qasm` 与 CPUQVM local probe circuit |
| connected color correlation between two probe sites | 左右 probe sites 的 connected color correlation | `correlation_slope_chart.txt`, `probe_color_correlation.png` | 给出后续 measurement circuit 的目标观测量 |
| ground-state fidelity loss | 参数扫描中基态结构变化的参考诊断 | `fidelity_loss_chart.txt`, `fidelity_loss.png` | 可用于验证参数化线路扫描的一致性 |

## 数学模型概述

每个 qubit 表示一个 fundamental SU(2) color spin。两体相互作用取 SU(2)-scalar 形式：

$$
S_i \cdot S_j = \frac{X_i X_j + Y_i Y_j + Z_i Z_j}{4}
$$

模型 Hamiltonian 为：

$$
\begin{aligned}
H(\delta)
&= \sum_i J\left[1 + \delta(-1)^i\right] S_i \cdot S_{i+1} \\
&\quad + \kappa \sum_{i<j} e^{-|i-j|/\xi} S_i \cdot S_j .
\end{aligned}
$$

其中 $\delta$ 控制 dimerization，$\kappa$ 和 $\xi$ 控制长程 color interaction 的强度与衰减。

## 项目亮点

- 物理结构保留 SU(2) 对称性。
- 输出 singlet/triplet sector resolved spectroscopy。
- 以 connected color correlation 作为 color-chain 诊断量。
- 示例规模小，便于教学、复现实验和算法展示。
- 同时包含数值模拟、图表生成、OpenQASM 导出和 pyQPanda CPUQVM 示例。

## 验证方式

```bash
python -m py_compile run_demo.py make_plots.py pyqpanda_demo.py
python Tutorials/SU2ColorChainTutorial/run_tutorial.py --num-qubits 6 --points 17
python Tutorials/SU2ColorChainTutorial/make_tutorial_plots.py
python -c "import sys; sys.path.insert(0, 'tests'); import test_su2_toy as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]; print('tests ok')"
```
