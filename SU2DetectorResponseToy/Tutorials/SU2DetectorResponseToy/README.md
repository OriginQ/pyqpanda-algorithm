# SU(2) Detector Response Toy Simulation Tutorial

## 简介

本教程展示一个小规模 SU(2)-对称量子多体模拟示例。模型以 qubit 表示 fundamental color spin，通过 SU(2)-scalar interaction 构造 color-chain Hamiltonian，并输出 singlet/triplet sector spectroscopy、局域 quench response 与 detector connected color correlation。

该教程适合作为量子算法创新应用、量子多体模拟教学和 pyQPanda CPUQVM 线路示例。数值教程只依赖 NumPy 与 Matplotlib；pyQPanda 只用于 CPUQVM 线路示例。

## 赛事指定工具使用说明

本教程使用 pyQPanda CPUQVM 构造并运行 singlet-pair preparation 与 central color-quench probe 线路。数值扫描提供参考观测量，pyQPanda 示例提供可执行线路，OpenQASM 文件提供可迁移线路原型。

```text
NumPy reference scan -> OpenQASM circuit prototype -> pyQPanda CPUQVM execution
```

## 任务类型

```text
【创新应用】SU(2) detector-response toy simulation demo
```

## 流程概览

```text
SU(2)-symmetric color-chain Hamiltonian
-> singlet ground state
-> singlet/triplet sector spectrum
-> central color-singlet quench
-> spectral weight redistribution
-> detector connected color correlation
-> OpenQASM / pyQPanda CPUQVM prototype
```

## 文件说明

```text
Tutorials/SU2DetectorResponseToy/
├── README.md
├── DERIVATION.md
├── CONTRIBUTION_NOTE.md
├── PR_DESCRIPTION.md
├── run_tutorial.py
├── make_tutorial_plots.py
└── pyqpanda_cpuqvm_example.py
```

可复用代码位于：

```text
src/eec_toy/
```

## 运行教程扫描

在项目根目录执行：

```bash
python Tutorials/SU2DetectorResponseToy/run_tutorial.py --num-qubits 6 --points 17
```

默认输出到：

```text
Tutorials/SU2DetectorResponseToy/outputs/
```

示例终端输出：

```text
SU(2) detector-response toy simulation 完成
输出目录：Tutorials/SU2DetectorResponseToy/outputs
singlet-triplet gap 最小窗口：delta = -0.750000
detector slope 诊断窗口：delta = -0.562500
fidelity dip 诊断窗口：delta = -0.187500
detector color correlation = -0.073882
```

主要输出：

```text
su2_spectrum_scan.csv
summary.json
singlet_triplet_gap_chart.txt
detector_slope_chart.txt
fidelity_loss_chart.txt
state_prep_singlet_pairs.qasm
color_quench_probe.qasm
```

## 生成图表

```bash
python Tutorials/SU2DetectorResponseToy/make_tutorial_plots.py
```

图表输出到：

```text
Tutorials/SU2DetectorResponseToy/outputs/figures/
```

包括：

```text
singlet_triplet_gap.png
detector_color_correlation.png
fidelity_loss.png
quench_delta_energy.png
summary_panels.png
```

这些图分别对应 sector gap、detector connected correlation、fidelity loss、quench energy injection 和汇总诊断面板。

## 数值诊断量与线路对应关系

| 数值或线路对象 | 教程作用 | 主要输出文件 | pyQPanda / OpenQASM 对应关系 |
|---|---|---|---|
| singlet-pair state preparation | 构造 SU(2) singlet 初态 | `state_prep_singlet_pairs.qasm` | pyQPanda CPUQVM 示例中的 pair preparation 线路 |
| singlet/triplet sector energies | 展示 sector-resolved spectroscopy | `su2_spectrum_scan.csv`, `singlet_triplet_gap_chart.txt` | 为后续 VQE 或谱测量线路提供参考量 |
| central color-singlet quench response | 展示中心 bond 局域扰动后的能量注入 | `summary.json`, `quench_delta_energy.png` | `color_quench_probe.qasm` 与 CPUQVM local probe circuit |
| detector connected color correlation | 展示左右 detector sites 的 connected response | `detector_slope_chart.txt`, `detector_color_correlation.png` | 给出后续 measurement circuit 的目标观测量 |
| ground-state fidelity loss | 展示参数扫描下基态结构变化 | `fidelity_loss_chart.txt`, `fidelity_loss.png` | 可用于验证参数化线路扫描的一致性 |

## pyQPanda CPUQVM 示例

```bash
python Tutorials/SU2DetectorResponseToy/pyqpanda_cpuqvm_example.py --num-qubits 6 --shots 1000
```

该脚本展示 singlet-pair preparation 与 central color-quench probe 的 pyQPanda 线路构造方式。运行后会输出 CPUQVM bitstring counts，并保存到：

```text
Tutorials/SU2DetectorResponseToy/outputs/pyqpanda_counts.json
```

示例终端输出：

```text
pyQPanda CPUQVM SU(2) singlet-pair demo 完成
输出文件：Tutorials/SU2DetectorResponseToy/outputs/pyqpanda_counts.json
非零 bitstring 数：...
```

## 理论说明

见：

```text
DERIVATION.md
```

该文档说明：

- SU(2)-scalar interaction 的构造。
- singlet/triplet sector 的识别方式。
- color-singlet quench 的线路含义。
- detector connected color correlation 的计算方式。
- 小规模量子模拟示例中的建模与测量难点。

## 示例输出摘要

默认 6-qubit 示例会输出：

```text
minimum singlet-triplet gap position
crossover window by detector-correlation slope
crossover window by fidelity loss
quench energy injection
connected detector color correlation
```

这些量用于展示有限尺寸 SU(2) color-chain 中的谱结构与关联结构变化。
