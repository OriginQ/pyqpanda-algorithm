# Contribution Note: SU(2) Detector Response Toy Simulation

## PR 标题

```text
【创新应用】SU(2) detector-response toy simulation demo
```

## 提交类型

```text
创新应用 / quantum simulation example
```

## 项目概述

本贡献提供一个小规模 SU(2)-对称量子多体模拟示例。项目使用 color-chain toy Hamiltonian 展示 SU(2)-scalar interaction、singlet/triplet sector spectroscopy、局域 color-singlet quench 和 detector connected color correlation。

该示例面向教学、算法展示和 pyQPanda 线路迁移，默认规模为 4 到 8 个 qubit，可在本地快速运行。

贡献定位为教程型创新应用：数值扫描部分提供参考观测量，pyQPanda CPUQVM 示例提供对应线路构造入口。

## 主要内容

1. `Tutorials/SU2DetectorResponseToy/README.md`
   - 教程入口与运行方式。

2. `Tutorials/SU2DetectorResponseToy/DERIVATION.md`
   - SU(2) 模型原理、sector 诊断和 detector correlation 说明。

3. `Tutorials/SU2DetectorResponseToy/run_tutorial.py`
   - 运行 SU(2) spectrum / detector correlation 扫描。

4. `Tutorials/SU2DetectorResponseToy/make_tutorial_plots.py`
   - 根据扫描结果生成图表。

5. `Tutorials/SU2DetectorResponseToy/pyqpanda_cpuqvm_example.py`
   - pyQPanda CPUQVM 线路示例。

6. `src/eec_toy/`
   - 可复用 SU(2) toy simulation 模块。

## 模型形式

Hamiltonian 由 SU(2)-scalar interaction 构成：

$$
S_i \cdot S_j = \frac{X_i X_j + Y_i Y_j + Z_i Z_j}{4}
$$

扫描参数 $\delta$：

$$
\begin{aligned}
H(\delta)
&= \sum_i J\left[1 + \delta(-1)^i\right] S_i \cdot S_{i+1} \\
&\quad + \kappa \sum_{i<j} e^{-|i-j|/\xi} S_i \cdot S_j .
\end{aligned}
$$

核心输出包括：

- singlet/triplet sector energies。
- singlet-triplet gap。
- ground-state fidelity loss。
- central color-singlet quench response。
- left-right detector connected color correlation。

## 创新点

- 使用 SU(2)-对称 toy model 展示 color-sector 诊断。
- 将谱结构、局域 quench 和 detector correlation 放在同一个小规模示例中。
- 同时提供数值扫描、图表、OpenQASM 线路和 pyQPanda CPUQVM 示例。
- 示例规模小，便于复现和教学。

## 本地验证

```bash
python Tutorials/SU2DetectorResponseToy/run_tutorial.py --num-qubits 6 --points 17
python Tutorials/SU2DetectorResponseToy/make_tutorial_plots.py
python -c "import sys; sys.path.insert(0, 'tests'); import test_eec_toy as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]; print('tests ok')"
```
