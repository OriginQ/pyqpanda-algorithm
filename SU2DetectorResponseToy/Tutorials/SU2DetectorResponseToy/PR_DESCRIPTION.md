# PR 正文草稿

## 标题

```text
【创新应用】SU(2) detector-response toy simulation demo
```

## 概要

本 PR 新增一个自包含的 SU(2)-对称量子模拟教程示例，并提供 pyQPanda CPUQVM 线路构造示例。

数值部分通过小规模 color-chain toy Hamiltonian 的精确对角化生成参考观测量；pyQPanda 示例使用 CPUQVM 构造并运行对应的 singlet 态制备与局域 probe 线路；OpenQASM 文件提供可迁移线路原型。该示例适合作为量子多体模拟、SU(2) sector 诊断和 pyQPanda CPUQVM 入门教程。

## 主要内容

- `Tutorials/SU2DetectorResponseToy/README.md`：教程入口与运行说明。
- `DERIVATION.md`：说明 SU(2)-scalar interaction、sector 诊断、局域 quench 与 connected correlation。
- `run_tutorial.py`：运行 4 到 8 个 qubit 的小规模有限尺寸扫描。
- `make_tutorial_plots.py`：根据扫描结果生成诊断图表。
- `pyqpanda_cpuqvm_example.py`：pyQPanda CPUQVM 线路原型，可运行并输出 bitstring counts。
- README 中补充赛事指定工具使用说明，以及数值诊断量、输出文件与 pyQPanda / OpenQASM 线路的对应关系。
- `src/eec_toy/`：教程使用的可复用 toy-model 工具。
- `tests/test_eec_toy.py`：基础正确性检查。

## 教程输出

教程会输出以下诊断量：

- singlet/triplet sector energies。
- singlet-triplet gap。
- ground-state fidelity loss。
- central color-singlet quench energy injection。
- left-right detector connected color correlation。
- 态制备与局域 probe 线路的 OpenQASM 原型。

## 运行方式

```bash
python Tutorials/SU2DetectorResponseToy/run_tutorial.py --num-qubits 6 --points 17
python Tutorials/SU2DetectorResponseToy/make_tutorial_plots.py
```

可选 pyQPanda CPUQVM 示例：

```bash
python Tutorials/SU2DetectorResponseToy/pyqpanda_cpuqvm_example.py --num-qubits 6 --shots 1000
```

`pyqpanda` 只用于 CPUQVM 示例。数值教程部分只需要 NumPy 与 Matplotlib 即可运行。

## 测试方式

- 编译检查所有 Python 脚本。
- 运行 6-qubit、17 个参数点的 tutorial scan。
- 根据 `summary.json` 生成图表。
- 运行 pyQPanda CPUQVM 示例，验证 singlet-pair preparation 与 local probe circuit 可执行，并输出 bitstring counts。
- 运行基础测试，检查 Hermiticity、sector diagnostics、quench normalization、scan summary fields 和 QASM circuit shape。

## 说明

生成的输出目录与 Python 缓存文件已通过 `.gitignore` 排除。
