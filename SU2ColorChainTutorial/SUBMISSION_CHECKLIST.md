# 提交检查清单

## PR 标题

```text
【创新应用】SU(2) color-chain quantum simulation tutorial
```

## 文件清单

- [x] `README.md`：项目总说明
- [x] `contest_pitch.txt`：项目简介与应用场景
- [x] `requirements.txt`：基础依赖列表
- [x] `requirements-pyqpanda.txt`：pyQPanda 可选依赖列表
- [x] `run_demo.py`：主扫描入口
- [x] `make_plots.py`：图表生成入口
- [x] `pyqpanda_demo.py`：pyQPanda CPUQVM 示例入口
- [x] `src/su2_toy/`：SU(2) toy simulation 模块
- [x] `tests/test_su2_toy.py`：基础测试
- [x] `Tutorials/SU2ColorChainTutorial/README.md`：教程说明
- [x] `Tutorials/SU2ColorChainTutorial/DERIVATION.md`：模型原理说明
- [x] `Tutorials/SU2ColorChainTutorial/CONTRIBUTION_NOTE.md`：贡献说明
- [x] `Tutorials/SU2ColorChainTutorial/PR_DESCRIPTION.md`：PR 正文草稿
- [x] `Tutorials/SU2ColorChainTutorial/run_tutorial.py`：教程运行入口
- [x] `Tutorials/SU2ColorChainTutorial/make_tutorial_plots.py`：教程画图入口
- [x] `Tutorials/SU2ColorChainTutorial/pyqpanda_cpuqvm_example.py`：教程版 pyQPanda 示例

## 本地验证命令

```bash
python -m py_compile run_demo.py make_plots.py pyqpanda_demo.py
python -m py_compile Tutorials/SU2ColorChainTutorial/run_tutorial.py Tutorials/SU2ColorChainTutorial/make_tutorial_plots.py Tutorials/SU2ColorChainTutorial/pyqpanda_cpuqvm_example.py
python Tutorials/SU2ColorChainTutorial/run_tutorial.py --num-qubits 6 --points 17
python Tutorials/SU2ColorChainTutorial/make_tutorial_plots.py
python -c "import sys; sys.path.insert(0, 'tests'); import test_su2_toy as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]; print('tests ok')"
```

## 输出文件检查

运行后生成的 `outputs/`、`outputs_pyqpanda/`、`__pycache__/`、`*.pyc` 请排除在提交内容之外。

## 提交前检查

- [ ] README 可以独立说明项目目的与运行方式。
- [ ] tutorial 文档可以独立运行。
- [ ] `DERIVATION.md` 只说明模型原理与算法设计。
- [ ] PR 正文使用中文为主，描述清楚教程定位、创新点、运行方式和测试方式。
- [ ] README 已说明 `pyqpanda` 只用于 CPUQVM 示例，数值教程可用 NumPy 与 Matplotlib 运行。
- [ ] README 已说明 NumPy reference、OpenQASM prototype 与 pyQPanda CPUQVM execution 的关系。
- [ ] pyQPanda CPUQVM 示例说明了 bitstring counts 输出位置。
- [ ] 提交内容不包含本地输出、缓存文件或个人配置。
