# QShadow：可观测量感知的经典阴影估计

QShadow 是面向 PyQPanda3 的多可观测量测量与后处理工具。它使用局域 Pauli
`X/Y/Z` 测量，从同一批量子态副本中同时估计多个 Pauli 可观测量；在给定目标
可观测量后，可优化每个量子比特的测量基概率，并用同时置信区间控制额外 shots。

> **准确定位**：经典阴影、局域偏置经典阴影和 Adaptive Pauli Shadows 都是已有
> 学术成果，本模块不声称发明这些算法。QShadow 的原创贡献范围仅限于本文末尾
> “原创性与引用边界”中列出的独立工程与算法设计。

## 功能

- `PauliTerm` / `PauliObservable`：显式 **q0-first** 的 Pauli 数据模型；
- `optimize_basis_probabilities`：以可观测量系数为权重、带严格概率下限的局域测量
  概率优化；
- `MeasurementPlan` / `ShadowDataset`：记录每个 shot 的测量基、结果和完整采样概率，
  可无偏合并不同测量轮次；
- `estimate_observables`：逆倾向（inverse-propensity）无偏估计与 Bonferroni 同时
  经验 Bernstein 区间；
- `ShadowBudgetController`：根据目标半宽、最小/最大 shots 和批大小给出下一批预算，
  **不会自动提交付费任务**；
- `simulate_pauli_measurements`：不依赖量子后端的 NumPy 小规模参考路径；
- `PyQPandaRunner`：CPUQVM 或用户提供的远程 executor；内置端序转换与
  `max_jobs` 硬上限。

## 快速开始

```python
import numpy as np
from pyqpanda_alg.QShadow import (
    PauliObservable,
    estimate_observables,
    optimize_basis_probabilities,
    sample_measurement_plan,
    simulate_pauli_measurements,
)

observables = [
    PauliObservable.from_terms(
        {"ZZ": -1.0, "XI": 0.25, "IX": 0.25}, name="energy"
    ),
    PauliObservable.from_terms({"ZI": 1.0}, name="magnetisation_q0"),
]

optimised = optimize_basis_probabilities(
    observables, probability_floor=0.03
)
plan = sample_measurement_plan(optimised.probabilities, shots=5000, seed=7)

# 小规模独立参考：|00>
dataset = simulate_pauli_measurements([1, 0, 0, 0], plan, seed=8)
results = estimate_observables(dataset, observables, family_confidence=0.95)
for name, result in results.items():
    print(name, result.value, (result.lower, result.upper))
```

Pauli 字符串的第 `q` 个字符作用于量子比特 `q`，所以 `"XI"` 是 `X(q0)`。
`ShadowDataset` 内的 bitstring 也采用 q0-first；`PyQPandaRunner` 会将 PyQPanda
通常使用的 MSB-first 计数字符串转换为该约定。

## PyQPanda3 本地执行

```python
from pyqpanda3.core import CNOT, H, QCircuit
from pyqpanda_alg.QShadow import PyQPandaRunner, run_measurement_plan


def prepare(qubits):
    return QCircuit() << H(qubits[0]) << CNOT(qubits[0], qubits[1])

runner = PyQPandaRunner(prepare, n_qubits=2)  # 默认 CPUQVM
dataset = run_measurement_plan(plan, runner, max_jobs=9)
```

`max_jobs` 在调用任何 runner **之前**检查，适合为远程后端设置硬任务数上限。

## 本源量子云 / 真机接入（后端无关，不保存密钥）

QShadow 不绑定云 SDK、服务端点或环境变量名称，也不接收和保存 API 密钥。应用层应使用
当前受支持的 SDK 在 QShadow 之外完成鉴权，并把一个
``executor(program, shots) -> counts`` 回调交给 ``PyQPandaRunner``。凭据只应从本地
环境变量或密钥存储读取，不得写入源码、Notebook、PR、日志或命令行参数。

下面是**接口合约伪代码**，其中 ``submit_once_and_wait`` 由应用层适配器实现；它不是
某个云 SDK 的可直接运行示例，也不应隐式重试或切换后端：

```python
from pyqpanda_alg.QShadow import PyQPandaRunner, run_measurement_plan

job_ids = []


def execute(program, shots):
    result = submit_once_and_wait(program, shots=shots)  # 应用层 SDK 适配器
    job_ids.append(result.task_id)                       # 任务 ID 非凭据
    return result.counts                                 # Mapping[str, int]

required_jobs = len(plan.grouped_shots)
if required_jobs > 3:
    raise RuntimeError("remote job budget exceeded")

runner = PyQPandaRunner(prepare, 2, executor=execute)
dataset = run_measurement_plan(plan, runner, max_jobs=3)
print(job_ids)
```

``max_jobs`` 会在任何 executor 调用前检查压缩后的唯一测量基数量。已验证的可选
``qpanda3_runtime==1.0.0`` 路径及非敏感原始证据见下文；该 SDK 不是 QShadow 的
核心依赖，服务地址和鉴权方式应以部署时的官方文档为准。

建议的真机证据链：

1. 固定代码版本、随机种子、后端名称、芯片/映射信息和 UTC 时间；
2. 先通过 NumPy 与 CPUQVM，再提交 1–3 比特、受控任务数和 shots 的真机实验；
3. 保存每个任务 ID、原始 counts、测量计划和采样概率；
4. 报告估计值、统计区间及理想模拟器偏差，不把统计区间误称为硬件噪声区间；
5. 如未做读出误差缓解，应明确披露。QShadow 当前不会隐式修正设备噪声。

### 已执行的 WK_C180 最小验证（2026-07-26）

在明确上限后，验证通过 ``qpanda3_runtime==1.0.0`` 向 ``WK_C180`` 提交了
X/Y/Z 各 512 shots，合计恰好 3 个任务、1536 shots，且没有自动重试。非敏感的
task ID、OriginIR、原始概率、counts 和分析保存在
[`qshadow_wukong_validation_20260726.json`](../../example/QShadow/qshadow_wukong_validation_20260726.json)，
解释说明见
[`qshadow_wukong_validation_20260726.md`](../../example/QShadow/qshadow_wukong_validation_20260726.md)。

这三个任务不只是云接口连通性测试：WK_C180 的原始 counts 经归一化和 q0-first
转换后进入同一个 ``ShadowDataset``，逐 shot 记录 ``(1/3, 1/3, 1/3)`` 概率，
并产生 X/Y/Z 的同时点估计与经验 Bernstein 区间；运行同时保持了 3 jobs、
1536 shots 和 0 retry 的硬预算。

硬件 X/Y/Z 估计 ``0.898438 / 0.832031 / 0.933594`` 的同时区间均未覆盖理想值
``0.599121 / 0.504633 / 0.621610``，独立 Bloch 向量范数为
``1.539826 > 1``。这使统计模型之外的设备或平台偏差在结果中可见，而没有被有限
shots 的区间掩盖。本次 Runtime 使用 ``is_amend=True``，QShadow 本身没有内建
读出缓解，因此不能据此声称硬件精度成功。

本次目标集合 X/Y/Z 完全对称，使用的是均匀计划，未比较偏置规划器与均匀规划的
真机表现。规划器的代理损失和概率下限行为由 CPUQVM 示例与测试验证；本记录验证的
是真实 counts 摄取、同时估计、置信区间和任务预算流程。

## 方法

### 无偏局域 Pauli 阴影估计

对 Pauli 项 `P`，第 `t` 次测量在量子比特 `q` 选择基
`B[t,q] ~ beta[t,q]`，得到本征值 `s[t,q] in {-1,+1}`。QShadow 使用

```text
X_t(P) = 1[B_t covers P] * product(s[t,q])
         / product(beta[t,q,P[q]])
```

作为 `<P>` 的无偏估计。分母保留每个 shot 当时的完整概率，因此先均匀测量、再
使用偏置测量的多轮数据可以直接合并。

### 可观测量感知规划

对目标 Pauli 项 `P_j`、系数 `c_j` 和用户权重 `w_j`，优化器最小化对角二阶矩代理

```text
L(beta) = sum_j w_j c_j^2
          / product(beta[q, P_j[q]], q in support(P_j)).
```

固定其余量子比特后，单个量子比特的子问题为
`min sum_a A_a / beta_a`；无下限时解析解满足 `beta_a ∝ sqrt(A_a)`。本实现加入
`beta_a >= probability_floor`，用活动集求出每个块的精确最优点并循环至收敛。
该目标是**状态无关的对角代理**，不是完整协方差，也不保证最小化未知量子态上的
真实方差。

### 可复现离线对照

[`qshadow_planning_benchmark.py`](../../example/QShadow/qshadow_planning_benchmark.py)
在相同三比特状态、目标可观测量和 shots 下比较均匀规划与 QShadow 规划。以下结果
使用固定种子、1000 轮 × 2000 shots；顺序预算部分另运行 100 轮：

```bash
PYTHONPATH=pyqpanda-algorithm python \
  pyqpanda-algorithm/example/QShadow/qshadow_planning_benchmark.py \
  --trials 1000 --shots 2000 --budget-trials 100 \
  --target-half-width 0.22 --seed 2000000
```

- 规划代理损失：``24.54 -> 5.964549``；
- 聚合 RMSE 下降 ``60.20%``；两个可观测量的 RMSE 分别下降 ``61.72%`` 和
  ``57.68%``；
- 平均同时区间半宽分别下降 ``62.35%`` 和 ``53.42%``；
- 达到相同 ``0.22`` 目标半宽时，平均总 shots 从 ``7284.9`` 降至 ``1892.8``，
  减少 ``74.02%``，且 100 轮均无硬上限违规；
- 本实验两方案的单项和 family-wise 经验覆盖率均为 ``1.000``，说明区间在该配置下
  较保守；这不是“精确校准为 95%”的声明。

1000 轮中优化方案两个偏差的 z-score 为 ``0.28`` 和 ``-0.26``，与零一致。上述
改善只针对固定状态、可观测量、概率下限和种子日程，不推广为所有问题上的普遍优势。

### 置信预算

对每个可观测量的逐 shot 样本，QShadow 使用

```text
radius = sqrt(2 s^2 log(3/delta) / n) + 3 R log(3/delta) / n
```

作为保守经验 Bernstein 半宽，并以 Bonferroni 分配 `delta`，形成多个可观测量的
同时区间。`R` 来自概率下限下的逆倾向确定性范围。高权重 Pauli 项可能令该范围很
大，因此区间常比高斯误差条更保守。

## 限制

- 目前仅支持实系数 Hermitian Pauli 和局域 `X/Y/Z` 测量；
- 不包含读出误差缓解、零噪声外推或完整状态层析；
- 经验 Bernstein 区间控制抽样随机性，不覆盖校准漂移、门噪声等系统偏差；
- 状态向量参考采样是指数复杂度，只用于小规模校验；
- 每个唯一 basis 通常对应一个后端任务，真机前必须检查
  `len(plan.grouped_shots)` 并设置 `max_jobs`；
- “规划损失下降”不等同于所有量子态、所有可观测量上的实际均方误差必然下降。

## 原创性与引用边界

### 已有工作（明确归因，绝不声称首创）

1. H.-Y. Huang, R. Kueng, J. Preskill, *Predicting many properties of a
   quantum system from very few measurements*, Nature Physics 16, 1050–1057
   (2020), [arXiv:2002.08953](https://arxiv.org/abs/2002.08953)。
2. C. Hadfield, S. Bravyi, R. Raymond, A. Mezzacapo, *Measurements of Quantum
   Hamiltonians with Locally-Biased Classical Shadows*,
   [arXiv:2006.15788](https://arxiv.org/abs/2006.15788)。
3. C. Hadfield, *Adaptive Pauli Shadows for Energy Estimation*,
   [arXiv:2105.12207](https://arxiv.org/abs/2105.12207)。这是已有自适应方法；
   QShadow 不把“自适应 Pauli 阴影”作为自己的发明。

### 本贡献可主张的原创范围

- 面向 PyQPanda3、q0-first 且后端无关的完整数据与 runner API；
- 基于上述明确代理目标、带概率下限的活动集块坐标实现及收敛诊断；
- 支持异构测量轮次的逐 shot 概率记录与无偏合并；
- 同时经验 Bernstein 区间到硬 shots 上限之间的显式预算控制；
- `max_jobs` 付费后端安全阈值、端序回归测试、NumPy/CPUQVM 双参考路径；
- 本仓库中的代码、测试、示例和文档均独立编写，未复制论文作者或其他框架的
  源文件。

## 测试

```bash
PYTHONPATH=pyqpanda-algorithm python -m pytest test/QShadow -q -o addopts=""
```

测试覆盖 Pauli 代数、规划损失单调性、概率下限、多轮无偏合并、置信预算、计数
校验、任务上限、端序，以及 CPUQVM 的 X/Y/Z 基符号。

---

## English summary

QShadow is a PyQPanda3-oriented workflow for estimating many Pauli observables
from local X/Y/Z measurements.  It provides observable-aware locally biased
planning, per-shot propensity records, unbiased post-processing, simultaneous
empirical-Bernstein intervals, explicit shot-budget decisions, and local/cloud
runner adapters.  Classical shadows and locally/adaptively biased Pauli shadows
are prior work cited above; this contribution claims only its independently
written PyQPanda integration, constrained coordinate optimiser, data model,
budget workflow, examples, documentation, and tests.  Device noise is not
silently mitigated, and real-hardware results must report backend metadata, job
IDs, raw counts, statistical uncertainty, and simulator discrepancy.  A disclosed
WK_C180 run on 2026-07-26 used exactly three 512-shot jobs with no retry.  Cloud
integration and budget controls passed, while hardware accuracy was not
demonstrated: all three simultaneous intervals missed the ideal values and the
independently estimated Bloch-vector norm exceeded one.  The run used the
platform option ``is_amend=True``; full raw evidence and limitations are stored
under ``example/QShadow/qshadow_wukong_validation_20260726.*``.
