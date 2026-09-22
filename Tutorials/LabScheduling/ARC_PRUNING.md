# 无支持候选删除：让一个超限组件进入可运行范围

保留实验室故障重排方向，新增可选 `--reduce --pruning arc`。它在经典预处理中
剔除不可能出现在任何可行排程里的候选，再按原方式运行真实 CPU QAOA。
默认 `singleton`、旧实验和预算结论保持原貌。

## 一条命令观察变化

按[主教程](README.md#1-三分钟演示)安装，在仓库根目录运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pyqpanda_alg.LabScheduling pyqpanda-algorithm/example/LabScheduling/data/support-chain.json --reduce --pruning arc --output reports/support-chain.json
```

预期 `status="feasible"`、`absolute_gap=0`、`quantum.best.objective=3`；
`quantum.reduction.component_qubits=[12]`，固定变量为空，四项任务都由量子样本决定。
删除 `--pruning arc` 重跑，将得到 `status="resource_limit"`、退出码 5：
原有单候选传播留下一个 18 比特组件，超过本应用配置的每组件 16 比特上限。
报告仍保留经典认证，不能把认证解当作实际量子输出。

[输入](../../pyqpanda-algorithm/example/LabScheduling/data/support-chain.json)是专门构造
的边界案例：四道按序实验共用烘箱，原预约开始于 0/2/4/6；设备在 `[0,1)`
停机，第一单必须改约。末工序候选只到时隙 6，前三道工序中的晚窗口不能接到
末工序，逐层删除。它不是新留出集，也不代表一般实例的平均收益。

| 阶段 | 候选/量子组件 | 固定任务 | 全部可行排程数 |
|---|---|---:|---:|
| 原始输入 | 19 个候选，含停机无效项 | — | 15 |
| 日历剪枝 + 原单候选传播 | 单组件 18 比特 | 0 | 15 |
| 追加弧一致性 | 单组件 12 比特 | 0 | 15 |

种子 7/19/42，p=1、每种子 2 次初始化、每次最多 60 次评价、512 shots，
三次实际 CPU 运行均得到成本 **3**、**0** 违反、**0** 最优差距；各实际评价 120 次。
只将第一单从 0 移到 1，偏好移动成本 1 + 改约成本 2，其余三单保持原预约。
原预约在无故障时经独立原始语义检查合法、成本 0；停机后违反一条日历约束。
这里的最优来自实际采样，随后独立穷举与原始输入 MILP 认证，不是经典修补。
这些固定种子的结果不构成一般 QAOA 最优保证。

## 规则、证明和审计字段

设当前存活候选域为 $D_t$，编译后的禁止候选对集合为 $E$。
对 $i\in D_t$，若存在另一任务 $u$ 满足

$$
\forall j\in D_u,\quad \{i,j\}\in E,
$$

则删除 $i$。任一全局可行排程必须从 $D_u$ 选一个候选，因而无法同时选 $i$。
从原始域出发逐次应用这个论证，删除始终不移除任何原可行排程。
该规则不依赖成本，因此保留全部可行排程及其原成本，包括所有并列最优解。
空域给出不可行证据；否则进入原单候选固定和独立分量分解，沿用原变量回填证明。

实现采用 AC-3 的有向弧队列：修改 $D_t$ 后重新检查可能失去支持的相邻域。
任务顺序、队列顺序、候选遍历及审计输出都确定；用集合去重待检查的弧。
无禁止对的任务之间所有候选互相支持，不需要加入队列。
典型 AC-3 最坏时间为 $O(e d^3)$（$e$ 为任务间有向弧数，$d$ 为最大候选域大小），
本应用编译上限仍为 256 候选。它是经典预处理，计入求解器耗时。
方法来自已有约束满足理论，参考
[Mackworth, Consistency in Networks of Relations (1977)](https://www.cs.ubc.ca/~mack/Publications/b2hd-AI77.html)，
不将 AC-3 本身列为原创量子算法。

`quantum.reduction.removals` 每条新增删除含：

- `variable`：原始编译模型中的候选索引。
- `rule="no_support"`：与原单候选规则的 `forced_by` 区分。
- `against_task`：没有任何支持选择的对方任务。
- `blockers`：删除当时对方**全部**存活候选及每对的原始冲突原因。

例如工序 t2 的晚窗口与 t3 的三个候选都违反前序，先删 t2 的两个晚窗口；
t1 的晚窗口随后失去全部支持，再向 t0 传播。见归档中的逐条原因链。
验证器按删除顺序重建域，拒绝缺失候选、虚构冲突原因和错误空域证明。
原模式的 `forced_by` 字段保持不变；报告 schema 仍为 2，`method` 标明使用弧一致性。

Python 调用分别为 `reduce_model(model, pruning="arc")`、
`solve_reduced(model, pruning="arc")`、`run_experiment(path, reduce=True, pruning="arc")`。
CLI 单独传 `--pruning arc` 而没有 `--reduce` 会明确报错，不静默忽略参数。

## 已知语料上的结构收益

在既有 V2 的全部 36 个输入上核验两种模式的**全部原始语义可行集合和成本**。
24 个可行、12 个不可行的结论保持一致；5 个可行实例的最大组件严格缩小，
其余 31 个没有最大组件收益，不隐藏这些结果。

| V2 实例 | 原组件比特数 | arc 组件比特数 | arc 固定任务 |
|---|---|---|---:|
| n3-s2-r1 | [7] | [5] | 1 |
| n4-s2-r1 | [9] | [] | 4 |
| n4-s2-r2 | [7] | [2] | 3 |
| n5-s2-r0 | [12] | [11] | 1 |
| n5-s2-r3 | [7] | [2] | 4 |

`[]` 表示全部由经典传播解决，不计为量子成功。V2 本来就已满足组件上限；
新增 18→12 案例另列，不混入随机语料平均值。两种模式可能以不同顺序发现
同一不可行问题，空域任务名称和删除链不必相同。
此处只比较精确结构化简，没有新跑 360 次训练，也没有借用旧结果宣称 arc 的
量子解质量更高。预处理重算组件 QUBO/罚系数，量子训练分布不要求保持一致。

## 复现与失败边界

[评价前范围记录](ARC_PROTOCOL.md)提交为 `086840d`。
[完整证据](results/support/results.json)保存 37 个输入的指纹、两种模式及其证明记录，
还有三个种子的完整量子 counts、参数、历史和独立证书；
[二次重放](results/support/replay.json)确认除 `_seconds` 计时字段外一致。
实现版本为 `67d07e7`；源码与环境指纹不同会使严格重放失败，
应使用该归档对应版本与锁定依赖。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python pyqpanda-algorithm/example/LabScheduling/support_pruning.py --output reports/support-pruning
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python pyqpanda-algorithm/example/LabScheduling/support_pruning.py --output reports/support-pruning-replay --verify-against reports/support-pruning/results.json
```

[回归测试](../../test/LabScheduling/support.test.py)额外核验种子 0–199 的 200 个小问题，
包括日历、清洗、小组、前序、空问题/空候选和固定成本；所有原始可行排程逐一对应。
包含两任务各有多个候选但全部互斥的不可行证书，以及**三任务共用两个时隙**：
每个候选对任意另一任务都有支持，弧一致性不删除任何项，但整个问题不可行。
因此局部一致性不是全局可行证书；仍必须检查真实采样结果并保留经典认证。

没有通用规模保证：过大的存活组件仍报资源限制。经典求解在这些规模更快；
本项贡献是保留可行集合的资源缩减、可检查的删除依据与完整端到端验证。
