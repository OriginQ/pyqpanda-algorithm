# 维护者与评委审阅路线

队伍 **youshen**。本应用处理设备故障后的实验预约重排：输入预约候选与约束，
在 CPU 上运行 QAOA，再独立核验排程。全部业务数据为合成，无真实用户试用声明。

## 五分钟主要演示路线（安装完成后）

按[主教程](README.md#1-三分钟演示)安装后，在仓库根目录运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python pyqpanda-algorithm/example/LabScheduling/tradeoff_demo.py --sensitivity
```

三个有前序关系的实验共享带清洗时间的烘箱。故障后可将三单整体后移，成本 12；
也可支付备用设备费用，保留后两单，成本 14。三项决策全部留在 8 比特组件中，
没有预处理固定项。默认种子的实际量子样本得到成本 12，MILP/穷举随后认证。
一般输入和其他参数不保证采到最优，脚本保留实际 gap，不能用经典解修补。

打开 `reports/tradeoff-demo/tradeoff.svg` 看原预约与采样结果；
`alternatives.csv` 列出全部 9 个可行排程；`outage-report.json` 含参数、counts、
原始约束复核及证书。`ablation.json` 分别列出各阶段资源与验证耗时。

接着看 `sensitivity/summary.csv` 与 `sensitivity/sensitivity.svg`：每单改约成本
从 0 增到 6，最优决策在 3 分处由整体后移切换为备用设备，阈值处两者同优。
全部 21 次实际量子结果和七组完整可行排程都可核验；详见
[权重与决策解释](EVALUATION_V2.md#改约成本敏感性什么时候值得使用备用设备)。

最后运行资源边界案例：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pyqpanda_alg.LabScheduling pyqpanda-algorithm/example/LabScheduling/data/support-chain.json --reduce --pruning arc --output reports/support-chain.json
```

查看 `component_qubits=[12]`、成本 3、零违反和零 gap。去掉 `--pruning arc`
会留下 18 比特组件并返回资源限制，退出码 5。只运行这一受控边界例，便可说明
精确化简带来的用途；不要将组件上限突破说成通用大规模量子调度能力。
五分钟是演示目标，实测环境与计时见 VALIDATION.md，首次安装下载另计。

讲解顺序：约一分钟交代故障与两种方案，一分钟解释权重切换，一分钟展示
18→12 的删除依据，再展示实际 counts、经典证书和失败状态。
若现场未采到最优，展示实际 gap；不能以认证解替换量子样本。

## 核心代码阅读顺序（约 15 分钟）

| 阅读顺序 | 文件 | 核查重点 |
|---|---|---|
| 1 | [problem.py](../../pyqpanda-algorithm/pyqpanda_alg/LabScheduling/problem.py) | 严格输入、整数成本、时间区间及单容量边界 |
| 2 | [model.py](../../pyqpanda-algorithm/pyqpanda_alg/LabScheduling/model.py) | 候选剪枝、禁止对只计一次、A 的充分界、独立解码 |
| 3 | [quantum.py](../../pyqpanda-algorithm/pyqpanda_alg/LabScheduling/quantum.py) | PyQPanda3 CPU、XY 单激发、相位简化、样本来源 |
| 4 | [reduction.py](../../pyqpanda-algorithm/pyqpanda_alg/LabScheduling/reduction.py) | 单候选传播、严格独立分量、原变量回填、超限状态 |
| 5 | [milp.py](../../pyqpanda-algorithm/pyqpanda_alg/LabScheduling/milp.py) | 从原始输入独立建模、限时不冒充最优 |
| 6 | [report.py](../../pyqpanda-algorithm/pyqpanda_alg/LabScheduling/report.py) | 求解后认证、独立语义检查、状态与 gap |

数学证明集中在[主教程](README.md)和[增强教程第 2–4 节](ENHANCEMENTS.md)。
可选[弧一致性化简](ARC_PRUNING.md)补强经典预处理；默认量子算法保持不变。

## 证据入口

- [无支持候选删除](ARC_PRUNING.md)：18→12 比特边界演示、200 个随机小问题与
  全部 V2 输入的完整可行集合验证，以及局部有支持但全局不可行的反例。
- [同总预算对照](MATCHED_BUDGET.md)：将总训练上限与 shots 对齐，新增化简后随机
  基线。512 shots 下随机也全部采到最优；低 shots 条件曲线与退化实例同时公开。
- [第二批协议](BENCHMARK_V2_PROTOCOL.md)：在评价前提交冻结；36 个问题，每格 4 个数据种子。
- [第二批分析](EVALUATION_V2.md)：正常、不可行、纯经典和退化结果一并解释。
- [逐实例表](results/v2/SUMMARY.md)：先看概览；再按实例/seed 查看
  [完整训练记录](results/v2/runs.jsonl)与[结构证明记录](results/v2/instances.jsonl)。
- [关键测试](../../test/LabScheduling/enhancement.test.py)：100 个随机问题的可行集合等价、
  CPU 相位等价和 MILP 状态边界；[新增案例测试](../../test/LabScheduling/evidence.test.py)
  覆盖 36 个异构实例的无故障合法性和全部可行集合，以及完整业务取舍。
- [验证记录](VALIDATION.md)：本地测试和 GitHub CI 分开标注。维护者批准前不能算上游 CI 通过。
- [输入输出回归](../../test/LabScheduling/io.test.py)：拒绝输入别名覆盖，验证求解
  数据与指纹一致，以及报告写入/替换失败时旧文件不被截断。

原始 JSONL 与生成数据占 diff 的大部分；可先审阅上表核心文件及测试，再抽查归档。
归档保留输入和源码指纹、全部失败及计时，不需要逐行阅读优化历史。

新增的[安装兼容性说明](COMPATIBILITY.md)记录 3.11 / 3.12 / 3.13 的实际验证。
兼容性轮次只调整安装约束、CI 和证据说明；后续预算修复见下方失败状态说明。

## 当前运行与历史归档如何核对

**验证当前代码是否可重复运行**：在同一 checkout、同一 `.venv` 中执行两次。
若已运行上面的主演示，直接执行第二条即可：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python pyqpanda-algorithm/example/LabScheduling/tradeoff_demo.py --sensitivity
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python pyqpanda-algorithm/example/LabScheduling/tradeoff_demo.py --sensitivity --output reports/tradeoff-replay --verify-sensitivity reports/tradeoff-demo/sensitivity/results.json
```

第二条命令严格比较 21 条敏感性记录及其输入、源码和环境指纹，只排除
`_seconds` 计时字段。成功退出表示这些记录一致；不表示不同 Python、依赖版本
或机器上的优化参数和 counts 必然相同。比较对象是 `sensitivity/results.json`，
不是整份演示目录或 SVG。

**核验提交在仓库中的历史实验**：先选择归档对应的源码版本。后续修复会改变
源码指纹，不能直接拿当前版本与旧归档严格比较，也不能删除指纹检查来“通过”。
以下提交同时包含对应实现与归档；2026-09-22 已逐项核对归档记录的源码指纹：

| 历史实验 | 包含匹配源码和归档的提交 | 匹配的源码/协议/配置指纹 |
|---|---|---:|
| 同总预算 360 条记录 | `a3716ed778ece61fcb7dfe13f18633f4c6164a62` | 17 |
| AC-3 结构证明与边界运行 | `4eb272a9f7c75167487a578752eb8f6fbda11c88` | 14 |
| 权重敏感性 21 条记录 | `384649588690fa3b100541b78a1ad82447df2388` | 17 |

例如，已取得本 PR 的 Git 历史后，在仓库根目录创建一个独立检出：

```bash
git worktree add --detach ../lab-sensitivity-archive 384649588690fa3b100541b78a1ad82447df2388
cd ../lab-sensitivity-archive
python3.12 -m venv .venv
.venv/bin/python -m pip install -c pyqpanda-algorithm/example/LabScheduling/constraints-py312.txt -e ./pyqpanda-algorithm -r pyqpanda-algorithm/example/LabScheduling/requirements-dev.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python pyqpanda-algorithm/example/LabScheduling/tradeoff_demo.py --sensitivity --output reports/archive-replay --verify-sensitivity Tutorials/LabScheduling/results/sensitivity/results.json
```

历史敏感性归档记录的解释器是 **Python 3.12.3**；这里的 `python3.12` 也须是该
补丁版本，并使用归档约束。其他版本仍可做当前环境内的重复运行，但严格环境
比较可能拒绝历史重放。指纹匹配是必要条件，不能代替实际执行比较器；历史完整
重放的实测记录见 [VALIDATION.md](VALIDATION.md)。如果使用 ZIP 或浅克隆而没有
这些提交，请先获取本 PR 的完整分支历史，再创建 worktree。

## 结论边界

可证明的是候选化简保持可行集合、固定成本回填、XY 恒零罚项删除的理想分布等价。
经典基线在这些规模更快；分量预算相加，分量最大深度不代表实际并行耗时。
360 个种子组合来自 36 个问题，不能当作 360 个独立业务实例，也不代表真实实验室部署。

## 答辩时需要回答的六个问题

| 问题 | 回答与可核验证据 |
|---|---|
| 为什么经典方法更快，还做量子调度？ | 本作品研究受限量子资源下如何形成可运行、可验证的调度子问题；没有商业性能或量子加速结论。同预算随机对照及失败记录均公开。 |
| 创新是不是只有把现成 QAOA 换个场景？ | 完整贡献包括最小扰动业务建模、可证明的候选化简和分量回填、可追溯成本/删除依据、独立原始输入认证与可复现流程。QAOA 和 AC-3 本身不是原创。 |
| 量子求解是否被经典最优解替代？ | 训练与采样先执行，最优证书随后计算；查看各组件 counts 与选中位串，再用原始输入检查器复核。纯经典传播明确单列。 |
| 18→12 比特说明什么？ | 在公开的构造案例中，可选 AC-3 将组件压到当前模拟预算内，四项决策仍在量子组件。V2 中只有 5/36 实例的最大组件缩小；不是普遍缩减比例。 |
| 业务成本有没有现实依据？ | 当前是可解释的合成偏好分。七档权重展示决策切换与并列最优；没有真实使用者反馈，不能称为经济收益测量。 |
| 没有最优证书、抽样失败或超过预算怎么办？ | 保留实际样本、违反数与明确状态；没有证书时 gap 为 null。MILP 超预算只跳过该认证路径，不丢弃已有结果，也不将结果标成最优。 |

建议以“受限量子资源下的可审计故障重排工具”概括作品。评委无需阅读全部历史
训练记录即可检查以上六点；原始证据保留供抽查。后续研究应由真实需求或实验
结果驱动，不以不断增加功能代替验证。

已准备[小范围使用者试用流程](USER_TRIAL.md)及[空白反馈表](TRIAL_FEEDBACK_TEMPLATE.md)，
复用现有 CSV 导入和求解接口。真实参与者、匿名事件数据与反馈尚未取得；材料就绪
不计为真实试用完成，也不改变以上证据边界。
