# 验证与复现记录

本文前半保留初版历史记录；当前增强版状态与验证见文末。

日期：2026-09-21。基线：`upstream/develop@5f973ef`。
环境：WSL2 Linux x86_64，Python 3.12.3，PyQPanda3 0.4.1，
pyqpanda_alg 2.0.0（本地源码），NumPy 2.5.3，SciPy 1.18.1。
完整包版本见 [constraints-py312.txt](../../pyqpanda-algorithm/example/LabScheduling/constraints-py312.txt)。

## 上游基线

1. 官方 requirements 安装后，初始 `pytest test` 因未安装 Allure 插件拒绝
   `--alluredir` 参数；`import pyqpanda_alg` 也因未声明的 pandas 依赖失败。
2. 依据实际导入和测试所需，在应用专属 requirements-dev 中补充
   pandas/scikit-learn/allure-pytest。没有修改全局 Python 或上游依赖文件。
3. `test/pytest.ini` 没有 `[pytest]` 节，当前 pytest 9 仍能读取；使用独立配置
   固定本次收集规则。有效测试是 **18 项**；其他多个文件主体为注释，不能算作测试通过。
4. 独立运行原有测试得到 18 passed（35.99 s）。之后并发模拟压力下的重复检查
   触发一个 20 秒超时且产生线程争用，该轮中止，不将它当成功结果。
   固定 OMP/BLAS 单线程后完整复核 **18 passed，9.08 s**。

最终基线命令（在仓库根目录）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
  -c test/LabScheduling/pytest.ini \
  test/QAOA test/QAlgBase test/QPCA test/QSVM test/QARM \
  -o 'python_files=Test_*.py' --timeout=60 -q
```

## 新增应用测试

**56 passed，覆盖率 98.46%（455 条可执行语句，7 条未命中）**。
`model.py` 和 `classical.py` 行覆盖为 100%。命令见主教程第 8 节；
聚焦 CI 设置 95% 最低覆盖率，保留真实量子模拟测试。

- 对 simple 的 64 态、medium 的 4096 态逐一验证
  `QUBO = 原始成本 + A*(恰选一次平方罚分 + 冲突对数)`。
- 对 medium 全部 4096 态验证 Ising 对角值与 QUBO 能量，覆盖 qubit 10/11，
  避免 `x10` 字符串排序把映射改变。
- 对真实 CPUQVM 做非对称位序测试，对 X 线路做独立 NumPy 态矢量交叉验证。
- 对不同层数和角度验证 XY 域内单激发守恒，以及零角度均匀 one-hot 概率。
- 同 seed 重复求解验证参数、counts、最佳样本一致；最佳结果必须出现在 counts。
- 分别验证语义不可行、空域、零任务、单候选、同对多原因冲突、清洗与停机边界、
  原预约变更成本、参数错误、重复键/未知字段、NaN 和规模保护。
- 使用可行 simple + X + 1 shot 的真实模拟验证“未采到可行解”与“问题不可行”
  的区别，并验证 CLI 退出码 0/2/3/4。

覆盖工具注意：`--cov` 传**源码目录路径**。本机 coverage 对包含二进制导入链的
包名预加载触发 NumPy 重复导入错误；路径模式正常工作。没有为通过覆盖率而
mock 量子库。覆盖率未包含子进程执行的 CLI 分支，实际子进程命令另有断言验证。

## 代码质量与安装

- Ruff：E/F/I/UP/B 检查通过，格式检查通过，配置范围仅新增应用和测试。
- Mypy：7 个模块检查通过；仅跳过 PyQPanda3 不完整的二进制导出类型定义。
- `pip check`：无依赖冲突。
- 构建 wheel 成功；用 `--no-deps --target .venv/wheel-check` 安装到项目内目录，
  从该安装目录导入运行 medium，得到目标值 4、0 违反、差距 0。
- `git diff --check`、新增 Python 公开函数标注/文档字符串、相对文档链接、
  缓存/密钥/旧 pyqpanda API 检查在最终提交前完成。
- 新增 GitHub Actions workflow 的等价命令已本地运行；远程 Actions 尚未触发，
  不声称已经通过 GitHub CI。

## 多规模、多种子实验

统一预算：p=1，shots=512，restarts=2，COBYLA maxiter=60；种子 7、19、42。
采用 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，CPU 无噪声模拟。
按独立函数定义分别计时，`quantum.runtime_seconds` 包含量子训练、采样及诊断，
`exact.runtime_seconds` 包含全组合枚举和原始约束检查，不含公共建模和解释器导入。
`report.runtime_seconds` 包含读入、建模和三种求解流程，也不含解释器导入。
这些指标不是硬件量子执行耗时。

| 实例/方法 | seed 7 可行概率 | seed 19 | seed 42 | 最佳目标（对应三种子） |
|---|---:|---:|---:|---|
| simple / XY | 0.384406 | 0.688144 | 0.420494 | 5 / 5 / 5 |
| simple / X | 0.013176 | 0.013175 | 0.013176 | 5 / 5 / 5 |
| medium / XY | 0.393214 | 0.221515 | 0.130439 | 4 / 4 / 4 |
| medium / X | 0.002105 | 0.002105 | 0.002105 | 8 / 18 / 13 |

全部上述最佳样本可行，违反数为 0；X 在 medium 的最优差距为 4/14/9。
XY 在 simple 的模拟求解计时约 0.10–0.11 s，medium 约 0.30–0.34 s；
实际端到端还需解释器导入和建模。具体逐次耗时见
[summary.csv](results/summary.csv)，不是稳定性能阈值。

同预算均匀 one-hot 抽样在所有这些小实例也命中经典最优值。
XY simple 两个种子的可行概率低于均匀 one-hot，不能称为普遍优势。
三个种子不足以推导统计显著性或可扩展性；精确基线在此规模明显更快。
可行概率提升展示约束编码的作用，不能推出实用量子加速。

[每次完整 JSON](results/)包含输入 SHA-256、环境、QUBO 系数/映射、剪枝与
冲突原因、counts、优化历史、终止状态、排程、精确认证和均匀基线。
所有种子均纳入结果，没有以最佳种子替代均值。

默认一层线路资源也经 PyQPanda3 `QProg.depth()` / `count_ops()` 实测：

| 场景/混合器 | 原生门深度 | CNOT 数 |
|---|---:|---:|
| simple / XY | 36 | 34 |
| simple / X | 23 | 22 |
| medium / XY | 45 | 78 |
| medium / X | 32 | 54 |

这些数值含初态、相位与混合器，未做真实芯片映射，不代表硬件编译后资源。

最终完整重放：再次运行 12 组实验，将每份 JSON 递归删除 `runtime_seconds`
后与归档比较，全部完全一致（包括参数、概率、counts、排程和诊断）。
最终再次 fetch 的 develop 仍为 `5f973ef`；刷新开放 PR 仍为 70 项，无新增标题。

## 增强版验证（2026-09-21，已更新 PR #92）

以上 56 项和 12 份结果为增强前的历史基线。当前新增模块后实测：
**71 passed，98.44% 行覆盖率（771 条语句，12 条未命中），6.11 s**；
原有有效测试再次运行 **18 passed，8.22 s**。Ruff lint/format 全部通过，
Mypy **11 个模块**通过。`bookings.py`、`reduction.py`、`milp.py` 在本轮覆盖为 100%。
覆盖率是辅助指标；核心证据是独立建模、可行集合等价和真实 CPU 线路测试。

- 100 个随机小问题：对全部可行排程集合进行化简/回填一一比对，验证固定成本，
  原始输入 MILP 的可行性及最优值与穷举一致。
- simple/medium 每例 30 组随机参数，p=1/2/3，full 与 auto 相位概率误差 <1e-10；
  另测单候选混合域，X 分支保持完整相位。未声称两个优化轨迹必然一致。
- 业务 CSV：故障重排成本 6、3 张改约、0 违反，量子组件 4 比特；无故障对照成本 0。
- `--reduce` 最大组件资源上限实测退出 5，量子评价次数 0；没有静默经典替换。
- MILP 限时状态、带 incumbent 的限时、错误/非整数 incumbent 等防御逻辑单独测试；
  正常求解使用真实 HiGHS。故障返回模拟仅用于不易稳定触发的状态分支测试。
- 12 个新合成评价实例 × 10 个优化种子全部保存；9 可行、3 不可行，两经典基线一致。
  120 个已训练参数上的 full/auto 最大概率差 2.220446049250313e-16。
  整次评测约 49.06 秒；峰值进程 RSS 269832 KiB，包含 Python、依赖、MILP 和模拟器。
  这不是单个线路内存或硬件耗时承诺。
- 两次完整 120 训练的参数、counts、分布指标及独立抽样结果在排除耗时后完全相同。
  所有输入/核心源码哈希与归档 provenance 一致，最佳位串均真实出现在 counts。
- 公开 Python API 的参数/返回类型及 docstring 经 AST 扫描无遗漏。

原始实验记录与源代码指纹：[增强结果目录](results/enhanced/)。
评测协议在首次评价前冻结，详见 [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)。

远程状态说明：用户批准后，增强提交 `9c23ddb` 已推送至 PR #92，完整 PR 描述已更新。
对应 [Actions 运行](https://github.com/OriginQ/pyqpanda-algorithm/actions/runs/35576581365)
状态为 `action_required`，jobs 数量为 0，等待上游维护者批准。
此前“尚未触发”表述仅适用于首次本地验证时；没有远程 CI 通过的证据，
本地测试不可替代上游审核。后续发布记录提交不改变应用代码及本地验证结论。

### 增强版全新 wheel 安装复核

使用 `.venv/bin/python -m pip wheel --no-deps --no-build-isolation ./pyqpanda-algorithm`
构建 wheel，然后在项目内新建 `.venv/enhanced-clean`，按相同 constraints 和
requirements-dev 安装 wheel 及依赖。确认导入来自该新环境的 site-packages，
而非 editable 源码；`pip check` 无依赖冲突。
新环境中 **71 项测试通过（4.70 s）**，CSV 连锁改约/无故障对照与 medium CLI
均成功，分别保持成本 6/0/4，0 约束违反。
最初尝试 `python -m build` 时环境未安装 build，改用现有 pip 的 wheel 构建流程，
没有全局安装包或跳过构建验证。

## 第二批评测与审阅优化（2026-09-21）

冻结提交 `94efa84` 先于评价执行。新增 36 个异构实例/360 条配对种子记录，
以及三项预约决策全部留在 8 比特组件的业务演示；核心应用源码未修改。

- 当前完整应用测试 **76 passed，6.95 s**，覆盖率 **98.44%**；
  包含全部 36 个输入的控制排程、可行集合/成本证明和独立 MILP 一致性。
- Ruff lint/format（29 文件）、Mypy（11 个核心模块）、pip check 全部通过。
- 已有纯 wheel 隔离环境再次运行当前 76 项测试：**76 passed，5.45 s**；
  新业务演示实际得到成本 12、零违反，全部决策仍由量子组件求解。
  因本轮核心包未修改，复用已验证 wheel，不将示例修改称作新算法版本。
- 第二次完整重放只剔除键名以 `_seconds` 结尾的时间字段后，36 份实例结构记录、
  360 份配对训练记录完全一致；源码/输入哈希均匹配，实际最佳位串来自 counts。
- 业务案例全部 9 个可行选择公开；高价备用设备方案成本 14，三单后移成本 12。
- 新增独立 fork 的分支 push CI 入口，运行与 PR 相同的 Python 3.12 安装、质量检查、
  CPU 测试和三类演示；其结果与上游维护者批准状态分开记录。

结果入口：[异构评测分析](EVALUATION_V2.md)、[审阅路线](REVIEW_GUIDE.md)。
个人 fork 的 [Actions 35583744393](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35583744393)
已在提交 `198ddfb` 上返回 **success**：Ubuntu / Python 3.12 新安装、Ruff/Mypy、
76 项真实 CPU 测试、medium CLI、CSV 连锁改约及新增取舍演示均通过。
官方 PR 对应运行 `35583752436` 仍为 `action_required`，等待上游维护者批准。
fork CI 成功不等于上游 CI 成功或 PR 合并；后续补录记录仅修改文档，
不改变已验证代码、数据、测试和工作流。

## 同总预算补充评价（2026-09-21）

- 协议先以 `eb9c5c3` 冻结；运行实现 `b140dab`。核心包与旧归档均未修改。
- **82 passed，7.86 s，核心行覆盖率 98.44%**；Ruff lint/format（32 文件）、
  Mypy（11 个核心模块）、pip check 均通过。
- 重新运行全部 36 实例 × 十种子，四条路径保留原始 counts 与状态；总训练评价
  上限 120、采样上限 512。首轮 90.81 s，第二轮 90.55 s，非计时字段完全一致。
- 第二轮 360 条直接 QAOA 记录还与旧 V2 归档一致（仅忽略计时），没有修改旧基准。
- 两次运行源码/输入/环境指纹一致，归档前源码哈希匹配，采样来源与预算上限复核通过。
- 22 个仍需量子的可行实例中：直接/分量 QAOA 最优 217/220、220/220；
  两个随机对照均为 220/220。低 shots 条件曲线包含分量 QAOA 低于随机的实例。
- 新增测试涵盖预算守恒、不足时停止、单分量与直接路径完全一致、分量回填、
  实际 CPU 样本来源、不可行域和重放比较器拒绝篡改 counts。

完整解释与复现命令见 [MATCHED_BUDGET.md](MATCHED_BUDGET.md)，原始记录见
[results/matched](results/matched/)。此处本地验证不代替当前提交的远程 CI。

本轮发布提交 `a3716ed` 的独立 fork
[Actions 35586823475](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35586823475)
实际返回 **success**：Ubuntu / Python 3.12.14 全新安装，Ruff/Mypy，
**82 passed，16.32 s，覆盖率 98.44%**，三类演示全部通过，取舍示例成本 12。
上游运行 `35586829560` 仍为 `action_required`。最终补录仅修改文档，
保留该成功运行验证过的代码、测试与数据不变。

## 输入快照与报告完整性修复（2026-09-21）

本轮在临时目录先复现两处实际问题：`--output` 同输入时覆盖原文件；输入在
求解后被修改时，报告哈希对应新文件而非求解数据。改为读取一次原始字节快照，
同一快照解析与哈希；`parse_problem(text)` 复用严格 JSON 校验。CLI 在求解前
检查同路径/符号链接/硬链接，报告写完同目录临时文件后才替换目标。

- 本地 **92 passed，9.39 s，核心覆盖率 98.87%**；新增十项 I/O 回归。
- Ruff lint/format（33 文件）、Mypy（11 模块）、pip check 全部通过。
- 原有 **18 passed，7.75 s**。首次误用应用 `*.test.py` 规则未收集到上游测试；
  显式采用上游 `Test_*.py` 规则后执行，未把空收集计为通过：

  ```bash
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -c test/LabScheduling/pytest.ini -o 'python_files=Test_*.py' test --ignore=test/LabScheduling --timeout=60
  ```

- 重新构建 wheel 并强制安装到已有 `.venv/enhanced-clean` 隔离环境，确认实际
  导入 site-packages。**92 passed，5.60 s**，取舍演示成本 12、最优差距 0。
- 写入/替换故障用 I/O 层故障注入验证；数据修改/删除测试仍运行真实 CPU 求解。
  CRLF 原字节哈希、有效输出符号链接保留、临时文件清理也已检查。

量子线路、优化器、QUBO 与冻结实验数据未改。旧归档中的源码指纹保留为对应
历史提交的记录（同预算归档为 `a3716ed`），不重写成修复后的源码指纹。
普通写入失败保护不等于断电持久性或多写入进程事务保证，接口说明见主教程。

发布提交 `36adc9f` 的个人 fork
[CI 35588841891](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35588841891)
全部通过：全新安装、质量检查、**92 passed，17.03 s，覆盖率 98.87%**、
三类 CPU 演示成功，取舍样本与证书仍为成本 12。上游 `35588846744` 仍待
维护者批准；最后补录 CI 的提交仅改变文档。

## 可选无支持候选化简（2026-09-21）

范围与边界数据先在 `086840d` 提交；默认模式保留，增加 `--reduce --pruning arc`。
实现采用确定性 AC-3，记录完整对方域及逐对冲突原因，不增加依赖或新量子变体。

- 本地 **102 passed，13.91 s，核心覆盖率 98.92%**（831 行，9 行未覆盖）；
  reduction.py 的 128 条可执行语句全部覆盖。
- 200 个固定种子小问题、全部既有 36 个 V2 输入和一个专门构造的边界案例，
  逐一核对原始语义全部可行集合、成本、回填映射与删除依据。
- 40 个输入的默认化简审计与上一发布 `f64a7ad` 完全相同；旧归档未改。
- 18 比特单组件降至 12 比特，无预先固定项；seed 7/19/42 均实际采到最优成本 3。
  V2 中仅 5/36 最大组件缩小，全部结果与无收益/不可行实例保留。
- 第二次完整评价与第一次所有非 `_seconds` 字段完全一致；14 项源码/协议指纹
  和全部 37 项输入指纹核对通过。没有用旧归档源码指纹替换当前指纹。
- Ruff lint/format（35 文件）、Mypy（11 核心模块）、pip check 通过。
- 原有 **18 passed，8.75 s**，使用此前记录的 `Test_*.py` 收集方式。
- 最新 wheel 构建并重新安装至项目内 `.venv/enhanced-clean`，确认来自 site-packages；
  最终完整 **102 passed，9.00 s**，包含新增模式真实 CPU 演示与全记录重放。
  wheel SHA-256：`d4cd80c87c604e0cedaf48b93e5200564e601d7e579b3658a5f044be7d2237aa`。
- 增加端到端 CLI 的聚焦 CI 步骤；源码/数据路径、相对文档链接、公开函数注解与
  文档字符串、Git diff 均已检查。

新评价脚本在初次运行中暴露输出目录未创建、内存 tuple 与 JSON list 比较问题，
均在归档前修复；新增命令级真实 CPU 重放测试，最终记录来自修复后的两次完整运行。
数学证明、经典算法出处、局部一致性不保证全局可行的反例及复现命令见
[ARC_PRUNING.md](ARC_PRUNING.md)。新模式的远程 CI 状态在发布后另行记录。

弧一致性版本 `4eb272a` 的个人 fork
[CI 35597416071](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35597416071)
全部成功：Ubuntu / Python 3.12.14 全新安装、Ruff/Mypy，**102 passed，19.82 s，
覆盖率 98.92%**，中等实例、连锁改约、成本取舍及新增资源边界四类 CPU 示例均通过。
官方运行 `35597421975` 仍待维护者批准。随后补录 CI 的提交只修改文档，
保留被远程验证的核心代码、测试、数据与工作流。

## 决策权重敏感性与发布候选（2026-09-21）

参数协议 `5ebcdc3` 在评价前提交；实现 `51a5067`。原核心模块、默认权重和
量子参数均未修改，只在现有 tradeoff 演示增加可选 `--sensitivity`。

- 权重 0–6、seed 7/19/42，共 21 次真实 CPU 求解，均采到认证最优、零违反；
  各权重完整原始可行集合均为相同的九个排程。w=3 处两个方案同优，代价 15。
- 全部输入、参数、counts、成本、状态及源码/环境指纹二次重放一致，仅忽略耗时。
  17 项源码/协议/配置指纹与七个派生输入哈希核对通过。
- 本地 **108 passed，31.90 s，核心行覆盖率 98.92%**。
  新增测试包含真实 21 次运行与重放、全部同优接受、非最优/无可行样本保留、
  权重零边界、成本变换不修改原输入，以及拒绝被改变的样本和源码指纹。
- Ruff lint/format（37 文件）、Mypy（11 核心模块）、pip check 通过。
- 创建 `51a5067` 的干净 detached worktree，在其中新建 `.venv`，按锁定依赖
  安装 wheel 和开发依赖；确认从 site-packages 导入，11 个安装核心文件与
  源码逐个 SHA-256 相同。**108 passed，18.78 s**，pip check 通过。
  同一干净环境下原有 **18 passed，9.35 s**。
- wheel SHA-256：`4bc41141574a11b949ca01256d707429317f5122f5da167d86a89247c61bfbac`。
- 安装完成后的命令级演示计时（含解释器启动）：tradeoff + 21 次敏感性 **6.98 s**，
  arc 资源边界 **2.40 s**，原模式资源限制 **1.82 s**，不可行案例 **2.25 s**；
  退出码依次为 0/0/5/3。总计约 13.45 s，满足本机五分钟演示目标。
  首次依赖下载另计，不将本机时间推广为跨机器性能承诺。

发布候选的命令、提交、导入位置与计时见
[release-check.json](results/sensitivity/release-check.json)。审阅路线已收敛到
REVIEW_GUIDE.md；权重说明和决策表并入原 EVALUATION_V2.md，没有新增平行教程。
完整实验保留为约 221 KB 的紧凑 JSON，便于减少生成数据的 diff 行数。

发布提交 `3846495` 的个人 fork
[CI 35600289671](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35600289671)
全部成功：Ubuntu / Python 3.12.14 全新安装、Ruff/Mypy、**108 passed，43.86 s，
覆盖率 98.92%**，全部 CPU 演示通过，包含新增的 21 次敏感性运行。
官方 `35600294948` 仍为 `action_required`，等待维护者批准。随后仅补录文档，
不改变该成功运行验证过的代码、数据、测试和工作流。

## 第七轮：Python 3.11 / 3.13 安装兼容性（2026-09-21）

在 `a27d784` 的两个干净 detached worktree 中各建项目 `.venv`，安装上一轮同一
wheel，SHA-256 仍为 `4bc41141574a11b949ca01256d707429317f5122f5da167d86a89247c61bfbac`。
实际导入来自 site-packages，11 个核心模块与 checkout 逐文件字节相同。

- Python 3.11.16：**108 passed，20.46 s，核心覆盖率 98.92%**；上游 **18 passed，10.86 s**。
- Python 3.13.11：**108 passed，22.79 s，核心覆盖率 98.92%**；上游 **18 passed，11.76 s**。
- 两个环境 `pip check` 通过，依赖逐项符合各自约束；medium 实际样本目标 4、
  AC-3 边界目标 3，零违反、零 gap；取舍与 21 次敏感性演示通过，超限/不可行
  命令分别返回 5/3。计时为本机记录，部分检查并行进行，不作性能比较。
- 原 3.12 约束在 3.11 安装预检失败；新增 `constraints-py311.txt` 仅更换 NumPy、
  SciPy、ContourPy、Sphinx 四个不支持 3.11 的固定版本。3.13 沿用原文件。
- 首次按包名统计覆盖率在两个环境均发生 9 项收集错误；改用 wheel 安装目录
  作为覆盖率来源后完整通过。错误、处理方法和环境边界保留于
  [兼容性说明](COMPATIBILITY.md)及[机器记录](results/compatibility.json)。

CI 扩展为三个 Python 版本，Mypy 按各目标版本检查，增加上游测试及两个失败
状态的显式核验。核心算法、示例、旧实验归档与 3.12 约束文件均未修改。
本地 Ruff 检查和 37 文件格式检查通过；Mypy 在新增两个目标版本均通过。

发布提交 `5678e41` 的个人 fork
[三版本 CI 35604218576](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35604218576)
已全部成功：

| Ubuntu Python | 应用测试 | 原有算法测试 | 核心覆盖率 |
|---|---|---|---|
| 3.11.16 | 108 passed，30.62 s | 18 passed，12.44 s | 98.92% |
| 3.12.14 | 108 passed，24.22 s | 18 passed，8.97 s | 98.92% |
| 3.13.15 | 108 passed，38.30 s | 18 passed，14.86 s | 98.92% |

三项任务的安装、pip check、Ruff/Mypy、全部成功演示和两个失败状态检查均通过。
3.11 另报告 15 项上游文档字符串转义的弃用警告；没有测试失败，未扩展为无关修复。
官方 [35604226057](https://github.com/OriginQ/pyqpanda-algorithm/actions/runs/35604226057)
仍为 `action_required`，待维护者批准。随后提交仅记录这些结果，未改变矩阵验证过的
代码、约束文件、测试或工作流。

## 第八轮：报告预算边界与最终审查（2026-09-21）

复现输入含 2050 个原始候选、日历剪枝后仅 2 比特，原报告在量子求解后因
MILP 的 2048 原始候选上限抛错，导致可用结果无法保存。实现提交 `1a54465`
在报告层预检独立基线预算，超过则记录 `budget_exceeded`、变量数/上限以及空证书。
直接 `solve_milp` API 的超限异常保持不变；量子算法与默认参数未修改。

- 新增五项回归：2048/2049 分界、双认证路径超预算、日历空域不可行及 CLI
  保存结果。修复前 **4 failed、1 passed**，修复后 **5 passed**。
- 完整应用 **113 passed，30.79 s，核心覆盖率 98.92%**。
- simple、medium、infeasible、tradeoff、support-chain 五条既有路径重放，
  全部非计时报告字段与修复前一致；未仅比较目标值，也未替换实际采样结果。
- Ruff lint/format（38 文件）、Mypy（11 模块）通过。

历史实验文件保持原样。此次 report.py / milp.py 源码指纹发生变化；严格对照
历史归档需使用其记录对应的提交和环境，不能忽略指纹差异假称同一构建重放。
当前版本同环境的独立重复运行仍由完整测试中的重放检查覆盖。

从 `1a54465` 建立干净 worktree、新建项目 `.venv`、安装新 wheel 与固定依赖，
确认 site-packages 导入和 11 个核心模块字节一致。完整 **113 passed，41.26 s，
覆盖率 98.92%**，原有 **18 passed，10.47 s**；pip check、medium、取舍及 21 次
敏感性运行、AC-3 和两个失败退出状态全部通过。wheel 指纹、各命令与结果见
[final-review.json](results/final-review.json)。

发布提交 `e89c78b` 的个人 fork
[CI 35606032579](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35606032579)
三版本全部成功，安装、质量检查、全部成功/失败演示均通过：

| Ubuntu Python | 应用测试 | 原有算法测试 | 核心覆盖率 |
|---|---|---|---|
| 3.11.16 | 113 passed，37.00 s | 18 passed，14.15 s | 98.92% |
| 3.12.14 | 113 passed，30.63 s | 18 passed，9.24 s | 98.92% |
| 3.13.15 | 113 passed，20.77 s | 18 passed，8.04 s | 98.92% |

3.11 仍有相同的 15 项上游文档字符串弃用警告。官方
[35606039667](https://github.com/OriginQ/pyqpanda-algorithm/actions/runs/35606039667)
仍待维护者批准运行；PR 为 Open、可合并，无评论或评审。后续仅补录验证记录，
不更改被验证代码、数据、测试与工作流。

## 评委入口与重放说明核查（2026-09-22）

本轮仅调整文档，核心代码仍与 `e89c78b` 的三版本 CI 相同，不新增测试数量或
覆盖率声明。以 `0a524ce` 的代码、现有 Python 3.12.3 项目环境运行两次
`tradeoff_demo.py --sensitivity`，分别输出至本地忽略目录
`reports/reviewer-audit/run-1` 和 `run-2`；第二次通过 `--verify-sensitivity`
比较第一次记录。两命令退出码均为 0，各包含 21 次敏感性求解，所有非计时字段
严格一致；主案例均为成本 12、gap 0、8 比特、无固定变量、9 个可行备选。

对审阅指南列出的三个历史提交，通过 `git show COMMIT:PATH` 获取文件字节并
重新计算 SHA-256，与各自原始归档中的源码指纹逐项比较：同预算 17 项、AC-3
14 项、敏感性 17 项全部一致。本轮未重新执行历史环境的完整量子实验，历史
重放结果仍以各原轮次记录为准。没有覆写旧归档，也没有忽略源码或环境差异。

文档本地链接及 Git diff 检查通过。PR #92 仍开放、可合并，无评论或评审；
官方运行 `35606039667` 仍为 `action_required`。平台作品回执尚未核验。

## 试用材料的合成演练（2026-09-22）

按 USER_TRIAL.md 原样执行四个 bash 命令块：创建本地案例、导入 CSV、运行
已有 CPU CLI、读取报告摘要，退出码全部为 0。沿用未修改的合成 chain.csv
和 outage.json，展开 10 个候选；报告目标值 6、零违反、gap 0，三项任务由
传播固定、一个 4 比特组件实际执行量子求解，输入字节与报告 SHA-256 一致。
命令日志保留在本地忽略目录 reports/trial-material-validation。

重复执行初始化和导入命令均因文件已存在而停止，案例目录所有文件字节保持不变。
模板反馈仍为“未开展真实试用”，未填入虚构参与者或意见；案例与反馈目录经
git check-ignore 确认不进入普通提交。本轮仅新增文档，未更改算法或现有测试，
不扩大既有 113 项测试和三版本 CI 的验证范围。


## 原始日历空域的构造预算检查（2026-09-22）

独立 MILP 现在先判断每项任务是否有日历可用的原始选项。若没有，报告明确
标出任务和原因、constraints=0，不调用 HiGHS；这与一般约束不可行的求解器
认证路径区分开。6 个回归覆盖停机、清洗、半开区间边界、保留可用候选，以及
512 任务 CLI 报告（退出 3，gap=null，量子线路调用 0）。

新 wheel 在项目内独立 `.venv` 安装，确认 site-packages 导入与全部 11 个核心
模块字节一致；**137 passed，32.30 s**（119 应用 + 18 原有测试），核心覆盖率
**98.93%**（838 行、9 行未覆盖）。Ruff/格式/Mypy/pip check 通过。中等实例、
主取舍案例及 21 次敏感性、AC-3 资源边界与显式失败状态均在安装包环境通过。

此处是本地结果；本轮三版本远程 CI 尚待推送后核对。历史源码指纹与实验记录
没有修改，严格重放仍应按 REVIEW_GUIDE 检出相应固定提交。


`b0a92d0` 的个人 fork [CI 35696834868](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35696834868)
三版本全部通过：3.11 / 3.12 / 3.13 各 119 项应用测试、18 项上游测试、98.93%
覆盖率及全部 CPU 演示。应用测试用时分别 37.59 / 45.03 / 36.12 s，上游测试
14.46 / 15.16 / 14.51 s。官方 [35696840057](https://github.com/OriginQ/pyqpanda-algorithm/actions/runs/35696840057)
仍为 action_required。该记录覆盖日历空域提前判定，尚未包含后续冗余配对行优化。


## 日历固定零变量的冗余配对行（2026-09-22）

新增两项真实 MILP 回归验证：省去上界固定为零变量的冗余配对行后，仍保留
原始变量数、原始选项编号和最优目标。修改前两项均失败，修改后完整应用
**121 passed，19.87 s，98.93%**。固定零变量不影响可行集合的证明见 ENHANCEMENTS。

更新后的 wheel 在另一个全新项目 `.venv` 安装，确认 site-packages 导入、11 个
模块字节相同；**139 passed，26.86 s**（121 应用 + 18 原有），核心覆盖率
**98.93%**（839 行、9 行未覆盖）。主取舍及 21 次敏感性通过；Ruff/format/Mypy/
pip check 通过。对应远程结果见下方最终核对。

## 最终交付核对（2026-09-22）

个人 fork [CI 35697552146](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35697552146)
验证代码提交 `6c43e4059c07516f5cd1df5d912f417faa37bdeb`，最终 conclusion=success。
收尾仅更新文档，不修改被测代码。

| Python | 应用测试 | 原有测试 | 应用覆盖率 | 通过轮次 |
| --- | --- | --- | --- | --- |
| 3.11 | 121，27.42 s | 18，11.00 s | 98.93% | 失败任务重跑一次 |
| 3.12 | 121，44.37 s | 18，15.34 s | 98.93% | 首次 |
| 3.13 | 121，38.23 s | 18，14.83 s | 98.93% | 首次 |

3.11 首次应用测试已通过（121 项，36.47 s），原有测试为 17 通过、1 失败：
`test/QAlgBase/Test_QUBO_GAS_origin_run.py:31` 的有限次随机 GAS 搜索返回
`[1,1,0]`，与断言的 `[[0,1,0]]` 不同。应用分支没有修改该测试、QUBO.py 或
Grover_core.py。保留失败日志后仅重跑失败矩阵任务一次，未修改测试断言或代码。
因此不能将本次记录描述为首次全绿，也不能由重跑通过断言已修复该随机失败。

原始日志保留于本地忽略目录 `reports/calendar-final-release/ci-attempt1.log`、
`ci-failed.log`、`ci-attempt2.log`；公开运行链接也可查看两轮结果。
官方 [35697558446](https://github.com/OriginQ/pyqpanda-algorithm/actions/runs/35697558446)
仍为 action_required，等待维护者批准。

当前应用与四个独立库修复的临时集成 `faee903` 在 Python 3.12 上通过
**250 项测试，27.68 s，应用覆盖率 98.93%**。它没有混入任何参赛 PR；此前
Python 3.11–3.13 各 242 项的记录仍对应应用旧提交，不扩大其验证范围。

截至本次核对，应用 #92 与库修复 #95/#96/#97/#98 均开放且可合并。这表示 Git
未检测到合并冲突，不表示维护者已经批准。停止主动扩展功能；平台提交回执、
正式审核及真实试用仍属外部待办，没有虚构完成状态。
