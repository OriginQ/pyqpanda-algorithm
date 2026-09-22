# 【创新应用】新增基于QUBO的共享实验室最小扰动重排应用

参赛队伍：**youshen**，已完成官网报名。关联 #13；目标分支 develop。

## 解决的问题与应用贡献

设备临时停机后，如何重排实验、满足清洗/小组互斥/工序前序，同时降低原预约
变更成本？新增完整离线应用 LabReschedule，从 JSON 或 CSV 预约窗口生成
可追溯 QUBO，在 PyQPanda3 CPU 上求解，再由原始输入 MILP 与小规模穷举核验。

完整取舍示例：烘箱故障后，三个实验可整体后移（成本 12、改约三单），也可
付费使用备用设备（成本 14、改约一单）。三项决策全部留在 8 比特组件，
没有预处理固定项；默认实际量子样本得到成本 12、零违反，随后由双经典基线认证。
原有 CSV 连锁传播案例也保留，展示被迫改约的删除原因链。

建议从[审阅路线](https://github.com/youshen2025/pyqpanda-algorithm/blob/contest/qubo-scheduling-app/Tutorials/LabScheduling/REVIEW_GUIDE.md)开始，按路线定位核心代码、
数学证明、关键测试及实验记录；数据与图表占 diff 的大部分。审阅路线还区分
当前代码重复运行与历史归档核验，列出匹配的源码提交和独立环境操作命令。
另提供[小范围试用流程](https://github.com/youshen2025/pyqpanda-algorithm/blob/contest/qubo-scheduling-app/Tutorials/LabScheduling/USER_TRIAL.md)
和空白反馈表；已通过合成数据演练，尚未取得真实使用者反馈。

首次审阅可先看 11 个核心模块与 11 个测试模块（另含 3 个检查配置），再按审阅路线抽查生成的实验
归档。最近修复报告层的基线预算处理，量子线路与默认训练/采样参数保持不变。

贡献集中在可证明的资源节省与可审计求解：

- 原始候选日历剪枝、单候选传播、严格独立冲突分量、固定成本和原变量回填；
  不启发式冻结“未直接受故障影响”的预约，不静默截断或以经典解替代量子样本。
- 可选 AC-3 删除没有任何对方候选支持的选择，保存完整删除依据；构造的
  四工序组件从 18 降到 12 比特，三个固定种子实际采到成本 3，全部决策仍在量子组件。
- 在现有业务演示中增加七组改约权重，解释何时使用备用设备；21 次真实量子
  运行与完整可行集合认证，阈值处接纳全部并列最优，不改变核心默认参数。
- XY 线路中删去可达子空间上恒零的 one-hot 罚项，保留完整 QUBO 审计和原缩放。
  medium 的原生 CNOT 从 78 降至 54，深度 45 降至 35；相同参数分布等价。
- 独立原始输入 MILP 不使用 QUBO 的剪枝表或冲突图；区分最优证书、不可行、
  限时 incumbent、量子抽样失败和组件超限，记录所有失败。
- 第一批 12 个实例 × 10 个种子保留；第二批独立冻结 36 个异构实例，每格 4 个
  数据种子、每例 10 个优化种子，比较日历剪枝/传播/分解/相位精简各阶段。
- 补充同总训练上限/采样预算对照，并加入化简后的均匀随机基线，区分结构化简
  与量子分布收益；完整保留随机持平或胜出的结果，不以额外预算制造优势。
- 输入只读一次，报告指纹绑定实际求解字节；CLI 拒绝输入/输出同文件（含链接），
  写完临时报告再替换，避免误覆盖输入或因写入失败破坏旧报告。
- 原始 MILP 超过候选预算时明确跳过该基线，保留量子结果和仍可用的穷举证书；
  两者都无法认证时 gap 为 null。回归覆盖 2048/2049 边界、不可行及双基线超限。

QUBO、XY、传播和 MILP 均来自已有理论；本贡献是应用建模、编译、验证与复现流程，
不声称发明新的通用量子算法或量子加速。题目区别于开放灌溉应用 #60；
已有调研快照和 scope 在 PLAN.md 与教程中说明。

## 数学与架构

每个候选一个二进制变量，成本为偏好加原预约变更费用。完整 QUBO 为
`Q = sum(c_i*x_i) + A*sum_t(sum_Dt(x_i)-1)^2 + A*sum_conflicts(x_i*x_j)`。
对非负成本取 `A=1+各任务最大存活候选成本之和`；存在可行解时，任一不可行
位串能量高于所有可行解。`x=(I-Z)/2` 转为 Ising，成本相位按同一 A 缩放。
W 初态与域内 XY 交换保持每任务单激发，故可只在 XY 相位中删除 one-hot 罚项。
X/H/RX 对照仍保留完整罚项。原生 CRY 与 CNOT 分别报告，不作硬件分解计数承诺。

默认传播只删与必选候选冲突的选择；可选弧一致性删除与另一任务所有存活候选
都冲突的选择。两者均保留全部可行排程；组件无跨界禁止对，回填与原可行排程一一对应，
成本等于固定偏移加组件成本。近似组件求解没有全局最优保证。

主要改动集中于 `pyqpanda_alg/LabScheduling/`、`example/LabScheduling/`、
`test/LabScheduling/`、`Tutorials/LabScheduling/` 及文档索引/聚焦 CI。
复用上游 W 态、XY 交换与成本相位组件，未修改上游算法或进行无关重构。

## 安装与演示

在仓库根目录使用 Python 3.12 虚拟环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -c pyqpanda-algorithm/example/LabScheduling/constraints-py312.txt -e ./pyqpanda-algorithm -r pyqpanda-algorithm/example/LabScheduling/requirements-dev.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python pyqpanda-algorithm/example/LabScheduling/tradeoff_demo.py --sensitivity
```

普通 JSON CLI：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pyqpanda_alg.LabScheduling pyqpanda-algorithm/example/LabScheduling/data/medium.json --reduce --output reports/medium.json
```

输出到项目 reports；不需要 Key、GPU、真机、Web 或数据库。pandas/scikit-learn
补充项用于上游导入链，声明限定在应用开发依赖内。

Linux x86_64 上 Python 3.11 / 3.13 也已分别通过独立 wheel 安装、108 项应用
测试、18 项原有算法测试和完整演示。3.11 安装时换用 `constraints-py311.txt`；
3.13 沿用上述约束。版本、失败记录和范围见
[安装兼容性](https://github.com/youshen2025/pyqpanda-algorithm/blob/contest/qubo-scheduling-app/Tutorials/LabScheduling/COMPATIBILITY.md)。

## 测试与实测证据

WSL / Python 3.12.3 / PyQPanda3 0.4.1：

- 当前 121 项应用测试通过，核心行覆盖率 98.93%；原有 18 项有效测试见下方矩阵。
- Ruff lint/format、Mypy 11 模块通过；真实 CPUQVM 测试，无模拟器 mock。
- 原始日历可直接证明某任务没有候选时，MILP 提前报告不可行，避免构造两两
  约束；不调用 HiGHS，也不读取 QUBO 剪枝结果。6 项新回归包括真实边界问题和
  512 项任务的 CLI 输出；有可用选项的问题继续建模与求解。另用 2 项真实 MILP
  回归验证固定零变量的冗余配对行可被省去，原始索引及最优值保持不变。
- 最新 wheel 在项目内全新的 `.venv` 安装，确认 site-packages 导入
  与核心文件哈希；121 项应用测试及 18 项原有测试通过。含敏感性分析、资源边界
  和两个失败状态的完整命令路线验证成功，安装记录见 VALIDATION.md。
- 原有 64/4096 态 QUBO/约束等价与 4096 态 Ising 验证保留。
- 100 个随机小问题核对化简前后全部可行集合和回填成本，再与原始 MILP 比较。
- 可选弧一致性在 200 个随机小问题及既有 V2 全部 36 个输入上核对全部原始
  可行集合与成本；5 个 V2 实例的最大组件缩小，其他 31 个没有该项收益。
  18→12 边界案例的三种子与结构审计完整重放；保留局部有支持但全局不可行反例。
  命令加 `--reduce --pruning arc`；证明和归档见 `Tutorials/LabScheduling/ARC_PRUNING.md`。
- 改约成本敏感性协议先以 `5ebcdc3` 冻结：w=0–6、seed 7/19/42，共 21 次真实
  QAOA 均采到经独立认证的最优；阈值 w=3 时两个方案同为 15，完整重放一致。
  这是已知合成案例上的业务解释，不是新留出集或量子优势证明。
- simple/medium 各 30 组 p=1/2/3 随机角度验证 XY 相位精简；单候选混合域另测。
- 12 个冻结评价实例含 9 个可行、3 个不可行，穷举和 MILP 结论全部一致。
  可行实例 86/90 个训练种子的单次最优概率高于均匀 one-hot，4 个退化，全部保留。
  原始/精简线路在相同已训练参数上的最大概率差为 2.23e-16 以下。
- 120 次训练再次复现，除耗时外参数、counts、分布指标和独立采样结果完全一致。
- 20/35/50 任务的合成链仅运行经典 MILP，目标 40/71/100；明确标记未执行量子求解。
- 第二批支持 2–4 候选、1–3 时隙时长、0–2 清洗时间和三类故障；36 个原排程
  均验证无故障时合法。故障后 24 可行、12 不可行，全部原始可行集合与回填成本一致。
- 第二批 240 个可行训练种子有 206 个最优概率高于均匀、34 个退化；一个实例连
  种子均值也退化，均保留。分解可能增加总训练/采样预算，不作同预算优势声明。
- 360 条配对求解记录完整重放，除耗时外完全一致；其中纯经典传播明确标注。
- 同总预算补充评价重新运行全部 360 组，并完整重放。22 个仍需量子的可行实例
  × 十种子中，分量 QAOA 最优 220/220、直接 QAOA 217/220；两条随机基线也为
  220/220，不宣称 512 shots 下量子优于随机。低 shots 条件曲线及退化实例见
  `Tutorials/LabScheduling/MATCHED_BUDGET.md`，曲线不包括训练成本。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -c test/LabScheduling/pytest.ini test/LabScheduling --cov=pyqpanda-algorithm/pyqpanda_alg/LabScheduling --cov-fail-under=95
```

证明、接口、协议、全部失败记录、资源与命中曲线见
`Tutorials/LabScheduling/ENHANCEMENTS.md`；安装及 CI 实际状态见 VALIDATION.md。
旧 schema 1 开发基线独立保留，新版为 schema 2，默认使用精简 XY。

个人 fork 的 [Ubuntu / Python 3.11–3.13 CI](https://github.com/youshen2025/pyqpanda-algorithm/actions/runs/35697552146)
在代码提交 `6c43e4059c07516f5cd1df5d912f417faa37bdeb` 上最终全部通过：各全新安装、
质量检查、121 项应用测试（98.93% 覆盖率）、18 项原有测试及全部 CPU 演示。
3.12 / 3.13 首次通过；3.11 首次应用测试通过，但未修改的上游 GAS 随机搜索
测试失败（17 通过、1 失败），仅重跑该失败矩阵任务一次后全部通过。保留首次
失败日志，未修改原测试或算法来规避失败；完整记录见 VALIDATION.md。
官方 [35697558446](https://github.com/OriginQ/pyqpanda-algorithm/actions/runs/35697558446)
仍待上游批准；fork CI 不代替上游审批。后续收尾提交仅更新验证文档。

## 限制

每个量子组件最多 16 比特，总编译候选最多 256；精确枚举最多一百万组合，
MILP 最多 2048 个原始候选。报告分别记录超预算跳过状态，限时或无证书不冒充最优。
只处理非负整数成本、确定时长、单位容量、单小组任务。训练与概率诊断为
指数复杂度的无噪声模拟；训练后的 shots 命中曲线不是端到端加速。
第一批每格一个数据种子，第二批每格四个，仍是小规模合成证据，没有实际实验室用户验证。
经典基线在此规模更快，量子效果并非每个种子改善，硬件噪声下不保证相位精简等价。


## 独立库兼容验证

另行修复的 #95/#96/#97/#98 分别处理 XY 边界、QAOA 寄存器、QUBO 符号位和
GAS 同值解；这些改动不包含在本应用 PR 中，本应用也不依赖它们才能运行。
2026-09-22 在独立 checkout 合并应用代码 `7e4e663` 与上述修复，Python 3.11 / 3.12 /
3.13 各通过 242 项测试，应用覆盖率均为 98.92%。这是本地兼容性证据，
没有扩大原 fork CI 的覆盖范围或宣称官方审核通过。

上述集成随后加入当前两项 MILP 构造优化，在 Python 3.12 上通过 250 项测试，
应用覆盖率 98.93%；三版本的 242 项历史记录仍对应先前明确标注的应用提交。
