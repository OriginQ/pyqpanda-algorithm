# 一次小范围实验室试用

这份材料用于收集真实使用反馈。**目前仅完成合成数据演练，尚未开展真实试用。**
建议由队伍联系一位熟悉仪器预约的使用者或管理者，选择一次已经发生的设备停机，
离线回放当时的 3–5 项预约。参与者核验结果后再决定是否用于实际工作。
一次试用只能说明该案例的适用性，不能证明一般性能或量子优势。

## 1. 先确定案例和评价方式

建议安装完成后预留约 20–30 分钟；这是安排建议，不是实测完成时间。
开始前填写[反馈模板](TRIAL_FEEDBACK_TEMPLATE.md)，记录以下内容：

- 用 T01、G01、R01 这类代号表示任务、小组和设备；保留真实约束，不填写姓名、
  联系方式、课题名称或原始预约截图。身份对应表由参与者保留，无需交给项目。
- 约定一个整数时隙代表多少分钟，以及时隙 0 的含义。时间不能精确落在网格上时，
  先记录模型不适配，不自行四舍五入后宣称排程可执行。
- 记录停机、设备清洗、允许替代设备、小组冲突、工序先后及不可移动预约。
  本模型只支持单位容量设备、单小组任务、不可拆分时长和给定候选。
  技能匹配、多人交叉参与、耗材或容量大于 1 等未表达约束须在反馈中列出。
- **看求解结果前**，记录参与者的人工排法、改约数量、考虑因素和大致用时，
  再填写允许窗口及改约/移动成本。若无法给出人工排法，填“未提供”，不补造基线。
- 保留首次运行，包括不可行、未采到可行解和资源超限。修改窗口或权重时，
  建立 case-002 等新目录，写明原因，不覆盖首次记录，也不只保留成功案例。

改约成本是参与者给出的相对偏好分，不自动解释为货币或节省的工时。
试用不以“必须优于人工”作为成功条件：发现一项真实遗漏约束也是有效反馈。

## 2. 创建本地工作副本

先按[安装说明](README.md#1-三分钟演示)准备项目 `.venv`，在仓库根目录运行：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from shutil import copyfile

trial = Path("reports/user-trial/case-001")
trial.mkdir(parents=True, exist_ok=False)
source = Path("pyqpanda-algorithm/example/LabScheduling/data/bookings")
copyfile(source / "chain.csv", trial / "bookings.csv")
copyfile(source / "outage.json", trial / "calendar.json")
copyfile("Tutorials/LabScheduling/TRIAL_FEEDBACK_TEMPLATE.md", trial / "feedback.md")
print(f"已复制合成模板至 {trial}；请先确认数据来源，再填写反馈。")
PY
```

目录存在时命令停止，避免覆盖之前的数据。模板可以直接演练；原样运行始终属于
**合成演练**。真实试用须用参与者确认的数据替换 CSV 和日历，修改案例名称，并在
反馈中注明来源。`reports/` 已被 Git 忽略；它只防止普通误提交，不是访问控制。

CSV 保持首行与列序，使用 UTF-8 保存，数字填整数：

| 字段 | 如何填写 |
|---|---|
| `id`, `group` | 唯一任务代号、小组代号；同组任务不能重叠 |
| `duration` | 实验持续时隙数，不包含设备清洗 |
| `original_resource`, `original_start` | 停机前的设备代号和开始时隙 |
| `allowed_resources` | 确实兼容的设备代号，以 `\|` 分隔 |
| `earliest`, `latest`, `step` | 允许开始时刻的含端点整数网格；固定时刻令两端相等 |
| `change_cost` | 改变设备或开始时刻时支付一次的偏好分 |
| `shift_cost` | 每移动一个时隙的偏好分，按绝对位移计费 |

允许设备与时间网格的所有组合都会生成候选；若不同设备有不同时间窗口，或有
设备专属费用，应改用[显式 JSON 候选](README.md#3-输入格式)，不能让 CSV 隐含
不存在的限制。固定设备还须将 `allowed_resources` 限为该设备。
`calendar.json` 填写 `horizon`、设备 `id/cleanup/downtime` 和 `precedence`。
清洗占用设备但不占用小组，必须在 horizon 内；前序只等待实验完成，不等待清洗。
CSV 最多展开 256 个原始候选，量子模拟上限是每个组件 16 比特。

## 3. 导入并运行

确认人工排法和权重已记录后，生成输入。文件以排他方式创建，重复转换须新建案例
目录，避免悄悄覆盖输入：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from pyqpanda_alg.LabScheduling.bookings import import_bookings

trial = Path("reports/user-trial/case-001")
calendar = json.loads((trial / "calendar.json").read_text(encoding="utf-8"))
problem = import_bookings(trial / "bookings.csv", calendar)
with (trial / "input.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(problem, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print("导入成功；候选数：", sum(len(task["options"]) for task in problem["tasks"]))
PY
```

一次 CPU 求解、原始约束复核及经典认证：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pyqpanda_alg.LabScheduling reports/user-trial/case-001/input.json --reduce --seed 7 --output reports/user-trial/case-001/report.json
```

查看紧凑结果，让参与者逐项核验任务、设备和时间：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("reports/user-trial/case-001/report.json").read_text(encoding="utf-8"))
quantum = report["quantum"]
best = quantum["best"]
print(json.dumps({
    "status": report["status"],
    "solver_status": quantum["status"],
    "objective": best["objective"] if best else None,
    "violations": best["violation_count"] if best else None,
    "schedule": best["schedule"] if best else None,
    "absolute_gap": report["absolute_gap"],
    "component_qubits": quantum["reduction"]["component_qubits"],
    "runtime_seconds": report["runtime_seconds"],
    "input_sha256": report["input_sha256"],
}, ensure_ascii=False, indent=2))
PY
```

未修改的合成模板应得到目标值 6、零违反、gap 0，三个烘箱任务被传播固定，另有
一个 4 比特量子组件。实际试用不预设目标值或最优命中。

| 状态 / CLI 退出码 | 如何记录和处理 |
|---|---|
| `feasible` / 0 | 模型内可行；仍须参与者确认未遗漏实际限制。gap=0 需有经典最优证书 |
| `infeasible` / 3 | 给定候选与约束不可行；保留原因，核对录入，不随意放宽硬约束 |
| `no_feasible_sample` / 4 | 有限量子样本没给出可行解，不等于问题不可行 |
| `resource_limit` / 5 | 组件超过 CPU 预算；记录规模，不截断候选后冒充同一问题 |
| 输入错误 / 2 | 按终端提示纠正格式；未生成有效报告时不记录为求解失败 |

`classical_propagation` 表示全部由经典传播解决，不计为量子成功。
gap 为 null 表示没有可用差距，不能填成零。程序运行时间不包含人工录入和访谈；
如果记录人工用时，须分别保留这些阶段，不能由一次演练得出效率优势。

## 4. 使用者复核与可交付证据

请参与者先独立判断能否执行，再阅读解释和经典证书，并填写反馈模板。人工排法
只有在同一候选、约束和成本定义下才适合比较；否则写明不可直接比较，不填收益比例。
若需定量核对人工排法，交由工程师使用现有 `validate_assignment` 对原始候选索引
独立检查，检查前只报告参与者的定性意见。

一次案例保留 `bookings.csv`、`calendar.json`、`input.json`、`report.json`、
`feedback.md` 及实际代码提交号（`git rev-parse HEAD`）。输入指纹已在报告中；
反馈标明是独立使用还是由队伍协助，记录失败、不同意见和缺失约束。
默认不公开真实试用材料。参与者明确同意公开的范围与去标识摘要单独记录后，
才能将其作为参赛证据；对外联系与公开由队伍协调，不由本流程自动发送。

没有实际参与者的数据或反馈时，将状态保持为“材料准备完成 / 合成演练通过”。
即使真实试用成功，也只描述这一案例和参与者意见，不称为部署、普遍收益或量子加速。
