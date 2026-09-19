# 批量真机测试

`python -m wheat_irrigation.hardware_batch_job` 使用一次 `QCloudBackend.run(list[QProg], shots, options)`，将多个场景电路提交为**一个云端任务**。清单保存一个父任务 ID，返回结果按电路列表拆分分析。

`python -m wheat_irrigation.hardware_batch` 为三个内置场景批量准备电路、提交独立真机任务并恢复查询。当前批量路径直接分析云端返回的小数概率，完成状态记为 `completed_probabilities`；原单任务路径仍要求真实整数计数，只有计数通过校验才记为 `finished`。两种完成状态的含义不同。

已在 `WK_C180` 完成三个场景真机任务，3/3 通过概率结果验证，0 个失败，其中区分物理目标最优方案与完整 QUBO 最优态。

## 运行一个批次

在包含 `pyproject.toml` 的项目目录中执行。基础依赖足以运行离线准备；在线提交和查询另外需要兼容当前 Python 与平台的 `pyqpanda3`，安装及单任务流程见[真机测试说明](hardware.md)。批量模块使用 `fcntl.flock` 防止多个进程同时操作同一目录，目前支持 macOS/Linux；Windows 还有待实现该批量锁，不能直接运行此批量入口。

```bash
# 仅本地优化、生成电路和清单，不读取密钥、不访问云端。
python -m wheat_irrigation.hardware_batch prepare --output outputs/hardware_batch_example

# 以下命令实际提交尚未提交的任务。
python -m wheat_irrigation.hardware_batch submit --output outputs/hardware_batch_example

# 用已保存的任务 ID 查询和分析，最长轮询 300 秒。
python -m wheat_irrigation.hardware_batch collect --output outputs/hardware_batch_example --timeout 300

# 只读取本地状态并刷新汇总，无须 SDK 或密钥。
python -m wheat_irrigation.hardware_batch status --output outputs/hardware_batch_example
```

默认配置为 `WK_C180` 芯片，`jointing`（拔节）、`heading`（抽穗）、`grain_filling`（灌浆）各 1 次，共 3 个任务；每个任务请求 1024 次测量，QAOA 深度 1，本地优化 `maxiter=100`、`restarts=2`、`seed=7`。每个默认场景含 11 比特，所有本地参数优化均使用 NumPy 理想模拟。提交前会查询芯片当前可用性，保存的芯片名称不保证账户权限或芯片一直可用。

也可以用 `run` 完成准备、提交和收集：

```bash
python -m wheat_irrigation.hardware_batch run --output outputs/hardware_batch_example --timeout 300
```

对已有批次执行 `prepare` 或 `run` 时，配置必须与保存值一致；更改实验设置应使用新的输出目录。`submit`、`collect` 和 `status` 直接使用 `batch.json` 中的配置，不用命令行的场景、重复数、shots 等默认值覆盖它。

## 凭据和 SDK

批量入口依次读取非空的 `QCLOUD_API_KEY`、兼容变量 `OPENAI_API_KEY`，最后读取 `--key-file` 指定的文件，文件默认路径是 `/tmp/wheat-qcloud-api-key`。无论使用哪个变量名，内容都必须是**本源量子云 API Key**。

在执行命令的同一个 Bash 终端中静默输入：

```bash
read -r -s QCLOUD_API_KEY
export QCLOUD_API_KEY
```

如果运行进程不能继承该终端的环境，可以保存为仅本人可读的本地文件，再使用 `--key-file`。例如：

```bash
(umask 077; printf '%s' "$QCLOUD_API_KEY" > /tmp/wheat-qcloud-api-key)
python -m wheat_irrigation.hardware_batch collect --output outputs/hardware_batch_example --key-file /tmp/wheat-qcloud-api-key
```

默认优先读取环境变量，指定 `--key-file` 不会覆盖已经存在的非空环境变量。密钥内容不作为命令行参数，也不写入任务清单或结果报告。在线查询的原生 SDK 日志仅在子进程内捕获，保存的 `cloud_response.json` 只包含经筛选的任务及测量字段。

`--sdk-path` 可指定包含 `pyqpanda3` 包的目录，并传递给只读查询子进程。必须使用与该 SDK 二进制架构匹配的 Python；该路径不会自动安装 SDK 或转换架构。此兼容查询实现处理旧 SDK 在完成结果解析时出错、但认证响应仍提供概率的情况，不扩大项目声明的 `pyqpanda3>=0.4.1,<0.5` 支持范围，也不代表所有 SDK 版本均已通过在线集成验证。

## 重复测量与参数

`--repeats` 为每个场景建立多个独立硬件任务。一个场景只进行一次本地参数优化，同场景各次测量使用完全相同的最终电路；不会随重复编号重新优化。比如只测试拔节和抽穗，每个场景重复 3 次，共 6 个任务：

```bash
python -m wheat_irrigation.hardware_batch prepare --output outputs/hardware_batch_repeated --scenarios jointing heading --repeats 3 --shots 1024
python -m wheat_irrigation.hardware_batch submit --output outputs/hardware_batch_repeated
python -m wheat_irrigation.hardware_batch collect --output outputs/hardware_batch_repeated --timeout 300
```

| 参数                       | 默认值与用途                                          |
| -------------------------- | ----------------------------------------------------- |
| `--output`                 | 必填，批次清单、场景计划和任务结果的目录              |
| `--chip`                   | `WK_C180`，准备时指定真实芯片                         |
| `--scenarios`              | 三个内置场景，可选择不重复的子集                      |
| `--repeats`                | `1`，每个场景的独立真机任务数                         |
| `--shots`                  | `1024`，每个最终电路请求的测量次数                    |
| `--depth`                  | `1`，QAOA 层数                                        |
| `--maxiter` / `--restarts` | `100` / `2`，每个场景本地参数优化设置                 |
| `--seed`                   | `7`，控制本地优化和对照抽样，不控制硬件噪声或测量结果 |
| `--timeout`                | `300` 秒，本次收集轮询时限；`0` 表示遍历查询一次      |
| `--poll-interval`          | `10` 秒，未完成任务的查询轮次间隔                     |
| `--key-file`               | `/tmp/wheat-qcloud-api-key`，环境变量无可用值时读取   |
| `--sdk-path`               | 可选，将给定 SDK 包目录加入导入路径                   |

轮询时限不是严格的进程运行时长上限：每轮依次查询所有未完成任务，单个只读查询子进程另有 45 秒超时。排队或本地等待结束不代表云端失败，也不会取消已经提交的任务。

## 持久化和恢复

每次提交前先将该任务记为 `submitting` 并写入 `batch.json`；获得云端任务 ID 后立即保存。已有 ID 的任务只通过该 ID 续查。若提交请求中断且没有保存 ID，程序保留 `submitting` 或 `submission_unknown`，并阻止自动重新提交整个批次，避免把结果不明的请求重复执行。

| 状态                      | 含义与后续动作                                            |
| ------------------------- | --------------------------------------------------------- |
| `prepared`                | 离线计划已准备；使用`submit` 提交                         |
| `submitting`              | 已开始提交；若没有 ID，须先核对云平台任务记录             |
| `submission_unknown`      | 提交结果不明且没有 ID；须核对云平台记录，不能直接重试     |
| `submitted`               | 已保存任务 ID；使用`collect` 查询                         |
| `pending`                 | 最近查询尚未得到已完成且验证通过的概率结果；继续`collect` |
| `completed_probabilities` | 云端完成，任务身份、电路和概率结果已通过验证              |
| `failed`                  | 云端明确报告失败，作为终止状态保留                        |

对无 ID 的不确定提交，应先确认云平台是否创建了对应任务，再人工核对批次条目；目前没有自动认领未知任务的 CLI。不要通过删除清单或创建新目录来直接重发该请求。已有 ID 时恢复命令如下，它不提交新电路：

```bash
python -m wheat_irrigation.hardware_batch collect --output outputs/hardware_batch_example --timeout 300
```

批量目录的操作使用操作系统锁；另一个进程持有该锁时会报错退出，进程中断后锁自动释放。退出码 `0` 只说明本次流程结束，批次仍可能处于排队、超时或查询延后状态；判断全部成功必须确认 `batch_summary.json` 中 `completed_jobs == planned_jobs` 且 `failed_jobs == 0`。查询错误仅记录异常类型并保留可续查状态，失败或不确定提交等情况可返回 `2`。

## 结果文件和指标

| 相对于批次输出目录的路径                           | 内容                                                                                                |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `batch.json`                                       | 保存配置、每个独立任务的 ID、状态、电路哈希和时间记录；`execution_environment` 记录 SDK/Python 版本 |
| `plans/<scenario>/hardware_job.json`               | 每个场景的可重放准备清单，含输入、参数及电路                                                        |
| `plans/<scenario>/qaoa.originir`                   | 含测量操作的逻辑电路                                                                                |
| `jobs/<scenario>_<repeat>/initial_sdk_result.json` | 初次提交时 SDK 成功返回的结果；SDK 解析异常时可能不存在                                             |
| `jobs/<scenario>_<repeat>/cloud_response.json`     | 最近取得的认证云端任务响应快照，保留原始概率值                                                      |
| `jobs/<scenario>_<repeat>/result_analysis.json`    | 完成后的概率校验、原始和归一化指标、理想模拟及候选分析                                              |
| `batch_summary.json` / `batch_summary.csv`         | 每次独立任务的状态和可比较指标，不合并为虚构计数                                                    |
| `report.md`                                        | 可读汇总表、任务 ID 和证据链接                                                                      |

概率分析验证任务 ID、芯片、完成状态、测量宽度，以及已保存输入、参数和电路的一致性。返回位串按十六进制解码，拒绝超出宽度的键、重复状态、非有限或不在 `[0, 1]` 内的值。概率总和与 1 的差不能超过 `返回条目数 × 1e-7`，该保守界覆盖逐项保留七位小数时的舍入或截断；超出容差的结果不会标记完成。

分析同时保存原始概率加总指标和显式除以概率总和后的归一化指标，并保存总和、容差及逐状态原始概率。汇总表的预算可行率和 TVD 使用归一化分布。预算可行率表示灌溉次数满足预算；零松弛残差率要求完整预算等式成立；完整 QUBO 最优态概率要求包含松弛位的整个状态达到最优能量。这些指标均与相同参数的理想模拟比较。

最佳候选仅从具有真实正概率且满足预算的观测状态中选择，松弛位不修复；若没有这样的观测，候选为空，经典最优解仍作为独立对照。`requested_shots` 是请求值；没有原始整数计数时，不能将其当作已经校验的实际测量总数。分析不把概率乘以 shots、四舍五入或重新抽样生成所谓原始计数。

TVD 包含有限测量、硬件、编译映射和导出精度的影响，不能直接视为芯片保真度。少量测量或命中经典最优候选均不足以证明量子优势。三个场景仍使用合成农业数据，结果不能解释为实际农田节水或增产；本地精确枚举及态矢量计算的 18 比特上限也仍然适用。

---

# 本源量子原生批量真机任务
