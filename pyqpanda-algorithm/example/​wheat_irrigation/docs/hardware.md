# 本源量子云真机测试

`--backend qcloud` 将灌溉调度的最终 QAOA 电路送到本源量子云指定的真实芯片，收集测量计数，并与同一问题的本地理想 QAOA 和经典精确解比较。这只涉及调度电路；ET₀、土壤张力预测，包括可选的 VQC 训练，仍在本地运行。

多个场景、独立重复测量以及云端小数概率的自动收集见[批量真机测试](hardware-batch.md)。批量概率结果使用 `completed_probabilities` 状态，下面的单任务整数计数管线及其 `finished` 校验规则保持不变。

## 执行流程

1. 读取场景或自定义输入，按需完成预测，构建包含供水预算松弛位的 QUBO 和归一化 Ising 模型。
2. 在 CPU 上优化 QAOA 参数。默认使用 `numpy`，可通过 `--qcloud-local-backend pyqpanda_alg` 使用算法包；两者均使用理想态期望值。
3. 导出固定参数的 `qaoa.originir` 电路和 `hardware_job.json` 任务清单。电路包括所有决策位、松弛位及测量操作。
4. 在线模式最终电路提交一次，保存任务 ID，再轮询云端任务。优化器不会为每次参数评估提交真机任务。
5. 成功获取有效测量计数后，按原有 QUBO、预算和灌溉评价规则计算真机候选及对照指标。

已完成实际 QPU 任务：`WK_C180`、11 比特、QAOA 深度 1，请求 1024 shots。云端导出的任务状态为 `Completed`，包含 705 个正概率位串。其最佳预算可行决策与经典精确解一致，但导出概率中的预算可行质量约为 16.11%，低于同参数理想分布的 74.82%。这次运行验证了任务提交、硬件完成和导出结果分析；不构成量子优势证据。

## 离线演示

进入包含 `pyproject.toml` 的项目目录，安装基础依赖：

```bash
python -m pip install -e .
wheat-irrigation --backend qcloud --qcloud-chip CHIP_NAME --qcloud-dry-run --depth 1 --maxiter 30 --restarts 1 --shots 1024 --no-plot --output outputs/qcloud_preview
```

将 `CHIP_NAME` 替换为计划使用的芯片名称。`--qcloud-dry-run` 使用默认 NumPy 本地后端时，不导入云端 SDK、不读取 API Key、不访问网络。它会运行本地参数优化并保存电路、输入和优化结果，便于检查比特数、门电路及计划测量次数。由于没有访问云端，预演不会确认账号权限、芯片可用性、队列长度或实际映射后的电路资源。

新的实验不会覆盖已有 `hardware_job.json`。重复预演或另做一组实验时，选择新的 `--output` 目录；已提交的任务用恢复命令继续查询。

三个内置场景默认需要 11 个量子比特：9 个灌溉决策位和 2 个预算松弛位。改变供水预算或输入后，比特数也可能改变。双比特相互作用需要在目标芯片上编译和映射，芯片的标称比特数并不能单独保证电路可运行或得到高质量结果。

## 配置并提交真机任务

安装云端 SDK。这个安装选项包含 `pyqpanda3>=0.4.1,<0.5`，默认本地优化使用 NumPy，无需安装 `pyqpanda_alg`：

```bash
python -m pip install -e '.[qcloud]'
```

直接从源码运行可改用 `python -m pip install -r requirements-qcloud.txt`。若本地参数优化需要 `pyqpanda_alg`，安装 `python -m pip install -e '.[qcloud,pyqpanda]'` 并添加 `--qcloud-local-backend pyqpanda_alg`。

使用本源量子云账号对应的 API Key。默认从 `QCLOUD_API_KEY` 环境变量读取，不通过 CLI 参数传递密钥。Bash 下可静默输入后导出：

```bash
# 输入 API Key 后按回车；输入过程不回显。
read -r -s QCLOUD_API_KEY
export QCLOUD_API_KEY
```

Windows PowerShell 可通过宿主环境或凭据管理工具设置同名环境变量。自定义变量名时使用 `--qcloud-token-env MY_QCLOUD_KEY`。任务清单和报告不保存密钥。

需要查询当前账号可见的真实芯片后端时，可以调用本项目的查询函数；此操作访问云端，但不提交任务：

```bash
python -c "from wheat_irrigation.hardware import list_hardware_backends; print(list_hardware_backends())"
```

返回字典中的 `True` 表示后端当前可用。首次提交还会检查芯片可用比特数，并启用云端映射和电路优化。名称包含 `amplitude`、`simulator` 或 `qvm` 的模拟器后端不能作为真实芯片提交。

在云平台确认账号拥有目标真实芯片的权限及可用机时，将下面的 `CHIP_NAME` 替换为其名称。以下命令会提交云端任务：

```bash
wheat-irrigation --backend qcloud --qcloud-chip CHIP_NAME --depth 1 --maxiter 30 --restarts 1 --shots 1024 --qcloud-timeout 300 --qcloud-poll-interval 5 --no-plot --output outputs/qcloud_run
```

`--shots` 是最终电路的测量次数。`--seed` 控制本地优化和均匀抽样等本地对照，不能使真实硬件测量确定性复现。本地模拟的最佳参数不保证在带噪芯片上仍然最佳；当前功能用于固定参数电路的真机对照，不实施硬件反馈的闭环参数优化。

## 等待与恢复

`--qcloud-timeout` 默认 300 秒，`--qcloud-poll-interval` 默认 5 秒。时限作用于本地轮询，单次 SDK 网络调用仍由 SDK 的传输超时控制，因此它不是进程总时长的严格上限。云端排队可能超过本次等待时间；本地等待结束不等于云端任务失败或取消。保留输出目录中的 `hardware_job.json`，使用原任务 ID 继续查询：

```bash
wheat-irrigation --qcloud-resume outputs/qcloud_run/hardware_job.json --qcloud-timeout 600 --no-plot
```

需要立即查询一次便返回时，可以指定 `--qcloud-timeout 0`。若首次运行使用此参数，仍会提交任务并立即查询一次；任务尚未完成则保存为 `pending`。

恢复流程使用清单保存的输入、参数和芯片，并校验模型、参数与电路的一致性；不重新训练预测器或优化参数，也不再次提交电路。它仍需可用的 SDK 和同一账号的凭据。无需再指定 `--backend` 或 `--qcloud-chip`；只允许调整输出目录、绘图开关、凭据环境变量和等待参数。默认更新清单所在目录；使用 `--output` 时，也会更新原始清单以保留最新任务状态。离线预演没有任务 ID，不能用恢复命令将预演转换为在线提交；需要选择新输出目录重新执行在线命令。

提交后的超时或网络中断应优先恢复已有任务。若提交请求失败且没有取得任务 ID，应先检查云平台任务记录，确认该请求是否已创建任务，再决定是否重新执行提交命令。

| 任务状态   | 含义                                                                                  |
| ---------- | ------------------------------------------------------------------------------------- |
| `dry_run`  | 完成离线准备，未提交云端任务                                                          |
| `prepared` | 保存了在线运行计划，尚未记录成功提交的任务 ID                                         |
| `pending`  | 已有任务 ID，尚未取得并校验整数计数；可能仍在计算，也可能因查询或解析中断而保留此状态 |
| `finished` | 云端任务完成，且整数测量计数通过校验                                                  |
| `failed`   | 云端明确报告任务失败                                                                  |

`pending` 超时返回退出码 `0`，表示本次提交或查询流程正常结束，**不表示真机已完成**。云端 `failed` 返回退出码 `2`；凭据、连接或结果校验错误也会以错误结束。自动化流程应读取 `summary.json` 的 `hardware.status`，不能只根据退出码认定实验完成。

## 参数一览

| 参数                                | 用途                                           |
| ----------------------------------- | ---------------------------------------------- |
| `--backend qcloud`                  | 启用真机测试流程                               |
| `--qcloud-chip CHIP_NAME`           | 首次运行指定目标真实芯片；恢复时从清单读取     |
| `--qcloud-dry-run`                  | 仅进行本地优化和电路、任务清单导出             |
| `--qcloud-resume PATH`              | 从已保存且有任务 ID 的清单继续查询             |
| `--qcloud-local-backend numpy`      | 默认本地理想态优化；可选`pyqpanda_alg`         |
| `--qcloud-token-env QCLOUD_API_KEY` | 指定保存 API Key 的环境变量名                  |
| `--qcloud-timeout 300`              | 本次轮询等待时限，单位为秒；`0` 表示仅查询一次 |
| `--qcloud-poll-interval 5`          | 查询任务状态的间隔，单位为秒                   |
| `--shots 1024`                      | 本例所用最终电路测量次数                       |

`--depth`、`--maxiter`、`--restarts`、`--seed`、场景与预测参数仍按原有含义工作。恢复时以保存的实验为准；重新指定不同输入或参数应作为新实验，使用新的输出目录。

## 结果解读

| 文件或字段                         | 内容                                                              |
| ---------------------------------- | ----------------------------------------------------------------- |
| `qaoa.originir`                    | 提交前的逻辑电路，包含所有测量操作                                |
| `hardware_job.json`                | 完整实验清单：电路、SHA-256、输入、优化参数、报告上下文和云端结果 |
| 清单的`report.hardware.job_id`     | 成功提交后立即保存的任务 ID                                       |
| `summary.json` 的 `qaoa_reference` | CPU 理想模拟的概率指标与参数优化记录                              |
| `summary.json` 的 `hardware`       | 芯片、状态、任务 ID、执行选项和`raw_counts`                       |
| `summary.json` 的 `qaoa`           | 仅在有效真机测量完成后写入的 QAOA 评价、计数和可行候选            |
| `report.md`、CSV 和图表            | 经典对照及已取得的真机结果；`--no-plot` 不生成图表                |

提交时关闭读出修正并请求整数测量计数。结果必须是完整宽度的二进制位串与非负整数计数，计数总和必须等于请求的 `shots`；返回浮点概率或不完整计数时会报错，不将其换算或重新抽样成所谓真机计数。若采用云端导出的概率开展独立分析，应保留原始概率总和、明确归一化方式，并区分请求的 `shots` 与未经原始计数验证的实际测量总数。电路测量 `q[i] → c[i]`，返回位串按 `c[n-1]...c[0]` 解读，最右位对应第 0 个 QUBO 变量。

真机候选从真实测量所得且满足供水预算的决策中选择，不通过修复原始位串或重新抽样制造可行方案。没有预算可行样本时，报告保留空候选。经典精确解可用作对照，不作为真机结果替补。

比较时关注同一套输入和优化参数下的目标值、预算可行样本比例以及最佳可行候选。`qaoa.total_variation_distance` 衡量真机经验频率与同参数理想分布的差异，取值为 0 到 1；它同时包含有限测量波动和硬件误差，不能直接解释成纯硬件噪声或芯片保真度。硬件噪声、读出误差和映射后的电路深度也会影响输出。一次真机运行或接近经典解的候选，不能证明量子优势、农田节水效果或产量提升。

真机模式仍依赖本地精确枚举和态矢量优化，继续受 18 比特规模上限约束；它没有扩展当前模型的可求解规模。所有内置农业参数仍为合成数据，模型及农学边界参见 [model.md](model.md) 与 [SOURCES.md](SOURCES.md)。

## 官方文档

当前 SDK 版本和平台支持请查阅[本源 pyqpanda3 文档](https://qcloud.originqc.com.cn/document/pyqpanda3-docs/zh/)。该官方入口提供 [QCloud 用户指南](https://qcloud.originqc.com.cn/document/pyqpanda3-docs/zh/tutorial/quantum-cloud-user-guide)及 [QCloud API](https://qcloud.originqc.com.cn/document/pyqpanda3-docs/zh/api/qcloud/)，用于核对账号认证、后端与任务接口。芯片名称、权限和可用性以实际账号及云平台返回结果为准。
