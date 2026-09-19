# 黄河流域缺水地区小麦灌溉量子计算应用

这是一个小麦灌溉预测与调度项目。以黄河流域渭河关中平原的缺水冬小麦田为应用背景，先预测参考蒸散量 **ET₀** 和土壤水分张力 **ψ**，再计算未来一周“哪个地块、哪天灌溉”，在有限渠首取水预算下比较 **QUBO + QAOA**、经典精确求解、缺水触发贪心和均匀抽样。也可直接提供 ET₀，使用直接输入调度模式。

主量子调度流程默认使用 `pyqpanda_alg` 的 `QAOA`，由 `pyqpanda3` 在 CPU 上模拟电路；需要安装算法包及量子依赖。NumPy QAOA 和经典求解可通过 `--backend numpy`、`--backend classical` 显式选择。`--backend qcloud` 支持本源量子云真机测试：先在本地优化参数，再将最终电路提交到指定芯片，使用云端测量计数与理想模拟、经典精确解比较。

三个内置场景的气象、土壤、作物系数、生育期日期和管理参数全部为合成数据。项目用于模型研究和计算演示，未接入实测农田数据，也未经过田间验证；计算结果不能解释为增产、节水比例。后期根据特定区域的实际情况和条件替换成实际数据。2026-09-18 已在 `WK_C180` 完成一次 11 比特真机任务，并分析云端导出的概率结果；SDK 的整数计数读取尚未完成，详见[本次测试记录](docs/hardware-test-2026-09-18.md)。[数据来源说明](docs/SOURCES.md)区分了农学概念依据与演示参数。

## 安装与运行

需要 Python 3.11 或以上；默认量子后端还需要适配当前平台的 `pyqpanda3`。进入本项目目录（含 `pyproject.toml`）后执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pyqpanda]'
wheat-irrigation
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活。`.[pyqpanda]` 安装应用及量子依赖，包括 `pyqpanda_alg`。本目录运行所需的三个小麦场景随 Python 包安装。若需要使用父仓库中的算法开发版本，可先执行 `python -m pip install -e ../..`。

以下三个入口等价，默认计算拔节期场景并输出报告、数据与图表：

```bash
wheat-irrigation
python -m wheat_irrigation
python irrigation.py
```

只需要 JSON、CSV 和 Markdown 报告时：

```bash
wheat-irrigation --no-plot
wheat-irrigation --backend classical --no-plot
```

只使用 NumPy QAOA 或经典求解时，可安装基础依赖并显式选择后端：

```bash
python -m pip install -e .
wheat-irrigation --backend numpy --no-plot
wheat-irrigation --backend classical --no-plot
```

直接从源码运行时，基础依赖也可用 `python -m pip install -r requirements.txt` 安装；默认量子流程可用 `python -m pip install -r requirements-pyqpanda.txt` 安装。经典调度配合 `ridge` 预测和 `--no-plot` 的实际运行仅需 NumPy；QAOA 参数优化和 `vqc` 预测训练额外使用 SciPy，图表使用 Matplotlib。基础安装会装齐这三个依赖，默认量子流程还需上述量子依赖。

## 量子计算机真机测试

本源量子云的**一个任务包含多个电路**使用 `python -m wheat_irrigation.hardware_batch_job`，详见[原生批量真机任务](docs/hardware-batch.md)。已在 `WK_C180` 完成包含三个场景电路的原生批量任务，同一响应返回通过验证的概率结果。

需要为多个场景或重复测量分别建立任务时，使用[多个独立单任务的批量测试](docs/hardware-batch.md)，入口为 `python -m wheat_irrigation.hardware_batch`。两种入口均保留云端原始概率，验证完成后记为 `completed_probabilities`，不从概率合成整数计数。

先离线生成待执行电路与任务清单。默认用 NumPy 在 CPU 上优化 QAOA 参数，此步骤只需基础依赖，不需要云端密钥或 `pyqpanda3`：

```bash
wheat-irrigation --backend qcloud --qcloud-chip CHIP_NAME --qcloud-dry-run --no-plot --output outputs/qcloud_preview
```

`CHIP_NAME` 替换为账号可访问的真实芯片名称。实际提交时安装 QCloud 依赖，将 API Key 写入环境变量，再运行：

```bash
python -m pip install -e '.[qcloud]'
# 输入 API Key 后按回车；输入过程不回显。
read -r -s QCLOUD_API_KEY
export QCLOUD_API_KEY
wheat-irrigation --backend qcloud --qcloud-chip CHIP_NAME --shots 1024 --qcloud-timeout 300 --no-plot --output outputs/qcloud_run
```

源码运行也可执行 `python -m pip install -r requirements-qcloud.txt`。在线命令会提交一次最终 QAOA 电路；参数优化和经典对照仍在 CPU 上完成。程序保存任务 ID，等待超过指定时间后可从清单继续查询，避免重复提交：

```bash
wheat-irrigation --qcloud-resume outputs/qcloud_run/hardware_job.json --no-plot
```

报告区分本地理想模拟和真机测量；自动报告仅在整数测量计数通过校验后写入 `finished`。本次真机任务已由云端导出确认为 `Completed`，但返回的是概率，计数管线仍保留 `pending`。新实验需使用未包含 `hardware_job.json` 的输出目录。芯片选择、凭据、参数、结果解读和恢复流程见[真机测试说明](docs/hardware.md)。

## ET₀、土壤水分预测接入调度

```bash
# 以拔节场景的合成历史数据训练，预测未来 7 天，再进行调度
wheat-irrigation --forecast-demo --backend classical --output outputs/forecast_ridge

# 同一流程使用可训练的 4 比特 VQC 预测器；调度仍可独立选择经典或 QAOA
wheat-irrigation --forecast-demo --forecast-model vqc --forecast-maxiter 30 --backend classical --output outputs/forecast_vqc

# 抽穗场景配合 QAOA 调度；默认调用 pyqpanda_alg
wheat-irrigation --scenario heading --forecast-demo --output outputs/heading_forecast

# 自有田块参数和独立预测数据
wheat-irrigation --input my_field.json --forecast-input my_forecast.json --output outputs/my_forecast
```

流程为 **历史气象与张力观测 → 按时间留出评估 → ET₀ 与无新增灌溉张力预测 → 标定曲线换算根区可利用水量 → 供水预算下的灌溉调度**。默认 `ridge` 预测器只使用 NumPy；`vqc` 是在 CPU 上训练旋转门参数的量子电路小演示。`--forecast-model` 选择预测器，`--backend` 选择调度求解器，两者独立。

`--forecast-demo` 为场景生成 84 天的合成历史、未来天气和土壤水分特征曲线，给定场景与种子可复现；`--forecast-input` 读取自己的预测 JSON，两者互斥。未来天气必须是规划时已经获得的预报，程序不调用在线天气服务。ET₀ 的训练标签由历史天气通过 FAO56 计算，未来同时导出学习模型预测和 FAO56 对照；它们都不等于实测蒸散量。土壤模型预测的是 kPa 单位的张力，须经本地标定的土壤水分特征曲线转换，才能进入 mm 单位的调度模型。

预测模块采用日尺度天气特征和滞后状态，提供 NumPy 岭回归与 4 比特可训练 VQC 两种实现。预测经过时间留出评估，再转换为固定的无新增灌溉水量基线，供调度模型使用。输入格式、时间对齐、评估与耦合假设见 [预测与调度说明](docs/forecasting.md)。

## 三个小麦生育期场景

```bash
wheat-irrigation --list-scenarios
wheat-irrigation --scenario heading
wheat-irrigation --scenario grain_filling
```

| 场景               | 生育期     | 输入文件                                                       |
| ------------------ | ---------- | -------------------------------------------------------------- |
| `jointing`（默认） | 拔节期     | [jointing.json](wheat_irrigation/data/jointing.json)           |
| `heading`          | 抽穗扬花期 | [heading.json](wheat_irrigation/data/heading.json)             |
| `grain_filling`    | 灌浆期     | [grain_filling.json](wheat_irrigation/data/grain_filling.json) |

三个场景分别描述独立的一周，不构成连续全生育期预测。它们使用不同初始水分、每日作物系数、目标水分及偏差权重。直接输入模式下，实际优化由 `crop_coefficient`、`stress_weights`、`target_mm` 等数值驱动；`growth_stage` 用于标记和汇总，不自动推导农学参数。预测模式使用学习得到的固定水分基线，Kc 等通量参数的作用范围见[耦合说明](docs/forecasting.md#日末土壤张力和水量基线)。

共同的演示设定为三个相邻、各 1 ha 的地块，规划 7 天；每地块开放 3 个灌溉日，共 9 个二进制决策。单次根区净灌水深 25 mm，输配水效率 75%，渠首总取水预算 700 m³：

```text
每次根区净水 = 25 mm × 1 ha × 10 = 250 m³
每次渠首取水 = 250 / 0.75 ≈ 333.33 m³
最多灌溉次数 = floor(700 / 333.33…) = 2
用满两次时：取水约 666.67 m³，根区净水 500 m³，预算剩余约 33.33 m³
```

这些数值用于演示预算离散化，并非当地灌溉定额。因为所有地块等面积、剂量和效率相同，体积预算才能精确转换成事件数上限；异面积、变剂量或分地块效率需要重新设计带权体积约束。

## 调整缺水条件与自定义地块

```bash
wheat-irrigation --water-budget-m3 350 --efficiency 0.8 --output outputs/reduced_supply
wheat-irrigation --scenario heading --export-input my_field.json --no-plot
wheat-irrigation --input my_field.json --output outputs/my_field
```

`--export-input` 保存本次生效的输入，并继续求解；随后可以编辑 JSON 的天气、土壤、窗口及阶段参数。启用预测时，它保存预测已经解析成 ET₀ 和水量调整项的调度输入；再次通过 `--input` 运行即可复现调度，无须重新训练。`--scenario` 和 `--input` 互斥。`--water-budget-m3` 覆盖毛取水预算，`--efficiency` 覆盖净水/取水比例；输入若还指定 `budget_events`，次数上限与体积上限同时生效，取较小者。

默认输出是**当前工作目录**下的 `outputs/<场景名>/`；自定义输入使用其文件名作为子目录。预测模式加上 `_forecast_<模型>` 后缀，例如 `outputs/jointing_forecast_ridge/`；真机模式再加 `_qcloud`，离线预演加 `_qcloud_dry_run`。内置数据通过包资源读取，与当前目录无关。普通本地运行会覆盖同目录内同名文件，需保留不同实验时显式设置 `--output`；新的真机实验会拒绝覆盖已有 `hardware_job.json`，须恢复旧任务或选择新目录。`--no-plot` 不更新已有图表，比较不同运行时应使用各自的输出目录。

量子求解参数示例：

```bash
wheat-irrigation --depth 3 --maxiter 200 --restarts 4 --shots 4096 --seed 42 --output outputs/qaoa_seed42
wheat-irrigation --help
```

`--restarts` 包含第一个固定初值；`--maxiter` 是每个起点的优化评估预算。`--shots` 只控制优化结束后的有限抽样，优化本身使用理想态的精确期望；在 `qcloud` 后端中，它同时指定最终电路的真机测量次数。`--depletion-fraction` 可统一覆盖贪心策略的允许耗水比例，默认采用各地块逐日参数；它不改变 QUBO 目标或报告中场景定义的水分阈值。

## 查看输出

| 文件                                | 内容                                                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| `report.md`                         | 中文方案报告：预算、各方法对照、QAOA 候选和经典精确日程                             |
| `summary.json`                      | 生效输入、运行配置、版本、原始抽样、优化状态、方案及缺水诊断                        |
| `schedules.csv`                     | 每个方法、地块、日期和生育期的灌溉、取水、净水及损失                                |
| `moisture.csv`                      | 根区水分轨迹、目标、耗水阈值、低于阈值及物理越界标记                                |
| `qubo.json`                         | 变量索引、QUBO 线性/二次系数、常数和 Ising 系数                                     |
| `comparison.png` / `comparison.svg` | 加权目标与根区水分轨迹图；`--no-plot` 时不生成                                      |
| `resolved_input.json`               | 本次实际使用的完整调度输入，可通过`--input` 再次运行                                |
| `forecast.json` / `forecast.csv`    | 启用预测时生成：ET₀、张力、无新增灌溉水量基线、时间留出评估及模型信息               |
| `forecast_input.json`               | 启用预测时保存的原始预测请求，可作为`--forecast-input` 模板                         |
| `forecast.png` / `forecast.svg`     | 启用预测时生成的预测图；`--no-plot` 时不生成                                        |
| `qaoa.originir`                     | `qcloud` 模式生成：最终参数对应的逻辑门电路和测量操作                               |
| `hardware_job.json`                 | `qcloud` 模式生成：电路校验值、输入、本地优化记录、任务 ID 与真机结果，用于恢复查询 |

`objective` 越小示当前加权目标越好，不是预测产量或费用。`gross_water_m3` 是渠首取水，`net_water_m3` 是根区净水。`budget_feasible` 只表示窗口内事件数满足供水约束；水分低于阈值、超过 TAW 或低于 0 仍可能发生，见对应诊断。空间冲突和连续灌溉是软惩罚。

QAOA 候选仅从实际抽到、预算可行的决策中选择；无可行样本时报告空候选，保留经典结果作为对照。原始松弛位不修复。`budget_feasible_probability` 与 `zero_slack_residual_probability` 分别衡量事件数可行性与完整预算等式，含义不同。并结合抽样命中率、优化成本与经典对照评价结果。

三个场景在**未启用预测**时的小规模精确对照如下，均取水约 666.67 m³、无水分物理越界；仍存在低于耗水阈值的日期。预测模式会改变水分基线，目标值和最优日程也可能变化：

| 场景     | 精确日程（规划日）               |     加权目标 |
| -------- | -------------------------------- | -----------: |
| 拔节     | Wheat-A 第 1 天；Wheat-B 第 2 天 |  7103.300119 |
| 抽穗扬花 | Wheat-A 第 1 天；Wheat-B 第 2 天 | 40521.998025 |
| 灌浆     | Wheat-B 第 2 天；Wheat-C 第 3 天 |  9008.408777 |

## 默认 pyqpanda_alg 后端

```bash
wheat-irrigation --backend pyqpanda_alg --output outputs/pyqpanda_alg
# 兼容旧参数名，也调用相同的 pyqpanda_alg 实现
wheat-irrigation --backend pyqpanda --output outputs/pyqpanda_alias
```

适配层将 QUBO 转成归一化 Ising `PauliOperator`，创建 `pyqpanda_alg.QAOA.qaoa.QAOA` 并设置完整问题维度，包括系数为零的变量和预算松弛位。COBYLA 的每次期望值评估都调用 `QAOA.run_qaoa_circuit(gammas, -betas, shots=-1)`，读取算法库返回的完整理想概率分布；负号用于对齐两种实现的混合角约定。优化结束后使用给定种子的 NumPy 多项分布抽样，并保留经典精确对照。数学约定见 [model.md](docs/model.md#ising-与-qaoa)。

所需版本为 `pyqpanda_alg>=2.0.0,<3` 和 `pyqpanda3>=0.4.1,<0.5`；量子依赖另包含算法包导入所需的 pandas、scikit-learn。依赖缺失时默认流程明确报错，不自动切换后端。当前默认 Python 环境未安装这些量子依赖，完整 `pyqpanda_alg` 原生后端仍需在匹配环境中验证。

## 项目结构与验证

```text
wheat_irrigation/
├── pyproject.toml                 安装配置与 wheat-irrigation 命令
├── wheat_irrigation/
│   ├── __main__.py / cli.py       模块入口、对照、导出与绘图
│   ├── model.py                  水分动态、供水预算、QUBO/Ising
│   ├── forecasting.py            ET₀/张力预测、留出评估与调度耦合
│   ├── forecast_physics.py       日 FAO56 与标定曲线换算
│   ├── forecast_regression.py    NumPy 岭回归、可训练 VQC 与评估指标
│   ├── forecast_demo.py / forecast_output.py  合成序列、预测导出与图表
│   ├── solvers.py                精确枚举、pyqpanda_alg 适配与 NumPy QAOA
│   ├── hardware.py / hardware_workflow.py  真机电路、云端任务与恢复查询
│   └── data/                     三个独立小麦场景
├── irrigation.py                 源码目录快捷入口
├── irrigation_model.py / solvers.py  旧导入路径兼容层
├── docs/                         数学、输入说明及参数来源
└── tests/                        独立物理回算、量子演化、预算、导出与可移植性
```

```bash
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --wheel-dir dist
```

测试核对逐日水量平衡与完整 QUBO/Ising 的能量等价、松弛编码、毛净水换算、阶段阈值、独立矩阵指数与量子模拟的一致性、抽样真实性，以及 FAO56 手算值、标定换算、递归预测的数据泄漏边界、VQC 电路训练和导出重放。另将应用包复制到临时目录，验证基础 NumPy/经典流程脱离父仓库运行；默认量子流程仍需可导入的 `pyqpanda_alg` 与原生依赖。

枚举与调度态矢量最多支持 **18 比特**，包括预算松弛位；默认场景共 11 比特。真机模式仍使用这些本地计算，不能通过选择云端芯片绕过此规模限制。较大田块数或更长时间窗需要其他求解与建模策略。模型采用未截断的线性根区水量平衡和对称平方水分目标；预测模式在固定的无新增灌溉基线上累加灌水，仍未建模灌水触发的非线性下渗、径流、泵站日容量或产量响应。本地优化使用理想态，不含硬件噪声模型；真机输出反映云端实际测量结果。

详见[预测与调度说明](docs/forecasting.md)、[真机测试说明](docs/hardware.md)、[模型与输入说明](docs/model.md)、[概念依据与数据来源](docs/SOURCES.md)。代码沿用 [Apache-2.0 许可证](LICENSE)，代码来源见 [NOTICE](NOTICE)。
