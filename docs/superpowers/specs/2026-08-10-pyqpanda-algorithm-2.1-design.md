# pyqpanda-algorithm 2.1.0 完整性与真机适配设计

日期：2026-08-10  
状态：已批准  
目标版本：2.1.0

## 1. 背景

`pyqpanda-algorithm` 是基于 pyqpanda3 的量子算法库。当前仓库包含 QAOA、QARM、QKmeans、QPCA、QSVM、QUBO、QCmp、QAE、QSVD、QSVR、Grover、QmRMR 和 QSEncode 等模块，但执行逻辑普遍直接创建 `CPUQVM`，尚未形成统一的真机执行边界。VQE、HHL 和 Shor 的源码缺失，仓库中仅残留部分注释入口、构建脚本条目和失效 notebook。

发版前审查还确认了以下问题：

- 当前 pytest 配置只收集 6 个测试，全量实际可收集 18 个测试；仓库中大量测试文件被整段注释。
- 本地 Python 3.13.11、pyqpanda3 0.4.1 环境下，实际收集的 18 个测试全部通过，但不足以证明全库完整。
- CI 使用 `|| true` 忽略测试失败，无法作为发布门禁。
- QARM 的旧 QCloud 路径调用不存在的 `QCloudResult.get_prob_dict()`。
- QAOA 接受 `Hamiltonian` 的路径调用不存在的 `Hamiltonian.terms()`。
- `QmRMR.__all__` 包含类对象而非字符串，星号导入失败。
- `extensions` 引用仓库中不存在的 `_EXTENSIONS`。
- 运行依赖缺少源码实际使用的 pandas 和 scikit-learn，同时把 mypy、Sphinx 等开发工具列为运行依赖。
- Python 支持声明、CI 构建矩阵、测试版本和 wheel 标签不一致。

因此，2.1.0 不仅要新增三个算法，还必须先建立可信的执行、测试和发布基础。

## 2. 目标

2.1.0 必须实现：

1. 保持现有有效 CPU API 兼容，旧调用默认继续使用 CPU。
2. 以 `qpanda3-runtime` 作为正式真机入口，不新增 `QCloudService` 或 `QPilotService` 适配器。
3. 为所有现有公开算法提供 runtime 执行路径；纯线路组件必须能通过目标设备转译。
4. 以稳定 API 发布 VQE、HHL 和 Shor，不使用 experimental 标记。
5. 同时提供同步 `run()` 和可恢复的 `submit()`。
6. 将 runtime 支持作为可选安装项 `pyqpanda_alg[runtime]`。
7. 建立无凭据 CI 契约测试与发版候选真机验收流程。
8. 修复已确认的打包、测试、文档和公开 API 完整性问题。

## 3. 非目标

2.1.0 不包含：

- `QPilotService` 或旧 `QCloudService` 的新执行适配器。
- 从分子几何、基组到 Hamiltonian 的完整量子化学建模流程。
- 任意规模 HHL 或 Shor 在当前真机上成功完成的承诺。
- 自研误差缓解框架；本版本仅使用 qpanda3-runtime 提供的测量修正能力。
- 自动选择最佳芯片、跨设备调度或常驻云端工作流服务。
- 对任意自定义优化器内部状态的完整恢复保证。
- runtime 失败后自动回退 CPU。

## 4. 关键决策

### 4.1 执行架构

采用能力型统一执行层，而不是在每个算法内部直接调用 `RuntimeService`。

```text
算法
 ├─ 输入验证
 ├─ 构建线路与 Observable
 ├─ 调用采样、期望估计或变分会话能力
 └─ 解析为算法领域结果
              │
       ExecutionBackend
       ├─ LocalBackend → CPUQVM
       └─ QPandaRuntimeBackend → qpanda3_runtime
                                 ├─ FakeBackend
                                 ├─ QDevice
                                 ├─ VQSession
                                 └─ QTaskManager
```

### 4.2 兼容策略

- 现有有效位置参数和 CPU 返回类型保持不变。
- 新增的 `backend` 与 `execution_options` 追加为仅限关键字参数。
- `backend=None` 等价于 `LocalBackend`。
- 算法内部不读取 API Key、不执行登录、不硬编码芯片 ID。
- 调用方负责构造已登录的 `RuntimeService` 和目标 `QDevice`，再创建 `QPandaRuntimeBackend`。
- QARM 的 `machine_type="QCloud"` 参数保留一个版本并发出弃用警告；该失效路径不被视为有效兼容承诺，调用者必须显式提供 runtime backend。

## 5. execution 包设计

新增 `pyqpanda_alg.execution` 包。

### 5.1 ExecutionBackend

`ExecutionBackend` 定义算法所需的能力，不暴露服务商细节。核心方法为：

```python
submit_sample(circuits, *, options) -> BackendTask
submit_estimate(circuit_observable, *, options) -> BackendTask
create_variational_session(ansatz, observable, *, options) -> VariationalSession
capabilities() -> BackendCapabilities
```

后端方法始终返回任务对象。同步执行由上层调用任务的 `result()` 实现，从而避免本地和远程后端出现两套语义。

### 5.2 LocalBackend

`LocalBackend` 基于 pyqpanda3 的 `CPUQVM`、本地期望值接口和 VQCircuit 能力。它返回立即完成的 `BackendTask`，并维持当前 CPU 结果的比特序和数值含义。

### 5.3 QPandaRuntimeBackend

`QPandaRuntimeBackend` 封装以下 qpanda3-runtime 能力：

- `RuntimeService.sample()`
- `RuntimeService.estimate()`
- `RuntimeService.vqsession()`
- `QDevice` 能力和配置查询
- `FakeBackend` 预验证
- `QTaskManager` 查询、checkpoint 和恢复

公共构造方式为 `QPandaRuntimeBackend(service, device)`；`service` 必须已经完成登录，`device` 是调用方明确选择的 `QDevice`。2.1.0 不在 backend 内自动登录或自动选择芯片。

该模块只在实际导入或构造 runtime backend 时加载 `qpanda3_runtime`。未安装可选依赖时抛出 `MissingRuntimeDependencyError`，并给出 `pip install pyqpanda_alg[runtime]` 提示；基础包导入不得失败。

### 5.4 ExecutionOptions

`ExecutionOptions` 统一包含：

- `shots`
- `specified_block`
- `is_amend`
- `is_mapping`
- `is_optimization`
- `timeout`
- FakeBackend 预验证级别
- 算法允许覆盖的设备执行选项

预验证级别至少包含：

- `none`：只做静态能力检查。
- `transpile_only`：只验证目标门集、拓扑和转译。
- `fake_execute`：在 FakeBackend 上执行后再允许提交真机。

`fake_execute` 必须遵守 qpanda3-runtime 的多进程入口约束。不满足约束时抛出专门错误并给出正确的 `if __name__ == "__main__":` 使用方式。

### 5.5 BackendCapabilities

能力对象至少声明：

- 是否支持采样、期望估计和变分会话。
- 是否支持状态向量或密度矩阵。
- 是否支持层析所需的测量组合。
- 可用量子比特数、基础门集和拓扑。
- 当前设备是否可提交任务。

算法必须在提交前完成能力检查。超出能力时不得静默换用其他后端。

### 5.6 BackendTask 与 AlgorithmTask

`BackendTask` 封装一次底层采样、估计或会话任务。

`AlgorithmTask[T]` 封装完整算法状态机，提供：

```python
id
status()
poll()
try_result()
result(timeout=None)
result_async(timeout=None)
checkpoint(path=None)
AlgorithmTask.resume(path, *, backend)
```

`AlgorithmTask.id` 是本地生成的算法级稳定标识；一个算法可能对应多个远程任务，全部底层 ID 通过 `backend_task_ids` 暴露并写入结果元数据。`resume()` 是类级恢复入口，必须由调用方重新提供兼容且已登录的 backend。

状态至少包括 `PENDING`、`RUNNING`、`SUCCEEDED`、`FAILED` 和 `TIMED_OUT`。

VQE、Shor、QAE、GroverAdaptiveSearch 等多轮算法不能由单一 QTaskManager 表示。它们的 `submit()` 提交第一批量子任务后返回 `AlgorithmTask`；后续由 `poll()`、`result()` 或 `result_async()` 推进经典计算和下一批量子任务。实现不得启动不可控的后台守护线程。

## 6. 公共调用方式

```python
# 保持原有 CPU 行为
result = algorithm.run(problem_input)

# 同步 runtime 执行
result = algorithm.run(
    problem_input,
    backend=runtime_backend,
    execution_options=options,
)

# 可恢复提交
task = algorithm.submit(
    problem_input,
    backend=runtime_backend,
    execution_options=options,
)
result = task.result()
```

所有具有完整执行流程的公开算法都提供 `submit()`。QCmp、plugin 和 QAOA/default_circuits 等纯线路组件不创建任务，但产出的线路必须能被 execution backend 接收。

## 7. VQE 设计

### 7.1 输入与 API

VQE 以 `pyqpanda3.hamiltonian.Hamiltonian` 为标准输入：

```python
solver = VQE(
    hamiltonian,
    ansatz=None,
    optimizer=None,
)

result = solver.run(initial_parameters=initial_parameters, backend=backend)
task = solver.submit(initial_parameters=initial_parameters, backend=backend)
```

- 默认 Ansatz 为硬件高效 Ansatz。
- 默认优化器来自 SciPy。
- 用户可传入自定义 Ansatz、优化器和停止条件。
- 本版本不负责从分子几何生成 Hamiltonian。

### 7.2 执行

- LocalBackend 使用本地期望值计算。
- QPandaRuntimeBackend 优先使用带 Observable 的 `VQSession`。
- 设备或 runtime 不支持 VQSession 时，使用批量 `estimate()`，但不得回退 CPU。
- 会话由 `AlgorithmTask` 持有，并在完成、异常或超时时释放。
- checkpoint 恢复时不复用可能过期的 session，而是根据参数和迭代历史创建新 session。

### 7.3 结果

`VQEResult` 包含：

- `energy`
- `optimal_parameters`
- `converged`
- `iterations`
- `energy_history`
- `optimal_circuit`
- `task_ids`
- 后端、设备、shots 和执行时间等元数据

内置优化器支持完整恢复。自定义优化器默认只保存当前参数、目标值和历史，并通过 `resume_supported` 声明是否可继续。

## 8. HHL 设计

### 8.1 输入与 API

HHL 支持有限的实数或复数 Hermitian 方阵与向量：

```python
solver = HHL(matrix, vector, precision=1e-3)
result = solver.run(backend=backend)
circuit = solver.build_circuit()
```

保留旧 notebook 暗示的兼容入口：

```python
build_HHL_circuit(matrix, vector, precision=1e-3)
expand_linear_equations(matrix, vector)
HHL_solve_linear_equations(matrix, vector, precision=1e-3)
```

输入验证包括：有限数值、方阵、Hermitian、维度匹配和非奇异性。非二次幂维度按公开且可逆的 padding 规则扩展。病态矩阵根据可配置条件数阈值拒绝执行，并在错误中报告估计条件数。

### 8.2 线路与执行

实现包含状态制备、相位估计、受控倒数旋转、逆相位估计和辅助量子比特后选择。一般矩阵的受控时间演化必须使用明确的矩阵编码/线路综合过程，并在提交前给出量子比特、深度和门数估算。

CPU 可以返回状态向量。runtime 默认只执行采样或用户请求的 Observable，不假设能低成本读取完整经典向量。

### 8.3 结果

`HHLSolution` 包含：

- HHL 线路
- 归一化因子
- 后选择成功概率
- 可计算时的残差
- 用户请求的 Observable 结果
- `task_ids` 和执行元数据
- 可选 `statevector`
- 可选 `classical_vector`

只有 CPU 或显式 `reconstruct=True` 时才生成完整经典向量。runtime 重建通过额外层析测量任务实现，并在提交前报告额外线路数和 shots；超出限制时拒绝提交。

## 9. Shor 设计

### 9.1 范围与 API

Shor 接受一般奇合数，但只对资源允许的小规模输入承诺执行：

```python
solver = Shor(modulus)
estimate = solver.estimate_resources()
result = solver.run(backend=backend)
task = solver.submit(backend=backend)
```

算法包含：

1. 偶数、质数和完全幂等经典快速处理。
2. 随机底数选择与最大公约数检查。
3. 通用模指数线路构造。
4. 量子阶寻找。
5. 连分数恢复候选阶。
6. 阶和因子验证。
7. 有界次数的重新尝试。

偶数和完全幂等输入可以由明确的经典快速路径直接返回；质数返回 `ShorResult` 并标记 `is_prime=True`，不提交量子任务。所有结果必须记录 `used_quantum` 和 `task_ids`，避免把经典快速路径误认为真机执行。只有进入量子阶寻找的结果才能计入 Shor 的真机 RC 验收。

线路生成不得使用已知因子或已知阶。针对 `N=15` 等小规模真机用例，可以依据公开的 `N` 和底数综合资源受控线路，但不能依据预期答案裁剪结果。

### 9.2 结果

`ShorResult` 包含：

- `factors`
- `order`
- `base`
- 尝试次数
- 线路资源估算
- `task_ids`
- `used_quantum`
- `is_prime`
- 成功状态或结构化失败原因

资源超限时抛出 `DeviceCapabilityError`，不得伪造结果，也不得静默使用经典分解替代量子阶寻找。

## 10. 现有算法迁移

| 类型 | 模块 | runtime 路径 |
|---|---|---|
| 线路组件 | QCmp、plugin、QAOA/default_circuits | 构建并转译线路 |
| 采样型 | Grover、QAE、QARM、QKmeans、QPCA、QSVM、QSVR、QSEncode | `submit_sample()` |
| 期望/变分型 | QAOA、QUBO_QAOA、QmRMR、QSVD | `submit_estimate()` 或变分会话 |
| 混合型 | QUBO_GAS、GroverAdaptiveSearch | 多轮采样和经典反馈 |

迁移要求：

- QKmeans、QSVM 和 QSVR 使用 ancilla/overlap 测量估计距离或核值。
- QPCA 使用相位估计采样分布，不直接读取真机状态向量。
- QSVD 使用可测量的 overlap/cost Observable 驱动优化。
- QSEncode 在 runtime 上返回采样分布或用户指定 Observable，不承诺完整振幅向量。
- QAOA、QUBO 和 QmRMR 共享统一期望估计与优化循环。
- QAE、Grover 和相关动态轮次必须进入可 checkpoint 的算法状态机。
- 后端层保留原始 counts 和 expectation；算法解析层统一概率归一化、比特序和领域结果。

“全库真机支持”指每个执行型公开算法都有真实的 runtime 路径。不得把纯经典结果包装为真机结果。超出设备能力的输入可以产生结构化能力错误，但每个算法仍必须至少通过一个小规模真机用例。

## 11. 数据流与恢复

```text
算法输入验证
  → 构建线路与 Observable
  → 估算量子比特、深度和任务数量
  → 查询 QDevice 能力
  → 按配置执行转译或 FakeBackend 预验证
  → sample / estimate / VQSession
  → QTaskManager 追踪
  → 算法状态机推进
  → 解析领域结果
```

checkpoint 保存：

- 算法名称、输入摘要和当前阶段。
- 当前迭代、参数、目标值和经典中间结果。
- channel、芯片 ID、任务 ID 和已完成任务。
- 可序列化线路表示。
- qpanda3-runtime checkpoint 引用。
- 库版本和 checkpoint 格式版本。

checkpoint 不保存 API Key、token、`RuntimeService` 对象或完整凭据。恢复时调用方重新提供已登录 backend。格式或库版本不兼容时给出明确迁移错误。

## 12. 错误模型

公共异常包括：

- `AlgorithmInputError`
- `MissingRuntimeDependencyError`
- `BackendUnavailableError`
- `DeviceCapabilityError`
- `TranspilationError`
- `TaskSubmissionError`
- `TaskTimeoutError`
- `TaskRecoveryError`
- `ResultDecodingError`

异常包含算法名、执行阶段、设备标识、任务 ID 和可恢复性，并通过异常链保留底层原因。错误信息必须过滤凭据和服务端敏感数据。

任务提交默认不自动重试，避免重复计费。只有幂等的状态查询可以进行次数有限的指数退避重试。同步超时必须保留任务 ID 和 checkpoint，使调用者可以继续查询。

## 13. 测试设计

### 13.1 标准 CI

标准 CI 不需要 API Key，也不访问真实设备。它运行：

- 数学与线路单元测试。
- LocalBackend 集成测试。
- runtime stub 契约测试。
- 任务状态、序列化结果、checkpoint 和恢复测试。
- 基础安装与 `[runtime]` 安装导入测试。
- sdist、wheel、依赖元数据和文档构建测试。
- 精选 notebook smoke test。

FakeBackend 依赖真实设备配置，不作为无凭据标准 CI 的强制能力。

### 13.2 算法正确性

- VQE 与精确对角化比较小型 1–4 qubit Hamiltonian。
- HHL 使用 2×2 和 4×4 Hermitian 系统，与 `numpy.linalg.solve` 比较归一化方向、保真度和残差。
- Shor 验证 15、21 等小规模合数，并覆盖质数、偶数、完全幂、无效输入和资源超限。
- 现有算法的有效测试必须恢复或重写；整文件注释的测试必须恢复或删除。
- 随机算法固定可复现的种子策略，并使用预先定义的统计容差。

### 13.3 覆盖率

- 新 execution/runtime 模块行覆盖率不低于 85%。
- 任务状态机、结果解析和异常转换的关键分支达到 100% 覆盖。
- 旧模块记录当前覆盖率基线，后续提交不得降低。
- 覆盖率不能替代数学正确性和统计断言。

### 13.4 版本与平台

Python 支持范围必须取 pyqpanda3、qpanda3-runtime 和本库实际验证过的公共兼容区间。基础包在 Windows、Linux、macOS 的支持版本上测试；`[runtime]` 只在 qpanda3-runtime 官方支持的平台上测试。如果依赖不支持当前声明的 Python 版本，必须收窄 classifiers 和文档，而不是保留未经验证的声明。

## 14. 真机发版候选验收

每个具有执行入口的公开算法至少完成一次真实 QPU 小规模任务。纯线路组件至少通过目标设备转译。

RC 验收记录包含：

- Git commit、包版本和依赖版本。
- 芯片 ID、设备配置摘要和校准时间。
- FakeBackend 预验证结果。
- 真机任务 ID、shots、原始结果和解析结果。
- 运行前确定的随机种子策略、置信区间、容差和通过阈值。
- 成功/失败判定和失败原因。

不得在看到结果后修改容差，也不得通过无限重试获得偶然成功。设备排队或离线不直接判定算法错误，但没有至少一次有效的 RC 真机记录时不得正式发版。验收材料不得包含 API Key 或 token。

## 15. 打包和发布门禁

- 使用单一版本源，并让 tag、包版本、Changelog 和文档版本保持一致。
- 用标准打包配置替代互相漂移的 `setup.py` 与 `setup-cython.py` 逻辑。
- 基础运行依赖、开发依赖和 `runtime` 可选依赖分组明确。
- 纯 Python wheel 使用正确的 `py3-none-any` 标签，不手工伪装平台 ABI。
- 构建 wheel 后在干净环境安装该 wheel，再运行发布测试。
- pytest 全量收集，测试失败必须阻断 CI；删除所有绕过失败的 `|| true`。
- Sphinx 文档和精选 notebook smoke test 必须通过。
- GitHub Release 只能在标准 CI、安装产物测试和 RC 真机验收记录齐备后创建。

## 16. 实施顺序

### 阶段 1：发布基线

- 修复测试收集、CI、依赖、打包和版本声明。
- 修复已确认的 QAOA、QmRMR 等问题。
- 清理失效 notebook 和构建残留。
- 从发行物移除无法导入且无源码可重建的 `extensions`，并在 Changelog 中说明。

### 阶段 2：统一执行层

- 实现任务、能力、选项、本地后端和 runtime 后端。
- 用 QAOA 和 Grover 验证期望估计、采样和多轮状态机。

### 阶段 3：现有算法迁移

- 按线路组件、单次采样、期望估计、多轮混合算法的顺序迁移。
- 每迁移一个模块，先恢复或重写有效测试，再移除内部直接创建的 `CPUQVM`。

### 阶段 4：新算法

- VQE 验证 Estimator 和 VQSession。
- HHL 验证线路综合、资源估算和可选层析。
- Shor 验证通用模指数、阶恢复和多次尝试状态机。

### 阶段 5：发布候选

- 完成平台、安装产物、文档、FakeBackend 和真实 QPU 验收。
- 生成脱敏真机报告和已知限制。
- 所有门禁通过后创建 2.1.0 tag 和正式 Release。

## 17. 完成定义

2.1.0 只有在以下条件全部满足时才算完成：

1. 13 个现有模块和 VQE、HHL、Shor 的公开 API、文档、CPU 测试及 runtime 路径全部存在。
2. 原有有效 CPU 调用保持兼容。
3. 每个执行型算法至少有一次有效真机 RC 记录。
4. 纯线路组件全部通过目标设备转译。
5. 没有整文件注释的占位测试、失效示例或绕过失败的 CI 步骤。
6. 安装元数据、版本、Changelog、文档和 tag 一致。
7. runtime 失败不会被伪装为 CPU 成功或经典计算成功。
8. checkpoint、日志和 RC 材料不包含凭据。
