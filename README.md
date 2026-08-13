Origin Quantum is pleased to support the open source community by making pyqpanda-algorithm available.   <br/>
Copyright (c) 2026 Origin Quantum. All rights reserved. <br/>
This source code is licensed under the Apache License Version 2.0 <br/>

<h1 align="center" style="text-align:center;">
  pyqpanda-algorithm 算法软件包
</h1>

<p align="center"> QPanda3框架 高开发效率 高可靠和稳定性 高性能 <br />集合了在量子算法中常用的基本量子算法和函数</p>





## 介绍

pyqpanda-algorithm 是由本源量子（Origin Quantum）开发的量子算法软件包，旨在为量子计算开发者提供一套标准化、模块化、高性能的基础算法库。该库集成了多种在金融、机器学习、组合优化、科学计算等领域广泛应用的量子算法，帮助用户快速实现从理论到代码的转化，提升开发效率并确保算法在不同量子平台上的可移植性。

软件包官网： [https://qcloud.originqc.com.cn/zh/programming/pyqpanda-algorithm]

------

## **核心特点**

1. **模块化与高复用性**
   所有算法以独立模块形式组织，便于开发者按需调用。例如，`QAOA`、`Grover`、`QSVM` 等算法均可独立导入与使用，支持在不同项目中重复利用。
2. **高性能实现**
   领域特定算法经过算法优化与工程加速，结合 QPanda3 的底层优化，显著提升在 CPU 模拟器上的执行效率。
3.  **统一的执行层**
   新增 `pyqpanda_alg.execution` 执行层：默认在本地 CPU 模拟器（`LocalBackend`，基于 pyqpanda3 的 CPUQVM）上同步执行，现有 CPU 调用方式保持不变；可选的 qpanda3-runtime 后端（通过 `pip install pyqpanda_alg[runtime]` 安装）用于远程任务提交，执行失败时不会静默回退到 CPU 或经典结果。执行层使用说明见 [execution 文档](Tutorials/source/execution.rst)。
4. **2.1.0 新增稳定算法**
   VQE（变分量子本征求解器）、HHL（线性方程组求解）、Shor（大数分解）三个算法在 2.1.0 中以稳定 API 发布：输入前置校验、默认本地 CPU 执行、可通过 `backend=` 切换至 qpanda3-runtime 远程后端。
5. **完善的文档与示例**
   提供详尽的 API 文档、使用示例与注释代码，降低学习门槛，特别适合初学者与研究者快速上手机器学习与组合优化任务。
6. **生态整合性强**
   与本源量子的框架与工具链（如 VQNet、本源量禹）协同工作，支持从算法设计到验证的完整开发流程。真实硬件支持以发布限定（release qualification）的形式交付，并作为发布创建前的强制门禁。

------

## **2.1.0 主要更新**

2.1.0 在保持既有 CPU 调用方式完全兼容的前提下，完成了执行架构、算法库与发布体系的全面升级。详细变更见 [Changelog](Tutorials/source/Changelog.rst)。

1. **新增三个稳定算法**：`pyqpanda_alg.VQE`、`pyqpanda_alg.HHL`、`pyqpanda_alg.Shor`，均以子模块方式导入（如 `from pyqpanda_alg.VQE import VQE`），配套教程见 [VQE](Tutorials/source/VQE.rst)、[HHL](Tutorials/source/HHL.rst)、[Shor](Tutorials/source/Shor.rst)。
2. **统一执行层 `pyqpanda_alg.execution`**：统一的任务协议（采样/期望估计/态矢量/变分会话）、`ExecutionOptions` 运行参数、检查点（checkpoint）与恢复（resume）、能力检查（`DeviceCapabilityError`）以及 10 个公开执行异常；运行时失败永不静默回退。
3. **全部既有算法迁移至执行层**：QAOA、QUBO、Grover、QAE/IQAE、QARM、QKmeans、QPCA、QSVM、QSVR、QSVD、QSEncode、QmRMR 等全部执行入口新增关键字参数 `backend=` 与 `execution_options=`（不传则保持原有 CPU 行为），不再直接构造 `CPUQVM`；QAOA、QUBO、QAE、QARM、QmRMR 等额外提供 `submit()` 任务化接口。
4. **行为变更与废弃**：QARM 的 `machine_type='QCloud'` 模式废弃（需显式传入 `backend=`）；QPCA 默认采样次数由 8192 调整为 1000（结果按实际采样数归一化，输出形状不变）。
5. **真实 QPU 发布限定门禁**：18 个固定小规模用例（预检 FakeBackend + 真实 QPU）经净化的限定清单（manifest）与 `check_release.py` 强制校验后，标签发布作业才会创建 GitHub Release；限定产出永不提交仓库。
6. **测试与文档**：CI 在 3 操作系统 × 3 Python 版本矩阵上对已安装 wheel 运行测试，并强制覆盖率基线；新增 VQE/HHL/Shor、执行层与算法迁移（[runtime_algorithms](Tutorials/source/runtime_algorithms.rst)）教程文档。

------

## 软件包种类

### 1. **优化与搜索算法包**

适用于组合优化、大规模搜索问题，在路径规划、资源调度、投资组合优化等领域有广泛应用。

- **QUBO（无约束二进制优化）**  
  将组合优化问题转化为二次无约束二元优化问题，是量子退火和变分量子算法的通用建模形式。

- **QAOA（量子近似优化算法）**  
  混合量子-经典变分算法，通过优化参数化量子线路（Ansatz）来近似求解 QUBO 问题，适用于最大割、最大满足等问题。

- **Grover 搜索算法**  
  在无结构数据库中实现目标项的二次加速搜索。通过振幅放大技术，将搜索复杂度从 $O(N)$ 降低至 $O(\sqrt{N})$。

### 2. **机器学习与数据挖掘算法**

将量子计算能力引入经典机器学习流程，提升分类、聚类、回归等任务的效率与精度。

- **QSVM（量子支持向量机）**  
  基于量子核函数的分类模型，可在高维空间中实现更优的分类边界。

- **QSVR（量子支持向量回归）**  
  用于拟合连续变量的回归模型，适用于时间序列预测等任务。

- **QKmeans（量子 K-均值聚类）**  
  利用量子加速实现大规模数据聚类，适用于高维数据聚类场景。

- **QPCA（量子主成分分析）**  
  通过量子线路提取数据主成分，实现降维加速。

- **QmRMR（量子最小冗余最大相关）**  
  实现高效特征选择，减少冗余特征影响。

- **QARM（量子关联规则挖掘）**  
  快速挖掘频繁项集与关联规则，适用于市场篮子分析等任务。

### 3. **科学计算与数值求解算法**

用于求解物理建模、工程仿真中的本征值、线性方程组、整数分解、矩阵分解等关键问题。

- **VQE（变分量子本征求解器）** `2.1.0 新增`  
  求解 pyqpanda3 `Hamiltonian` 基态能量的变分量子算法：输入前置校验（厄米性与系数有限性），内置硬件高效 ansatz（RY/RZ 旋转层 + 线性 CNOT 链）或自定义 ansatz，配合经典优化器迭代收敛，适用于量子化学与材料模拟。API：`VQE` / `VQEConfig` / `VQEResult`，导入方式 `from pyqpanda_alg.VQE import VQE`。

- **HHL（线性方程组求解算法）** `2.1.0 新增`  
  求解有限厄米线性方程组 `A x = b` 的量子算法：构造时校验矩阵与右端向量（方阵、厄米、可逆、条件数阈值可配置），非 2 的幂维数自动进行可逆块对角 padding；远程后端路径通过 Pauli 基采样重建解并报告重建保真度。API：`HHL` / `HHLConfig` / `HHLSolution` 等，导入方式 `from pyqpanda_alg.HHL import HHL`。

- **Shor（大数分解算法）** `2.1.0 新增`  
  通过量子阶求解分解奇数合数的 Shor 算法：偶数、素数、完全幂模数由经典预处理直接求解（如实报告 `used_quantum=False`），其余模数驱动有界次数的量子阶求解尝试，并用连分数恢复乘法阶。API：`Shor` / `ShorConfig` / `ShorResult` / `classical_preprocess` / `recover_order`，导入方式 `from pyqpanda_alg.Shor import Shor`。

- **QSVD（量子变分奇异值分解）**  
  在变分框架下提取矩阵的奇异值与奇异向量，用于降维与推荐系统。

### 4. **通用工具与基础组件**

提供量子振幅估计算法、比较器、稀疏编码等底层工具。

- **QAE（量子振幅估计算法）**  
  精确估算目标态的振幅或测量概率，具备二次加速优势，常用于金融衍生品定价、风险评估等。

- **Comparator（量子比较器，模块 `QCmp`）**  
  实现数值大小比较或阈值判定，支持构建量子决策逻辑。

- **QSEncode（稀疏幅度编码，类 `QSpare_Code`）**  
  高效将稀疏向量编码为量子态，减少量子资源消耗，适用于数据预处理。

------

## 安装

pyqpanda_alg是基于pyqpanda3的算法扩展模块。它的安装和使用需要依赖pyqpanda3。pyqpanda3的接口用法请参考[pyqpanda3](https://qcloud.originqc.com.cn/document/qpanda-3/cn/index.html)。

如果已经安装了python环境和pip工具，在终端或控制台中输入如下命令：`pip install pyqpanda_alg`

可选：如需使用 qpanda3-runtime 远程执行后端，请安装含可选依赖的版本：`pip install pyqpanda_alg[runtime]`

#### 注意：

如果你在linux下遇到权限问题，你需要添加sudo（superuser do）。

------

## 运行保障与发布门禁

### qpanda3-runtime 远程执行

可选的 qpanda3-runtime 后端通过 `pip install pyqpanda_alg[runtime]` 安装。使用远程执行时，必须显式注入已登录的 `RuntimeService` 与明确选定的设备：`QPandaRuntimeBackend(service, device)` 不会自行认证，也不会替用户选择设备。

### 能力错误

请求设备未提供的能力（例如 statevector 或变分会话）会抛出 `DeviceCapabilityError`；远程执行失败时，执行层永远不会静默回退到 CPU 或经典替代结果。

### 小规模保证

远程与真实硬件相关的能力保证，仅针对发布限定（release qualification）中固定的、小规模的用例（固定的线路、采样数与统计阈值）作出。这些保证不构成对任意规模问题或任意设备的一般性声明。

### QPU 发布门禁

2.1.0 的发布创建强制要求真实 QPU 限定：手动触发的 `runtime-rc` 工作流在明确选定的设备上运行固定的预检与 QPU 用例，生成经净化的限定清单（manifest）；标签发布作业只有在清单与候选提交、wheel 摘要、版本、设备记录以及全部用例结论完全匹配时，才会创建 GitHub Release。限定产出永不提交到仓库（提交会改变清单所证明的提交）；本仓库不对任何特定设备的通过情况作出超出清单记录的声明。

------

## 环境配置

pyqpanda_alg采用Python作为主要语言，对系统的环境要求如下：

### 		Windows

| software                                                     | version            |
| ------------------------------------------------------------ | ------------------ |
| [Microsoft Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) | 2019               |
| Python                                                       | >= 3.11 && <= 3.13 |

### 		Linux

| software | version            |
| -------- | ------------------ |
| GCC      | >= 7.5             |
| Python   | >= 3.11 && <= 3.13 |

------

## 仓库分支说明
本项目采用「稳定分支 + 开发分支」的协作模式，不同分支承担不同职责，方便你根据需求选择对应分支操作：

### 📌 main（主分支）
- **核心定位**：main 分支是项目的**稳定发布分支**，存放的是经过充分测试、可直接使用的开源量子算法代码。
- **使用场景**：如果你想研究、学习量子算法的核心实现，或直接基于成熟代码进行二次开发，可直接拉取/使用 main 分支的内容。
- **重要说明**：main 分支不接收任何直接提交或 Pull Request（PR），确保核心代码始终稳定、无已知问题。

### 🛠️ develop（开发分支）
- **核心定位**：develop 分支是项目的**协作开发分支**，用于整合社区贡献、迭代新功能、修复问题。
- **提交场景**：如果你想参与项目贡献，包括但不限于：提交 Bug 修复代码、完善项目文档、补充量子算法的示例代码、提出/实现新的功能建议、优化现有算法的性能或可读性，请将所有 Pull Request（PR）**统一提交至 develop 分支**。
- **协作说明**：我们会定期审核 develop 分支的贡献内容，经测试验证后合并至 main 分支，让优质贡献同步到稳定版本中。

感谢你的理解与配合！

------

## 开源许可

使用 [Apache License 2.0](https://gitee.com/OriginQ/alg/blob/master/LICENSE)，对 公司、团队、个人 等 商用、非商用 都自由免费且非常友好，请放心使用和登记。


------

## 致谢

感谢所有贡献者、测试者与社区支持者。特别鸣谢本源量子研究院在算法设计与性能优化方面的技术支持。

------

## **联系方式**

- **官方邮箱**：[qcloud@originqc.com](mailto:qcloud@originqc.com)

- **售前咨询链接**：https://contact.originqc.com.cn/

- **官方微信**：搜索“本源量子云社区”，关注开源项目动态
<p align="center">
  <img src="my-folder/服务号.png" alt="本源量子云社区服务号" width="50%">
</p>

- **官方小助手**：可扫描下方二维码，添加官方小助手，获取更多支持
<p align="center">
  <img src="my-folder/本源量子云小助手.jpg" alt="本源量子官方小助手" width="30%">
</p>

