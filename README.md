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
   域名特定算法经过算法优化与工程加速，结合 QPanda3 的底层优化（如 OriginBIS 指令集、硬件感知编译），显著提升在模拟器与真实量子硬件上的执行效率。
3.  **跨平台兼容性**
   与 QPanda3 框架深度集成，支持在 CPU 模拟器、量子云服务（如本源悟空）及真实量子处理器上运行，实现“一次编写，多端部署”。
4. **完善的文档与示例**
   提供详尽的 API 文档、使用示例与注释代码，降低学习门槛，特别适合初学者与研究者快速上手机器学习与组合优化任务。
5. **生态整合性强**
   与本源量子的其他工具链（如 VQNet、本源悟空、本源量禹）无缝对接，支持从算法设计到实际运行的完整工作流。

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

- **QKMeans（量子 K-均值聚类）**  
  利用量子加速实现大规模数据聚类，适用于高维数据聚类场景。

- **QPCA（量子主成分分析）**  
  通过量子线路提取数据主成分，实现降维加速。

- **QMRMR（量子最小冗余最大相关）**  
  实现高效特征选择，减少冗余特征影响。

- **QARM（量子关联规则挖掘）**  
  快速挖掘频繁项集与关联规则，适用于市场篮子分析等任务。

### 3. **科学计算与数值求解算法**

用于求解物理建模、工程仿真中的本征值、线性方程组、矩阵分解等关键问题。

- **QSVD（量子变分奇异值分解）**  
  在变分框架下提取矩阵的奇异值与奇异向量，用于降维与推荐系统。

### 4. **通用工具与基础组件**

提供量子振幅估计算法、比较器、稀疏编码等底层工具。

- **QAE（量子振幅估计算法）**  
  精确估算目标态的振幅或测量概率，具备二次加速优势，常用于金融衍生品定价、风险评估等。

- **Comparator（量子比较器）**  
  实现数值大小比较或阈值判定，支持构建量子决策逻辑。

- **SparseAmp（稀疏幅度编码）**  
  高效将稀疏向量编码为量子态，减少量子资源消耗，适用于数据预处理。

------

## 安装

pyqpanda_alg是基于pyqpanda3的算法扩展模块。它的安装和使用需要依赖pyqpanda3。pyqpanda3的接口用法请参考[pyqpanda3](https://qcloud.originqc.com.cn/document/qpanda-3/cn/index.html)。

如果已经安装了python环境和pip工具，在终端或控制台中输入如下命令：`pip install pyqpanda_alg`

#### 注意：

如果你在linux下遇到权限问题，你需要添加sudo（superuser do）。

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
