# Origin Quantum pyqpanda-algorithm Quantum Algorithm Software Package

## Introduction
pyqpanda-algorithm is a quantum algorithm software package developed by Origin Quantum, designed to provide quantum computing developers with a standardized, modular, and high-performance foundational algorithm library. This library integrates a variety of quantum algorithms widely used in finance, machine learning, combinatorial optimization, scientific computing, and other fields. It helps users quickly translate theories into code, improve development efficiency, and ensure algorithm portability across different quantum platforms.

Official Website: [https://qcloud.originqc.com.cn/zh/programming/pyqpanda-algorithm]

------

## Core Features
1. **Modularity & High Reusability**
   All algorithms are organized as independent modules for on-demand invocation. For example, QAOA, Grover's, and QSVM can be imported and used independently, supporting reuse across different projects.
2. **High-Performance Implementation**
   Domain-specific algorithms are optimized and engineered for acceleration. Combined with QPanda3's underlying optimizations (e.g., OriginBIS instruction set, hardware-aware compilation), execution efficiency on simulators and real quantum hardware is significantly improved.
3. **Cross-Platform Compatibility**
   Deeply integrated with the QPanda3 framework, it supports running on CPU simulators, quantum cloud services (e.g., Origin Wukong), and real quantum processors, enabling "write once, deploy anywhere".
4. **Comprehensive Documentation & Examples**
   Detailed API docs, usage examples, and annotated code lower the learning barrier, making it ideal for beginners and researchers to quickly start with machine learning and combinatorial optimization tasks.
5. **Strong Ecosystem Integration**
   Seamlessly integrates with other Origin Quantum toolchains (e.g., VQNet, Origin Wukong, Origin Liangyu), supporting the full workflow from algorithm design to actual execution.

------

## Package Categories
### 1. Optimization & Search Algorithms
Suitable for combinatorial optimization and large-scale search problems (e.g., path planning, resource scheduling, portfolio optimization).
- **QUBO (Unconstrained Binary Optimization)**  
  Converts combinatorial optimization problems into quadratic unconstrained binary optimization problems (a universal modeling form for quantum annealing and variational quantum algorithms).
- **QAOA (Quantum Approximate Optimization Algorithm)**  
  A hybrid quantum-classical variational algorithm that approximates QUBO solutions by optimizing parameterized quantum circuits (Ansatz), applicable to Max-Cut, Max-SAT, etc.
- **Grover's Search Algorithm**  
  Achieves quadratic speedup for target item search in unstructured databases. Amplitude amplification reduces search complexity from $O(N)$ to $O(\sqrt{N})$.

### 2. Machine Learning & Data Mining Algorithms
Integrates quantum computing into classical machine learning to improve efficiency and accuracy of classification, clustering, regression, etc.
- **QSVM (Quantum Support Vector Machine)**  
  A classification model based on quantum kernel functions, enabling optimal classification boundaries in high-dimensional spaces.
- **QSVR (Quantum Support Vector Regression)**  
  A regression model for fitting continuous variables (e.g., time series prediction).
- **QKMeans (Quantum K-Means Clustering)**  
  Quantum-accelerated large-scale data clustering for high-dimensional data scenarios.
- **QPCA (Quantum Principal Component Analysis)**  
  Extracts data principal components via quantum circuits to accelerate dimensionality reduction.
- **QMRMR (Quantum Minimum Redundancy Maximum Relevance)**  
  Implements efficient feature selection to reduce redundant feature interference.
- **QARM (Quantum Association Rule Mining)**  
  Rapidly mines frequent itemsets and association rules (e.g., market basket analysis).

### 3. Scientific Computing & Numerical Solution Algorithms
Solves key problems in physical modeling and engineering simulation (e.g., eigenvalues, linear equations, matrix decomposition).
- **QSVD (Quantum Variational Singular Value Decomposition)**  
  Extracts matrix singular values/vectors under a variational framework (for dimensionality reduction and recommendation systems).

### 4. General Tools & Basic Components
Provides underlying tools for quantum computing workflows.
- **QAE (Quantum Amplitude Estimation)**  
  Precisely estimates target state amplitude/measurement probability with quadratic speedup (e.g., financial derivative pricing, risk assessment).
- **Comparator (Quantum Comparator)**  
  Implements numerical comparison or threshold judgment to build quantum decision logic.
- **SparseAmp (Sparse Amplitude Encoding)**  
  Efficiently encodes sparse vectors into quantum states, reducing quantum resource consumption (for data preprocessing).

------

## Installation
pyqpanda_alg is an algorithm extension module based on pyqpanda3. Its installation and usage depend on pyqpanda3. Refer to [pyqpanda3](https://qcloud.originqc.com.cn/document/qpanda-3/cn/index.html) for interface usage.

If Python and pip are installed, run the following command in the terminal/console:  
`pip install pyqpanda_alg`

#### Note:
Add `sudo` if you encounter permission issues on Linux.

------

## Environment Configuration
pyqpanda_alg is primarily developed in Python with the following system requirements:

### Windows
| Software                                                     | Version            |
| ------------------------------------------------------------ | ------------------ |
| [Microsoft Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) | 2019               |
| Python                                                       | >= 3.11 && <= 3.13 |

### Linux
| Software | Version            |
| -------- | ------------------ |
| GCC      | >= 7.5             |
| Python   | >= 3.11 && <= 3.13 |

------

## Open Source License
Licensed under [Apache License 2.0](https://gitee.com/OriginQ/alg/blob/master/LICENSE), free and friendly for commercial/non-commercial use by companies, teams, and individuals. Feel free to use and register.

------

## Acknowledgements
Thanks to all contributors, testers, and community supporters. Special thanks to the Origin Quantum Research Institute for technical support in algorithm design and performance optimization.

------


## Contact Us

- **Official Email**：[qcloud@originqc.com](mailto:qcloud@originqc.com)

- **Pre-sales Consultation**：https://contact.originqc.com.cn/

- **Official WeChat**：Search for "本源量子云社区"，follow Open-Source Project Updates.
<p align="center">
  <img src="my-folder/服务号.png" alt="本源量子云社区服务号" width="50%">
</p>

- **The official assistant**：scan the QR code below to add the official assistant for quantum cloud support and event updates
<p align="center">
  <img src="my-folder/本源量子云小助手.jpg" alt="本源量子官方小助手" width="30%">
</p>
