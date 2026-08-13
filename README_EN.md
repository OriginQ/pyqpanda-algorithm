# Origin Quantum pyqpanda-algorithm Quantum Algorithm Software Package

## Introduction
pyqpanda-algorithm is a quantum algorithm software package developed by Origin Quantum, designed to provide quantum computing developers with a standardized, modular, and high-performance foundational algorithm library. This library integrates a variety of quantum algorithms widely used in finance, machine learning, combinatorial optimization, scientific computing, and other fields. It helps users quickly translate theories into code, improve development efficiency, and ensure algorithm portability across different quantum platforms.

Official Website: [https://qcloud.originqc.com.cn/zh/programming/pyqpanda-algorithm]

------

## Core Features
1. **Modularity & High Reusability**
   All algorithms are organized as independent modules for on-demand invocation. For example, QAOA, Grover's, and QSVM can be imported and used independently, supporting reuse across different projects.
2. **High-Performance Implementation**
   Domain-specific algorithms are optimized and engineered for acceleration. Combined with QPanda3's underlying optimizations, execution efficiency on the CPU simulator is significantly improved.
3. **Unified Execution Layer**
   The new `pyqpanda_alg.execution` package provides a capability-based execution layer: by default, algorithms run synchronously on the local CPU backend (`LocalBackend`, based on pyqpanda3's CPUQVM), and existing CPU calls keep working unchanged. An optional qpanda3-runtime backend (installed via `pip install pyqpanda_alg[runtime]`) submits work to a remote service and never silently falls back to the CPU or to a classical substitute. See the [execution documentation](Tutorials/source/execution.rst).
4. **New Stable Algorithms in 2.1.0**
   VQE (variational quantum eigensolver), HHL (linear system solver), and Shor (integer factorization) ship with stable APIs in 2.1.0: validated inputs, local CPU execution by default, and an optional `backend=` argument to switch to a qpanda3-runtime remote backend.
5. **Comprehensive Documentation & Examples**
   Detailed API docs, usage examples, and annotated code lower the learning barrier, making it ideal for beginners and researchers to quickly start with machine learning and combinatorial optimization tasks.
6. **Strong Ecosystem Integration**
   Works with other Origin Quantum frameworks and toolchains (e.g., VQNet, Origin Liangyu), supporting the full workflow from algorithm design to verification. Real-hardware support is delivered as release qualification and is mandatory before any release can be created.

------

## 2.1.0 Highlights
2.1.0 upgrades the execution architecture, the algorithm library, and the release process while keeping every existing CPU call compatible. See the [Changelog](Tutorials/source/Changelog.rst) for the full record.

1. **Three new stable algorithms**: `pyqpanda_alg.VQE`, `pyqpanda_alg.HHL`, and `pyqpanda_alg.Shor`, imported as submodules (e.g. `from pyqpanda_alg.VQE import VQE`), with tutorials [VQE](Tutorials/source/VQE.rst), [HHL](Tutorials/source/HHL.rst), and [Shor](Tutorials/source/Shor.rst).
2. **Unified execution layer `pyqpanda_alg.execution`**: one task contract (sampling / expectation estimation / state vectors / variational sessions), `ExecutionOptions`, checkpoint & resume, capability checks (`DeviceCapabilityError`), and 10 public execution errors; a runtime failure never silently falls back.
3. **All existing algorithms migrated to the execution layer**: QAOA, QUBO, Grover, QAE/IQAE, QARM, QKmeans, QPCA, QSVM, QSVR, QSVD, QSEncode, QmRMR, and the other executable entry points now take keyword-only `backend=` / `execution_options=` (omitting them keeps the historical CPU behaviour) and no longer construct `CPUQVM` directly; QAOA, QUBO, QAE, QARM, and QmRMR also expose a `submit()` task interface.
4. **Behavior changes and deprecations**: QARM's `machine_type='QCloud'` mode is deprecated (an explicit `backend=` is now required); QPCA's default sampling shots changed from 8192 to 1000 (results are normalized by the actual shot count, preserving the output shape).
5. **Mandatory real-QPU release qualification gate**: 18 fixed small-scale cases (FakeBackend preflight plus real QPU) produce a sanitized qualification manifest that `check_release.py` verifies against the exact commit, wheel digest, version, device record, and every case verdict before a tag release job can create a GitHub Release; qualification output is never committed.
6. **Testing and documentation**: CI tests the installed wheel on a 3 OS x 3 Python version matrix and enforces a coverage baseline; new tutorials cover VQE/HHL/Shor, the execution layer, and the per-algorithm backend contract ([runtime_algorithms](Tutorials/source/runtime_algorithms.rst)).

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
- **QKmeans (Quantum K-Means Clustering)**  
  Quantum-accelerated large-scale data clustering for high-dimensional data scenarios.
- **QPCA (Quantum Principal Component Analysis)**  
  Extracts data principal components via quantum circuits to accelerate dimensionality reduction.
- **QmRMR (Quantum Minimum Redundancy Maximum Relevance)**  
  Implements efficient feature selection to reduce redundant feature interference.
- **QARM (Quantum Association Rule Mining)**  
  Rapidly mines frequent itemsets and association rules (e.g., market basket analysis).

### 3. Scientific Computing & Numerical Solution Algorithms
Solves key problems in physical modeling and engineering simulation (e.g., eigenvalues, linear equations, integer factorization, matrix decomposition).
- **VQE (Variational Quantum Eigensolver)** `new in 2.1.0`  
  A variational algorithm for the ground-state energy of a pyqpanda3 `Hamiltonian`: inputs are validated up front (Hermiticity, finite coefficients), a built-in hardware-efficient ansatz (RY/RZ rotation layers plus a linear CNOT chain) or a custom ansatz is optimized with a classical optimizer. API: `VQE` / `VQEConfig` / `VQEResult`, imported via `from pyqpanda_alg.VQE import VQE`.
- **HHL (Linear System Solver)** `new in 2.1.0`  
  Solves finite Hermitian linear systems `A x = b` through the HHL subroutine: the matrix and right-hand vector are validated at construction (square, Hermitian, invertible, configurable condition-number threshold), and non-power-of-two dimensions are padded by reversible block-diagonal augmentation. The remote backend path reconstructs the solution from Pauli-basis sampling and reports a reconstruction fidelity. API: `HHL` / `HHLConfig` / `HHLSolution` and more, imported via `from pyqpanda_alg.HHL import HHL`.
- **Shor (Integer Factorization)** `new in 2.1.0`  
  Factors odd composite integers through quantum order finding: even, prime, and perfect-power moduli are resolved by classical preprocessing (honestly reporting `used_quantum=False`); every other modulus drives a bounded number of quantum order-finding attempts whose phases are turned into a multiplicative order via continued fractions. API: `Shor` / `ShorConfig` / `ShorResult` / `classical_preprocess` / `recover_order`, imported via `from pyqpanda_alg.Shor import Shor`.
- **QSVD (Quantum Variational Singular Value Decomposition)**  
  Extracts matrix singular values/vectors under a variational framework (for dimensionality reduction and recommendation systems).

### 4. General Tools & Basic Components
Provides underlying tools for quantum computing workflows.
- **QAE (Quantum Amplitude Estimation)**  
  Precisely estimates target state amplitude/measurement probability with quadratic speedup (e.g., financial derivative pricing, risk assessment).
- **Comparator (Quantum Comparator, module `QCmp`)**  
  Implements numerical comparison or threshold judgment to build quantum decision logic.
- **QSEncode (Sparse Amplitude Encoding, class `QSpare_Code`)**  
  Efficiently encodes sparse vectors into quantum states, reducing quantum resource consumption (for data preprocessing).

------

## Installation
pyqpanda_alg is an algorithm extension module based on pyqpanda3. Its installation and usage depend on pyqpanda3. Refer to [pyqpanda3](https://qcloud.originqc.com.cn/document/qpanda-3/cn/index.html) for interface usage.

If Python and pip are installed, run the following command in the terminal/console:  
`pip install pyqpanda_alg`

Optional: to use the qpanda3-runtime remote execution backend, install with the optional dependency:  
`pip install pyqpanda_alg[runtime]`

#### Note:
Add `sudo` if you encounter permission issues on Linux.

------

## Runtime Guarantees and Release Gate

### qpanda3-runtime remote execution
The optional qpanda3-runtime backend is installed with `pip install pyqpanda_alg[runtime]`. Remote execution requires explicit injection of an already logged-in `RuntimeService` and an explicitly selected device: `QPandaRuntimeBackend(service, device)` never authenticates or selects a device on its own.

### Capability errors
Requesting a capability the device does not advertise (e.g. statevector or variational sessions) raises `DeviceCapabilityError`; a remote failure never silently falls back to the CPU backend or to a classical substitute result.

### Small-scale guarantees
Remote and real-hardware guarantees are scoped to the fixed, small-scale cases committed for release qualification (fixed circuits, shots, and statistical thresholds). They are not a general claim about arbitrary problem sizes or arbitrary devices.

### Mandatory QPU release gate
Release creation for 2.1.0 requires real-QPU qualification: the manually dispatched `runtime-rc` workflow runs the fixed preflight and QPU cases against an explicitly selected device and uploads a sanitized qualification manifest; the tag release job refuses to create a GitHub Release unless the manifest matches the candidate commit, wheel digest, version, device record, and every case verdict. Qualification output is never committed (committing it would change the commit the manifest attests to), and this repository does not claim any particular device has passed beyond what the manifest records.

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

## Branch Explanation
This project follows a "stable branch + development branch" collaboration model, with different branches serving distinct purposes to help you choose the right one for your needs.

### 📌 main (Main Branch)
- **Core Purpose**: The `main` branch is the project's **stable release branch**, containing fully tested, production-ready open-source quantum algorithm code.
- **Usage Scenario**: If you want to study, learn the core implementation of quantum algorithms, or directly build upon mature code for secondary development, you can pull and use the content from the `main` branch.
- **Important Note**: The `main` branch does **not** accept any direct commits or Pull Requests (PRs), ensuring the core code remains stable and bug-free.

### 🛠️ develop (Development Branch)
- **Core Purpose**: The `develop` branch is the project's **collaborative development branch**, used for integrating community contributions, iterating new features, and fixing issues.
- **Submission Scenario**: If you wish to contribute to the project, including but not limited to:Submitting bug fixes、Improving project documentation (e.g., README, code comments)、Adding example code for quantum algorithms、Proposing or implementing new feature suggestions、Optimizing the performance or readability of existing algorithms，please submit all Pull Requests (PRs) **to the `develop` branch only**.
- **Collaboration Note**: We regularly review contributions on the `develop` branch. After testing and validation, approved changes will be merged into the `main` branch, so high-quality contributions are synced to the stable release.

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

