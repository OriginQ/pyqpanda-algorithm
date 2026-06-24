# QReupload — 数据重上传变分量子学习 / Data Re-uploading Variational Quantum Learning

> 中文简介在前，英文详细参考在后。 *(Chinese summary first, full English reference below.)*

---

## 中文

### 简介

`QReupload` 为 `pyqpanda-algorithm` 新增了**首个电路参数针对监督任务损失进行训练**的
机器学习估计器（sklearn 风格 `fit`）。`QSVM`/`QSVR` 只是在一个*无参数*的量子特征映射上
训练经典 SVM；而 `QReupload` 是一个**可训练**的数据重上传电路，带有可学习的输入缩放因子、
可选的纠缠环、精确伴随梯度训练，以及两个让模型行为可检视的诊断工具。

### 为什么

数据重上传模型将数据编码旋转门与可训练的单比特门块交替堆叠。每个量子比特对应一个特征、
重上传 `n_layers` 次时，它在输入上实现了一个**可控阶数的截断傅里叶级数**
（Schuld, Sweke & Meyer, *PRA* 103, 032430, 2021）。可训练的输入缩放因子让模型把可达频率
对齐到数据；纠缠环让它能表达"单比特读出之和"无法表达的多特征**交叉频率**项。

### 组件

| 对象 | 用途 |
|---|---|
| `QReuploadRegressor` | sklearn 风格 `fit / predict / score` 回归器 |
| `QReuploadClassifier` | 二分类器（`predict_proba`，sigmoid + 交叉熵）|
| `fourier_spectrum(model)` | 还原已拟合单特征模型学到的傅里叶谐波 |
| `trainability_scan(qubit_range)` | 贫瘠高原诊断：`Var[∂⟨Z⟩/∂θ]` 随比特数的变化 |

训练使用 `pyqpanda3.vqcircuit.VQCircuit` + `DiffMethod.ADJOINT_DIFF`（精确解析梯度），
由 NumPy 实现的 Adam + 余弦退火学习率优化。

### 用法

```python
import numpy as np
from pyqpanda_alg.QReupload import QReuploadRegressor, fourier_spectrum

X = np.linspace(-np.pi, np.pi, 80)
y = np.sin(2 * X) + 0.5 * np.cos(3 * X)

reg = QReuploadRegressor(n_layers=5).fit(X, y)
print(reg.score(X, y))                 # ~0.9999

k, mag = fourier_spectrum(reg)         # 峰值出现在 k = 2 与 3 —— 即目标频率
```

完整演示见 `example/QAlgBase/testeg_qreupload.py`（频谱读出、交叉频率目标上的纠缠消融、
贫瘠高原扫描），每一项都与交叉验证后的经典傅里叶岭回归基线并列展示。

### 设计要点（在 pyqpanda3 0.3.5 上验证）

- **数据作为常数进入电路**，而非第二个参数：`Param(λ) * float(x)`。两个占位符相乘
  （`Param * Param`）会返回*零*伴随梯度，因此每个样本被烘焙进各自的电路。
- **输入缩放因子初始化在 1.0 附近**；初始化在 0 附近会把编码塌缩为恒等映射并使训练停滞。
- **读出使用局部可观测量** `⟨Z_q⟩`；梯度方差仍随比特数大致指数衰减（运行
  `trainability_scan` 测量你配置下的速率），因此应有意识地扩展比特数，而非盲目增加。

### 诚实说明

在带限数据上，一个傅里叶感知、正确正则化的经典模型是很强的基线；示例将其并列展示而非隐藏。
`QReupload` 在此的价值是*可训练的编码*、*可读的频谱*和*内置的可训练性警告*，
而不是在原始误差上击败该基线。

---

## English

`QReupload` adds the first ML estimator to `pyqpanda-algorithm` whose **circuit
parameters are trained against a supervised task loss** (sklearn-style `fit`).
`QSVM`/`QSVR` fit only a classical SVM over a *parameter-free* quantum feature
map; `QReupload` is a **trainable** data re-uploading circuit with a learnable
input-scaling factor, an optional entangling ring, exact adjoint-gradient
training, and two diagnostics that make the model's behaviour inspectable.

### Why

A data re-uploading model interleaves data-encoding rotations with trainable
single-qubit blocks. With one feature per qubit and `n_layers` re-uploads it
realises a **truncated Fourier series of controllable degree** in the inputs
(Schuld, Sweke & Meyer, *PRA* 103, 032430, 2021). A trainable input-scaling
factor lets the model align its accessible frequencies to the data; an
entangling ring lets it represent multi-feature **cross-frequency** terms that a
sum of per-qubit readouts cannot.

### Components

| object | purpose |
|---|---|
| `QReuploadRegressor` | sklearn-style `fit / predict / score` regressor |
| `QReuploadClassifier` | binary classifier (`predict_proba`, sigmoid + BCE) |
| `fourier_spectrum(model)` | recover which Fourier harmonics a fitted 1-feature model learned |
| `trainability_scan(qubit_range)` | barren-plateau diagnostic: `Var[∂⟨Z⟩/∂θ]` vs qubit count |

Training uses `pyqpanda3.vqcircuit.VQCircuit` with `DiffMethod.ADJOINT_DIFF`
(exact analytic gradients), optimised with a NumPy Adam + cosine-annealed LR.

### Usage

```python
import numpy as np
from pyqpanda_alg.QReupload import QReuploadRegressor, fourier_spectrum

X = np.linspace(-np.pi, np.pi, 80)
y = np.sin(2 * X) + 0.5 * np.cos(3 * X)

reg = QReuploadRegressor(n_layers=5).fit(X, y)
print(reg.score(X, y))                 # ~0.9999

k, mag = fourier_spectrum(reg)         # peaks at k = 2 and 3 — the target frequencies
```

See `example/QAlgBase/testeg_qreupload.py` for a full demo (spectrum readout,
entanglement ablation on a cross-frequency target, barren-plateau scan), each
shown next to a cross-validated classical Fourier-ridge baseline.

### Design notes (verified on pyqpanda3 0.3.5)

- **Data enters as a constant**, not a second parameter: `Param(λ) * float(x)`.
  A product of two placeholders (`Param * Param`) returns a *zero* adjoint
  gradient, so each sample is baked into its own circuit.
- **Input-scaling factors are initialised near 1.0**; initialising them near 0
  collapses the encoding to identity and stalls training.
- **Readout uses local observables** `⟨Z_q⟩`; the gradient variance still decays
  roughly exponentially in qubit count (run `trainability_scan` to measure the
  rate for your config — e.g. ~`exp(-0.8 · n)` for the default depth-3 probe at
  `seed=0`), so scale qubits deliberately, not blindly.

### Honesty

On band-limited data a Fourier-aware, properly-regularised classical model is a
strong baseline; the example reports it side by side rather than hiding it. The
value of `QReupload` here is the *trainable encoding*, the *readable spectrum*,
and the *built-in trainability warning* — not beating that baseline on raw error.
