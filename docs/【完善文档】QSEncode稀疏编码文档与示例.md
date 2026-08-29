## 【QSEncode 稀疏编码模块文档补充】

---

### **1. 算法介绍**

**QSEncode（量子稀疏态编码，Quantum Sparse State Encoding）** 是 `pyqpanda_alg` 中用于将经典概率分布高效加载到量子态的模块，对外提供 `QSpare_Code` 类。它面向"目标分布在某个变换基下近似稀疏"的场景：先在 Walsh-Hadamard 或 Fourier 基下保留占主导的少数系数（稀疏截断），再通过振幅编码与相应的逆变换量子线路，在量子态上重建原分布的稀疏近似，从而以较少的量子资源完成量子态制备。

---

### **2. 核心思想**

经典概率分布直接做振幅编码往往需要与维度成正比的线路资源。QSEncode 的核心思想是：

1. 许多实际分布（如正态分布等平滑分布）在 **Walsh-Hadamard 基**或 **Fourier 基** 下能量集中于少数系数；
2. 只保留模长最大的 `cut_length` 个变换系数（其余置零），得到的稀疏向量仍是原分布在对应基下的良好近似；
3. 对该稀疏向量做振幅编码后，再施加 H 门阵列（walsh 模式，$H^{\otimes n}$ 即归一化 Walsh-Hadamard 变换）或 QFT（fourier 模式），即可在计算基底下还原出原分布的稀疏近似。

---

### **3. 数学原理与实现步骤**

设输入概率分布为 $p=(p_0,\dots,p_{N-1})$（$N=2^n$，不足时自动补零），振幅向量 $a=\sqrt{p}$（逐元素开方）。

**Walsh-Hadamard 模式（`mode='walsh'`）**：

$$
W_k = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} (-1)^{k\cdot j}\, a_j
$$

其中 $k\cdot j$ 表示 $k$ 与 $j$ 的二进制按位内积（模 2）。实现上借助快速 Walsh-Hadamard 变换（FWHT）完成。

**Fourier 模式（`mode='fourier'`）**：

$$
F_k = \sum_{j=0}^{N-1} a_j\, e^{-2\pi i k j / N}
$$

实现上调用 `scipy.fft.fft`（未归一化约定）。

**完整实现步骤**：

1. 输入校验与预处理：概率列表归一化，长度补零至 $2^n$，$n=\lceil\log_2 N\rceil$ 为所需量子比特数；
2. 对振幅向量做所选基变换（FWHT 或 FFT）；
3. 按模长保留 top-`cut_length` 个系数（缺省 `cut_length = 2n`），其余置零，并重新归一化；
4. 使用 `Encode.amplitude_encode` 将该稀疏向量编码为量子态；
5. walsh 模式追加 $H^{\otimes n}$，fourier 模式追加 QFT，得到近似还原原分布的量子态；
6. `Quantum_Res()` 在 `CPUQVM` 上模拟运行线路并返回测量概率分布。

---

### **4. 应用领域与应用方式**（举例说明）

QSEncode 适用于量子机器学习、量子金融、量子模拟等需要把经典概率分布加载到量子态的场景，可在变换基近似稀疏的前提下降低态制备的资源开销。

```python
# 导入模块
import numpy as np
import matplotlib.pyplot as plt
from pyqpanda_alg.QSEncode import QSpare_Code

# 构造标准正态分布的概率密度（离散采样 2^10 个点）
mu = 0
sigma = 1
x = np.linspace(-3, 3, 2 ** 10)
pdf_normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# 归一化后平方得到合法的概率分布（元素须为 float 且总和为 1）
ini = pdf_normal / np.linalg.norm(pdf_normal)

# walsh 模式，仅保留 20 个主成分做稀疏编码，模拟运行并取回概率分布
res_walsh = QSpare_Code(ini ** 2, mode='walsh', cut_length=20).Quantum_Res()
# fourier 模式同理
res_fourier = QSpare_Code(ini ** 2, mode='fourier', cut_length=20).Quantum_Res()

# 对比原始分布与稀疏编码重建结果
plt.plot(x, ini ** 2, label='origin')
plt.plot(x, res_walsh, label='walsh sparse')
plt.plot(x, res_fourier, label='fourier sparse')
plt.legend()
plt.show()
```

参数说明：

- `prob_list`：`list[float]` 或 `np.ndarray`，元素必须为非负 float，且总和与 1 的偏差不超过 0.001（超出会抛出 `Warning` 异常）；长度不必为 2 的幂，内部自动补零；
- `cut_length`：保留的主成分个数，须为正整数，缺省为 `2 * ceil(log2(len(prob_list)))`；
- `mode`：`'walsh'`（默认）或 `'fourier'`，其余取值会在构建线路/执行变换时抛出 `ValueError`。

---

### **5. 当前限制与挑战**

- **截断误差**：稀疏近似与原始分布存在偏差，`cut_length` 越小偏差越大，需要根据分布的稀疏性权衡资源与精度；
- **输入约束较严格**：元素类型须为 float（int 列表会被拒绝），概率和偏差超过 0.001 会直接抛出 `Warning` 异常而非告警；
- **基变换模式有限**：当前仅支持 walsh 与 fourier 两种基，且 `mode` 的合法性在变换/构建线路阶段才校验，构造时不校验；
- **采样规模**：`Quantum_Res()` 固定以 1000 次测量在模拟器上运行，暂未开放采样次数配置。

---

### **6. 参考文献**

1. Origin Quantum. pyqpanda-algorithm 官方文档：https://qcloud.originqc.com.cn/document/pyqpanda-algorithm/index.html
2. pyQPanda 官方文档（振幅编码 Encode / QFT 等接口）：https://pyqpanda-toturial.readthedocs.io/
3. Fino, B. J., & Algazi, V. R. "Unified matrix treatment of the fast Walsh-Hadamard transform." *IEEE Transactions on Computers*, 1976.
