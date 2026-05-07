# SU(2) Detector Response Toy Simulation：模型原理说明

## 1. 目标

本教程构造一个小规模 SU(2)-对称量子多体模拟示例，用于展示以下内容：

1. qubit 上的 fundamental color spin 表示。
2. SU(2)-scalar interaction 的 Hamiltonian 构造。
3. singlet / triplet sector 的能谱诊断。
4. color-singlet local quench 的线路含义。
5. detector connected color correlation 的计算。

该示例保持模型规模较小，适合教学、算法展示和 pyQPanda CPUQVM 运行。

## 2. qubit 与 SU(2) color spin

每个 qubit 表示一个 spin-1/2 color degree of freedom。单个 site 上定义：

$$
S_a(i) = \frac{\sigma_a(i)}{2}, \qquad a \in \{x, y, z\} .
$$

其中 $\sigma_a$ 是 Pauli operator。两个 site 的 SU(2)-scalar interaction 定义为：

$$
\begin{aligned}
\mathbf{S}(i) \cdot \mathbf{S}(j)
&= S_x(i)S_x(j) + S_y(i)S_y(j) + S_z(i)S_z(j) \\
&= \frac{X_i X_j + Y_i Y_j + Z_i Z_j}{4} .
\end{aligned}
$$

该相互作用在全局 SU(2) rotation 下保持不变，因此适合用作 SU(2)-对称量子多体示例的基本 building block。

## 3. Toy Hamiltonian

教程使用的 Hamiltonian 为：

$$
\begin{aligned}
H(\delta)
&= \sum_i J\left[1 + \delta(-1)^i\right] \mathbf{S}_i \cdot \mathbf{S}_{i+1} \\
&\quad + \kappa \sum_{i<j} e^{-|i-j|/\xi} \mathbf{S}_i \cdot \mathbf{S}_j .
\end{aligned}
$$

参数含义：

- `J`：最近邻 color exchange coupling。
- $\delta$：dimerization 参数，用于调节奇偶 bond 的强度。
- $\kappa$：长程 color interaction 强度。
- $\xi$：长程 interaction 的衰减长度。

第一项描述 dimerized SU(2) color chain。第二项提供一个随距离衰减的长程 color interaction，使示例不仅包含最近邻交换，也能展示更丰富的谱结构变化。

## 4. singlet / triplet sector 诊断

SU(2) 模型的重要信息之一是态所属的 total spin sector。定义：

$$
\mathbf{S}_{\mathrm{tot}} = \sum_i \mathbf{S}_i .
$$

Casimir operator 为：

$$
\begin{aligned}
\mathbf{S}_{\mathrm{tot}}^2
&= \sum_i \mathbf{S}_i^2 + 2\sum_{i<j} \mathbf{S}_i \cdot \mathbf{S}_j \\
&= \frac{3N}{4} + 2\sum_{i<j} \mathbf{S}_i \cdot \mathbf{S}_j .
\end{aligned}
$$

如果一个本征态属于 spin-S sector，则：

$$
\mathbf{S}_{\mathrm{tot}}^2 = S(S+1) .
$$

因此可以从 $\mathbf{S}_{\mathrm{tot}}^2$ 的期望值估计 sector：

$$
S = \frac{-1 + \sqrt{1 + 4\langle \mathbf{S}_{\mathrm{tot}}^2 \rangle}}{2} .
$$

在输出中：

- $S \approx 0$ 对应 singlet sector。
- $S \approx 1$ 对应 triplet sector。
- `singlet_triplet_gap` 表示最低 singlet 与最低 triplet 的能量差，即 $E_{\mathrm{triplet}} - E_{\mathrm{singlet}}$。

## 5. color-singlet local quench

局域 quench 作用在中心 bond 上，使用 SU(2)-scalar operator：

$$
O_q = \mathbf{S}_c \cdot \mathbf{S}_{c+1} .
$$

对应 unitary：

$$
U_q(\theta) = \exp\left(-i\theta O_q\right) .
$$

该 quench 与 Hamiltonian 使用相同的 SU(2)-scalar building block，适合展示中心局域扰动对低能谱权重的影响。

代码记录 quench 后能量注入：

$$
\Delta E = \langle \psi_q | H | \psi_q \rangle - E_0 .
$$

以及低能级 spectral weights：

$$
w_n = \left|\langle E_n | \psi_q \rangle\right|^2 .
$$

## 6. detector connected color correlation

左右 detector site 的 connected color correlation 定义为：

$$
\begin{aligned}
C_{LR}
&= \sum_a \left[\langle S_a(L)S_a(R)\rangle - \langle S_a(L)\rangle\langle S_a(R)\rangle\right] \\
&= \langle \mathbf{S}(L) \cdot \mathbf{S}(R) \rangle - \langle \mathbf{S}(L) \rangle \cdot \langle \mathbf{S}(R) \rangle .
\end{aligned}
$$

该量是 SU(2)-scalar observable，可用于展示两个空间分离位置之间的 connected correlation。代码还会计算 $C_{LR}$ 随 $\delta$ 变化的最大斜率窗口，用于定位有限尺寸系统中的关联结构变化区域。

## 7. 运行输出

默认 6-qubit 示例会输出：

```text
minimum_singlet_triplet_gap
crossover_by_detector_slope
crossover_by_fidelity_dip
largest_singlet_quench_response
```

这些结果用于展示 SU(2)-对称 color-chain 中的能谱结构、态保真度变化、局域 quench 响应和 detector connected correlation。

## 8. 小规模模拟中的技术点

本示例虽然规模较小，但仍包含若干量子模拟中的关键技术点：

1. Hamiltonian 由多个 Pauli-string 组合而成。
2. sector 诊断需要额外构造 $\mathbf{S}_{\mathrm{tot}}^2$。
3. connected correlation 需要组合 one-point 和 two-point expectation values。
4. quench 后态需要投影到能量本征态以分析 spectral weights。
5. pyQPanda 示例需要将态制备和 quench probe 写成可执行线路。

这些设计使该项目既可作为物理模型示例，也可作为量子线路构造与观测量分析的教学案例。
