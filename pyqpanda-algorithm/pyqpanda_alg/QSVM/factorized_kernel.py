# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""仅依赖测量计数，对固定且理想的 QSVM 特征映射进行分量采样。"""

from collections.abc import Mapping
from itertools import combinations, islice, product
from numbers import Integral

import numpy as np
from pyqpanda3.core import CNOT, QProg, U1, U2, measure


__all__ = ["FactorizedQuantumKernel"]


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _as_data(values, name):
    if values is None:
        raise ValueError(f"{name} must not be None")
    try:
        data = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric array") from exc
    if data.ndim not in (1, 2):
        raise ValueError(f"{name} must be a 1D or 2D array")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"{name} must have at least two features")
    if data.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        data = np.array(data, dtype=np.float64, copy=True)
        if not np.isfinite(data).all():
            raise ValueError(f"{name} must contain finite float64 values")
        # 特征值即使有限，仍可能使固定映射的门参数溢出。
        phase = 2.0 * (np.pi - data[:, 0]) * (np.pi - data[:, 1])
        if not np.isfinite(2.0 * data).all() or not np.isfinite(phase).all():
            raise ValueError(f"{name} produces non-finite feature-map angles")
    return data


def _overlap_program(x, y):
    """将原有重叠电路的门顺序限制到单个连通分量。"""
    width = len(x)
    program = QProg(width)
    qubits = program.qubits()
    phase_x = phase_y = 0.0
    if width == 2:
        phase_x = 2.0 * (np.pi - x[0]) * (np.pi - x[1])
        phase_y = 2.0 * (np.pi - y[0]) * (np.pi - y[1])
    for layer in range(2):
        for q, value in zip(qubits, x):
            program << U2(q, 0, np.pi) << U1(q, 2.0 * value)
        if width == 2:
            program << CNOT(qubits[0], qubits[1]) << U1(qubits[1], phase_x)
            if layer == 1:
                # 与原实现一致，仅消去中间相邻的一对 CNOT。
                program << U1(qubits[1], -phase_y)
            program << CNOT(qubits[0], qubits[1])
    for layer in range(2):
        for q, value in zip(qubits, y):
            program << U1(q, -2.0 * value) << U2(q, 0, np.pi)
        if width == 2 and layer == 0:
            program << CNOT(qubits[0], qubits[1])
            program << U1(qubits[1], -phase_y) << CNOT(qubits[0], qubits[1])
    program << measure(qubits, qubits)
    return program


def _zero_frequency(counts, width, shots):
    if not isinstance(counts, Mapping):
        raise ValueError("run_counts must return a counts mapping")
    total = 0
    for bits, count in counts.items():
        if not isinstance(bits, str) or len(bits) != width or any(b not in "01" for b in bits):
            raise ValueError(f"counts keys must be {width}-bit binary strings")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0:
            raise ValueError("counts values must be non-negative integers")
        total += int(count)
    if total != shots:
        raise ValueError(f"counts total must equal requested shots ({shots}), got {total}")
    return int(counts.get("0" * width, 0)) / shots


class FactorizedQuantumKernel:
    """使用至多两个量子比特的电路对现有 QSVM 映射采样。

    这是需显式选用的*理想、固定映射*估计器，并非通用的含噪回波核。
    此映射有两层编码，仅使特征 0 和 1 发生纠缠，其后各特征保持独立。
    每个非恒等分量均由调用方制备并测量；其全零结果频率用于乘积计算。
    此处不使用模拟器、量子态、概率查询接口或半正定（PSD）投影。
    ``QuantumKernel_vqnet`` 保持不变。

    参数
    ----
    run_counts : callable
        必须提供 ``run_counts(program, shots) -> Mapping[str, int]``。
        从全零初始化出发，执行所提供的带测量 QProg，并返回恰好
        ``shots`` 次观测。键为局部宽度（1 或 2）的二进制字符串，
        值为非负整数。缺少全零键表示成功次数为零。
        执行函数负责执行与重置，必须保证不同分量位置之间的样本独立，
        且在整个求值过程中保持平稳的统计行为。执行函数的异常直接向上传播。
    shots : int，默认值为 1024
        每个不同的非恒等分量的采样次数，而非每个矩阵元素的采样次数。
    batch_size : int，默认值为 100
        一次缓冲的矩阵索引对数量上限。对执行函数的调用仍按顺序进行，
        每次均采样 ``shots`` 次；此参数既不限制单次求值内的缓存大小，
        也不将设备提交合并成批次。
    n_qbits : int 或 None，默认值为 None
        可选的预期特征数（>=2）。为 None 时，每次调用 evaluate 时推断。
        此值是原始特征宽度，而非执行电路的宽度。

    说明
    ----
    分量输入在转换为 float64 后若完全相等，则无需执行，直接贡献 1。
    这一恒等性质与对称性以理想映射为前提；它们不校准制备、量子门
    或测量噪声。同样，局部测量未必能复现完整含噪设备电路的结果。

    单次求值仅在分量位置相同且有序输入对完全相等时复用已测频率。
    不同位置即使数值相同，也保持独立。复用与对称镜像会使矩阵元素
    之间产生相关性；两者都不会产生额外样本。调用结束后不保留任何
    统计量，调用失败时亦然。不同次执行函数调用不得重放同一条
    公共随机样本流。

    相互独立且无偏的分量频率，其乘积是理想核的无偏估计，
    但有限采样次数下的 Gram 矩阵不一定半正定。
    假设条件与资源成本参见 ``docs/QSVM_factorized_kernel.md``。
    """

    def __init__(self, run_counts, *, shots=1024, batch_size=100, n_qbits=None):
        if not callable(run_counts):
            raise TypeError("run_counts must be callable")
        self._run_counts = run_counts
        self._shots = _positive_integer(shots, "shots")
        self._batch_size = _positive_integer(batch_size, "batch_size")
        self._n_qbits = None if n_qbits is None else _positive_integer(n_qbits, "n_qbits", 2)

    def evaluate(self, x_vec, y_vec=None):
        """返回形状为 (n_x, n_y)、类型为 float64 的采样核矩阵；可用作 SVC 的可调用核函数。

        接受元素为有限实数的一维（单个样本）或二维类数值数组，
        特征维数须一致且满足 d>=2。显式形状为 (0, d) 的空数组返回空矩阵；
        [] 没有有效的特征维度，因此会被拒绝。x_vec=None、参差不齐或
        非数值数组、非有限值或角度，以及不一致的特征宽度，均会在
        调用任何执行函数之前引发 ValueError。y_vec=None 表示使用 x_vec。
        两个数据集相等时，仅计算一个三角区域，再精确镜像到另一侧。
        """
        x = _as_data(x_vec, "x_vec")
        y = x if y_vec is None else _as_data(y_vec, "y_vec")
        if x.shape[1] != y.shape[1]:
            raise ValueError("x_vec and y_vec have incompatible feature dimensions")
        if self._n_qbits is not None and x.shape[1] != self._n_qbits:
            raise ValueError("feature dimension must equal n_qbits")
        symmetric = np.array_equal(x, y)
        kernel = np.empty((len(x), len(y)), dtype=np.float64)
        if symmetric:
            np.fill_diagonal(kernel, 1.0)
            pairs = combinations(range(len(x)), 2)
        else:
            pairs = product(range(len(x)), range(len(y)))
        components = [(0, 2)] + [(j, j + 1) for j in range(2, x.shape[1])]
        cache = {}
        batch_limit = min(self._batch_size, kernel.size)
        while True:
            batch = list(islice(pairs, batch_limit))
            if not batch:
                break
            for i, j in batch:
                value = 1.0
                for start, stop in components:
                    cx = tuple(x[i, start:stop])
                    cy = tuple(y[j, start:stop])
                    if cx == cy:
                        continue  # 理想固定映射的恒等性质，而非含噪测量。
                    key = (start, stop, cx, cy)
                    if key not in cache:
                        program = _overlap_program(cx, cy)
                        counts = self._run_counts(program, self._shots)
                        cache[key] = _zero_frequency(counts, stop - start, self._shots)
                    value *= cache[key]
                kernel[i, j] = value
                if symmetric:
                    kernel[j, i] = value
        return kernel
