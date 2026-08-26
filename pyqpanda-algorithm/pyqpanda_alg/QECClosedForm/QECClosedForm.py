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

"""
QECClosedForm —— 量子纠错码闭式参数预测（零电路、零模拟）

对 AG 完备码族（Reed-Muller CSS 码）[[2^m, k, 2^{r+1}]]，由组合闭式直接
给出全套纠错参数：编码率、损失标度、零损失边界、逻辑算符计数、检测率。

设计定位：与 QECNoise（QPanda3 模拟验证 θ⁴ 标度律）互补——本模块是
"预测层"（闭式秒算参数），QECNoise 是"验证层"（模拟确认标度）。两者
闭环：预测 loss(θ)=c_d·θ^d，模拟复现 log-log 斜率 ≈ d。

定理来源：
  - 码参数     [[2^m, n-2·dim RM(r,m), 2^{r+1}]]       （10.30）
  - fail(w0)   1 - Pr/(v_r·P(w0)) - Pr1/(v_r1·P(w0))    （引理 10.35.2.07）
  - κ_r(m)     2^{(r+1)(m-r-1)} / [m choose r+1]_2       （引理 10.35.2.10）
  - loss(θ)    c_d·θ^d, c_d = C(n,w0)·P(w0)·fail(w0)·κ·2^{-2w0}（定理 10.35.1.07）
  - 零损失      k ≤ ⌊(d-1)/2⌋                             （定理 10.31.1.01）
  - 逻辑计数    N = 2^{m-r-1}·[m choose r+1]_2            （定理 10.30.2.04）
  - 检测率      p_det(θ) = sin²(θ/2)                      （10.29 预言 2a）
"""

from math import comb
from fractions import Fraction
import math


def _gb(m, k):
    """高斯二项 [m k]_2（正整数）"""
    if k < 0 or k > m:
        return 0
    num = den = 1
    for i in range(k):
        num *= (1 << (m - i)) - 1
        den *= (1 << (k - i)) - 1
    return num // den


def _flats(m, k):
    """m 维 F_2 空间中 k-平坦数 = 2^{m-k}·[m k]_2"""
    return (1 << (m - k)) * _gb(m, k)


_E_CACHE = {}


def _E(k, s):
    """k 维平坦内、仿射包恰 k 维的 s 点子集数（递推）"""
    key = (k, s)
    if key in _E_CACHE:
        return _E_CACHE[key]
    if s == 1:
        r = 1 if k == 0 else 0
        _E_CACHE[key] = r
        return r
    if k == 0 or s > (1 << k):
        _E_CACHE[key] = 0
        return 0
    total = comb(1 << k, s)
    for j in range(k):
        total -= _flats(k, j) * _E(j, s)
    _E_CACHE[key] = total
    return total


class QECClosedForm:
    """AG 完备码族 [[2^m, k, 2^{r+1}]] 闭式纠错参数预测器。

    Parameters
        m : ``int``\n
            仿射空间维数（码长 n = 2^m）。
        r : ``int``\n
            RM(r,m) 的阶数（码距 d = 2^{r+1}）。

    示例::

        cf = QECClosedForm(10, 3)          # [[1024, 672, 16]]
        cf.code()                          # (1024, 672, 16)
        cf.encoding_rate()                 # 0.65625
        cf.loss(0.01)                      # 1.05e-24
    """

    def __init__(self, m, r):
        self.m = m
        self.r = r
        self.n = 1 << m
        self.d = 1 << (r + 1)
        self.w0 = 1 << r
        dim_rm = sum(comb(m, i) for i in range(r + 1))
        self.k = self.n - 2 * dim_rm
        if self.k < 1:
            raise ValueError(f"参数 m={m}, r={r} 给出非正逻辑比特数 k={self.k}")

        # fail(w0) 的代数成分（引理 10.35.2.07）
        self.Pw = Fraction(_flats(m, r + 1) * _E(r + 1, self.w0) + _flats(m, r),
                           comb(self.n, self.w0))
        self.Pr = Fraction(_flats(m, r), comb(self.n, self.w0))
        self.Pr1 = Fraction(_flats(m, r + 1) * _E(r + 1, self.w0),
                            comb(self.n, self.w0))
        v_r, v_r1 = 1 << (m - r), 2
        self.fail = float(Fraction(1) - self.Pr / (v_r * self.Pw)
                          - self.Pr1 / (v_r1 * self.Pw))
        # κ_r(m)（引理 10.35.2.10）
        self.kap = (1 << ((r + 1) * (m - r - 1))) / _gb(m, r + 1)
        # loss 系数 c_d（定理 10.35.1.07）
        self.c_d = float(Fraction(comb(self.n, self.w0)) * self.Pw * self.fail
                         / (1 << (2 * self.w0))) * self.kap

    def code(self):
        """返回码参数 (n, k, d)。"""
        return self.n, self.k, self.d

    def encoding_rate(self):
        """编码率 k/n。"""
        return self.k / self.n

    def zero_loss_boundary(self):
        """注入零损失边界 k_max = ⌊(d-1)/2⌋（定理 10.31.1.01）。

        注入 ≤ k_max 比特相干旋转，最优纠错后损失恒为 0。
        """
        return (self.d - 1) // 2

    def loss(self, theta):
        """逻辑损失闭式 loss(θ) = c_d·θ^d（定理 10.35.1.07）。

        Parameters
            theta : ``float``\n
                单比特相干旋转角度上限 θ_max。
        """
        return self.c_d * (theta ** self.d)

    def logical_operator_count(self):
        """权重 d 逻辑算符数（定理 10.30.2.04）:
        N = 2^{m-r-1}·[m choose r+1]_2 = AG(m,2) 的 (r+1)-平坦数。"""
        return (1 << (self.m - self.r - 1)) * _gb(self.m, self.r + 1)

    @staticmethod
    def detection_rate(theta):
        """检测率闭式 p_det(θ) = sin²(θ/2)（10.29 预言 2a，与码无关）。"""
        return math.sin(theta / 2) ** 2

    def summary(self):
        """返回一行可读的参数摘要。"""
        return (f"[[{self.n},{self.k},{self.d}]] rate={self.encoding_rate():.4f} "
                f"w0={self.w0} fail={self.fail:.4f} κ={self.kap:.4f} "
                f"c_d={self.c_d:.4g} zero-loss≤{self.zero_loss_boundary()} "
                f"logicals={self.logical_operator_count()}")


__all__ = ["QECClosedForm"]
