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

"""用于 Z/ZZ 均值目标、仅依赖采样的反向光锥 QAOA。

此执行器需显式选用，采用 |+> 初始化、exp(-i gamma H_cost) 和
RX(-2 beta)，与现有 QAOA 实现的默认门一致。
仅复用不可变的依赖计划。调用方必须提供返回测量计数的执行函数；
本模块从不创建量子后端，也不查询其状态。"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import fsum, isfinite
from numbers import Integral, Real
from types import MappingProxyType

from pyqpanda3.core import CNOT, H, QProg, RX, RZ, measure

__all__ = [
    "QAOALightCone", "ZTerm", "LightConeLayer", "LightConePlan",
    "LightConeProgram", "LightConeEstimate",
]


@dataclass(frozen=True)
class ZTerm:
    """已复制并规范化的全局支撑集及其有限实系数。"""

    qubits: tuple[int, ...]
    coefficient: float


@dataclass(frozen=True)
class LightConeLayer:
    """保留的代价项与混合器量子比特，均使用全局索引。"""

    cost_terms: tuple[ZTerm, ...]
    mixer_qubits: tuple[int, ...]


@dataclass(frozen=True)
class LightConePlan:
    """单个目标的不可变计划；各层按正向执行顺序排列。

    ``global_qubits[local_index]`` 是 ``global_to_local`` 的逆映射。
    此计划仅描述结构，不针对任何特定的 gamma 或 beta 值。
    """

    target: ZTerm
    global_qubits: tuple[int, ...]
    layers: tuple[LightConeLayer, ...]

    @property
    def global_to_local(self) -> Mapping[int, int]:
        """从原始标签到连续局部索引的只读映射。"""
        return MappingProxyType({q: i for i, q in enumerate(self.global_qubits)})


@dataclass(frozen=True)
class LightConeProgram:
    """新建且归调用方所有的 QProg，以及其不可变的解释信息。

    QProg 本身可变；它从不被存回执行器。
    所有局部量子比特均测量到同索引的经典比特。宽度为 ``w`` 时，
    计数字典的键必须严格采用 ``c[w-1]...c[1]c[0]`` 格式（c0 在最右侧）。
    """

    program: QProg
    plan: LightConePlan

    @property
    def local_target(self) -> tuple[int, ...]:
        """目标量子比特从全局索引重映射到局部索引后的结果。"""
        mapping = self.plan.global_to_local
        return tuple(mapping[q] for q in self.plan.target.qubits)

    @property
    def parity_cbits(self) -> tuple[int, ...]:
        """其异或值决定目标 +/-1 测量结果的经典比特。"""
        return self.local_target

    @property
    def measurement_cbits(self) -> tuple[int, ...]:
        """所有被测经典比特的索引，按递增顺序排列。"""
        return tuple(range(len(self.plan.global_qubits)))


@dataclass(frozen=True)
class LightConeEstimate:
    """单次求值的采样均值与采样次数统计。

    ``term_means`` 按 ``executor.plans`` 的顺序存储未加权的奇偶均值。
    常数项不占条目，也不需要采样。``total_shots`` 统计所有程序的
    总制备次数，而非单个全局电路的采样次数。
    """

    mean: float
    term_means: tuple[float, ...]
    shots_per_program: int
    total_shots: int


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite as a float") from error
    if not isfinite(result):
        raise ValueError(f"{name} must be finite as a float")
    return result


def _objective_sum(values: list[float]) -> float:
    """保留 fsum 的常规路径；溢出时恢复相消后仍有限的求和结果。"""
    try:
        result = fsum(values)
    except OverflowError:
        # 这些值是已经舍入的有限加权样本均值，而非量子振幅。
        # 此处精确求和可避免中间和溢出，同时不会因缩放而丢失
        # 大数相消后的微小残差。
        exact = sum((Fraction.from_float(value) for value in values), Fraction(0))
        try:
            result = float(exact)
        except OverflowError as error:
            raise ValueError("sampled objective is outside the finite float range; shots were consumed") from error
    if not isfinite(result):
        raise ValueError("sampled objective is outside the finite float range; shots were consumed")
    return result


def _parameters(values: Iterable[Real], depth: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be an iterable of real numbers")
    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of real numbers") from error
    if len(values) != depth:
        raise ValueError(f"{name} must contain exactly depth={depth} values")
    return tuple(_real(value, f"{name}[{i}]") for i, value in enumerate(values))


def _copy_terms(n_qubits: int, terms: Mapping) -> tuple[ZTerm, ...]:
    if not isinstance(terms, Mapping):
        raise TypeError("terms must map qubit tuples to real coefficients")
    copied = {}
    for support, coefficient in terms.items():
        if not isinstance(support, tuple):
            raise TypeError("each term key must be a tuple of qubit indices")
        if len(support) > 2:
            raise ValueError("only constants, Z fields and ZZ interactions are supported")
        qubits = tuple(_integer(q, "qubit index") for q in support)
        if len(set(qubits)) != len(qubits):
            raise ValueError("a term must not repeat a qubit index")
        if any(q >= n_qubits for q in qubits):
            raise ValueError("qubit indices must be less than n_qubits")
        qubits = tuple(sorted(qubits))
        if qubits in copied:
            raise ValueError("duplicate term after canonicalizing qubit order")
        copied[qubits] = _real(coefficient, "coefficient")
    return tuple(ZTerm(qubits, coefficient) for qubits, coefficient in sorted(copied.items())
                 if coefficient != 0.0)


def _make_plan(target: ZTerm, terms: tuple[ZTerm, ...],
               incident: Mapping[int, list[int]], depth: int) -> LightConePlan:
    support = set(target.qubits)
    backwards = []
    for _ in range(depth):
        # 遍历此可交换代价层之前，先固定进入该层时的支撑集。
        # 若在选择项时迭代扩展支撑集，就会错误地在单层内
        # 沿链传播至任意远处。
        mixers = tuple(sorted(support))
        selected = sorted({index for q in mixers for index in incident[q]})
        costs = tuple(terms[index] for index in selected)
        backwards.append(LightConeLayer(costs, mixers))
        for term in costs:
            support.update(term.qubits)
    return LightConePlan(target, tuple(sorted(support)), tuple(reversed(backwards)))


def _parity_mean(counts: Mapping[str, int], width: int,
                 parity_cbits: tuple[int, ...], shots: int) -> float:
    if not isinstance(counts, Mapping):
        raise TypeError("run_counts must return a mapping of bitstrings to integer counts")
    total = 0
    signed_total = 0
    for key, count in counts.items():
        if not isinstance(key, str) or len(key) != width or any(bit not in "01" for bit in key):
            raise ValueError(f"counts keys must be {width}-bit binary strings with c0 at the right")
        count = _integer(count, "count")
        total += count
        parity = sum(key[-1 - cbit] == "1" for cbit in parity_cbits) % 2
        signed_total += -count if parity else count
    if total != shots:
        raise ValueError(f"counts sum to {total}, but {shots} shots were requested")
    return signed_total / shots


@dataclass(frozen=True, init=False)
class QAOALightCone:
    """用于 ``H = constant + sum a_S Z_S`` 的可复用局部采样计划。

    参数
    ----
    n_qubits : int
        显式指定的非负全局宽度。允许使用稀疏的全局标签。
    terms : Mapping[tuple[int, ...], real]
        ``()`` 表示常数；``(i,)`` 表示 Z_i；``(i, j)`` 表示 Z_i Z_j。
        同一项内的索引必须互不相同，且均位于 ``range(n_qubits)`` 内。
        比特对的顺序会被规范化；规范化后支撑集重复的项将被拒绝。
        权重被复制为有限浮点数；零系数项被省略。
        不接受布尔值或复数（即使其虚部为零）。
    depth : int
        先执行代价门、再执行混合器的层数，须非负。深度为零时，
        非常数目标仍从局部制备的 |+> 态中采样。

    说明
    ----
    每个非零且非常数的项对应一个计划，按支撑集排序。
    不提供分组、优化器、CVaR/Gibbs 目标、完整比特串重建、
    后端选择或期望值缓存。修改权重或深度需要新建实例；
    修改角度时仅复用计划。
    """

    n_qubits: int
    terms: tuple[ZTerm, ...]
    depth: int
    constant: float
    plans: tuple[LightConePlan, ...]

    def __init__(self, n_qubits: int, terms: Mapping[tuple[int, ...], Real], depth: int):
        n_qubits = _integer(n_qubits, "n_qubits")
        depth = _integer(depth, "depth")
        copied = _copy_terms(n_qubits, terms)
        nonconstant = tuple(term for term in copied if term.qubits)
        constant = next((term.coefficient for term in copied if not term.qubits), 0.0)
        incident = {}
        for index, term in enumerate(nonconstant):
            for q in term.qubits:
                incident.setdefault(q, []).append(index)
        plans = tuple(_make_plan(target, nonconstant, incident, depth) for target in nonconstant)
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "terms", copied)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "plans", plans)

    def build_programs(self, gammas: Iterable[Real], betas: Iterable[Real]) -> tuple[LightConeProgram, ...]:
        """为恰好 ``depth`` 对角度新建带测量的局部 QProg。

        每个程序先将光锥内的各量子比特制备为 |+> 态，再执行保留的
        代价门（RZ(2 gamma a) 或 CNOT/RZ/CNOT）和 RX(-2 beta) 混合器。
        常数项仅产生全局相位，因此被省略。此处不调用任何设备。
        所有参数及计算得到的门角度都必须是有限的实浮点数。
        """
        gamma_values = _parameters(gammas, self.depth, "gammas")
        beta_values = _parameters(betas, self.depth, "betas")
        mixer_angles = tuple(_real(-2.0 * beta, "mixer angle") for beta in beta_values)
        jobs = []
        for plan in self.plans:
            mapping = plan.global_to_local
            width = len(plan.global_qubits)
            program = QProg(width)
            for q in range(width):
                program << H(q)
            for layer, gamma, mixer_angle in zip(plan.layers, gamma_values, mixer_angles):
                for term in layer.cost_terms:
                    local = tuple(mapping[q] for q in term.qubits)
                    angle = _real((gamma * term.coefficient) * 2.0, "cost angle")
                    if len(local) == 1:
                        program << RZ(local[0], angle)
                    else:
                        program << CNOT(local[0], local[1])
                        program << RZ(local[1], angle)
                        program << CNOT(local[0], local[1])
                for q in layer.mixer_qubits:
                    program << RX(mapping[q], mixer_angle)
            for q in range(width):
                program << measure(q, q)
            jobs.append(LightConeProgram(program, plan))
        return tuple(jobs)

    def estimate(self, gammas: Iterable[Real], betas: Iterable[Real], shots: int,
                 run_counts: Callable[[QProg, int], Mapping[str, int]]) -> LightConeEstimate:
        """仅通过 ``run_counts(program, shots)`` 采样估计目标均值。

        ``shots`` 是每个程序的正整数采样次数，而非总预算。
        回调必须返回非负整数计数，其总和须恰好为 ``shots``；
        键为与电路宽度一致的比特串 ``c[w-1]...c[0]``。不接受概率、
        整数键、空白、前缀、隐式补位或准计数。
        可以省略计数为零的结果。

        执行函数必须为每个程序重新制备并采样。错误直接向上传播，
        不切换备用后端，也不重试。若此前已有调用，失败时可能已
        消耗采样次数；不返回部分估计。即使目标仅含常数项，
        也必须提供可调用对象，但不会调用它。
        """
        shots = _integer(shots, "shots", minimum=1)
        if not callable(run_counts):
            raise TypeError("run_counts must be a callable; no default backend is provided")
        # 提交任何程序之前，先验证并构建整个批次。
        jobs = self.build_programs(gammas, betas)
        means = tuple(_parity_mean(run_counts(job.program, shots),
                                   len(job.plan.global_qubits), job.parity_cbits, shots)
                      for job in jobs)
        mean = _objective_sum([self.constant] + [plan.target.coefficient * value
                                      for plan, value in zip(self.plans, means)])
        return LightConeEstimate(mean, means, shots, shots * len(jobs))
