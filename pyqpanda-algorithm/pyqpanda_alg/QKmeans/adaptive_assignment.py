# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""针对现有双特征 SWAP 编码，在预算约束下仅依据测量计数进行分配。

这一可选决策层不改变 ``QuantumKmeans.fit``。它依据辅助量子比特测得 1 的
概率进行选择，而不是依据一般的欧氏距离。
测量执行器由调用方提供；此处不选择后端，也不会回退到精确概率计算。
"""
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log, pi, sqrt
from numbers import Integral, Real
from typing import Callable, Iterable, Optional, Tuple

from pyqpanda3.core import CNOT, H, QProg, T, U3, measure

__all__ = ["AssignmentResult", "adaptive_assign", "build_assignment_program"]


@dataclass(frozen=True)
class AssignmentResult:
    """不可变快照，其中元组按候选的原始顺序索引。

    仅当状态为 ``separated`` 或 ``single_candidate`` 时，``winner`` 才是索引。
    未测量候选的均值为 ``None``，区间为 ``(0, 1)``。
    已淘汰候选保留其最后的累计统计量；``active`` 包含尚未淘汰的候选索引，
    并不表示排名或估计的优胜者。
    """
    status: str
    winner: Optional[int]
    means: Tuple[Optional[float], ...]
    intervals: Tuple[Tuple[float, float], ...]
    shots_per_centroid: Tuple[int, ...]
    total_shots: int
    active: Tuple[int, ...]


def _point_snapshot(values, name):
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must contain two real coordinates")
    try:
        coordinates = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must contain two real coordinates") from exc
    if len(coordinates) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    for value in coordinates:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} coordinates must be real numbers, not bools")
        if not -1 <= value <= 1 or not isfinite(value):
            raise ValueError(f"{name} coordinates must be finite and in [-1, 1]")
    return tuple(float(value) for value in coordinates)


def _shot_count(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a bool")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _append_fredkin(program, control, left, right):
    # 门序列为 CX(right,left) CCX(control,left,right) CX(right,left)。
    # CCX 采用标准的精确 H/T/CX 分解，而非相对相位分解。
    # 仅生成单量子比特门和单控制比特的非门。
    program << CNOT(right, left)
    program << H(right) << CNOT(left, right) << T(right).dagger()
    program << CNOT(control, right) << T(right) << CNOT(left, right)
    program << T(right).dagger() << CNOT(control, right)
    program << T(left) << T(right) << H(right)
    program << CNOT(control, left) << T(control) << T(left).dagger()
    program << CNOT(control, left)
    program << CNOT(right, left)


def build_assignment_program(point: Iterable[Real], centroid: Iterable[Real]) -> QProg:
    """新建三量子比特 SWAP 测试程序，仅将 q2 的测量结果写入 c0。

    每个输入包含两个位于 ``[-1, 1]`` 内的有限实数坐标 ``(x, y)``。
    与原有实现一样，在 q0/q1 上使用 ``U3((y+1)*pi/2, (x+1)*pi/2, 0)``
    制备状态。受控 SWAP 采用精确的 H/T/CNOT 分解。
    返回程序的所有权归调用方。每次采样都需重新制备 |000>，执行环境需支持
    标准量子门及其厄米共轭操作，测量计数中只能包含 c0。
    """
    point = _point_snapshot(point, "point")
    centroid = _point_snapshot(centroid, "centroid")
    program = QProg(3)
    program << H(2)
    for qubit, (x, y) in enumerate((point, centroid)):
        program << U3(qubit, (y + 1) * pi / 2, (x + 1) * pi / 2, 0)
    _append_fredkin(program, 2, 0, 1)
    program << H(2) << measure(2, 0)
    return program


def _ones_from_counts(counts, shots):
    if not isinstance(counts, Mapping):
        raise TypeError("run_counts must return a mapping of single-bit counts")
    total = 0
    ones = 0
    for bit, count in counts.items():
        if not isinstance(bit, str) or bit not in ("0", "1"):
            raise ValueError("count keys must be '0' or '1' for c0 only")
        count = _shot_count(count, "count", 0)
        total += count
        if bit == "1":
            ones = count
    if total != shots:
        raise ValueError("counts must sum exactly to the requested shots")
    return ones


def adaptive_assign(
    point: Iterable[Real],
    centroids: Iterable[Iterable[Real]],
    run_counts: Callable[[QProg, int], Mapping],
    *,
    batch_shots: int = 64,
    max_shots: int = 4096,
    alpha: float = 0.05,
) -> AssignmentResult:
    """使用固定批量的累计测量计数，淘汰已由区间分离判定的候选。

    ``max_shots`` 是请求采样总数的全局硬上限，而不是每个候选各自的上限。
    尚未淘汰的候选按原始索引顺序轮询。绝不请求不足一批的采样。
    执行器失败或计数无效时立即抛出异常，不重试，也不回退到其他后端。

    ``run_counts(program, shots)`` 必须返回映射，键只能为 ``'0'`` 和 ``'1'``，
    值必须是非负整数（不能是布尔值），且总和严格等于 ``shots``。
    缺失结果按零次计数。每次回调都接收一个新建的、归调用方所有的程序；
    在任何回调之前，都会对输入坐标创建快照。统计量仅属于本次调用。

    对于 K 个候选，令 L = floor(max_shots / batch_shots)，在每个候选的固定
    样本量 n = j*batch_shots 处，使用半径 sqrt(log(2*K*max(1,L)/alpha)/(2*n))。
    联合界覆盖所有可能的 K*L 次检查，因此，只要对每个固定候选的采样都是
    独立且分布平稳的伯努利采样，自适应淘汰就有效。对于一次分配，返回一个
    错误的、状态为 separated 的优胜者的概率至多为 alpha。这不是以已经返回
    优胜者为条件的概率界，也不是硬件保真度保证，更不代表欧氏聚类质量。
    alpha 只涵盖一次分配；多个数据点或多轮迭代需要另行分配错误概率预算。
    参见 docs/qkmeans_adaptive_assignment.md。
    """
    point = _point_snapshot(point, "point")
    if isinstance(centroids, (str, bytes, Mapping)):
        raise TypeError("centroids must be a nonempty iterable of points")
    try:
        centers = tuple(_point_snapshot(c, "centroid") for c in centroids)
    except TypeError as exc:
        raise TypeError("centroids must be a nonempty iterable of real points") from exc
    if not centers:
        raise ValueError("centroids must be nonempty")
    batch_shots = _shot_count(batch_shots, "batch_shots", 1)
    max_shots = _shot_count(max_shots, "max_shots", 0)
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real number, not a bool")
    if not 0 < alpha < 1 or not isfinite(alpha):
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be representable as a float strictly between 0 and 1")
    if not callable(run_counts):
        raise TypeError("run_counts must be callable")

    k = len(centers)
    ones = [0] * k
    shots = [0] * k
    means = [None] * k
    intervals = [(0.0, 1.0)] * k
    active = tuple(range(k))
    total = 0

    def result(status, winner=None):
        return AssignmentResult(status, winner, tuple(means), tuple(intervals),
                                tuple(shots), total, active)

    if k == 1:
        return result("single_candidate", 0)
    looks = max_shots // batch_shots
    # 在对数域中计算，避免 alpha 极小时比值溢出。
    log_factor = log(2) + log(k) + log(max(1, looks)) - log(alpha)
    while total + batch_shots <= max_shots:
        for index in active:
            # 本轮遍历的元组已固定；较早的检查可能淘汰
            # 本轮尚未轮到的候选。
            if index not in active:
                continue
            if total + batch_shots > max_shots:
                return result("undecided")
            program = build_assignment_program(point, centers[index])
            observed = _ones_from_counts(run_counts(program, batch_shots), batch_shots)
            ones[index] += observed
            shots[index] += batch_shots
            total += batch_shots
            mean = ones[index] / shots[index]
            radius = sqrt(log_factor / (2 * shots[index]))
            means[index] = mean
            intervals[index] = (max(0.0, mean - radius), min(1.0, mean + radius))
            best_upper = min(intervals[i][1] for i in active)
            active = tuple(i for i in active if intervals[i][0] <= best_upper)
            if len(active) == 1:
                return result("separated", active[0])
    return result("undecided")
