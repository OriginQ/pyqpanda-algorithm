QShadow：可观测量感知的经典阴影
====================================

QShadow 从同一批局域 Pauli 测量中估计多个可观测量。它适合变分算法能量读出、
相关函数扫描和小规模真机实验的测量后处理。经典阴影与局域偏置方法是已有研究；
本模块的 PyQPanda3 集成、带下限规划器、置信预算流程和测试为独立实现。

基本工作流
----------

.. code-block:: python

   from pyqpanda_alg.QShadow import (
       PauliObservable,
       optimize_basis_probabilities,
       sample_measurement_plan,
       simulate_pauli_measurements,
       estimate_observables,
   )

   targets = [
       PauliObservable.from_terms(
           {"ZZ": -1.0, "XI": 0.2, "IX": 0.2}, name="energy"
       ),
       PauliObservable.from_terms({"ZI": 1.0}, name="magnetisation"),
   ]
   result = optimize_basis_probabilities(targets, probability_floor=0.03)
   plan = sample_measurement_plan(result.probabilities, 5000, seed=7)
   data = simulate_pauli_measurements([1, 0, 0, 0], plan, seed=8)
   estimates = estimate_observables(data, targets, family_confidence=0.95)

约定与无偏性
------------

Pauli 字符串和数据集统一使用 q0-first：``"XI"`` 表示 X 作用于 q0。对目标 Pauli
项，只有测量基覆盖其非恒等支撑时，该 shot 才产生非零贡献；贡献除以对应测量基
概率的乘积。数据集保存每个 shot 的完整 X/Y/Z 概率，因此不同规划轮次可以无偏
合并。

规划器优化的是系数加权的对角二阶矩代理，并通过 ``probability_floor`` 防止某个
基完全不可观测及逆倾向权重发散。代理损失下降不是任意未知态上实际方差下降的
保证。

离线重复基准
--------------

``example/QShadow/qshadow_planning_benchmark.py`` 使用固定状态、可观测量和种子，
对均匀规划与 QShadow 规划进行 1000 轮、每轮 2000 shots 的对照。规划损失从
``24.54`` 降至 ``5.964549``，聚合 RMSE 降低 ``60.20%``，两个可观测量的平均
区间半宽分别降低 ``62.35%`` 和 ``53.42%``。在另一个 100 轮顺序预算实验中，
达到 ``0.22`` 目标半宽所需平均 shots 从 ``7284.9`` 降至 ``1892.8``，减少
``74.02%``，且没有硬上限违规。

两种方案在该配置中的单项和 family-wise 经验覆盖率均为 ``1.000``，说明区间较
保守，不表示精确 95% 校准。结果只适用于该固定基准，不构成所有状态与可观测量上
普遍优势的声明。

置信预算
--------

``ShadowBudgetController`` 用 Bonferroni 同时经验 Bernstein 半宽决定下一批 shots，
同时遵守 ``max_shots``。它只返回建议，不自动提交任务。远程执行还可通过
``run_measurement_plan(..., max_jobs=N)`` 在任何任务提交前限制唯一测量线路数量。

真机结果必须同时报告设备/后端、时间、shots、任务 ID、原始 counts、采样概率、
统计区间及理想模拟器偏差。当前实现不含读出误差缓解，因此经验 Bernstein 区间
不覆盖硬件系统偏差。

WK_C180 实测证据
----------------

2026-07-26 的最小验证在 ``WK_C180`` 上执行 X/Y/Z 各 512 shots，总计恰好
3 个任务、1536 shots，且没有自动重试。真实 counts 经 q0-first 转换后进入同一个
QShadow 数据集，按逐 shot 的 ``(1/3, 1/3, 1/3)`` 概率同时估计 X/Y/Z，并生成
family-wise 经验 Bernstein 区间。机器可读证据位于
``example/QShadow/qshadow_wukong_validation_20260726.json``，配套解释位于同目录
``qshadow_wukong_validation_20260726.md``。

这次运行验证了从真实后端 counts 到同时估计、置信区间和 3-job/1536-shot 预算的
完整工作流。硬件 X/Y/Z 区间均未覆盖理想值，Bloch 向量范数为
``1.539826 > 1``；本次平台配置包含 ``is_amend=True``，QShadow 未执行读出误差
缓解，因此不把结果解释为硬件精度成功。由于 X/Y/Z 目标完全对称并采用均匀计划，
本实验也不比较偏置规划器与均匀规划的真机优势；规划器行为由 CPUQVM 示例和测试
覆盖。

引用
----

* Huang, Kueng, Preskill, *Predicting many properties of a quantum system from
  very few measurements*, Nature Physics 16 (2020), arXiv:2002.08953.
* Hadfield, Bravyi, Raymond, Mezzacapo, *Measurements of Quantum Hamiltonians
  with Locally-Biased Classical Shadows*, arXiv:2006.15788.
* Hadfield, *Adaptive Pauli Shadows for Energy Estimation*, arXiv:2105.12207.

完整 API、云后端安全示例、限制和原创性边界见
``pyqpanda_alg/QShadow/README.md``。
