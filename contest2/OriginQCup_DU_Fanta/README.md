# 量子风险分析：基于振幅估计的 VaR 与预期损失

**参赛队伍**：DU_Fanta ｜ **任务类型**：创新应用 ｜ **关联 Issue**：#13

---

## 一、这个应用做什么

在险价值（Value at Risk, VaR）和预期损失（Expected Shortfall, ES）是巴塞尔协议下最核心的两个风险指标。银行通常用蒙特卡洛模拟计算它们：要把误差降到 `eps`，需要 `O(1/eps^2)` 次采样。

本应用把 `pyqpanda_alg` 中已有的两个组件组合成一条完整的风险计量流水线：

| 组件 | 作用 |
| --- | --- |
| `QCmp.int_comparator` | 构造可逆判定电路 `L >= t` |
| `QAE.IQAE` | 估计该判定成立的概率，即尾部概率 `P(L >= t)` |

迭代振幅估计只需 `O(1/eps)` 次 oracle 查询，相对经典蒙特卡洛在**该子过程**上具有平方级优势。

### 关键设计：整条流水线只依赖尾部概率

本应用的核心观察是：**VaR 与 ES 都可以只用尾部概率表示**，因此一个量子原语就足够，不需要为 ES 单独设计带线性幅度旋转的电路。

对整数取值的损失 `L`：

```
E[(L - t)^+] = sum_{k > t} P(L >= k)

ES_alpha = VaR_alpha + E[(L - VaR_alpha)^+] / P(L >= VaR_alpha)
```

于是：

- **VaR** ：对 `t` 做二分搜索，只需 `O(log N)` 次振幅估计（而非线性扫描的 `O(N)` 次）；
- **ES** ：VaR 之上若干个尾部概率求和即可。

两个恒等式均在 `Test_quantum_risk.py` 中对照精确值验证。

---

## 二、文件说明

| 文件 | 内容 |
| --- | --- |
| `quantum_risk.py` | 核心模块：`LossDistribution`、`QuantumRiskAnalyzer` |
| `example_credit_risk.py` | 信贷组合的完整演示 |
| `Test_quantum_risk.py` | 26 个单元测试，全部对照经典精确值 |

---

## 三、快速开始

```bash
pip install pyqpanda3 pyqpanda_alg numpy
cd contest2/OriginQCup_DU_Fanta
python example_credit_risk.py          # 运行演示
python -m pytest Test_quantum_risk.py  # 运行测试
```

最小示例：

```python
from quantum_risk import LossDistribution, QuantumRiskAnalyzer, recommended_epsilon

# 5 个债务人，各自的违约概率与整数敞口
dist = LossDistribution.from_credit_portfolio(
    default_probabilities=[0.08, 0.15, 0.04, 0.22, 0.10],
    exposures=[2, 1, 3, 1, 2],
    unit=1e5,
)

confidence = 0.99
analyzer = QuantumRiskAnalyzer(dist, epsilon=recommended_epsilon(confidence))
summary = analyzer.report(confidence)

print(summary['var_monetary'])                 # 500000.0
print(summary['expected_shortfall_monetary'])  # 约 530000
```

---

## 四、实测结果

演示所用组合：5 个债务人，总敞口 90 万元，期望损失 8.5 万元；损失分布占 4 个量子比特（16 个取值），加上比较器共 **8 个量子比特**。

### 尾部概率（量子估计 vs 精确值，eps = 0.01）

| 阈值 | 精确值 | 量子估计 | 绝对误差 |
| ---: | ---: | ---: | ---: |
| 1 | 0.47299 | 0.48212 | 0.00913 |
| 2 | 0.23135 | 0.23830 | 0.00695 |
| 3 | 0.10074 | 0.10271 | 0.00198 |
| 4 | 0.03092 | 0.02816 | 0.00276 |
| 5 | 0.01056 | 0.01015 | 0.00041 |
| 6 | 0.00278 | 0.00247 | 0.00032 |

### 风险指标

| 置信度 | eps | VaR（量子） | VaR（精确） | ES（量子） | ES（精确） | ES 误差 | 振幅估计次数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.90 | 0.0100 | 3 | 3 | 3.3785 | 3.4459 | 0.0673 | 18 |
| 0.95 | 0.0100 | 3 | 3 | 3.4660 | 3.4459 | 0.0201 | 18 |
| 0.99 | 0.0020 | 5 | 5 | 5.2986 | 5.3256 | 0.0271 | 16 |

三个置信度下 VaR 均与精确值完全一致。

### 查询次数对比

| 精度 eps | 经典蒙特卡洛采样数 |
| ---: | ---: |
| 0.05 | 738 |
| 0.01 | 18,445 |
| 0.005 | 73,778 |
| 0.001 | 1,844,440 |

经典方法为 `O(1/eps^2)`，迭代振幅估计为 `O(1/eps)`。

---

## 五、一个必须说明的精度条件

VaR 的二分搜索需要判断 `P(L >= t+1) <= 1 - alpha`。**如果估计精度 `eps` 与 `1 - alpha` 相当，这个比较就会被估计噪声主导**，返回的 VaR 可能相差一档。

开发过程中在 `alpha = 0.99`、`eps = 0.01` 下确实观察到了该现象：真实 `P(L >= 5) = 0.01056`，与判定阈值 `0.01` 的差距小于 `eps`，导致 VaR 被低估为 4。

因此模块提供 `recommended_epsilon(alpha) = 0.2 * (1 - alpha)`，并在 `eps` 过粗时发出 `RuntimeWarning`。上表中 `alpha = 0.99` 使用 `eps = 0.002` 后结果正确。

---

## 六、本应用不主张的内容

- **不主张端到端的量子加速。** 平方级优势存在于尾部概率这一子过程的查询复杂度上。整条流水线还包含经典的分布构造与二分搜索；在模拟器上，经典计算本身更快。
- **不主张已可用于生产。** 演示规模为 4 个损失比特（16 个取值）。真实组合需要更多比特、相关性违约模型，以及在含噪声硬件上的误差缓解。
- **未在真实量子硬件上运行。** `QAE.IQAE` 目前仅支持 CPU 模拟器后端。

---

## English summary

**Quantum risk analysis: Value at Risk and Expected Shortfall via amplitude estimation.**

This application composes two existing `pyqpanda_alg` components — `QCmp.int_comparator` (the reversible predicate `L >= t`) and `QAE.IQAE` (amplitude estimation) — into a complete risk-measurement workflow.

The key design point is that **both VaR and Expected Shortfall reduce to tail probabilities**, so one quantum primitive suffices. For integer-valued losses:

```
E[(L - t)^+] = sum_{k > t} P(L >= k)
ES_alpha     = VaR_alpha + E[(L - VaR_alpha)^+] / P(L >= VaR_alpha)
```

VaR therefore comes from a bisection over tail probabilities in `O(log N)` amplitude estimations instead of an `O(N)` scan, and ES from a short sum of them. No separate estimation primitive with linear-amplitude rotations is needed.

**Results** on a 5-obligor credit portfolio (8 qubits: 4 loss + 4 comparator): VaR matches the exact value at all three confidence levels, Expected Shortfall to within 0.07 loss levels, using 16–18 amplitude estimations.

**Accuracy condition.** The VaR bisection compares tail probabilities against `1 - alpha`, so `epsilon` must be small relative to that gap. This was observed concretely at `alpha = 0.99, epsilon = 0.01`, where the true `P(L >= 5) = 0.01056` sits within `epsilon` of the `0.01` decision threshold and VaR was under-estimated. The module exposes `recommended_epsilon(alpha) = 0.2 * (1 - alpha)` and warns when `epsilon` is too coarse.

**Not claimed:** end-to-end quantum speedup (the quadratic advantage is in the query complexity of the tail-probability subroutine only), production readiness, or any hardware run — `QAE.IQAE` currently supports the CPU simulator backend only.

**Tests:** 26 unit tests, every quantum estimate checked against the exact classical value.

## References

1. S. Woerner, D. J. Egger. *Quantum risk analysis.* npj Quantum Information **5**, 15 (2019).
2. D. Grinko, J. Gacon, C. Zoufal, S. Woerner. *Iterative quantum amplitude estimation.* npj Quantum Information **7**, 52 (2021).
