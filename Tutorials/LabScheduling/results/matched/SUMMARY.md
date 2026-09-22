# 同总预算对照结果

全部 36 个实例、每实例 10 种子；Q=直接量子，R=分量量子，U=直接均匀随机，S=分量均匀随机。

下表为实际 512 总 shots 下采到精确最优的种子数；不可行行无最优值，记 0。

| 实例 | 类别 | 最优值 | Q | R | U | S |
|---|---|---:|---:|---:|---:|---:|
| n3-s0-r0.json | quantum_required | 10 | 10 | 10 | 10 | 10 |
| n3-s0-r1.json | quantum_required | 19 | 10 | 10 | 10 | 10 |
| n3-s0-r2.json | quantum_required | 10 | 10 | 10 | 10 | 10 |
| n3-s0-r3.json | quantum_required | 14 | 10 | 10 | 10 | 10 |
| n3-s1-r0.json | classical_only | 25 | 10 | 10 | 10 | 10 |
| n3-s1-r1.json | quantum_required | 10 | 10 | 10 | 10 | 10 |
| n3-s1-r2.json | infeasible | — | 0 | 0 | 0 | 0 |
| n3-s1-r3.json | infeasible | — | 0 | 0 | 0 | 0 |
| n3-s2-r0.json | infeasible | — | 0 | 0 | 0 | 0 |
| n3-s2-r1.json | quantum_required | 7 | 10 | 10 | 10 | 10 |
| n3-s2-r2.json | classical_only | 27 | 10 | 10 | 10 | 10 |
| n3-s2-r3.json | infeasible | — | 0 | 0 | 0 | 0 |
| n4-s0-r0.json | quantum_required | 22 | 10 | 10 | 10 | 10 |
| n4-s0-r1.json | quantum_required | 18 | 10 | 10 | 10 | 10 |
| n4-s0-r2.json | quantum_required | 4 | 9 | 10 | 10 | 10 |
| n4-s0-r3.json | quantum_required | 10 | 9 | 10 | 10 | 10 |
| n4-s1-r0.json | infeasible | — | 0 | 0 | 0 | 0 |
| n4-s1-r1.json | infeasible | — | 0 | 0 | 0 | 0 |
| n4-s1-r2.json | quantum_required | 18 | 10 | 10 | 10 | 10 |
| n4-s1-r3.json | quantum_required | 15 | 10 | 10 | 10 | 10 |
| n4-s2-r0.json | quantum_required | 7 | 10 | 10 | 10 | 10 |
| n4-s2-r1.json | quantum_required | 3 | 10 | 10 | 10 | 10 |
| n4-s2-r2.json | quantum_required | 32 | 10 | 10 | 10 | 10 |
| n4-s2-r3.json | infeasible | — | 0 | 0 | 0 | 0 |
| n5-s0-r0.json | quantum_required | 17 | 9 | 10 | 10 | 10 |
| n5-s0-r1.json | quantum_required | 11 | 10 | 10 | 10 | 10 |
| n5-s0-r2.json | quantum_required | 13 | 10 | 10 | 10 | 10 |
| n5-s0-r3.json | quantum_required | 3 | 10 | 10 | 10 | 10 |
| n5-s1-r0.json | infeasible | — | 0 | 0 | 0 | 0 |
| n5-s1-r1.json | infeasible | — | 0 | 0 | 0 | 0 |
| n5-s1-r2.json | quantum_required | 20 | 10 | 10 | 10 | 10 |
| n5-s1-r3.json | infeasible | — | 0 | 0 | 0 | 0 |
| n5-s2-r0.json | quantum_required | 4 | 10 | 10 | 10 | 10 |
| n5-s2-r1.json | infeasible | — | 0 | 0 | 0 | 0 |
| n5-s2-r2.json | infeasible | — | 0 | 0 | 0 | 0 |
| n5-s2-r3.json | quantum_required | 44 | 10 | 10 | 10 | 10 |

## 同总 shots 的条件命中概率

仅 22 个仍需量子的可行实例，先平均每实例的十种子，再等权平均实例。
这是训练后理论曲线，低 shots 未重新训练或实际抽样，不包括训练成本。

| 总 shots | Q | R | U | S | R>S 实例 | R<S 实例 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.565738 | 0.713846 | 0.345499 | 0.484328 | 22 | 0 |
| 16 | 0.728238 | 0.842681 | 0.524584 | 0.694213 | 21 | 1 |
| 32 | 0.843777 | 0.921471 | 0.709892 | 0.857838 | 20 | 2 |
| 64 | 0.909796 | 0.965687 | 0.857092 | 0.947985 | 17 | 5 |
| 128 | 0.946055 | 0.989306 | 0.950215 | 0.986030 | 13 | 6 |
| 512 | 0.986578 | 0.999852 | 0.999787 | 0.999963 | 2 | 7 |

R/S 包含精确化简与组合收益；R 与 S 才是相同结构上的量子/随机对照。
不能将 R 相对 U 的全部改善归于量子，也不能用相等预算上限声称相等实际计算成本。
实际评价数、采样数、串行运行时间见 summary.csv；完整失败与 counts 见 runs.jsonl。
