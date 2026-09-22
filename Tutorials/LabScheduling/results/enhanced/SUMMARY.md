# 冻结语料逐实例汇总

完整训练记录见 runs.jsonl，每行一次训练；包含全部不可行实例。

| 实例 | 状态 | 平均/最差 p_opt | 均匀 p_opt | 胜出种子/10 | CNOT 原始→精简 |
|---|---|---|---|---|---|
| evaluation-n3-d0.json | optimal | 0.265351 / 0.261244 | 0.125000 | 10 | 19 → 13 |
| evaluation-n3-d1.json | optimal | 0.546045 / 0.531842 | 0.125000 | 10 | 21 → 15 |
| evaluation-n3-d2.json | optimal | 0.246273 / 0.054195 | 0.125000 | 9 | 23 → 17 |
| evaluation-n5-d0.json | optimal | 0.140080 / 0.071053 | 0.031250 | 10 | 33 → 23 |
| evaluation-n5-d1.json | optimal | 0.183046 / 0.044921 | 0.031250 | 10 | 39 → 29 |
| evaluation-n5-d2.json | infeasible | 0.000000 / 0.000000 | 0.000000 | 0 | 57 → 47 |
| evaluation-n6-d0.json | optimal | 0.033681 / 0.018933 | 0.015625 | 10 | 34 → 22 |
| evaluation-n6-d1.json | optimal | 0.184659 / 0.181925 | 0.031250 | 10 | 44 → 32 |
| evaluation-n6-d2.json | infeasible | 0.000000 / 0.000000 | 0.000000 | 0 | 60 → 48 |
| evaluation-n8-d0.json | optimal | 0.008020 / 0.000766 | 0.003906 | 7 | 42 → 26 |
| evaluation-n8-d1.json | optimal | 0.025118 / 0.006623 | 0.003906 | 10 | 56 → 40 |
| evaluation-n8-d2.json | infeasible | 0.000000 / 0.000000 | 0.000000 | 0 | 78 → 62 |
