# 共享实验室最小扰动重排

完整中文教程、输入规范、QUBO 推导、结果和演示流程：
[LabReschedule](../../../Tutorials/LabScheduling/README.md)。

在**仓库根目录**安装本地包和应用依赖后运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pyqpanda_alg.LabScheduling pyqpanda-algorithm/example/LabScheduling/data/medium.json --output reports/medium.json
```

默认 XY / seed 7：中等样例目标值 4、违反数 0、经典最优差距 0。
数据为合成数据；不需要 API Key、GPU 或真实量子硬件。

增强演示（CSV 预约→连锁改约→4 比特 CPU 求解→双经典认证→变更表/甘特图）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python pyqpanda-algorithm/example/LabScheduling/booking_demo.py
```

[增强教程与全部实验](../../../Tutorials/LabScheduling/ENHANCEMENTS.md)解释安全传播、
组件回填、XY 相位精简、MILP、12 实例 × 10 种子评测及已知失败。
## 最新审阅与异构案例

安装后可运行 `tradeoff_demo.py`，查看三个实验整体后移与使用备用设备的完整取舍。
`benchmark_v2.py` 执行冻结的 36 个异构问题及两条求解路径；
`summarize_v2.py reports/lab-v2` 汇总所有结果，旧版评价入口继续保留。
详见[审阅指南](../../../Tutorials/LabScheduling/REVIEW_GUIDE.md)。

`matched_budget.py` 与 `summarize_matched.py` 提供同总预算及化简后随机基线对照。
运行、结论边界和全量重放见[补充评价](../../../Tutorials/LabScheduling/MATCHED_BUDGET.md)。
