# QECClosedForm —— 量子纠错码闭式参数预测

`QECClosedForm` 对 **AG 完备码族**（Reed-Muller CSS 码）`[[2^m, k, 2^{r+1}]]`
由**组合闭式**直接给出全套纠错参数——**不需要电路、不需要模拟**，一台普通
电脑秒级完成。

## 定位：与 QECNoise 闭环

| 模块 | 层 | 做什么 |
|---|---|---|
| `QECNoise`（[PR #48](https://github.com/OriginQ/pyqpanda-algorithm/pull/48)） | **验证层** | QPanda3 模拟：复现相干噪声 θ⁴ 损失标度（log-log 斜率 ≈ 4） |
| `QECClosedForm`（本模块） | **预测层** | 闭式秒算：loss(θ)=c_d·θ^d 的指数与系数 |

两者闭环：`QECClosedForm` **预测** loss(θ) = c_d·θ^d（指数 d、系数 c_d 全闭式），
`QECNoise` **模拟确认** log-log 斜率 ≈ d —— 预测与验证一致，一台电脑完成
纠错码设计全流程。

## 能力

```python
from pyqpanda_alg.QECClosedForm import QECClosedForm

cf = QECClosedForm(10, 3)          # [[1024, 672, 16]]
cf.code()                          # (1024, 672, 16)
cf.encoding_rate()                 # 0.65625
cf.zero_loss_boundary()            # 7（注入 ≤7 比特旋转零损失）
cf.loss(0.01)                      # 1.05e-24（逻辑损失闭式）
cf.logical_operator_count()        # 逻辑算符计数
QECClosedForm.detection_rate(0.1)  # 0.002498 = sin²(0.05)
```

## 演示

```sh
# 1. 闭式参数表（秒算）
python3 example/QECClosedForm/demo_closedform.py

# 2. 预测 vs 模拟对比（闭式指数 × QECNoise 模拟斜率，~25s）
python3 example/QECClosedForm/demo_predict_vs_simulate.py

# 3. 可视化（两张图 → figs/）
python3 example/QECClosedForm/plot_closedform_vs_sim.py
```

- demo_closedform：7 个精选码完整参数表，`[[16,6,4]]` 损失 3e-08 →
  `[[1024,252,32]]` 损失 **7.5e-57**（跨 49 个数量级）
- demo_predict_vs_simulate：QECNoise 模拟 slope ≈ 4 与闭式指数 4 一致
- plot：fig1 预测曲线 vs 模拟散点；fig2 损失 vs 码距指数下降

## 测试

```sh
python3 -m pytest test/test_qecclosedform.py -q
```

验证闭式与已发布的精确值一致（10.30/10.35）：编码率、fail(w0)、loss 系数、
逻辑算符计数（RM(1,5)→1240、RM(1,6)→10416）、检测率。

## 理论来源

- 码参数 `[[2^m, n-2·dim RM(r,m), 2^{r+1}]]` —— 10.30
- 损失闭式 `loss(θ) = c_d·θ^d`，`c_d = C(n,w0)·P(w0)·fail(w0)·κ·2^{-2w0}` —— 定理 10.35.1.07
- 零损失边界 `k ≤ ⌊(d-1)/2⌋` —— 定理 10.31.1.01
- 逻辑算符计数 `2^{m-r-1}·[m choose r+1]_2` —— 定理 10.30.2.04
- 检测率 `p_det(θ) = sin²(θ/2)` —— 10.29 预言 2a
