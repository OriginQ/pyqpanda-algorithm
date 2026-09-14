# QECNoise：量子纠错相干噪声 θ⁴ 损失标度演示

## 简介

本模块演示量子纠错码在**相干单比特旋转噪声**下的一个可检验标度律：

- 相干旋转注入后，最优纠错损失满足
  \[
  L(\theta_{\max}) = c\,\theta_{\max}^{4} + O(\theta_{\max}^{6}),
  \]
  log-log 斜率 ≈ 4。
- 作为对照，随机 Pauli（非相干）噪声下损失近似 \( \sim p^{2} \)，斜率 ≈ 2。

该现象对应几何论文章 10.29 的预言：不可恢复成分权重 ≥ 3，恢复干涉导致损失主导阶为 θ⁴。

## 文件

- `pyqpanda_alg/QECNoise/QECNoise.py`：核心模块（精确态矢量模拟 + 查表恢复）
- `pyqpanda_alg/QECNoise/__init__.py`：模块导出
- `example/QECNoise/demo_theta4_scaling.py`：可直接运行的示例

## 运行

```bash
python3 example/QECNoise/demo_theta4_scaling.py
```

输出示例：

```text
Coherent single-qubit rotation noise (expected slope ~4):
  [[5,1,3]]: loss=[...], log-log slope=3.9x
  [[7,1,3]]: loss=[...], log-log slope=4.0x

Random-Pauli incoherent control (expected slope ~2 at low p):
  [[7,1,3]]: loss=[...], log-log slope=~2
```

## 支持码型

- `[[5,1,3]]` 五比特码
- `[[7,1,3]]` Steane 码
- `[[9,1,3]]` Shor 码

## 依赖

- Python ≥ 3.11
- `numpy`
- `pyqpanda3`（用于电路预览；核心模拟为 numpy 态矢量）
