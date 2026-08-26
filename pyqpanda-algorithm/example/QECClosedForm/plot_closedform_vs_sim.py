#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_closedform_vs_sim.py —— 预测 vs 模拟可视化（QECClosedForm × QECNoise）

生成两张图：
  fig1: 损失 vs θ（对数坐标）—— 闭式曲线 loss(θ)=c_d·θ^d vs QECNoise 模拟散点
  fig2: 损失 vs 码距（AG 完备族）—— 闭式秒算的跨码距趋势

运行:
  python3 example/QECClosedForm/plot_closedform_vs_sim.py [--out_dir figs]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyqpanda_alg.QECClosedForm import QECClosedForm
from pyqpanda_alg.QECNoise import run_theta4_scan


def main():
    out_dir = "figs"
    if len(sys.argv) > 1 and sys.argv[1].startswith("--out_dir"):
        out_dir = sys.argv[1].split("=")[1]
    os.makedirs(out_dir, exist_ok=True)

    # ---- fig1: [[5,1,3]] / [[7,1,3]] 预测闭式 vs 模拟 ----
    thetas = np.array([0.05, 0.1, 0.2, 0.4])
    fig, ax = plt.subplots(figsize=(7, 5))

    # 模拟数据（trials=10，10.29 标准）
    for code, c_d, d in [("[[5,1,3]]", 0.06, 4), ("[[7,1,3]]", 0.10, 4)]:
        losses, slope = run_theta4_scan(code, trials=10, seed=42)
        ax.loglog(thetas, losses, "o", label=f"{code} 模拟 (slope={slope:.2f})")
        # 闭式预测（10.29 系数量级：[[5,1,3]]≈0.06, [[7,1,3]]≈0.10）
        pred = c_d * thetas ** d
        ax.loglog(thetas, pred, "--", label=f"{code} 闭式 {c_d}·θ^{d}")

    ax.set_xlabel(r"$\theta_{\max}$")
    ax.set_ylabel("logical loss $L(\\theta)$")
    ax.set_title("预测（闭式）vs 模拟（QPanda3/态矢量）")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_predict_vs_sim.png"), dpi=150)
    print(f"已保存 fig1 → {out_dir}/fig1_predict_vs_sim.png")

    # ---- fig2: AG 完备族 loss vs 码距（闭式秒算） ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    codes = []
    for m, r in [(4, 1), (6, 1), (6, 2), (8, 2), (8, 3), (10, 2), (10, 3), (10, 4)]:
        cf = QECClosedForm(m, r)
        n, k, d = cf.code()
        codes.append((d, cf.loss(0.01), f"[[{n},{k},{d}]]"))
    ds = [c[0] for c in codes]
    losses = [c[1] for c in codes]
    labels = [c[2] for c in codes]
    ax2.semilogy(ds, losses, "o-", color="tab:red")
    for x, y, lab in zip(ds, losses, labels):
        ax2.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6),
                     fontsize=7, rotation=15)
    ax2.set_xlabel("code distance $d$")
    ax2.set_ylabel("loss(θ=0.01)")
    ax2.set_title("AG 完备码族：损失 vs 码距（闭式秒算）")
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "fig2_loss_vs_distance.png"), dpi=150)
    print(f"已保存 fig2 → {out_dir}/fig2_loss_vs_distance.png")


if __name__ == "__main__":
    main()
