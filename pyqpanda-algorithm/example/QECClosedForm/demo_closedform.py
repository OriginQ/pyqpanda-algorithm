#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_closedform.py —— QECClosedForm 演示：AG 完备码族闭式参数表（零电路零模拟）

运行:
  python3 example/QECClosedForm/demo_closedform.py
"""
import os
import sys

# 保证可独立运行（无论从仓库根还是 example 目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyqpanda_alg.QECClosedForm import QECClosedForm


def main():
    print("QECClosedForm —— AG 完备码族闭式纠错参数（零电路零模拟）")
    print("=" * 78)

    print("\n[一] 精选码族的完整参数（秒算）")
    print(f"{'code':<18} {'rate':>7} {'w0':>3} {'fail':>7} {'κ':>7} "
          f"{'c_d':>9} {'zero-loss≤':>10} {'logicals':>9}")
    print("-" * 78)
    for m, r in [(4, 1), (6, 1), (6, 2), (8, 3), (10, 1), (10, 3), (10, 4)]:
        cf = QECClosedForm(m, r)
        n, k, d = cf.code()
        print(f"$[[{n},{k},{d}]]$".ljust(18)
              + f"{cf.encoding_rate():>7.3f} {cf.w0:>3} {cf.fail:>7.4f} "
              + f"{cf.kap:>7.4f} {cf.c_d:>9.3g} {cf.zero_loss_boundary():>10} "
              + f"{cf.logical_operator_count():>9}")

    print("\n[二] 损失闭式 loss(θ) = c_d·θ^d（θ = 0.01）")
    print(f"{'code':<18} {'d':>3} {'loss(0.01)':>16}")
    print("-" * 42)
    for m, r in [(4, 1), (6, 2), (8, 3), (10, 3), (10, 4)]:
        cf = QECClosedForm(m, r)
        n, k, d = cf.code()
        print(f"$[[{n},{k},{d}]]$".ljust(18) + f"{d:>3} {cf.loss(0.01):>16.4g}")

    print("\n[三] 检测率闭式 p_det(θ) = sin²(θ/2)（与码无关）")
    print(f"{'θ':>6} {'p_det':>10}")
    print("-" * 20)
    for th in (0.01, 0.05, 0.1, 0.2, 0.4):
        print(f"{th:>6.2f} {QECClosedForm.detection_rate(th):>10.6f}")

    print("\n结论：[[1024,252,32]] 在 θ=0.01 时损失 7.5e-57 —— "
          "闭式秒算，无需电路或模拟。与 QECNoise 的模拟验证闭环。")
    print("\n示例: cf = QECClosedForm(10, 3); cf.loss(0.01) -> "
          f"{QECClosedForm(10, 3).loss(0.01):.3e}")


if __name__ == "__main__":
    main()
