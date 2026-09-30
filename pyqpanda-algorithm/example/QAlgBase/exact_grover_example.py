# -*- coding: utf-8 -*-
# 精确 Grover 搜索（100% 成功率）示例：
# 在 4 个量子比特（N = 16 个元素）的搜索空间中，全方位对比
# 标准 Grover 与 LongExactGrover（任意相位匹配）：
#   (1) 迭代次数（oracle 调用次数）对比
#   (2) 成功概率对比（理论值 + 模拟值）
#   (3) 全空间 16 个元素的完整概率分布对比
#
# 理论预期：
#   M=1 : 标准 Grover  J=3 次迭代, 成功率 ~0.9613
#         LongExactGrover  K=3 次迭代, phi~2.1951, 成功率 = 1.0
#   M=2 : 标准 Grover  J=2 次迭代, 成功率 ~0.9453
#         LongExactGrover  K=2 次迭代, phi~2.1269, 成功率 = 1.0

from pyqpanda3.core import CPUQVM, QProg
from pyqpanda_alg import Grover
import numpy as np

m = CPUQVM()
q_state = list(range(4))          # 4 个量子比特，N = 16
N = 2 ** len(q_state)


# ======================= 工具函数 =======================

def all_basis_strings(n):
    """生成全部 2^n 个计算基比特串（按数值升序）"""
    return [format(i, '0%db' % n) for i in range(2 ** n)]


def run_and_get_probs(circuit, shots=100000):
    """运行线路并返回全空间概率字典"""
    prog = QProg()
    prog << circuit
    m.run(prog, shots)
    return m.result().get_prob_dict(q_state)


def print_full_distribution(prob_dict, marks, label):
    """
    打印全空间每个元素的概率分布（两列并排，16 个元素分 8 行）
    目标态用 ' <== mark' 标出，其余为坏态
    """
    marks = [marks] if isinstance(marks, str) else marks
    keys = all_basis_strings(len(q_state))
    print(f'  {label} full distribution (N={N}):')
    half = N // 2
    for row in range(half):
        left = keys[row]
        right = keys[row + half]
        l_str = f"{left}: {prob_dict.get(left, 0.0):.6f}"
        r_str = f"{right}: {prob_dict.get(right, 0.0):.6f}"
        if left in marks:
            l_str += ' <== mark'
        if right in marks:
            r_str += ' <== mark'
        print(f'    {l_str:<28} {r_str}')
    p_good = sum(prob_dict.get(s, 0.0) for s in marks)
    p_bad = 1.0 - p_good
    print(f'    --> P_good = {p_good:.6f},  P_bad = {p_bad:.6f}\n')


def print_comparison_table(M, J_std, K_exact, p_std_theory, p_std_sim,
                           p_exact_sim, phi):
    """打印单个情形下两个算法的迭代次数 / 成功率对比表"""
    print(f'  Comparison table (N={N}, M={M}):')
    print(f'  +-------------------+---------------+---------------+')
    print(f'  |                   | Standard      | Exact Grover   |')
    print(f'  +-------------------+---------------+---------------+')
    print(f'  | iterations        | {J_std:^13d} | {K_exact:^13d} |')
    print(f'  | oracle calls      | {J_std:^13d} | {K_exact:^13d} |')
    print(f'  | P_good (theory)   | {p_std_theory:^13.6f} | {1.0:^13.6f} |')
    print(f'  | P_good (simulated)| {p_std_sim:^13.6f} | {p_exact_sim:^13.6f} |')
    if phi is not None:
        print(f'  | matched phase phi | {"pi":^13s} | {round(phi, 4):^13.4f} |')
    print(f'  +-------------------+---------------+---------------+\n')


# ================= M = 1：单目标 '1010' =================
M1, mark1 = 1, '1010'
theta1 = 2 * np.arcsin(np.sqrt(M1 / N))
print(f'=================== Case 1: N={N}, M={M1}, mark="{mark1}" '
      f'(theta={theta1:.4f}) ===================\n')

# ---- 1. 标准 Grover ----
J_std1 = Grover.iter_num(q_num=len(q_state), sol_num=M1)            # = 3
p_std1_theory, _ = Grover.iter_analysis(q_num=len(q_state), sol_num=M1,
                                        iternum=J_std1)             # ~0.9613
std1 = Grover.Grover(mark_data=mark1)
probs_std1 = run_and_get_probs(std1.cir(q_input=q_state, iternum=J_std1))
p_std1_sim = probs_std1.get(mark1, 0.0)

print(f'>>> Standard Grover: J = {J_std1} iterations '
      f'(theory P = {p_std1_theory:.6f})')
print_full_distribution(probs_std1, mark1, 'Standard Grover')

# ---- 2. ExactGrover ----
exact1 = Grover.LongExactGrover(mark_data=mark1)
probs_exact1 = run_and_get_probs(exact1.cir(q_input=q_state))
p_exact1_sim = probs_exact1.get(mark1, 0.0)

print(f'>>> LongExactGrover: K = {exact1.last_iternum} iterations, '
      f'phi = {exact1.last_phase:.4f}')
print_full_distribution(probs_exact1, mark1, 'LongExactGrover')

# ---- 3. 对比表 ----
print_comparison_table(M1, J_std1, exact1.last_iternum,
                       p_std1_theory, p_std1_sim, p_exact1_sim,
                       exact1.last_phase)

# ================= M = 2：双目标 '1010' 与 '0110' =================
M2, mark2 = 2, ['1010', '0110']
theta2 = 2 * np.arcsin(np.sqrt(M2 / N))
print(f'=================== Case 2: N={N}, M={M2}, marks={mark2} '
      f'(theta={theta2:.4f}) ===================\n')

# ---- 1. 标准 Grover ----
J_std2 = Grover.iter_num(q_num=len(q_state), sol_num=M2)            # = 2
p_std2_theory, _ = Grover.iter_analysis(q_num=len(q_state), sol_num=M2,
                                        iternum=J_std2)             # ~0.9453
std2 = Grover.Grover(mark_data=mark2)
probs_std2 = run_and_get_probs(std2.cir(q_input=q_state, iternum=J_std2))
p_std2_sim = sum(probs_std2.get(s, 0.0) for s in mark2)

print(f'>>> Standard Grover: J = {J_std2} iterations '
      f'(theory P = {p_std2_theory:.6f})')
print_full_distribution(probs_std2, mark2, 'Standard Grover')

# ---- 2. LongExactGrover ----
exact2 = Grover.LongExactGrover(mark_data=mark2)
probs_exact2 = run_and_get_probs(exact2.cir(q_input=q_state))
p_exact2_sim = sum(probs_exact2.get(s, 0.0) for s in mark2)

print(f'>>> LongExactGrover: K = {exact2.last_iternum} iterations, '
      f'phi = {exact2.last_phase:.4f}')
print_full_distribution(probs_exact2, mark2, 'LongExactGrover')

# ---- 3. 对比表 ----
print_comparison_table(M2, J_std2, exact2.last_iternum,
                       p_std2_theory, p_std2_sim, p_exact2_sim,
                       exact2.last_phase)

# ======================= 总结论 =======================
print('=================== Summary (N=16) ===================')
print(f'  M=1: Standard Grover  {J_std1} iters, P ~ {p_std1_theory:.4f}')
print(f'       Exact Grover      {exact1.last_iternum} iters, P = 1.0')
print(f'  M=2: Standard Grover  {J_std2} iters, P ~ {p_std2_theory:.4f}')
print(f'       Exact Grover      {exact2.last_iternum} iters, P = 1.0')
print('  -> Same iteration depth, success rate improved to exactly 1.0')
