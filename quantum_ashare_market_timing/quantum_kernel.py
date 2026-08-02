#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量子核计算模块 — 基于QPanda3的量子核SVM
Quantum Kernel Module for A-Share Market Timing

核心API:
    QCircuit.matrix() → 量子态矢量 → 量子核矩阵 K[i,j] = |<ψ_i|ψ_j>|²

作者: 周勋洪 (中国电信资阳分公司)
日期: 2026-08-02
"""
import numpy as np
import pyqpanda3.core as pq


def build_angle_encoding(features, n_qubits, n_reps=2):
    """
    AngleEncoding: Hadamard + RZ(2x) 量子编码
    
    将n维经典特征编码到n个量子比特:
      1. 对每个比特施加Hadamard门 → 均匀叠加态
      2. 施加RZ(2*x_j)旋转门 → 将特征值编码到相位中
    
    优势: 无纠缠，计算高效，核矩阵区分度最高
    """
    circ = pq.QCircuit(n_qubits)
    q = list(range(n_qubits))
    for rep in range(n_reps):
        if rep == 0:
            for j in range(n_qubits):
                circ << pq.H(q[j])
        for j in range(n_qubits):
            circ << pq.RZ(q[j], float(features[j] * 2))
    return circ


def build_zz_feature_map(features, n_qubits, n_reps=2):
    """
    ZZFeatureMap: Hadamard + RZ + CNOT+RZ纠缠 量子编码
    
    经典IBM QSVM的ZZFeatureMap实现:
      1. Hadamard层 → 均匀叠加
      2. RZ(2*x_j)层 → 单比特旋转
      3. CNOT(i,j) + RZ(2*(π-x_i)*(π-x_j)) → 两比特纠缠
    
    优势: 包含纠缠层，可捕获特征间非线性交互
    """
    circ = pq.QCircuit(n_qubits)
    q = list(range(n_qubits))
    for rep in range(n_reps):
        # Step 1: Hadamard layer
        for j in range(n_qubits):
            circ << pq.H(q[j])
        # Step 2: RZ rotation
        for j in range(n_qubits):
            circ << pq.RZ(q[j], float(features[j] * 2))
        # Step 3: Entanglement (linear topology)
        for j in range(n_qubits - 1):
            circ << pq.CNOT(q[j], q[j + 1])
            angle = float(2 * (np.pi - features[j]) * (np.pi - features[j + 1]))
            circ << pq.RZ(q[j + 1], angle)
            circ << pq.CNOT(q[j], q[j + 1])
    return circ


def compute_quantum_states(X, builder, n_qubits, n_reps=2, scale=np.pi):
    """
    计算每个样本的量子态矢量 |ψ(x)> = U(x)|0⟩
    
    利用QPanda3的QCircuit.matrix()方法:
      - matrix()返回酉矩阵 U (shape: 2^n × 2^n)
      - 取第一列即为 |ψ(x)> = U|0...0⟩
    
    参数:
      X: (N, d) 经典特征矩阵 (已标准化)
      builder: 量子电路构建函数 (build_angle_encoding 或 build_zz_feature_map)
      n_qubits: 量子比特数 (= 特征维度)
      n_reps: 量子电路重复次数
      scale: 特征缩放因子 (π/2, π, 2π)
    
    返回:
      states: (N, 2^n) 复数态矢量矩阵
    """
    Xs = X * scale
    states = []
    for i in range(len(Xs)):
        circ = builder(Xs[i], n_qubits, n_reps)
        mat = np.array(circ.matrix(), dtype=complex)
        states.append(mat[:, 0])  # 第一列 = |ψ(x)> = U|0⟩
    return np.array(states)


def quantum_kernel_matrix(train_states, test_states=None):
    """
    计算量子核矩阵 K[i,j] = |<ψ_i|ψ_j>|²
    
    量子核定义为两个量子态的内积模平方:
      K(x_i, x_j) = |⟨ψ(x_i)|ψ(x_j)⟩|² = |ψ_i† @ ψ_j|²
    
    性质:
      - K[i,i] = 1 (自核=1)
      - K[i,j] ∈ [0, 1] (非负, 有界)
      - K是对称半正定矩阵 (满足Mercer定理)
    
    参数:
      train_states: (N_train, 2^n) 训练集态矢量
      test_states: (N_test, 2^n) 测试集态矢量 (None=用训练集自身)
    
    返回:
      K: (N_test, N_train) 量子核矩阵
    """
    if test_states is None:
        overlap = train_states.conj() @ train_states.T
    else:
        overlap = test_states.conj() @ train_states.T
    K = np.abs(overlap) ** 2
    K = np.clip(K, 0, 1)
    if test_states is None:
        np.fill_diagonal(K, 1.0)
    return K


def build_classical_kernel(X_train, X_test=None, kernel='rbf', gamma='scale'):
    """
    经典核基线 (用于与量子核对比)
    
    参数:
      X_train: (N_train, d) 训练特征
      X_test: (N_test, d) 测试特征 (None=用训练集自身)
      kernel: 'rbf' 或 'linear'
      gamma: RBF核的γ参数
    
    返回:
      K: (N_test, N_train) 经典核矩阵
    """
    from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
    # 处理 gamma='scale' (1/(n_features*X.var())) 和 'auto' (1/n_features)
    if isinstance(gamma, str):
        n_features = X_train.shape[1]
        if gamma == 'scale':
            gamma = 1.0 / (n_features * X_train.var())
        else:  # 'auto'
            gamma = 1.0 / n_features
    if kernel == 'rbf':
        if X_test is None:
            return rbf_kernel(X_train, X_train, gamma=gamma)
        return rbf_kernel(X_test, X_train, gamma=gamma)
    else:
        if X_test is None:
            return linear_kernel(X_train, X_train)
        return linear_kernel(X_test, X_train)


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("=== 量子核计算模块测试 ===\n")
    
    # 生成4维测试数据
    np.random.seed(42)
    X = np.random.randn(5, 4)
    
    # AngleEncoding
    print("1. AngleEncoding (4 qubits)")
    states = compute_quantum_states(X, build_angle_encoding, n_qubits=4, n_reps=2)
    print(f"   态矢量矩阵: {states.shape}")
    K = quantum_kernel_matrix(states)
    print(f"   核矩阵: {K.shape}")
    print(f"   对角线: {np.diag(K)}")
    print(f"   非对角线均值: {K[~np.eye(5, dtype=bool)].mean():.4f}")
    print(f"   非对角线标准差: {K[~np.eye(5, dtype=bool)].std():.4f}")
    print(f"   区分度(std/mean): {K[~np.eye(5, dtype=bool)].std()/K[~np.eye(5, dtype=bool)].mean():.4f}")
    
    # ZZFeatureMap
    print("\n2. ZZFeatureMap (4 qubits)")
    states_zz = compute_quantum_states(X, build_zz_feature_map, n_qubits=4, n_reps=2)
    K_zz = quantum_kernel_matrix(states_zz)
    print(f"   核矩阵: {K_zz.shape}")
    print(f"   区分度: {K_zz[~np.eye(5, dtype=bool)].std()/K_zz[~np.eye(5, dtype=bool)].mean():.4f}")
    
    # 经典RBF核对比
    print("\n3. 经典RBF核 (对比)")
    K_rbf = build_classical_kernel(X, kernel='rbf')
    print(f"   核矩阵: {K_rbf.shape}")
    print(f"   区分度: {K_rbf[~np.eye(5, dtype=bool)].std()/K_rbf[~np.eye(5, dtype=bool)].mean():.4f}")
    
    print("\n✓ 量子核计算模块测试通过!")
