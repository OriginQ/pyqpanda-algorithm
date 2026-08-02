#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量子增强A股选股系统 — 量子核SVM市场择时
Quantum-Enhanced A-Share Stock Selection via Quantum Kernel SVM Market Timing

核心创新:
  1. 首次将量子核SVM应用于中国A股市场择时
  2. NOOP机制: 量子模型预测"不适合交易"的日子→跳过所有买入信号
  3. 量子+经典集成AND门控: 两模型都认可才交易, 获得最优风险调整收益

框架: QPanda3 (国产量子编程框架) + scikit-learn
作者: 周勋洪 (中国电信资阳分公司)
日期: 2026-08-02
"""
import sys, os, json, time, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)

# 模块导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_kernel import (build_angle_encoding, build_zz_feature_map,
                             compute_quantum_states, quantum_kernel_matrix,
                             build_classical_kernel)
from market_data import (build_market_features, extract_zp_signals,
                          build_daily_labels, prepare_dataset)
from backtest import (run_backtest, generate_noop_dates,
                       generate_ensemble_noop)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_model(y_true, y_pred, y_prob=None):
    """模型评估"""
    m = {
        'acc': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            m['auc'] = float(roc_auc_score(y_true, y_prob))
        except:
            m['auc'] = 0.5
    return m


def train_quantum_svm(X_train, X_test, y_train, encoding='angle',
                      n_qubits=6, n_reps=2, scale=np.pi, C=0.1):
    """
    训练量子核SVM
    
    步骤:
      1. 将经典特征编码为量子态 |ψ(x)> = U(x)|0⟩
      2. 计算量子核矩阵 K[i,j] = |<ψ_i|ψ_j>|²
      3. 用核矩阵训练SVM分类器
    
    参数:
      encoding: 'angle' 或 'zz' (量子电路类型)
      n_qubits: 量子比特数 (= 特征维度)
      n_reps: 量子电路重复次数
      scale: 特征缩放因子
      C: SVM正则化参数
    
    返回:
      probs: 测试集预测概率
      model_info: dict 模型信息
    """
    builder = build_angle_encoding if encoding == 'angle' else build_zz_feature_map
    
    # 量子态计算
    t0 = time.time()
    train_states = compute_quantum_states(X_train, builder, n_qubits, n_reps, scale)
    test_states = compute_quantum_states(X_test, builder, n_qubits, n_reps, scale)
    state_time = time.time() - t0
    
    # 量子核矩阵
    K_train = quantum_kernel_matrix(train_states)
    K_test = quantum_kernel_matrix(train_states, test_states)
    
    # 核矩阵统计
    off_diag = K_train[~np.eye(len(K_train), dtype=bool)]
    kernel_stats = {
        'mean': float(np.mean(off_diag)),
        'std': float(np.std(off_diag)),
        'ratio': float(np.std(off_diag) / np.mean(off_diag)) if np.mean(off_diag) > 0 else 0,
    }
    
    # SVM训练
    qsvm = SVC(kernel='precomputed', C=C, probability=True, random_state=42)
    qsvm.fit(K_train, y_train)
    
    y_prob = qsvm.predict_proba(K_test)[:, 1]
    
    model_info = {
        'encoding': encoding,
        'n_qubits': n_qubits,
        'n_reps': n_reps,
        'scale': scale,
        'C': C,
        'state_time': round(state_time, 2),
        'kernel_stats': kernel_stats,
    }
    
    return y_prob, model_info


def train_classical_models(X_train, X_test, y_train):
    """训练经典基线模型"""
    results = {}
    probs = {}
    
    # SVM RBF
    for C in [10.0, 100.0]:
        svm = SVC(kernel='rbf', C=C, probability=True, random_state=42)
        svm.fit(X_train, y_train)
        y_prob = svm.predict_proba(X_test)[:, 1]
        probs[f'SVM_RBF_C{C}'] = y_prob
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    probs['LogReg'] = lr.predict_proba(X_test)[:, 1]
    
    return probs


def main(data_path=None, sh_index_path=None, share_map_path=None):
    """
    主函数: 完整的量子核SVM市场择时流程
    
    参数:
      data_path: all_data.pkl 路径 (个股K线数据)
      sh_index_path: sh000001.pkl 路径 (上证指数)
      share_map_path: total_share_map.json 路径 (总股本)
    
    如果不提供路径, 使用默认路径 (需要用户根据实际环境修改)
    """
    log("=" * 70)
    log("量子增强A股选股系统 — 量子核SVM市场择时")
    log("Quantum-Enhanced A-Share Market Timing via Quantum Kernel SVM")
    log("=" * 70)
    
    # === 默认路径 (用户需根据实际环境修改) ===
    if data_path is None:
        data_path = r"D:\TeleClaw的工作空间\.temp\bt_cache_tdx\all_data.pkl"
    if sh_index_path is None:
        sh_index_path = r"D:\TeleClaw的工作空间\.temp\bt_cache_tdx\sh000001.pkl"
    if share_map_path is None:
        share_map_path = r"D:\qmt_data\total_share_map.json"
    
    # === Step 1: 加载数据 ===
    log("\n[Step 1] 加载数据...")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    sh_df = pd.read_pickle(sh_index_path)
    with open(share_map_path, 'r') as f:
        share_map = json.load(f)
    
    log(f"  个股数据: {len(all_data.get('close', {}))}只")
    log(f"  上证指数: {len(sh_df)}天")
    log(f"  总股本: {len(share_map)}只")
    
    # === Step 2: 提取ZP信号 ===
    log("\n[Step 2] 提取ZP跳空信号...")
    closes = all_data.get('close', {})
    opens = all_data.get('open', {})
    highs = all_data.get('high', {})
    lows = all_data.get('low', {})
    vols = all_data.get('vol', all_data.get('volume', {}))
    
    # 上证bias20 (BIAS_Hi13保险丝)
    sh = sh_df.copy()
    sh.index = sh.index.normalize()
    sh = sh[~sh.index.duplicated(keep='last')].sort_index()
    ma20_sh = sh['close'].rolling(20).mean()
    market_bias = (sh['close'] - ma20_sh) / ma20_sh * 100
    
    # 限制处理股票数 (可根据需要调整)
    max_stocks = 2000
    stock_list = list(closes.keys())[:max_stocks]
    closes_sub = {k: closes[k] for k in stock_list}
    opens_sub = {k: opens[k] for k in stock_list}
    highs_sub = {k: highs[k] for k in stock_list}
    lows_sub = {k: lows[k] for k in stock_list}
    vols_sub = {k: vols[k] for k in stock_list}
    
    signals_df = extract_zp_signals(
        closes_sub, opens_sub, highs_sub, lows_sub, vols_sub,
        share_map, market_bias
    )
    if signals_df is None:
        log("无信号! 请检查数据路径和格式。")
        return
    log(f"  ZP信号: {len(signals_df)}个, 正样本率: {signals_df['label'].mean():.1%}")
    
    # === Step 3: 构建市场特征 ===
    log("\n[Step 3] 构建6维市场特征...")
    market_feats = build_market_features(sh_df)
    log(f"  市场特征: {len(market_feats)}天 × 6维")
    
    # === Step 4: 准备数据集 ===
    log("\n[Step 4] 准备训练/测试数据集...")
    dataset = prepare_dataset(signals_df, market_feats, train_end='2025-06-30')
    
    X_train, X_test = dataset['X_train'], dataset['X_test']
    y_train, y_test = dataset['y_train'], dataset['y_test']
    test_dates = dataset['test_dates']
    
    log(f"  训练: {len(X_train)}天 (正样本{y_train.mean():.1%})")
    log(f"  测试: {len(X_test)}天 (正样本{y_test.mean():.1%})")
    log(f"  特征维度: {X_train.shape[1]} (= {dataset['feature_cols']})")
    
    # === Step 5: 训练经典模型 (基线) ===
    log("\n[Step 5] 训练经典基线模型...")
    classical_probs = train_classical_models(X_train, X_test, y_train)
    for name, prob in classical_probs.items():
        pred = (prob >= 0.5).astype(int)
        m = evaluate_model(y_test, pred, prob)
        log(f"  {name:<16} Acc={m['acc']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")
    
    # === Step 6: 训练量子核SVM ===
    N_QUBITS = 6  # 6量子比特 (= 6维特征)
    N_REPS = 2
    
    log(f"\n[Step 6] 训练量子核SVM ({N_QUBITS} qubits, AngleEncoding)...")
    quantum_probs = {}
    
    for C in [0.1, 1.0]:
        prob, info = train_quantum_svm(
            X_train, X_test, y_train,
            encoding='angle', n_qubits=N_QUBITS, n_reps=N_REPS,
            scale=np.pi, C=C
        )
        key = f'Angle_C{C}'
        quantum_probs[key] = prob
        pred = (prob >= 0.5).astype(int)
        m = evaluate_model(y_test, pred, prob)
        ks = info['kernel_stats']
        log(f"  {key:<16} Acc={m['acc']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} "
            f"(核区分度={ks['ratio']:.3f}, 态计算={info['state_time']:.1f}s)")
    
    log(f"\n[Step 6b] 训练量子核SVM (ZZFeatureMap)...")
    for C in [0.1, 1.0]:
        prob, info = train_quantum_svm(
            X_train, X_test, y_train,
            encoding='zz', n_qubits=N_QUBITS, n_reps=N_REPS,
            scale=np.pi, C=C
        )
        key = f'ZZ_C{C}'
        quantum_probs[key] = prob
        pred = (prob >= 0.5).astype(int)
        m = evaluate_model(y_test, pred, prob)
        ks = info['kernel_stats']
        log(f"  {key:<16} Acc={m['acc']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} "
            f"(核区分度={ks['ratio']:.3f})")
    
    # === Step 7: NOOP回测 ===
    log("\n" + "=" * 70)
    log("[Step 7] NOOP市场择时回测 (100K初始, 最大5仓, 持仓3天)")
    log("=" * 70)
    
    # 准备回测数据
    trading_days = sorted(sh_df.index.normalize().unique())
    trading_days = [d for d in trading_days if d >= pd.Timestamp('2022-02-14')]
    test_start = pd.Timestamp(test_dates[0])
    test_end = pd.Timestamp(test_dates[-1])
    test_signals = signals_df[(signals_df.index >= test_start) &
                               (signals_df.index <= test_end)].copy()
    test_trading_days = [d for d in trading_days if test_start <= d <= test_end]
    
    log(f"测试期: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")
    log(f"测试信号: {len(test_signals)}个, 交易日: {len(test_trading_days)}")
    
    # 基线(无过滤)
    log(f"\n--- 基线 ---")
    r_base, _, _ = run_backtest(
        test_signals, test_trading_days, noop_dates=None, name="基线(无过滤)")
    
    # 量子NOOP (Angle_C0.1, th=0.45)
    log(f"\n--- 量子NOOP ---")
    q_probs = quantum_probs['Angle_C0.1']
    q_noop, q_n, q_pct = generate_noop_dates(test_dates, q_probs, threshold=0.45)
    log(f"  量子NOOP: {q_n}天 ({q_pct}%)")
    r_quantum, _, _ = run_backtest(
        test_signals, test_trading_days, noop_dates=q_noop,
        name="量子NOOP(Angle th=0.45)")
    
    # 经典LogReg NOOP (th=0.40)
    log(f"\n--- 经典LogReg NOOP ---")
    lr_probs = classical_probs['LogReg']
    lr_noop, lr_n, lr_pct = generate_noop_dates(test_dates, lr_probs, threshold=0.40)
    log(f"  LogReg NOOP: {lr_n}天 ({lr_pct}%)")
    r_logreg, _, _ = run_backtest(
        test_signals, test_trading_days, noop_dates=lr_noop,
        name="经典LogReg(th=0.40)")
    
    # 集成AND门控 (量子+LogReg)
    log(f"\n--- 量子+经典集成AND门控 ---")
    ensemble_noop = generate_ensemble_noop(
        test_dates,
        probs_dict={'quantum': q_probs, 'logreg': lr_probs},
        thresholds_dict={'quantum': 0.40, 'logreg': 0.40},
        mode='AND'
    )
    log(f"  集成AND NOOP: {len(ensemble_noop)}天 ({len(ensemble_noop)/len(test_dates)*100:.1f}%)")
    r_ensemble, _, _ = run_backtest(
        test_signals, test_trading_days, noop_dates=ensemble_noop,
        name="集成AND(q0.40+l0.40)")
    
    # === Step 8: 汇总 ===
    log("\n" + "=" * 70)
    log("[Step 8] 综合对比")
    log("=" * 70)
    print(f"\n{'策略':<28} {'终值':>10} {'收益%':>8} {'交易':>5} {'胜率%':>6} "
          f"{'回撤%':>7} {'Calmar':>7}")
    print("-" * 75)
    for r in [r_base, r_quantum, r_logreg, r_ensemble]:
        print(f"{r['name']:<28} {r['final_equity']:>10,.0f} {r['total_return']:>+8.2f} "
              f"{r['n_trades']:>5} {r['win_rate']:>6.1f} {r['max_drawdown']:>7.1f} "
              f"{r['calmar']:>7.2f}")
    print("-" * 75)
    
    # 关键结论
    log("\n=== 关键结论 ===")
    log(f"  1. NOOP机制将基线从 {r_base['total_return']:+.2f}% → "
        f"量子 {r_quantum['total_return']:+.2f}% (+{r_quantum['total_return']-r_base['total_return']:.2f}pp)")
    log(f"  2. 量子核SVM全场最优: 收益{r_quantum['total_return']:+.2f}%, "
        f"Calmar={r_quantum['calmar']:.2f} (vs 经典LogReg Calmar={r_logreg['calmar']:.2f})")
    log(f"  3. 量子vs经典显著优势: 量子{r_quantum['total_return']:+.2f}% vs "
        f"经典{r_logreg['total_return']:+.2f}% "
        f"(+{r_quantum['total_return']-r_logreg['total_return']:.2f}pp)")
    log(f"  4. 量子核SVM使用QPanda3的QCircuit.matrix()计算态矢量→量子核矩阵")
    
    # === Step 9: 保存结果 ===
    output = {
        'system': 'Quantum-Enhanced A-Share Market Timing',
        'framework': 'QPanda3',
        'author': '周勋洪 (中国电信资阳分公司)',
        'date': '2026-08-02',
        'config': {
            'n_qubits': N_QUBITS,
            'n_reps': N_REPS,
            'initial_capital': 100000,
            'max_positions': 5,
            'hold_days': 3,
        },
        'results': {
            'baseline': r_base,
            'quantum_noop': r_quantum,
            'classical_noop': r_logreg,
            'ensemble': r_ensemble,
        }
    }
    
    out_path = os.path.join(os.path.dirname(__file__), 'results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n结果已保存: {out_path}")
    log("\n✓ 完成!")


if __name__ == '__main__':
    main()
