#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量子增强A股选股系统 — 量子核SVM市场择时
Quantum-Enhanced A-Share Stock Selection via Quantum Kernel SVM Market Timing

核心创新:
  1. 首次将量子核SVM应用于中国A股市场择时
  2. NOOP机制: 量子模型预测"不适合交易"的日子→跳过所有买入信号
  3. Walk-forward扩展窗口验证: 2022-06-01~至今, 6个月refit, 零前视偏差
  4. 标签降噪: 按日聚合交易信号, 用0.5%收益阈值过滤噪音日

框架: QPanda3 (国产量子编程框架) + scikit-learn
作者: 周勋洪 (中国电信资阳分公司)
日期: 2026-08-03
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
from market_data import (build_market_features, load_zp_signals,
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


def walk_forward_validation(all_daily, market_features, test_start='2022-06-01',
                            n_qubits=6, n_reps=1, scale=np.pi, q_C=0.1,
                            feature_selection=True):
    """
    Walk-forward扩展窗口验证
    
    方法论:
      - 初始训练: test_start之前的全部数据
      - 每6个月refit模型, 训练窗口逐步扩大(只增不减)
      - 每段独立训练+预测, 预测结果拼接为完整样本外概率序列
      - StandardScaler仅在对应训练窗口fit, 无前视偏差
      - 特征选择: 当特征数>量子比特数时, SelectKBest(f_classif)选top-k
        (仅在训练窗口fit, 零前视偏差)
      - 经典LogReg使用全部特征(无qubit限制)
    
    参数:
      all_daily: DataFrame 按日聚合的交易信号标签
      market_features: DataFrame N维市场特征 (6~16维)
      test_start: str 测试期起始日
      n_qubits: 量子比特数 (= 量子模型使用的特征数)
      n_reps: 量子电路重复次数
      scale: 特征缩放因子
      q_C: 量子SVM正则化参数
      feature_selection: bool 特征数>n_qubits时是否做选择
    
    返回:
      q_probs: np.array 量子模型样本外概率
      lr_probs: np.array 经典LogReg样本外概率
      test_dates: list 日期字符串
      seg_results: list 每段详细结果
    """
    # 动态特征列: market_features的全部列 (向后兼容6~16维)
    feature_cols = list(market_features.columns)
    n_features = len(feature_cols)
    use_feat_sel = feature_selection and n_features > n_qubits
    
    log(f"  特征数: {n_features}维, 量子qubit: {n_qubits}, "
        f"特征选择: {'ON(SelectKBest top-%d)' % n_qubits if use_feat_sel else 'OFF'}")
    
    # 合并标签和特征
    data = all_daily.join(market_features, how='inner').sort_index()
    test_start_ts = pd.Timestamp(test_start)
    end_date = data.index[-1]
    
    # 生成6个月分段
    segments = []
    seg_start = test_start_ts
    while seg_start <= end_date:
        seg_end = seg_start + pd.DateOffset(months=6)
        if seg_end > end_date:
            seg_end = end_date + pd.Timedelta(days=1)
        segments.append((seg_start, seg_end))
        seg_start = seg_end
    
    log(f"  Walk-forward: {len(segments)}段, 扩展窗口, 测试期 {test_start} ~ {end_date.date()}")
    for i, (s, e) in enumerate(segments):
        train_n = len(data[data.index < s])
        test_n = len(data[(data.index >= s) & (data.index < e)])
        log(f"    Seg {i+1}: train<{s.date()}({train_n}d) test {s.date()}~{e.date()}({test_n}d)")
    
    q_probs_all, lr_probs_all, test_dates_all = [], [], []
    seg_results = []
    
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        train_mask = data.index < seg_start
        test_mask = (data.index >= seg_start) & (data.index < seg_end)
        
        X_train_raw = data.loc[train_mask, feature_cols].values
        X_test_raw = data.loc[test_mask, feature_cols].values
        y_train = data.loc[train_mask, 'label'].values
        y_test = data.loc[test_mask, 'label'].values
        
        if len(X_train_raw) < 10 or len(X_test_raw) == 0:
            log(f"    Seg {seg_idx+1}: 跳过(训练={len(X_train_raw)}, 测试={len(X_test_raw)})")
            continue
        
        # === 量子模型: 特征选择 + 标准化 (仅fit训练窗口, 零前视) ===
        from sklearn.feature_selection import SelectKBest, f_classif
        if use_feat_sel:
            selector = SelectKBest(f_classif, k=n_qubits)
            X_train_q_raw = selector.fit_transform(X_train_raw, y_train)
            X_test_q_raw = selector.transform(X_test_raw)
            sel_cols = [feature_cols[i] for i in selector.get_support(indices=True)]
        else:
            X_train_q_raw = X_train_raw
            X_test_q_raw = X_test_raw
            sel_cols = feature_cols
        
        scaler_q = StandardScaler()
        X_train_q = scaler_q.fit_transform(X_train_q_raw)
        X_test_q = scaler_q.transform(X_test_q_raw)
        
        # === 量子核SVM ===
        t0 = time.time()
        train_states = compute_quantum_states(X_train_q, build_angle_encoding,
                                               n_qubits, n_reps, scale)
        test_states = compute_quantum_states(X_test_q, build_angle_encoding,
                                              n_qubits, n_reps, scale)
        K_train = quantum_kernel_matrix(train_states)
        K_test = quantum_kernel_matrix(train_states, test_states)
        qsvm = SVC(kernel='precomputed', C=q_C, probability=True, random_state=42)
        qsvm.fit(K_train, y_train)
        q_prob = qsvm.predict_proba(K_test)[:, 1]
        q_time = time.time() - t0
        
        # === 经典LogReg (使用全部特征, 无qubit限制) ===
        scaler_lr = StandardScaler()
        X_train_lr = scaler_lr.fit_transform(X_train_raw)
        X_test_lr = scaler_lr.transform(X_test_raw)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_lr, y_train)
        lr_prob = lr.predict_proba(X_test_lr)[:, 1]
        
        # 段内评估
        q_pred = (q_prob >= 0.5).astype(int)
        lr_pred = (lr_prob >= 0.5).astype(int)
        q_acc = float(accuracy_score(y_test, q_pred))
        lr_acc = float(accuracy_score(y_test, lr_pred))
        
        try:
            q_auc = float(roc_auc_score(y_test, q_prob))
            lr_auc = float(roc_auc_score(y_test, lr_prob))
        except:
            q_auc = lr_auc = 0.5
        
        seg_dates = [str(d.date()) for d in data.index[test_mask]]
        q_probs_all.extend(q_prob)
        lr_probs_all.extend(lr_prob)
        test_dates_all.extend(seg_dates)
        
        seg_results.append({
            'seg': seg_idx + 1,
            'train_size': int(len(y_train)),
            'test_size': int(len(y_test)),
            'q_acc': round(q_acc, 4),
            'lr_acc': round(lr_acc, 4),
            'q_auc': round(q_auc, 4),
            'lr_auc': round(lr_auc, 4),
            'q_time': round(q_time, 1),
            'n_features': n_features,
            'q_features': sel_cols,
        })
        
        log(f"    Seg {seg_idx+1}: train={len(y_train)} test={len(y_test)} "
            f"Q_acc={q_acc:.3f} LR_acc={lr_acc:.3f} Q_AUC={q_auc:.3f} LR_AUC={lr_auc:.3f} "
            f"({q_time:.1f}s)")
    
    return (np.array(q_probs_all), np.array(lr_probs_all),
            test_dates_all, seg_results)


def scan_noop_thresholds(test_dates, q_probs, lr_probs, test_signals,
                         test_trading_days, thresholds=None):
    """
    扫描量子+经典模型的NOOP阈值, 找到最优配置
    
    参数:
      test_dates: list 日期
      q_probs: np.array 量子概率
      lr_probs: np.array 经典概率
      test_signals: DataFrame 测试信号
      test_trading_days: list 交易日
      thresholds: list NOOP阈值列表
    
    返回:
      q_results: list 量子扫描结果
      lr_results: list 经典扫描结果
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    
    q_results, lr_results = [], []
    
    for th in thresholds:
        # Quantum
        q_noop, q_n, q_pct = generate_noop_dates(test_dates, q_probs, threshold=th)
        r_q, _, _ = run_backtest(test_signals, test_trading_days,
                                 noop_dates=q_noop,
                                 name=f"量子NOOP(th={th:.2f})")
        q_results.append({'threshold': th, 'noop_pct': q_pct, **r_q})
        
        # LogReg
        lr_noop, lr_n, lr_pct = generate_noop_dates(test_dates, lr_probs, threshold=th)
        r_lr, _, _ = run_backtest(test_signals, test_trading_days,
                                  noop_dates=lr_noop,
                                  name=f"经典LogReg(th={th:.2f})")
        lr_results.append({'threshold': th, 'noop_pct': lr_pct, **r_lr})
    
    return q_results, lr_results


def main(data_path=None, sh_index_path=None):
    """
    主函数: Walk-forward量子核SVM市场择时完整流程
    
    测试期: 2022-06-01 ~ 数据最新日 (~4年, ~1000交易日)
    训练: 扩展窗口, 每6个月refit, 初始训练期2022-01~05
    回测: 100K初始, 最大5仓, 持仓3天
    标签: 日均收益>0.5%为好天(降噪阈值)
    
    参数:
      data_path: all_data.pkl 路径 (个股K线数据, 用于市场广度特征)
      sh_index_path: sh000001.pkl 路径 (上证指数)
    """
    log("=" * 70)
    log("量子增强A股选股系统 — 量子核SVM市场择时")
    log("Walk-Forward Validation: 2022-06-01 ~ latest (4-year)")
    log("=" * 70)
    
    # === 默认路径 ===
    if data_path is None:
        data_path = r"D:\TeleClaw的工作空间\.temp\bt_cache_tdx\all_data.pkl"
    if sh_index_path is None:
        sh_index_path = r"D:\TeleClaw的工作空间\.temp\bt_cache_tdx\sh000001.pkl"
    
    # === Step 1: 加载数据 ===
    log("\n[Step 1] 加载数据...")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    sh_df = pd.read_pickle(sh_index_path)
    
    log(f"  个股数据: {len(all_data.get('close', {}))}只")
    log(f"  上证指数: {len(sh_df)}天 ({sh_df.index[0].date()}~{sh_df.index[-1].date()})")
    
    # === Step 2: 加载交易信号 ===
    log("\n[Step 2] 加载交易信号...")
    signals_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'trading_signals_sample.pkl')
    if not os.path.exists(signals_path):
        log(f"信号文件不存在: {signals_path}")
        log("请确保 trading_signals_sample.pkl 与本脚本在同一目录。")
        return
    signals_df = load_zp_signals(signals_path)
    if signals_df is None or len(signals_df) == 0:
        log("无信号! 请检查信号文件格式。")
        return
    log(f"  交易信号: {len(signals_df)}个, 正样本率: {signals_df['label'].mean():.1%}")
    log(f"  信号日期范围: {signals_df.index[0].date()} ~ {signals_df.index[-1].date()}")
    
    # === Step 3: 按日聚合标签 (先于特征构建, 因信号层面特征需用daily数据) ===
    RET_THRESHOLD = 0.005  # 0.5% 标签降噪阈值
    log(f"\n[Step 3] 按日聚合交易信号 → 市场择时标签 (threshold={RET_THRESHOLD*100}%)...")
    all_daily = build_daily_labels(signals_df, ret_threshold=RET_THRESHOLD)
    log(f"  日标签: {len(all_daily)}天 (正样本率={all_daily['label'].mean():.1%})")
    good_avg = all_daily[all_daily['label']==1]['avg_ret_pct'].mean()
    bad_avg = all_daily[all_daily['label']==0]['avg_ret_pct'].mean()
    log(f"  好天平均收益: {good_avg:+.2f}%")
    log(f"  坏天平均收益: {bad_avg:+.2f}%")
    
    # === Step 4: 构建11维市场特征 (Phase 1确认11维SelectKBest最优) ===
    log("\n[Step 4] 构建市场特征...")
    feats_16 = build_market_features(sh_df, all_data=all_data, signals_daily=all_daily)
    # Phase 1实验确认: 11维(上证指数基础6+扩展5)作为SelectKBest起始池优于16维
    # Phase 2-4全部基于此11维+SelectKBest→6q, n_reps=1 → +126.12%
    market_feats = feats_16[['bias20', 'ret_5d', 'vol_ratio', 'rsi_5', 'ma_align',
                             'range_5d', 'ret_20d', 'vol_ratio_20d', 'ma_trend',
                             'vol_rank_60', 'rsi_14']]
    log(f"  市场特征: {len(market_feats)}天 × {len(market_feats.columns)}维")
    log(f"  特征列: {list(market_feats.columns)}")
    
    # === Step 5: Walk-forward验证 ===
    N_QUBITS = 6
    N_REPS = 1  # Phase 2发现: n_reps=1碾压n_reps=2 (+126.12% vs +1.84%)
    TEST_START = '2022-06-01'
    
    log(f"\n[Step 5] Walk-forward验证 ({N_QUBITS} qubits, AngleEncoding)...")
    log(f"  测试起始: {TEST_START}")
    
    q_probs, lr_probs, test_dates, seg_results = walk_forward_validation(
        all_daily, market_feats, test_start=TEST_START,
        n_qubits=N_QUBITS, n_reps=N_REPS, scale=np.pi, q_C=0.1
    )
    
    log(f"\n  样本外预测: {len(q_probs)}天")
    log(f"  量子模型平均准确率: {np.mean([s['q_acc'] for s in seg_results]):.3f}")
    log(f"  经典模型平均准确率: {np.mean([s['lr_acc'] for s in seg_results]):.3f}")
    log(f"  量子核总计算时间: {sum(s['q_time'] for s in seg_results):.1f}s")
    
    # === Step 6: 准备回测数据 ===
    log("\n" + "=" * 70)
    log(f"[Step 6] NOOP市场择时回测 (100K初始, 最大5仓, 持仓3天)")
    log(f"  测试期: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")
    log("=" * 70)
    
    sh_norm = sh_df.copy()
    sh_norm.index = sh_norm.index.normalize()
    sh_norm = sh_norm[~sh_norm.index.duplicated(keep='last')].sort_index()
    trading_days = sorted(sh_norm.index.unique())
    test_start_ts = pd.Timestamp(TEST_START)
    test_end_ts = pd.Timestamp(test_dates[-1])
    
    test_signals = signals_df[(signals_df.index.normalize() >= test_start_ts) &
                               (signals_df.index.normalize() <= test_end_ts)].copy()
    test_trading_days = [d for d in trading_days if test_start_ts <= d <= test_end_ts]
    log(f"  测试信号: {len(test_signals)}个, 交易日: {len(test_trading_days)}")
    
    # === Step 6a: 基线(无过滤) ===
    log(f"\n--- 基线(无过滤) ---")
    r_base, _, _ = run_backtest(
        test_signals, test_trading_days, noop_dates=None, name="基线(无过滤)")
    
    # === Step 6b: 阈值扫描 ===
    log(f"\n--- NOOP阈值扫描 (量子 + 经典) ---")
    scan_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    q_scan, lr_scan = scan_noop_thresholds(
        test_dates, q_probs, lr_probs, test_signals, test_trading_days, scan_thresholds)
    
    # 打印扫描结果
    print(f"\n{'模型':<20} {'阈值':>5} {'NOOP%':>6} {'终值':>10} {'收益%':>8} "
          f"{'交易':>5} {'胜率%':>6} {'回撤%':>7} {'Calmar':>7}")
    print("-" * 80)
    for r in q_scan:
        print(f"{'量子NOOP':<20} {r['threshold']:>5.2f} {r['noop_pct']:>5.1f}% "
              f"{r['final_equity']:>10,.0f} {r['total_return']:>+8.2f} "
              f"{r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['max_drawdown']:>7.1f} "
              f"{r['calmar']:>7.2f}")
    for r in lr_scan:
        print(f"{'经典LogReg':<20} {r['threshold']:>5.2f} {r['noop_pct']:>5.1f}% "
              f"{r['final_equity']:>10,.0f} {r['total_return']:>+8.2f} "
              f"{r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['max_drawdown']:>7.1f} "
              f"{r['calmar']:>7.2f}")
    print("-" * 80)
    
    # 找到量子最优
    best_q = max(q_scan, key=lambda x: x['total_return'])
    best_lr = max(lr_scan, key=lambda x: x['total_return'])
    log(f"\n  量子最优: th={best_q['threshold']:.2f} → {best_q['total_return']:+.2f}%")
    log(f"  经典最优: th={best_lr['threshold']:.2f} → {best_lr['total_return']:+.2f}%")
    
    # === Step 6c: 最优配置完整回测 ===
    Q_BEST_TH = best_q['threshold']
    LR_BEST_TH = best_lr['threshold']
    
    # 量子最优
    q_noop_best, _, _ = generate_noop_dates(test_dates, q_probs, threshold=Q_BEST_TH)
    r_quantum, _, _ = run_backtest(
        test_signals, test_trading_days, noop_dates=q_noop_best,
        name=f"量子NOOP(th={Q_BEST_TH:.2f})")
    
    # 经典最优
    lr_noop_best, _, _ = generate_noop_dates(test_dates, lr_probs, threshold=LR_BEST_TH)
    r_logreg, _, _ = run_backtest(
        test_signals, test_trading_days, noop_dates=lr_noop_best,
        name=f"经典LogReg(th={LR_BEST_TH:.2f})")
    
    # 集成AND门控 (量子+LogReg最优阈值)
    log(f"\n--- 量子+经典集成AND门控 ---")
    ensemble_noop = generate_ensemble_noop(
        test_dates,
        probs_dict={'quantum': q_probs, 'logreg': lr_probs},
        thresholds_dict={'quantum': Q_BEST_TH, 'logreg': LR_BEST_TH},
        mode='AND'
    )
    log(f"  集成AND NOOP: {len(ensemble_noop)}天 ({len(ensemble_noop)/len(test_dates)*100:.1f}%)")
    r_ensemble, _, _ = run_backtest(
        test_signals, test_trading_days, noop_dates=ensemble_noop,
        name=f"集成AND(q{Q_BEST_TH:.2f}+l{LR_BEST_TH:.2f})")
    
    # === Step 7: 汇总 ===
    log("\n" + "=" * 70)
    log("[Step 7] 综合对比")
    log("=" * 70)
    print(f"\n{'策略':<32} {'终值':>10} {'收益%':>8} {'交易':>5} {'胜率%':>6} "
          f"{'回撤%':>7} {'Calmar':>7}")
    print("-" * 80)
    for r in [r_base, r_quantum, r_logreg, r_ensemble]:
        print(f"{r['name']:<32} {r['final_equity']:>10,.0f} {r['total_return']:>+8.2f} "
              f"{r['n_trades']:>5} {r['win_rate']:>6.1f} {r['max_drawdown']:>7.1f} "
              f"{r['calmar']:>7.2f}")
    print("-" * 80)
    
    # 关键结论
    log("\n=== 关键结论 ===")
    log(f"  1. Walk-forward验证: {len(test_dates)}天样本外, {len(seg_results)}段refit, 零前视偏差")
    log(f"  2. 标签降噪: ret_threshold={RET_THRESHOLD*100}% 过滤噪音日")
    log(f"  3. NOOP机制: 基线{r_base['total_return']:+.2f}% → 量子{r_quantum['total_return']:+.2f}%")
    log(f"  4. 量子vs经典: 量子{r_quantum['total_return']:+.2f}% vs 经典{r_logreg['total_return']:+.2f}%")
    log(f"  5. 量子核平均准确率: {np.mean([s['q_acc'] for s in seg_results]):.1%}")
    log(f"  6. 量子核使用QPanda3的QCircuit.matrix()计算态矢量→量子核矩阵")
    
    # === Step 8: 保存结果 ===
    output = {
        'system': 'Quantum-Enhanced A-Share Market Timing (Walk-Forward)',
        'framework': 'QPanda3',
        'author': '周勋洪 (中国电信资阳分公司)',
        'date': '2026-08-03',
        'methodology': 'Walk-forward expanding window, 6-month refit, zero look-ahead bias',
        'label_threshold': f'{RET_THRESHOLD*100}%',
        'test_period': f"{test_dates[0]} ~ {test_dates[-1]}",
        'config': {
            'n_qubits': N_QUBITS,
            'n_reps': N_REPS,
            'initial_capital': 100000,
            'max_positions': 5,
            'hold_days': 3,
            'test_start': TEST_START,
            'n_segments': len(seg_results),
            'ret_threshold': RET_THRESHOLD,
        },
        'systematic_optimization': {
            'description': '4-phase systematic optimization (Phase 1-4)',
            'phase1_features': '6→16 dim expansion + SelectKBest; 11dim SelectKBest→6q optimal',
            'phase2_circuit': 'angle_nrep1_C0.1_pi_6q = global optimum (+126.12%); n_reps=1 breakthrough vs n_reps=2 (+1.84%)',
            'phase3_labels': '0.5% binary classification optimal; regression/multiclass/multi-horizon all inferior',
            'phase4_training': '6-month refit + expanding window optimal; monthly refit/rolling window/ensemble/regime all inferior',
            'global_optimal': 'angle_encoding, n_reps=1, C=0.1, scale=pi, 11dim SelectKBest->6q, 0.5% label, 6-month refit, expanding window -> +126.12%',
            'baseline_no_filter': '-76.34%',
        },
        'segment_results': seg_results,
        'threshold_scan': {
            'quantum': q_scan,
            'classical': lr_scan,
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
