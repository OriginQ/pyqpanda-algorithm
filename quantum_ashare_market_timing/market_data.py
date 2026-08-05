#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场数据与特征工程模块
Market Data & Feature Engineering for A-Share Market Timing

功能:
  1. 构建11维市场特征 (零前视: T-1收盘数据, 09:25前可见)
  2. 加载交易信号 → 按日聚合 → 市场择时标签
  3. 市场广度计算 (涨跌家数比/涨停家数/跌停家数)

注: 交易信号由外部策略引擎预计算并持久化，
    本模块仅负责加载信号 + 构建市场特征 + 生成标签，
    不包含任何策略选股/择时逻辑。

作者: 周勋洪 (中国电信资阳分公司)
日期: 2026-08-03
"""
import numpy as np
import pandas as pd


def build_market_features(sh_df, all_data=None, signals_daily=None):
    """
    构建市场特征 (全部T-1口径, 次日09:25前100%可见, 零前视偏差)
    
    基础6维 + 扩展5维 (上证指数T-1) + 市场广度3维 (全市场T-1) + 信号层面2维 (交易信号T-1)
    向后兼容: 不传all_data/signals_daily时仅输出基础+扩展11维
    
    参数:
      sh_df: DataFrame(open, high, low, close, volume) 上证指数日线
      all_data: dict (可选) 个股K线数据, 用于计算市场广度
      signals_daily: DataFrame (可选) 按日聚合的交易信号(含n_signals, avg_gap_pct), 用于信号层面特征
    
    返回:
      features: DataFrame, 11~16列特征, index=日期
    """
    sh = sh_df.copy()
    sh.index = sh.index.normalize()
    sh = sh[~sh.index.duplicated(keep='last')]
    sh = sh.sort_index()

    # 均线
    ma5 = sh['close'].rolling(5).mean()
    ma10 = sh['close'].rolling(10).mean()
    ma20 = sh['close'].rolling(20).mean()
    ma60 = sh['close'].rolling(60).mean()

    # ========== 基础6维 (T-1 shifted) ==========

    # 1. bias20 (T-1)
    bias20 = (sh['close'] - ma20) / ma20 * 100
    feat_bias20 = bias20.shift(1)

    # 2. 5日涨幅 (T-1)
    ret_5d = (sh['close'] / sh['close'].shift(5) - 1) * 100
    feat_ret_5d = ret_5d.shift(1)

    # 3. 量比 = 今日量 / 5日均量 (T-1)
    vol_5d = sh['volume'].rolling(5).mean()
    feat_vol_ratio = (sh['volume'] / vol_5d).shift(1)

    # 4. RSI(5) (T-1)
    delta = sh['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(5).mean()
    avg_loss = loss.rolling(5).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    feat_rsi = rsi.shift(1)

    # 5. MA5偏离MA10 (T-1)
    feat_ma_align = ((ma5 - ma10) / ma10 * 100).shift(1)

    # 6. 5日振幅 (T-1)
    high_5d = sh['high'].rolling(5).max()
    low_5d = sh['low'].rolling(5).min()
    feat_range = ((high_5d - low_5d) / sh['close'] * 100).shift(1)

    features = pd.DataFrame({
        'bias20': feat_bias20,
        'ret_5d': feat_ret_5d,
        'vol_ratio': feat_vol_ratio,
        'rsi_5': feat_rsi,
        'ma_align': feat_ma_align,
        'range_5d': feat_range,
    })

    # ========== 扩展5维 (上证指数, T-1 shifted) ==========

    # 7. ret_20d: 20日涨幅%
    ret_20d = (sh['close'] / sh['close'].shift(20) - 1) * 100
    feat_ret_20d = ret_20d.shift(1)

    # 8. vol_ratio_20d: 成交量/20日均量
    vol_20d = sh['volume'].rolling(20).mean()
    feat_vol_ratio_20d = (sh['volume'] / vol_20d).shift(1)

    # 9. ma_trend: (MA20-MA60)/MA60*100
    feat_ma_trend = ((ma20 - ma60) / ma60 * 100).shift(1)

    # 10. vol_rank_60: 20日已实现波动率在过去60日的分位
    daily_ret = sh['close'].pct_change()
    real_vol_20d = daily_ret.rolling(20).std() * np.sqrt(252) * 100
    vol_rank = real_vol_20d.rolling(60).rank(pct=True)
    feat_vol_rank = vol_rank.shift(1)

    # 11. rsi_14: 14日RSI
    delta14 = sh['close'].diff()
    gain14 = delta14.clip(lower=0)
    loss14 = (-delta14).clip(lower=0)
    avg_gain14 = gain14.rolling(14).mean()
    avg_loss14 = loss14.rolling(14).mean()
    rs14 = avg_gain14 / avg_loss14.replace(0, np.nan)
    rsi14 = 100 - 100 / (1 + rs14)
    feat_rsi_14 = rsi14.shift(1)

    features['ret_20d'] = feat_ret_20d
    features['vol_ratio_20d'] = feat_vol_ratio_20d
    features['ma_trend'] = feat_ma_trend
    features['vol_rank_60'] = feat_vol_rank
    features['rsi_14'] = feat_rsi_14

    # ========== 市场广度3维 (全市场主板个股, T-1 shifted) ==========
    if all_data is not None:
        closes_dict = all_data.get('close', {})
        breadth = compute_market_breadth(closes_dict)
        if breadth is not None:
            breadth_shifted = breadth.shift(1)
            features = features.join(breadth_shifted, how='inner')

    # ========== 信号层面2维 (交易信号聚合, T-1 shifted) ==========
    if signals_daily is not None:
        sig_feat = pd.DataFrame(index=signals_daily.index)
        sig_feat['n_sig_prev'] = signals_daily['n_signals']
        if 'avg_gap_pct' in signals_daily.columns:
            sig_feat['avg_gap_prev'] = signals_daily['avg_gap_pct']
        else:
            sig_feat['avg_gap_prev'] = np.nan
        sig_feat_shifted = sig_feat.shift(1)
        features = features.join(sig_feat_shifted, how='inner')

    features = features.dropna()
    return features


def compute_market_breadth(closes_dict):
    """
    计算市场广度指标 (全部T-1口径, 次日09:25前可见)
    
    从个股收盘价计算每日:
    - adv_dec_ratio: 上涨家数/下跌家数
    - limit_up_count: 涨停家数 (主板≥9.5%涨幅)
    - limit_down_count: 跌停家数 (主板≤-9.5%跌幅)
    
    仅使用主板股票 (排除300创业板/688科创板/8开头北交所)
    
    参数:
      closes_dict: dict {code: pd.Series} 个股收盘价
    
    返回:
      breadth: DataFrame(adv_dec_ratio, limit_up_count, limit_down_count)
    """
    # 收集主板股票收盘价
    close_cols = {}
    for code in closes_dict:
        if code.startswith('300') or code.startswith('688') or code.startswith('8'):
            continue
        s = closes_dict[code].dropna()
        if len(s) < 30:
            continue
        close_cols[code] = s

    if len(close_cols) < 100:
        return None

    # 构建宽表 (日期 × 股票), pandas自动对齐索引
    close_df = pd.DataFrame(close_cols)
    close_df.index = close_df.index.normalize()
    close_df = close_df.sort_index()

    # 日收益率
    ret_df = close_df.pct_change()

    # 市场广度
    adv = (ret_df > 0).sum(axis=1)
    dec = (ret_df < 0).sum(axis=1)
    adv_dec_ratio = adv / dec.replace(0, np.nan)

    # 涨停/跌停 (主板10%限制, 用9.5%作为阈值避免四舍五入差异)
    limit_up = (ret_df >= 0.095).sum(axis=1)
    limit_down = (ret_df <= -0.095).sum(axis=1)

    breadth = pd.DataFrame({
        'adv_dec_ratio': adv_dec_ratio,
        'limit_up_count': limit_up,
        'limit_down_count': limit_down,
    }).dropna()

    return breadth


def load_zp_signals(signals_path):
    """
    加载预计算的交易信号
    
    交易信号由外部策略引擎预计算并通过pickle持久化，
    本函数仅负责反序列化加载。信号包含:
      - date (index): 信号触发日期
      - code: 股票代码
      - gap_pct: 信号强度指标
      - buy_price: 买入价格
      - sell_price: 卖出价格
      - sell_ret: 收益率(卖出/买入)
      - label: 盈亏标签(1=盈利, 0=亏损)
    
    信号生成逻辑属于独立商业策略，不在本模块中实现。
    
    参数:
      signals_path: str 信号pickle文件路径
    
    返回:
      signals_df: DataFrame(code, gap_pct, buy_price, sell_price, sell_ret, label)
    """
    with open(signals_path, 'rb') as f:
        import pickle
        signals_df = pickle.load(f)
    return signals_df


def build_daily_labels(signals_df, ret_threshold=0.005):
    """
    按日聚合交易信号 → 市场择时标签
    
    对每个交易日:
      - label = 1 (好天): 该日所有信号的平均收益 > ret_threshold
      - label = 0 (坏天): 该日所有信号的平均收益 <= ret_threshold
    
    标签降噪原理:
      个股盈利预测(单笔)噪声大(AUC≈0.5)
      但日均收益跨信号平均显著降噪, 使分类器可学习
      ret_threshold=0.5% 过滤掉"微涨"噪音日, 只标记有意义的上涨日
    
    参数:
      signals_df: DataFrame 交易信号
      ret_threshold: float 正标签阈值(日均收益率), 默认0.5%
                     0%太弱(4年正样本率45%, 分离度仅0.1%)
                     0.5%最优(正样本率41%, 模型可学到有效信号)
    
    返回:
      daily: DataFrame(n_signals, avg_ret_pct, label, ...)
    """
    daily = signals_df.groupby(signals_df.index.normalize()).agg(
        n_signals=('label', 'count'),
        n_profit=('label', 'sum'),
        avg_sell_ret=('sell_ret', 'mean'),
        avg_ret_pct=('sell_ret', lambda x: np.mean(x) - 1.0),
        avg_gap_pct=('gap_pct', 'mean'),
        wr=('label', 'mean'),
    )
    daily['label'] = (daily['avg_ret_pct'] > ret_threshold).astype(int)
    
    return daily


def prepare_dataset(signals_df, market_features, train_end='2025-06-30'):
    """
    准备训练/测试数据集
    
    时间分割:
      训练集: ~ train_end (约3年)
      测试集: train_end ~ (约1年, 样本外)
    
    返回:
      X_train, X_test, y_train, y_test: 特征和标签
      daily_rets_train, daily_rets_test: 日均收益率(用于NOOP评估)
      n_sigs_train, n_sigs_test: 日信号数
      test_dates: 测试期日期列表
    """
    daily = build_daily_labels(signals_df)
    daily = daily.join(market_features, how='inner')
    
    # 动态特征列: market_features的全部列即为特征 (向后兼容6~16维)
    feature_cols = list(market_features.columns)
    X = daily[feature_cols].values.astype(np.float64)
    y = daily['label'].values.astype(np.int32)
    daily_rets = daily['avg_ret_pct'].values
    n_sigs = daily['n_signals'].values
    
    train_mask = daily.index <= pd.Timestamp(train_end)
    test_mask = ~train_mask
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    rets_train, rets_test = daily_rets[train_mask], daily_rets[test_mask]
    n_sigs_train, n_sigs_test = n_sigs[train_mask], n_sigs[test_mask]
    test_dates = [str(d.date()) for d in daily.index[test_mask]]
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'rets_train': rets_train, 'rets_test': rets_test,
        'n_sigs_train': n_sigs_train, 'n_sigs_test': n_sigs_test,
        'test_dates': test_dates,
        'daily': daily,
        'scaler': scaler,
        'feature_cols': feature_cols,
    }
