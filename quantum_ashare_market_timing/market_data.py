#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场数据与特征工程模块
Market Data & Feature Engineering for A-Share Market Timing

功能:
  1. 构建上证指数6维市场特征 (零前视: T-1收盘数据, 09:25前可见)
  2. 提取ZP跳空策略信号 (均线多头+跳空2-5%+温和放量)
  3. 按日聚合信号 → 市场择时标签

作者: 周勋洪 (中国电信资阳分公司)
日期: 2026-08-02
"""
import numpy as np
import pandas as pd


def build_market_features(sh_df):
    """
    构建6维市场特征 (全部T-1口径, 次日09:25前100%可见, 零前视偏差)
    
    特征清单:
      1. bias20: (close-MA20)/MA20*100  — 均值回归偏离度
                 ★BIAS_Hi13保险丝的核心(>13%时大盘过热)
      2. ret_5d: 5日涨幅%                — 短期动量
      3. vol_ratio: 成交量/5日均量        — 量能变化
      4. rsi_5: 5日RSI                   — 超买超卖指标
      5. ma_align: (MA5-MA10)/MA10*100  — 短期趋势方向
      6. range_5d: (5日最高-5日最低)/close*100 — 近期波动率
    
    参数:
      sh_df: DataFrame(open, high, low, close, volume) 上证指数日线
    
    返回:
      features: DataFrame, 6列特征, index=日期
    """
    sh = sh_df.copy()
    sh.index = sh.index.normalize()
    sh = sh[~sh.index.duplicated(keep='last')]
    sh = sh.sort_index()

    # 均线
    ma5 = sh['close'].rolling(5).mean()
    ma10 = sh['close'].rolling(10).mean()
    ma20 = sh['close'].rolling(20).mean()

    # 特征1: bias20 (T-1)
    bias20 = (sh['close'] - ma20) / ma20 * 100
    feat_bias20 = bias20.shift(1)

    # 特征2: 5日涨幅 (T-1)
    ret_5d = (sh['close'] / sh['close'].shift(5) - 1) * 100
    feat_ret_5d = ret_5d.shift(1)

    # 特征3: 量比 = 今日量 / 5日均量 (T-1)
    vol_5d = sh['volume'].rolling(5).mean()
    feat_vol_ratio = (sh['volume'] / vol_5d).shift(1)

    # 特征4: RSI(5) (T-1)
    delta = sh['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(5).mean()
    avg_loss = loss.rolling(5).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    feat_rsi = rsi.shift(1)

    # 特征5: MA5偏离MA10 (T-1)
    feat_ma_align = ((ma5 - ma10) / ma10 * 100).shift(1)

    # 特征6: 5日振幅 (T-1)
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
    }).dropna()

    return features


def extract_zp_signals(closes, opens, highs, lows, vols, share_map,
                       market_bias, gap_min=2.0, gap_max=5.0,
                       mcap_min=50, mcap_max=2000, min_price=3.0,
                       hold_days=3):
    """
    提取ZP跳空策略买入信号
    
    ZP策略买入条件 (全部09:25可见, 零前视):
      1. 均线多头: MA5 > MA10 > MA20, MA20上行
      2. 温和放量: 近5日/近10日量比 ∈ [0.8, 1.5]
      3. 站稳MA5: T-1收盘 >= T-1 MA5
      4. 跳空2.0~5.0%: T日open / T-1 close - 1 ∈ [2%, 5%]
      5. 市值50~2000亿 (排除微盘股和超大盘)
      6. 股价 >= 3元
      7. 排除一字板/300创业板/688科创板/8开头北交所/ST
    
    卖出: 固定持仓hold_days天后以收盘价卖出
    排序: 跳空%降序 (ZP铁律: 跳空越大=信号越强)
    
    参数:
      closes/opens/highs/lows/vols: dict {code: pd.Series} 个股OHLCV
      share_map: dict {code: total_share} 总股本(亿股)
      market_bias: pd.Series 上证bias20 (用于BIAS_Hi13保险丝)
      gap_min/gap_max: 跳空区间
      mcap_min/mcap_max: 市值区间(亿)
      min_price: 最低股价
      hold_days: 持仓天数
    
    返回:
      signals: DataFrame(code, gap_pct, sell_ret, label, ...)
    """
    signals = []
    
    for code in closes:
        # 排除创业板/科创板/北交所/ST
        if code.startswith('300') or code.startswith('688') or code.startswith('8'):
            continue
        if 'ST' in code.upper():
            continue

        close_s = closes[code]
        open_s = opens[code]
        high_s = highs[code]
        low_s = lows[code]
        vol_s = vols[code]

        if len(close_s) < 30:
            continue

        # 个股数据
        df = pd.DataFrame({
            'open': open_s, 'high': high_s, 'low': low_s,
            'close': close_s, 'vol': vol_s
        }).dropna()
        df.index = df.index.normalize()

        # 均线
        ma5 = df['close'].rolling(5).mean()
        ma10 = df['close'].rolling(10).mean()
        ma20 = df['close'].rolling(20).mean()

        # 总股本
        code_6 = code.split('.')[0]
        total_share = share_map.get(code_6)
        if total_share is None:
            continue

        for i in range(21, len(df) - hold_days - 1):
            # T-1数据 (i)
            pc = df['close'].iloc[i]
            pv = df['vol'].iloc[i]

            # 均线多头
            if not (ma5.iloc[i] > ma10.iloc[i] > ma20.iloc[i]):
                continue
            # MA20上行
            if ma20.iloc[i] <= ma20.iloc[i - 1]:
                continue
            # 站稳MA5
            if pc < ma5.iloc[i]:
                continue
            # 温和放量
            v5 = df['vol'].iloc[i - 4:i + 1].mean()
            v10 = df['vol'].iloc[i - 9:i + 1].mean()
            if v5 <= 0 or v10 <= 0:
                continue
            vr = v5 / v10
            if vr < 0.8 or vr > 1.5:
                continue

            # T日跳空 (i+1)
            t_open = df['open'].iloc[i + 1]
            gap_pct = (t_open / pc - 1) * 100
            if gap_pct < gap_min or gap_pct > gap_max:
                continue

            # 一字板排除
            t_high = df['high'].iloc[i + 1]
            t_low = df['low'].iloc[i + 1]
            if t_open >= t_high * 0.999 and t_open <= t_low * 1.001:
                continue

            # 市值过滤
            mcap = pc * total_share / 1e8  # 亿
            if mcap < mcap_min or mcap > mcap_max:
                continue

            # 股价过滤
            if pc < min_price:
                continue

            # 卖出: hold_days后收盘价
            sell_idx = min(i + 1 + hold_days, len(df) - 1)
            sell_price = df['close'].iloc[sell_idx]
            buy_price = t_open
            sell_ret = sell_price / buy_price  # 卖出价/买入价
            label = 1 if sell_ret > 1.0 else 0

            date = df.index[i + 1]
            signals.append({
                'date': date,
                'code': code,
                'gap_pct': gap_pct,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'sell_ret': sell_ret,
                'label': label,
            })

    if len(signals) == 0:
        return None

    sig_df = pd.DataFrame(signals)
    sig_df.set_index('date', inplace=True)
    sig_df.sort_index(inplace=True)
    return sig_df


def build_daily_labels(signals_df, ret_threshold=0.005):
    """
    按日聚合ZP信号 → 市场择时标签
    
    对每个交易日:
      - label = 1 (好天): 该日所有ZP信号的平均收益 > ret_threshold
      - label = 0 (坏天): 该日所有ZP信号的平均收益 <= ret_threshold
    
    标签降噪原理:
      个股盈利预测(单笔)噪声大(AUC≈0.5)
      但日均收益跨信号平均显著降噪, 使分类器可学习
      ret_threshold=0.5% 过滤掉"微涨"噪音日, 只标记有意义的上涨日
    
    参数:
      signals_df: DataFrame ZP信号
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
    
    feature_cols = ['bias20', 'ret_5d', 'vol_ratio', 'rsi_5', 'ma_align', 'range_5d']
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
