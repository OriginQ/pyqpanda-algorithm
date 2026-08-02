#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
资金池回测模块 — NOOP择时回测引擎
Portfolio Backtest Engine with NOOP Mechanism

核心创新: NOOP (No-Operation) 市场择时
  - 量子/经典模型预测"今天不适合交易" → 跳过当日所有买入信号
  - 持仓按固定天数到期退出 (不影响卖出)
  - 资金池模式: 最大N仓, 每仓=总资产/N, 动态分仓

作者: 周勋洪 (中国电信资阳分公司)
日期: 2026-08-02
"""
import numpy as np
import pandas as pd


def run_backtest(signals_df, trading_days, noop_dates=None,
                 initial_capital=100000, max_positions=5, hold_days=3,
                 name=""):
    """
    简化资金池回测
    
    交易规则:
      1. 每日开盘: 先平仓到期持仓
      2. 检查NOOP: 如果今日在noop_dates中 → 不买入, 继续下一日
      3. 买入: 按跳空%降序选股, 每仓=总资产/max_positions
      4. 卖出: 持仓hold_days天后以收盘价卖出
    
    参数:
      signals_df: DataFrame(code, gap_pct, sell_ret, ...) ZP信号
      trading_days: list 交易日列表 (pd.Timestamp)
      noop_dates: set NOOP日期集合 (字符串 "YYYY-MM-DD")
      initial_capital: 初始资金
      max_positions: 最大持仓数
      hold_days: 持仓天数
      name: 策略名称(用于输出)
    
    返回:
      result: dict 终值/收益/笔数/胜率/回撤等
      equity_df: DataFrame 每日权益曲线
      trades: DataFrame 交易明细
    """
    noop_dates = noop_dates or set()

    # 按日分组 + 跳空%降序排序 (ZP铁律)
    signals_df = signals_df.copy()
    signals_df['_date'] = signals_df.index.normalize()
    signals_df = signals_df.sort_values(['_date', 'gap_pct'], ascending=[True, False])

    # 交易日索引
    day_to_idx = {d: i for i, d in enumerate(trading_days)}

    portfolio = initial_capital
    positions = []  # [{exit_idx, capital, sell_ret_pct, code, gap_pct}]
    equity_curve = []
    trade_log = []

    for i, today in enumerate(trading_days):
        # 1. 平仓到期持仓
        new_positions = []
        for pos in positions:
            if pos['exit_idx'] == i:
                portfolio += pos['capital'] * (1 + pos['sell_ret_pct'])
            else:
                new_positions.append(pos)
        positions = new_positions

        # 2. 可用仓位
        available = max_positions - len(positions)
        current_equity = portfolio + sum(p['capital'] for p in positions)
        
        if available <= 0:
            equity_curve.append({'date': today, 'equity': current_equity})
            continue

        # 3. NOOP检查 (核心创新: 模型判断今天不适合交易就跳过)
        today_str = today.strftime('%Y-%m-%d')
        if today_str in noop_dates:
            equity_curve.append({'date': today, 'equity': current_equity})
            continue

        # 4. 当日信号
        day_sigs = signals_df[signals_df['_date'] == today]
        if len(day_sigs) == 0:
            equity_curve.append({'date': today, 'equity': current_equity})
            continue

        # 5. 买入 (跳空%降序已排好)
        day_sigs = day_sigs.head(available)
        alloc = portfolio / max_positions

        for _, sig in day_sigs.iterrows():
            if portfolio < alloc:
                break
            portfolio -= alloc
            exit_idx = min(i + hold_days, len(trading_days) - 1)
            positions.append({
                'exit_idx': exit_idx,
                'capital': alloc,
                'sell_ret_pct': sig['sell_ret'] - 1.0,
                'code': sig['code'],
                'gap_pct': sig['gap_pct'],
            })
            trade_log.append({
                'date': today_str,
                'code': sig['code'],
                'gap_pct': sig['gap_pct'],
                'sell_ret_pct': sig['sell_ret'] - 1.0,
                'capital': alloc,
            })

        equity_curve.append({
            'date': today,
            'equity': portfolio + sum(p['capital'] * (1 + p['sell_ret_pct']) for p in positions)
        })

    # 清算剩余持仓
    for pos in positions:
        portfolio += pos['capital'] * (1 + pos['sell_ret_pct'])
    positions = []

    # 统计
    trades = pd.DataFrame(trade_log)
    n_trades = len(trades)
    if n_trades > 0:
        win_rate = (trades['sell_ret_pct'] > 0).mean()
        avg_ret = trades['sell_ret_pct'].mean()
        total_pnl = sum(t['capital'] * t['sell_ret_pct'] for t in trade_log)
    else:
        win_rate = 0
        avg_ret = 0
        total_pnl = 0

    equity_df = pd.DataFrame(equity_curve)
    if len(equity_df) > 1:
        peak = equity_df['equity'].cummax()
        dd = (equity_df['equity'] - peak) / peak
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    total_return = (portfolio / initial_capital - 1) * 100
    calmar = abs(total_return / (max_dd * 100)) if max_dd != 0 else 0

    result = {
        'name': name,
        'final_equity': round(portfolio, 2),
        'total_return': round(total_return, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate * 100, 2),
        'avg_ret_per_trade': round(avg_ret * 100, 2),
        'total_pnl': round(total_pnl, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 2),
    }

    print(f"  [{name}]")
    print(f"    终值: {result['final_equity']:,.0f} (收益 {result['total_return']:+.2f}%)")
    print(f"    交易: {result['n_trades']}笔, 胜率={result['win_rate']:.1f}%, "
          f"单笔均值={result['avg_ret_per_trade']:+.2f}%")
    print(f"    总盈亏: {result['total_pnl']:,.0f}, "
          f"最大回撤: {result['max_drawdown']:.1f}%, Calmar={result['calmar']:.2f}")

    return result, equity_df, trades


def generate_noop_dates(test_dates, probs, threshold=0.45):
    """
    根据模型预测概率生成NOOP日期集合
    
    规则: prob < threshold → 该日NOOP (不买入)
    
    参数:
      test_dates: list 测试期日期 ["YYYY-MM-DD", ...]
      probs: np.array 模型预测概率 (好天的概率)
      threshold: NOOP阈值 (概率低于此值则跳过交易)
    
    返回:
      noop_dates: set NOOP日期集合
      noop_count: int NOOP天数
      noop_pct: float NOOP比例
    """
    noop_dates = set(d for d, p in zip(test_dates, probs) if p < threshold)
    noop_pct = len(noop_dates) / len(test_dates) * 100 if len(test_dates) > 0 else 0
    return noop_dates, len(noop_dates), round(noop_pct, 1)


def generate_ensemble_noop(test_dates, probs_dict, thresholds_dict,
                           mode='AND'):
    """
    集成多模型NOOP (AND门控或OR门控)
    
    AND门控: 任一模型说NOOP → 跳过 (保守, 交易更少, 质量更高)
    OR门控: 所有模型都NOOP → 才跳过 (激进, 交易更多)
    
    参数:
      test_dates: list 测试期日期
      probs_dict: dict {model_name: probabilities}
      thresholds_dict: dict {model_name: threshold}
      mode: 'AND' 或 'OR'
    
    返回:
      noop_dates: set NOOP日期集合
    """
    noop_sets = []
    for model_name, probs in probs_dict.items():
        th = thresholds_dict[model_name]
        noop_set = set(d for d, p in zip(test_dates, probs) if p < th)
        noop_sets.append(noop_set)
    
    if mode == 'AND':
        # Union: 任一模型NOOP → 跳过
        noop_dates = set().union(*noop_sets) if noop_sets else set()
    else:
        # Intersection: 所有模型NOOP → 才跳过
        noop_dates = set.intersection(*noop_sets) if noop_sets else set()
    
    return noop_dates
