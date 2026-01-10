#!/usr/bin/env python3
"""PnL attribution analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


def calculate_pnl_attribution(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    latency_bars: int = 1
) -> Dict[str, Any]:
    """
    Decompose PnL into signal, costs, and other components.
    
    Args:
        trades: DataFrame with trade records
        prices: DataFrame with price data
        commission_pct: Commission percentage
        slippage_pct: Slippage percentage
        latency_bars: Execution latency in bars
        
    Returns:
        Dict with PnL attribution components
    """
    
    if trades.empty:
        return {
            'total_pnl': 0,
            'signal_pnl': 0,
            'commission_cost': 0,
            'slippage_cost': 0,
            'latency_cost': 0,
            'rebalance_cost': 0,
            'num_trades': 0
        }
    
    # Calculate raw signal PnL (no costs)
    signal_pnl = 0
    commission_cost = 0
    slippage_cost = 0
    latency_cost = 0
    
    for _, trade in trades.iterrows():
        # Get entry and exit prices
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        size = abs(trade.get('size', 0))
        side = trade.get('side', 1)  # 1 for long, -1 for short
        
        if entry_price > 0 and exit_price > 0:
            # Raw PnL
            raw_pnl = side * size * (exit_price - entry_price)
            signal_pnl += raw_pnl
            
            # Commission cost (both legs)
            commission = size * (entry_price + exit_price) * commission_pct
            commission_cost += commission
            
            # Slippage cost (both legs)
            slippage = size * (entry_price + exit_price) * slippage_pct
            slippage_cost += slippage
            
            # Latency cost (approximation)
            if 'entry_time' in trade and 'exit_time' in trade:
                # Estimate price movement during latency
                try:
                    entry_idx = prices.index.get_loc(trade['entry_time'])
                    if entry_idx + latency_bars < len(prices):
                        latency_price = prices.iloc[entry_idx + latency_bars].get('close', entry_price)
                        latency_impact = abs(latency_price - entry_price) / entry_price
                        latency_cost += size * entry_price * latency_impact
                except:
                    pass
    
    # Calculate rebalance cost (turnover)
    if 'turnover' in trades.columns:
        avg_turnover = trades['turnover'].mean()
        rebalance_cost = avg_turnover * commission_pct * len(trades)
    else:
        rebalance_cost = 0
    
    # Total PnL
    total_pnl = signal_pnl - commission_cost - slippage_cost - latency_cost - rebalance_cost
    
    # Calculate percentages
    if abs(signal_pnl) > 0:
        commission_pct_of_signal = (commission_cost / abs(signal_pnl)) * 100
        slippage_pct_of_signal = (slippage_cost / abs(signal_pnl)) * 100
        latency_pct_of_signal = (latency_cost / abs(signal_pnl)) * 100
    else:
        commission_pct_of_signal = 0
        slippage_pct_of_signal = 0
        latency_pct_of_signal = 0
    
    return {
        'total_pnl': total_pnl,
        'signal_pnl': signal_pnl,
        'commission_cost': commission_cost,
        'slippage_cost': slippage_cost,
        'latency_cost': latency_cost,
        'rebalance_cost': rebalance_cost,
        'num_trades': len(trades),
        'commission_pct_of_signal': commission_pct_of_signal,
        'slippage_pct_of_signal': slippage_pct_of_signal,
        'latency_pct_of_signal': latency_pct_of_signal,
        'cost_ratio': (commission_cost + slippage_cost + latency_cost) / abs(signal_pnl) if signal_pnl != 0 else 0
    }


def generate_pnl_attribution_report(
    attribution: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """Generate PnL attribution report."""
    
    report = """# PnL Attribution Report

## Summary

"""
    
    # Overall metrics
    report += f"- **Total PnL**: ${attribution['total_pnl']:,.2f}\n"
    report += f"- **Signal PnL**: ${attribution['signal_pnl']:,.2f}\n"
    report += f"- **Number of Trades**: {attribution['num_trades']}\n\n"
    
    # Cost breakdown
    report += "## Cost Breakdown\n\n"
    report += "| Component | Amount ($) | % of Signal PnL |\n"
    report += "|-----------|------------|----------------|\n"
    report += f"| Commission | {attribution['commission_cost']:,.2f} | {attribution['commission_pct_of_signal']:.1f}% |\n"
    report += f"| Slippage | {attribution['slippage_cost']:,.2f} | {attribution['slippage_pct_of_signal']:.1f}% |\n"
    report += f"| Latency | {attribution['latency_cost']:,.2f} | {attribution['latency_pct_of_signal']:.1f}% |\n"
    report += f"| Rebalance | {attribution['rebalance_cost']:,.2f} | - |\n"
    
    total_costs = (attribution['commission_cost'] + attribution['slippage_cost'] + 
                   attribution['latency_cost'] + attribution['rebalance_cost'])
    
    report += f"| **Total Costs** | **{total_costs:,.2f}** | **{(total_costs/abs(attribution['signal_pnl'])*100) if attribution['signal_pnl'] != 0 else 0:.1f}%** |\n\n"
    
    # PnL waterfall
    report += "## PnL Waterfall\n\n"
    report += "```\n"
    report += f"Signal PnL:        ${attribution['signal_pnl']:>12,.2f}\n"
    report += f"- Commission:      ${-attribution['commission_cost']:>12,.2f}\n"
    report += f"- Slippage:        ${-attribution['slippage_cost']:>12,.2f}\n"
    report += f"- Latency:         ${-attribution['latency_cost']:>12,.2f}\n"
    report += f"- Rebalance:       ${-attribution['rebalance_cost']:>12,.2f}\n"
    report += f"{'='*40}\n"
    report += f"Total PnL:         ${attribution['total_pnl']:>12,.2f}\n"
    report += "```\n\n"
    
    # Key insights
    report += "## Key Insights\n\n"
    
    cost_ratio = attribution.get('cost_ratio', 0)
    if cost_ratio > 0.5:
        report += f"⚠️ **High cost ratio**: {cost_ratio:.1%} of signal PnL consumed by costs\n"
    elif cost_ratio > 0.3:
        report += f"⚡ **Moderate costs**: {cost_ratio:.1%} of signal PnL consumed by costs\n"
    else:
        report += f"✅ **Low costs**: {cost_ratio:.1%} of signal PnL consumed by costs\n"
    
    if attribution['num_trades'] > 0:
        avg_pnl_per_trade = attribution['total_pnl'] / attribution['num_trades']
        report += f"- Average PnL per trade: ${avg_pnl_per_trade:.2f}\n"
    
    if attribution['signal_pnl'] > 0 and attribution['total_pnl'] < 0:
        report += "- ⚠️ Positive signal PnL turned negative by costs\n"
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def analyze_pnl_by_period(
    trades: pd.DataFrame,
    period: str = 'D'
) -> pd.DataFrame:
    """
    Analyze PnL by time period.
    
    Args:
        trades: DataFrame with trades
        period: Period for grouping ('D', 'W', 'M')
        
    Returns:
        DataFrame with period-wise PnL
    """
    if trades.empty:
        return pd.DataFrame()
    
    # Group by period
    trades['period'] = pd.to_datetime(trades.get('exit_time', trades.index))
    trades_grouped = trades.set_index('period').resample(period)
    
    results = []
    for period_name, group in trades_grouped:
        if len(group) > 0:
            period_pnl = group['pnl'].sum() if 'pnl' in group.columns else 0
            results.append({
                'period': period_name,
                'pnl': period_pnl,
                'num_trades': len(group),
                'win_rate': (group['pnl'] > 0).mean() if 'pnl' in group.columns else 0
            })
    
    return pd.DataFrame(results)