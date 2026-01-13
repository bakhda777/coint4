#!/usr/bin/env python3
"""Test PnL attribution analysis."""

import pytest
import pandas as pd
import numpy as np

pytest.importorskip(
    "coint2.analytics.pnl_attribution",
    reason="coint2.analytics is not available in this codebase",
)

from coint2.analytics.pnl_attribution import (
    calculate_pnl_attribution,
    generate_pnl_attribution_report,
    analyze_pnl_by_period
)


def test_calculate_pnl_attribution_empty():
    """Test PnL attribution with empty trades."""
    trades = pd.DataFrame()
    prices = pd.DataFrame()
    
    result = calculate_pnl_attribution(trades, prices)
    
    assert result['total_pnl'] == 0
    assert result['signal_pnl'] == 0
    assert result['num_trades'] == 0


def test_calculate_pnl_attribution_single_trade():
    """Test PnL attribution with single trade."""
    trades = pd.DataFrame([{
        'entry_price': 100,
        'exit_price': 105,
        'size': 1,
        'side': 1,  # Long
        'entry_time': pd.Timestamp('2024-01-01'),
        'exit_time': pd.Timestamp('2024-01-02')
    }])
    
    prices = pd.DataFrame()
    
    result = calculate_pnl_attribution(
        trades, prices,
        commission_pct=0.001,
        slippage_pct=0.0005
    )
    
    # Signal PnL should be 5 (105 - 100)
    assert result['signal_pnl'] == 5.0
    
    # Commission on both legs: 1 * (100 + 105) * 0.001 = 0.205
    assert abs(result['commission_cost'] - 0.205) < 0.01
    
    # Slippage on both legs: 1 * (100 + 105) * 0.0005 = 0.1025
    assert abs(result['slippage_cost'] - 0.1025) < 0.01
    
    # Total PnL = Signal - Costs
    expected_total = 5.0 - 0.205 - 0.1025
    assert abs(result['total_pnl'] - expected_total) < 0.01


def test_calculate_pnl_attribution_multiple_trades():
    """Test PnL attribution with multiple trades."""
    trades = pd.DataFrame([
        {
            'entry_price': 100,
            'exit_price': 105,
            'size': 1,
            'side': 1,  # Long
        },
        {
            'entry_price': 105,
            'exit_price': 103,
            'size': 1,
            'side': -1,  # Short
        },
        {
            'entry_price': 103,
            'exit_price': 107,
            'size': 2,
            'side': 1,  # Long
        }
    ])
    
    prices = pd.DataFrame()
    
    result = calculate_pnl_attribution(trades, prices)
    
    # Calculate expected signal PnL
    # Trade 1: 1 * (105 - 100) = 5
    # Trade 2: -1 * (103 - 105) = 2
    # Trade 3: 1 * 2 * (107 - 103) = 8
    expected_signal = 5 + 2 + 8
    assert abs(result['signal_pnl'] - expected_signal) < 0.1
    
    assert result['num_trades'] == 3
    assert result['commission_cost'] > 0
    assert result['slippage_cost'] > 0


def test_generate_pnl_attribution_report():
    """Test PnL attribution report generation."""
    attribution = {
        'total_pnl': 1000,
        'signal_pnl': 1500,
        'commission_cost': 300,
        'slippage_cost': 150,
        'latency_cost': 50,
        'rebalance_cost': 0,
        'num_trades': 50,
        'commission_pct_of_signal': 20.0,
        'slippage_pct_of_signal': 10.0,
        'latency_pct_of_signal': 3.3,
        'cost_ratio': 0.33
    }
    
    report = generate_pnl_attribution_report(attribution)
    
    assert '# PnL Attribution Report' in report
    assert 'Total PnL' in report and '1,000' in report
    assert 'Signal PnL' in report and '1,500' in report
    assert 'Number of Trades' in report and '50' in report
    assert 'Commission | 300.00' in report
    assert 'PnL Waterfall' in report


def test_generate_pnl_attribution_report_high_costs():
    """Test report with high cost warning."""
    attribution = {
        'total_pnl': -100,
        'signal_pnl': 500,
        'commission_cost': 400,
        'slippage_cost': 150,
        'latency_cost': 50,
        'rebalance_cost': 0,
        'num_trades': 10,
        'commission_pct_of_signal': 80.0,
        'slippage_pct_of_signal': 30.0,
        'latency_pct_of_signal': 10.0,
        'cost_ratio': 1.2
    }
    
    report = generate_pnl_attribution_report(attribution)
    
    assert 'High cost ratio' in report
    assert 'Positive signal PnL turned negative' in report


def test_analyze_pnl_by_period_empty():
    """Test period analysis with empty trades."""
    trades = pd.DataFrame()
    
    result = analyze_pnl_by_period(trades, 'D')
    
    assert result.empty


def test_analyze_pnl_by_period_daily():
    """Test daily PnL analysis."""
    trades = pd.DataFrame([
        {
            'exit_time': pd.Timestamp('2024-01-01'),
            'pnl': 100
        },
        {
            'exit_time': pd.Timestamp('2024-01-01'),
            'pnl': -50
        },
        {
            'exit_time': pd.Timestamp('2024-01-02'),
            'pnl': 200
        }
    ])
    
    result = analyze_pnl_by_period(trades, 'D')
    
    assert len(result) == 2
    assert result.iloc[0]['pnl'] == 50  # 100 - 50
    assert result.iloc[0]['num_trades'] == 2
    assert result.iloc[1]['pnl'] == 200
    assert result.iloc[1]['num_trades'] == 1


def test_analyze_pnl_by_period_weekly():
    """Test weekly PnL analysis."""
    trades = pd.DataFrame([
        {
            'exit_time': pd.Timestamp('2024-01-01'),
            'pnl': 100
        },
        {
            'exit_time': pd.Timestamp('2024-01-05'),
            'pnl': 200
        },
        {
            'exit_time': pd.Timestamp('2024-01-08'),
            'pnl': 150
        }
    ])
    
    result = analyze_pnl_by_period(trades, 'W')
    
    assert len(result) == 2
    # Week 1: 100 + 200 = 300
    assert result.iloc[0]['pnl'] == 300
    assert result.iloc[0]['num_trades'] == 2
    # Week 2: 150
    assert result.iloc[1]['pnl'] == 150
    assert result.iloc[1]['num_trades'] == 1
