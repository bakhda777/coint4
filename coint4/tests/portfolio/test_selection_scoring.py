#!/usr/bin/env python3
"""Unit tests for pair selection and scoring algorithms."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from coint2.portfolio.optimizer import PortfolioOptimizer, PortfolioConfig


@pytest.fixture
def diverse_metrics():
    """Diverse pair metrics for selection testing."""
    return pd.DataFrame({
        'exp_return': [0.12, 0.10, 0.08, 0.09, 0.11, 0.07],
        'vol': [0.25, 0.20, 0.15, 0.18, 0.22, 0.12],
        'psr': [0.9, 0.8, 0.6, 0.7, 0.85, 0.4],
        'est_fee_per_turnover': [0.001, 0.002, 0.001, 0.0015, 0.001, 0.003],
        'est_slippage_per_turnover': [0.0005, 0.001, 0.0005, 0.0008, 0.0005, 0.002],
        'turnover_baseline': [0.08, 0.12, 0.10, 0.11, 0.09, 0.15],
        'adv_proxy': [2000000, 1500000, 1000000, 1200000, 1800000, 500000],
        'cap_per_pair': [0.15, 0.12, 0.18, 0.14, 0.16, 0.10]
    }, index=['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT', 'LINK/USDT'])


def test_score_calculation():
    """Test composite score calculation logic."""
    
    config = PortfolioConfig(
        alpha_fee=0.5,
        beta_slip=0.3,
        method="score_topN"
    )
    
    metrics = pd.DataFrame({
        'psr': [1.0, 0.8],
        'est_fee_per_turnover': [0.001, 0.002],
        'est_slippage_per_turnover': [0.0005, 0.001],
        'turnover_baseline': [0.1, 0.1]
    }, index=['HIGH_PSR', 'LOW_PSR'])
    
    optimizer = PortfolioOptimizer(config)
    
    # Manually calculate expected scores
    fee_cost_high = 0.001 * 0.1
    slip_cost_high = 0.0005 * 0.1
    expected_score_high = 1.0 - 0.5 * fee_cost_high - 0.3 * slip_cost_high
    
    fee_cost_low = 0.002 * 0.1
    slip_cost_low = 0.001 * 0.1
    expected_score_low = 0.8 - 0.5 * fee_cost_low - 0.3 * slip_cost_low
    
    # Get actual selection (should prefer high PSR)
    selected = optimizer._score_top_n(metrics)
    
    assert 'HIGH_PSR' in selected
    assert len(selected) <= config.top_n


def test_diversification_enforcement(diverse_metrics):
    """Test base currency diversification rules."""
    
    config = PortfolioConfig(
        diversify_by_base=True,
        max_per_base=2,
        top_n=4
    )
    
    optimizer = PortfolioOptimizer(config)
    selected = optimizer._select_pairs(diverse_metrics)
    
    # Count pairs per base currency
    base_counts = {}
    for pair in selected:
        if '/' in pair:
            base = pair.split('/')[0]
            base_counts[base] = base_counts.get(base, 0) + 1
    
    # No base should exceed max_per_base
    for base, count in base_counts.items():
        assert count <= config.max_per_base, f"Base {base} has {count} pairs, max allowed {config.max_per_base}"


def test_selection_stability():
    """Test that selection is stable with same inputs."""
    
    config = PortfolioConfig(seed=42, top_n=3)
    
    metrics = pd.DataFrame({
        'psr': [0.7, 0.8, 0.6, 0.9, 0.5],
        'est_fee_per_turnover': [0.001] * 5,
        'est_slippage_per_turnover': [0.0005] * 5,
        'turnover_baseline': [0.1] * 5
    }, index=['A', 'B', 'C', 'D', 'E'])
    
    optimizer1 = PortfolioOptimizer(config)
    optimizer2 = PortfolioOptimizer(config)
    
    selected1 = optimizer1._select_pairs(metrics)
    selected2 = optimizer2._select_pairs(metrics)
    
    assert selected1 == selected2, "Selection should be deterministic with same seed"


def test_top_n_enforcement(diverse_metrics):
    """Test that top_n limit is enforced."""
    
    config = PortfolioConfig(top_n=3, diversify_by_base=False)
    
    optimizer = PortfolioOptimizer(config)
    selected = optimizer._select_pairs(diverse_metrics)
    
    assert len(selected) <= config.top_n


def test_min_pairs_requirement():
    """Test minimum pairs requirement."""
    
    # Very restrictive selection that should trigger fallback
    single_pair = pd.DataFrame({
        'psr': [0.5],
        'est_fee_per_turnover': [0.001],
        'est_slippage_per_turnover': [0.0005],
        'turnover_baseline': [0.1]
    }, index=['ONLY_PAIR'])
    
    config = PortfolioConfig(min_pairs=3, top_n=1)
    
    optimizer = PortfolioOptimizer(config)
    result = optimizer.optimize_portfolio(single_pair)
    
    # Should trigger fallback due to insufficient pairs
    assert 'fallback' in result.method_used


def test_greedy_diversify_method():
    """Test greedy diversification method."""
    
    config = PortfolioConfig(
        method="greedy_diversify",
        top_n=4
    )
    
    optimizer = PortfolioOptimizer(config)
    selected = optimizer._select_pairs(diverse_metrics)
    
    # Should return some pairs (current implementation uses score_topN as fallback)
    assert len(selected) > 0
    assert len(selected) <= config.top_n


def test_scoring_with_missing_data():
    """Test scoring handles missing data gracefully."""
    
    incomplete_metrics = pd.DataFrame({
        'psr': [0.8, np.nan, 0.6],
        'est_fee_per_turnover': [0.001, 0.001, np.nan],
        'est_slippage_per_turnover': [0.0005, np.nan, 0.0005],
        'turnover_baseline': [0.1, 0.1, 0.1]
    }, index=['COMPLETE', 'PARTIAL', 'MOSTLY_COMPLETE'])
    
    config = PortfolioConfig(top_n=2)
    optimizer = PortfolioOptimizer(config)
    
    # Should handle NaN values without crashing
    selected = optimizer._select_pairs(incomplete_metrics)
    
    assert len(selected) > 0
    # Should prefer complete data
    assert 'COMPLETE' in selected


def test_base_currency_extraction():
    """Test base currency extraction logic."""
    
    config = PortfolioConfig(diversify_by_base=True, max_per_base=1)
    optimizer = PortfolioOptimizer(config)
    
    # Test different pair formats
    test_pairs = pd.DataFrame({
        'psr': [0.8, 0.7, 0.9, 0.6],
        'est_fee_per_turnover': [0.001] * 4,
        'est_slippage_per_turnover': [0.0005] * 4,
        'turnover_baseline': [0.1] * 4
    }, index=['BTC/USDT', 'BTCETH', 'ETH/BTC', 'ETHUSDT'])
    
    selected = optimizer._apply_base_diversification(
        test_pairs.sort_values('psr', ascending=False)
    )
    
    # Should extract base currencies correctly
    extracted_bases = set()
    for pair in selected:
        if '/' in pair:
            base = pair.split('/')[0]
        elif 'USDT' in pair:
            base = pair.replace('USDT', '')
        else:
            base = pair[:3]
        extracted_bases.add(base)
    
    # Should have diversification across bases
    assert len(extracted_bases) >= len(selected) // config.max_per_base


def test_selection_ordering():
    """Test that selection follows score ordering."""
    
    # Create clear ranking by PSR
    ranked_metrics = pd.DataFrame({
        'psr': [0.9, 0.8, 0.7, 0.6, 0.5],  # Descending order
        'est_fee_per_turnover': [0.001] * 5,  # Equal costs
        'est_slippage_per_turnover': [0.0005] * 5,
        'turnover_baseline': [0.1] * 5
    }, index=['BEST', 'SECOND', 'THIRD', 'FOURTH', 'WORST'])
    
    config = PortfolioConfig(
        top_n=3, 
        diversify_by_base=False,  # Disable diversification
        alpha_fee=0.0,  # No fee penalty
        beta_slip=0.0   # No slip penalty
    )
    
    optimizer = PortfolioOptimizer(config)
    selected = optimizer._score_top_n(ranked_metrics)
    
    # Should select top 3 by PSR
    expected_top3 = ['BEST', 'SECOND', 'THIRD']
    for pair in expected_top3:
        assert pair in selected, f"Expected {pair} in top 3 selection"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])