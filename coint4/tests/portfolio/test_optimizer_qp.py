#!/usr/bin/env python3
"""Unit tests for portfolio optimizer quadratic programming."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from coint2.portfolio.optimizer import PortfolioOptimizer, PortfolioConfig


@pytest.fixture
def simple_metrics():
    """Simple 2-asset metrics for closed-form validation."""
    return pd.DataFrame({
        'exp_return': [0.10, 0.08],
        'vol': [0.20, 0.15],
        'psr': [0.8, 0.6],
        'est_fee_per_turnover': [0.001, 0.001],
        'est_slippage_per_turnover': [0.0005, 0.0005],
        'turnover_baseline': [0.1, 0.1],
        'adv_proxy': [1000000, 800000],
        'cap_per_pair': [0.15, 0.15]
    }, index=['PAIR_A', 'PAIR_B'])


@pytest.fixture
def default_config():
    """Default portfolio configuration."""
    return PortfolioConfig(
        method="score_topN",
        top_n=2,
        min_pairs=2,
        lambda_var=0.0,  # No risk aversion for simple test
        gamma_cost=0.0,  # No cost penalty for simple test
        max_gross=1.0,
        net_target=0.0,
        max_weight_per_pair=0.6,
        seed=42
    )


def test_portfolio_config_creation():
    """Test PortfolioConfig creation and defaults."""
    config = PortfolioConfig()
    
    assert config.method == "score_topN"
    assert config.top_n == 12
    assert config.min_pairs == 5
    assert config.lambda_var == 2.0
    assert config.gamma_cost == 1.0
    assert config.max_gross == 1.0
    assert config.net_target == 0.0


def test_pair_selection_scoring(simple_metrics, default_config):
    """Test pair selection with scoring method."""
    optimizer = PortfolioOptimizer(default_config)
    
    # Test scoring
    selected = optimizer._select_pairs(simple_metrics)
    
    assert len(selected) == 2
    assert 'PAIR_A' in selected  # Should have higher PSR
    assert 'PAIR_B' in selected


def test_portfolio_optimization_simple(simple_metrics, default_config):
    """Test portfolio optimization with simple setup."""
    
    # Use numpy fallback for deterministic test
    optimizer = PortfolioOptimizer(default_config)
    
    result = optimizer.optimize_portfolio(simple_metrics)
    
    assert result.success
    assert len(result.weights) == 2
    assert result.method_used in ['numpy_fallback', 'cvxpy']
    
    # Check constraints
    gross_exposure = result.weights.abs().sum()
    net_exposure = result.weights.sum()
    max_weight = result.weights.abs().max()
    
    assert gross_exposure <= default_config.max_gross + 1e-6
    assert abs(net_exposure - default_config.net_target) < 1e-6
    assert max_weight <= default_config.max_weight_per_pair + 1e-6


def test_capacity_constraints(simple_metrics):
    """Test capacity constraint enforcement."""
    
    # Set very restrictive ADV constraint
    config = PortfolioConfig(
        max_adv_pct=0.001,  # Very small ADV allowance
        max_weight_per_pair=0.5
    )
    
    optimizer = PortfolioOptimizer(config)
    result = optimizer.optimize_portfolio(simple_metrics)
    
    assert result.success
    
    # Check that capacity analysis was performed
    if 'capacity_analysis' in result.diagnostics:
        capacity = result.diagnostics['capacity_analysis']
        assert 'capacity_warnings' in capacity


def test_base_currency_diversification():
    """Test base currency diversification logic."""
    
    metrics = pd.DataFrame({
        'exp_return': [0.10, 0.09, 0.08, 0.07],
        'vol': [0.20, 0.18, 0.15, 0.17],
        'psr': [0.8, 0.7, 0.6, 0.5],
        'est_fee_per_turnover': [0.001] * 4,
        'est_slippage_per_turnover': [0.0005] * 4,
        'turnover_baseline': [0.1] * 4,
        'adv_proxy': [1000000] * 4,
        'cap_per_pair': [0.15] * 4
    }, index=['BTC/USDT', 'BTC/ETH', 'ETH/USDT', 'ETH/ADA'])
    
    config = PortfolioConfig(
        diversify_by_base=True,
        max_per_base=1,
        top_n=4
    )
    
    optimizer = PortfolioOptimizer(config)
    selected = optimizer._apply_base_diversification(
        metrics.sort_values('psr', ascending=False)
    )
    
    # Should limit to 1 pair per base currency
    bases = set()
    for pair in selected:
        if '/' in pair:
            base = pair.split('/')[0]
        else:
            base = pair[:3]
        bases.add(base)
    
    # Each base should appear at most once
    assert len(selected) <= len(bases)


def test_fallback_behavior():
    """Test fallback when optimization fails."""
    
    # Create problematic metrics that might cause solver issues
    bad_metrics = pd.DataFrame({
        'exp_return': [np.inf, -np.inf],  # Bad values
        'vol': [0, np.nan],
        'psr': [np.nan, np.nan],
        'est_fee_per_turnover': [0.001, 0.001],
        'est_slippage_per_turnover': [0.0005, 0.0005],
        'turnover_baseline': [0.1, 0.1],
        'adv_proxy': [1000000, 800000],
        'cap_per_pair': [0.15, 0.15]
    }, index=['BAD_A', 'BAD_B'])
    
    config = PortfolioConfig(fallback="vol_target")
    optimizer = PortfolioOptimizer(config)
    
    result = optimizer.optimize_portfolio(bad_metrics)
    
    # Should fallback gracefully
    assert result.success
    assert result.method_used.startswith('fallback')


def test_empty_selection_fallback():
    """Test fallback when too few pairs selected."""
    
    metrics = pd.DataFrame({
        'exp_return': [0.01],  # Very low returns
        'vol': [0.50],         # High volatility
        'psr': [-0.5],         # Negative PSR
        'est_fee_per_turnover': [0.01],  # High fees
        'est_slippage_per_turnover': [0.01],
        'turnover_baseline': [0.5],
        'adv_proxy': [100],    # Low ADV
        'cap_per_pair': [0.05]
    }, index=['BAD_PAIR'])
    
    config = PortfolioConfig(min_pairs=5)  # Require more pairs than available
    optimizer = PortfolioOptimizer(config)
    
    result = optimizer.optimize_portfolio(metrics)
    
    assert result.success
    assert result.method_used.startswith('fallback')


def test_diagnostics_content():
    """Test that diagnostics contain expected keys."""
    
    metrics = pd.DataFrame({
        'exp_return': [0.08, 0.06],
        'vol': [0.15, 0.12],
        'psr': [0.6, 0.4],
        'est_fee_per_turnover': [0.001, 0.001],
        'est_slippage_per_turnover': [0.0005, 0.0005],
        'turnover_baseline': [0.1, 0.1],
        'adv_proxy': [1000000, 800000],
        'cap_per_pair': [0.15, 0.15]
    }, index=['PAIR_X', 'PAIR_Y'])
    
    config = PortfolioConfig()
    optimizer = PortfolioOptimizer(config)
    
    result = optimizer.optimize_portfolio(metrics)
    
    # Check required diagnostic keys
    required_keys = [
        'gross_exposure', 'net_exposure', 'max_weight', 'active_pairs'
    ]
    
    for key in required_keys:
        assert key in result.diagnostics, f"Missing diagnostic key: {key}"
    
    # Check value ranges
    assert 0 <= result.diagnostics['gross_exposure'] <= 1.01
    assert -1.01 <= result.diagnostics['net_exposure'] <= 1.01
    assert 0 <= result.diagnostics['max_weight'] <= 1.01
    assert 0 <= result.diagnostics['active_pairs'] <= len(metrics)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])