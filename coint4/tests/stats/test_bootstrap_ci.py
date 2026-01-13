#!/usr/bin/env python3
"""Unit tests for bootstrap confidence intervals."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

bootstrap_module = pytest.importorskip(
    "coint2.stats.bootstrap",
    reason="Legacy bootstrap module not available."
)
MetricsBootstrap = bootstrap_module.MetricsBootstrap


@pytest.fixture
def bootstrap_analyzer():
    """Bootstrap analyzer with small sample size for testing."""
    return MetricsBootstrap(
        block_size=5,
        n_bootstrap=20,  # Small for speed
        confidence_levels=[0.05, 0.50, 0.95],
        seed=42
    )


@pytest.fixture
def sample_returns():
    """Sample return series for testing."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 50)


def test_sharpe_calculation(bootstrap_analyzer, sample_returns):
    """Test Sharpe ratio calculation."""
    sharpe = bootstrap_analyzer.calculate_sharpe(sample_returns)
    
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)
    assert -5 < sharpe < 5  # Reasonable bounds


def test_psr_calculation(bootstrap_analyzer, sample_returns):
    """Test PSR calculation."""
    psr = bootstrap_analyzer.calculate_psr(sample_returns)
    
    assert isinstance(psr, float)
    assert 0 <= psr <= 1  # PSR is probability
    assert not np.isnan(psr)


def test_bootstrap_metrics(bootstrap_analyzer, sample_returns):
    """Test bootstrap metric distributions."""
    bootstrap_dist = bootstrap_analyzer.bootstrap_metrics(sample_returns)
    
    assert 'sharpe' in bootstrap_dist
    assert 'psr' in bootstrap_dist
    assert 'dsr' in bootstrap_dist
    
    # Check array lengths
    assert len(bootstrap_dist['sharpe']) == 20
    assert len(bootstrap_dist['psr']) == 20
    assert len(bootstrap_dist['dsr']) == 20


def test_confidence_intervals(bootstrap_analyzer, sample_returns):
    """Test confidence interval calculation."""
    ci = bootstrap_analyzer.calculate_confidence_intervals(sample_returns, 'sharpe')
    
    assert 'p05' in ci
    assert 'p50' in ci
    assert 'p95' in ci
    assert 'observed' in ci
    
    # Check monotonicity
    assert ci['p05'] <= ci['p50'] <= ci['p95']


def test_empty_returns():
    """Test handling of empty returns."""
    bootstrap = MetricsBootstrap(n_bootstrap=5)
    empty_returns = np.array([])
    
    bootstrap_dist = bootstrap.bootstrap_metrics(empty_returns)
    
    # Should return dummy distributions
    assert len(bootstrap_dist['sharpe']) == 5
    assert all(s == 0 for s in bootstrap_dist['sharpe'])


def test_constant_returns():
    """Test handling of constant returns."""
    bootstrap = MetricsBootstrap(n_bootstrap=5)
    constant_returns = np.ones(20) * 0.01
    
    sharpe = bootstrap.calculate_sharpe(constant_returns)
    assert sharpe == 0  # No volatility = zero Sharpe


@pytest.mark.smoke
def test_portfolio_uncertainty_analysis():
    """Smoke test for portfolio uncertainty analysis."""
    returns_data = pd.DataFrame({
        'PAIR_A': np.random.normal(0.001, 0.02, 30),
        'PAIR_B': np.random.normal(0.0005, 0.015, 30)
    })
    
    weights = {'PAIR_A': 0.6, 'PAIR_B': 0.4}
    
    bootstrap = MetricsBootstrap(n_bootstrap=10, seed=42)
    
    results = bootstrap.analyze_portfolio_uncertainty(returns_data, weights)
    
    assert 'PAIR_A' in results
    assert 'PAIR_B' in results
    assert 'PORTFOLIO' in results
    
    # Check portfolio has metrics
    portfolio_metrics = results['PORTFOLIO']
    assert 'sharpe' in portfolio_metrics
    assert 'psr' in portfolio_metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
