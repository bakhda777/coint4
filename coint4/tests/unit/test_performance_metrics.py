"""Tests for performance metrics."""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Direct import to avoid __init__ issues
import importlib.util
spec = importlib.util.spec_from_file_location("performance", "src/coint2/core/performance.py")
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)

deflated_sharpe_ratio = performance.deflated_sharpe_ratio
probabilistic_sharpe_ratio = performance.probabilistic_sharpe_ratio


class TestDeflatedSharpeRatio:
    """Test deflated Sharpe ratio calculation."""
    
    def test_deflated_sharpe_basic(self):
        """Test basic DSR calculation."""
        returns = np.random.normal(0.001, 0.01, 252)  # Daily returns for 1 year
        dsr = deflated_sharpe_ratio(returns, trials=100)
        assert isinstance(dsr, float)
        assert -50 < dsr < 50  # Wider bounds for DSR
    
    def test_deflated_sharpe_zero_returns(self):
        """Test DSR with zero returns."""
        returns = np.zeros(100)
        dsr = deflated_sharpe_ratio(returns, trials=10)
        assert dsr < 0  # Should be negative for zero returns
    
    def test_deflated_sharpe_high_trials(self):
        """Test DSR with many trials."""
        returns = np.random.normal(0.002, 0.01, 252)
        dsr_10 = deflated_sharpe_ratio(returns, trials=10)
        dsr_1000 = deflated_sharpe_ratio(returns, trials=1000)
        assert dsr_1000 < dsr_10  # More trials should deflate more
    
    def test_deflated_sharpe_empty_returns(self):
        """Test DSR with empty returns."""
        returns = np.array([])
        dsr = deflated_sharpe_ratio(returns, trials=10)
        assert dsr == 0.0


class TestProbabilisticSharpeRatio:
    """Test probabilistic Sharpe ratio calculation."""
    
    def test_probabilistic_sharpe_basic(self):
        """Test basic PSR calculation."""
        returns = np.random.normal(0.001, 0.01, 252)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.5)
        assert isinstance(psr, float)
        assert 0 <= psr <= 1  # Probability bounds
    
    def test_probabilistic_sharpe_high_benchmark(self):
        """Test PSR with high benchmark."""
        returns = np.random.normal(0.001, 0.01, 252)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=3.0)
        assert psr < 0.5  # Should be low probability
    
    def test_probabilistic_sharpe_negative_benchmark(self):
        """Test PSR with negative benchmark."""
        returns = np.random.normal(0.001, 0.01, 252)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=-1.0)
        assert psr > 0.5  # Should be high probability
    
    def test_probabilistic_sharpe_empty_returns(self):
        """Test PSR with empty returns."""
        returns = np.array([])
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.5)
        assert psr == 0.0
    
    def test_probabilistic_sharpe_with_skew_kurt(self):
        """Test PSR with skewness and kurtosis."""
        returns = np.random.normal(0.001, 0.01, 252)
        psr_normal = probabilistic_sharpe_ratio(returns, benchmark_sr=0.5, skew=0.0, kurt=3.0)
        psr_skewed = probabilistic_sharpe_ratio(returns, benchmark_sr=0.5, skew=0.5, kurt=4.0)
        assert isinstance(psr_normal, float)
        assert isinstance(psr_skewed, float)
        # Both should be valid probabilities
        assert 0 <= psr_normal <= 1
        assert 0 <= psr_skewed <= 1