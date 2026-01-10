"""Unit tests for Hurst exponent calculation."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from coint2.pipeline.pair_scanner import hurst_exponent


class TestHurstExponent:
    """Test Hurst exponent calculation."""
    
    def test_hurst_random_walk(self):
        """Test Hurst for pure random walk."""
        np.random.seed(42)
        # Random walk should have Hurst ≈ 0.5
        series = np.cumsum(np.random.randn(1000))
        
        hurst = hurst_exponent(series)
        
        # Should be close to 0.5
        assert abs(hurst - 0.5) < 0.15
    
    def test_hurst_mean_reverting(self):
        """Test Hurst for mean-reverting series."""
        # AR(1) with strong mean reversion
        n = 1000
        phi = -0.5
        
        np.random.seed(42)
        series = np.zeros(n)
        series[0] = np.random.randn()
        
        for i in range(1, n):
            series[i] = (1 + phi) * series[i-1] + np.random.randn() * 0.1
        
        hurst = hurst_exponent(series)
        
        # Should be < 0.5 (mean-reverting)
        assert hurst < 0.5
    
    def test_hurst_trending(self):
        """Test Hurst for trending series."""
        # Series with trend
        t = np.linspace(0, 10, 1000)
        trend = t + np.random.randn(1000) * 0.1
        
        hurst = hurst_exponent(trend)
        
        # Should be > 0.5 (trending)
        assert hurst > 0.5
    
    def test_hurst_short_series(self):
        """Test Hurst with short series."""
        series = np.random.randn(10)
        
        hurst = hurst_exponent(series)
        
        # Should return default 0.5
        assert hurst == 0.5
    
    def test_hurst_bounds(self):
        """Test Hurst bounds."""
        np.random.seed(42)
        
        # Test various series
        for _ in range(10):
            series = np.random.randn(100)
            hurst = hurst_exponent(series)
            
            # Should be bounded [0, 1]
            assert 0.0 <= hurst <= 1.0