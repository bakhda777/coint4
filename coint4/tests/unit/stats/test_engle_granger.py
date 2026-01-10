"""Unit tests for Engle-Granger cointegration test."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from coint2.pipeline.pair_scanner import test_cointegration, count_mean_crossings


class TestEngleGranger:
    """Test Engle-Granger cointegration method."""
    
    def test_cointegrated_series(self):
        """Test with truly cointegrated series."""
        np.random.seed(42)
        n = 500
        
        # Generate cointegrated series
        # y = beta * x + stationary_error
        x = np.cumsum(np.random.randn(n))  # Random walk
        beta = 2.0
        error = np.random.randn(n) * 0.5  # Stationary error
        y = beta * x + error
        
        config = {
            'criteria': {
                'coint_pvalue_max': 0.05,
                'hl_min': 5,
                'hl_max': 200,
                'hurst_min': 0.2,
                'hurst_max': 0.6,
                'min_cross': 10,
                'beta_drift_max': 0.15
            }
        }
        
        result = test_cointegration(y, x, config)
        
        # Should detect cointegration
        assert result['pvalue'] < 0.05
        assert abs(result['beta'] - beta) / beta < 0.1  # Beta close to true value
    
    def test_non_cointegrated_series(self):
        """Test with non-cointegrated series."""
        np.random.seed(42)
        n = 500
        
        # Two independent random walks
        x = np.cumsum(np.random.randn(n))
        y = np.cumsum(np.random.randn(n))
        
        config = {'criteria': {}}
        
        result = test_cointegration(y, x, config)
        
        # Should not detect cointegration
        assert result['pvalue'] > 0.05
    
    def test_beta_stability(self):
        """Test beta stability calculation."""
        np.random.seed(42)
        n = 500
        
        # Series with changing beta
        x = np.cumsum(np.random.randn(n))
        beta1 = 2.0
        beta2 = 2.5
        
        # First half with beta1, second half with beta2
        y = np.zeros(n)
        y[:n//2] = beta1 * x[:n//2] + np.random.randn(n//2) * 0.1
        y[n//2:] = beta2 * x[n//2:] + np.random.randn(n - n//2) * 0.1
        
        config = {'criteria': {}}
        
        result = test_cointegration(y, x, config)
        
        # Should detect beta drift
        expected_drift = abs(beta2 - beta1) / beta1
        assert abs(result['beta_drift'] - expected_drift) < 0.1
    
    def test_mean_crossings(self):
        """Test mean crossing counting."""
        # Oscillating series
        spread = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        
        crossings = count_mean_crossings(spread)
        
        # Should count 7 crossings
        assert crossings == 7
    
    def test_mean_crossings_no_cross(self):
        """Test mean crossings with no crossings."""
        # Always above mean
        spread = np.array([2, 3, 4, 5, 6, 7, 8, 9])
        
        crossings = count_mean_crossings(spread)
        
        # Should count 0 crossings
        assert crossings == 0