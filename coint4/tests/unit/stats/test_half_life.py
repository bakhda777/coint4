"""Unit tests for half-life estimation."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from coint2.pipeline.pair_scanner import estimate_half_life


class TestHalfLife:
    """Test half-life estimation."""
    
    def test_half_life_ar1_known(self):
        """Test half-life with known AR(1) process."""
        # Create AR(1) with known phi
        phi = -0.1  # Half-life = -log(2)/log(0.9) ≈ 6.58
        n = 1000
        
        # Generate AR(1) process
        np.random.seed(42)
        spread = np.zeros(n)
        spread[0] = np.random.randn()
        
        for i in range(1, n):
            spread[i] = (1 + phi) * spread[i-1] + np.random.randn() * 0.1
        
        # Estimate half-life
        hl = estimate_half_life(spread)
        
        # Should be close to theoretical value
        expected = -np.log(2) / np.log(1 + phi)
        assert abs(hl - expected) / expected < 0.5  # Within 50% tolerance
    
    def test_half_life_no_mean_reversion(self):
        """Test half-life for non-mean-reverting series."""
        # Random walk
        np.random.seed(42)
        spread = np.cumsum(np.random.randn(1000))
        
        hl = estimate_half_life(spread)
        
        # Should return very large value
        assert hl > 500 or np.isinf(hl)
    
    def test_half_life_perfect_mean_reversion(self):
        """Test half-life for perfect mean reversion."""
        # Oscillating series
        spread = np.array([1, -1, 1, -1, 1, -1] * 100)
        
        hl = estimate_half_life(spread)
        
        # Should return small value
        assert hl < 10
    
    def test_half_life_short_series(self):
        """Test half-life with insufficient data."""
        spread = np.array([1, 2])
        
        hl = estimate_half_life(spread)
        
        # Should return infinity
        assert np.isinf(hl)
    
    def test_half_life_constant_series(self):
        """Test half-life with constant series."""
        spread = np.ones(100)
        
        hl = estimate_half_life(spread)
        
        # Should return infinity (no variance)
        assert np.isinf(hl)