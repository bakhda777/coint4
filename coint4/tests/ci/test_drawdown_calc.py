"""Tests for drawdown calculation in CI gates."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.ci_gates import CIGateChecker


class TestDrawdownCalculation:
    """Test drawdown calculation function."""
    
    def test_flat_equity(self):
        """Test drawdown on flat equity curve."""
        checker = CIGateChecker(verbose=False)
        equity = np.array([100, 100, 100, 100, 100])
        dd = checker.calc_drawdown_pct(equity)
        assert dd == 0.0, "Flat equity should have 0% drawdown"
    
    def test_monotonic_growth(self):
        """Test drawdown on monotonically increasing equity."""
        checker = CIGateChecker(verbose=False)
        equity = np.array([100, 110, 120, 130, 140])
        dd = checker.calc_drawdown_pct(equity)
        assert dd == 0.0, "Monotonic growth should have 0% drawdown"
    
    def test_single_drawdown(self):
        """Test single drawdown and recovery."""
        checker = CIGateChecker(verbose=False)
        equity = np.array([100, 110, 120, 90, 100, 110])
        dd = checker.calc_drawdown_pct(equity)
        # Drawdown from 120 to 90 = 30/120 = 25%
        assert abs(dd - 0.25) < 0.01, f"Expected 25% drawdown, got {dd:.2%}"
    
    def test_drawdown_no_recovery(self):
        """Test drawdown without recovery."""
        checker = CIGateChecker(verbose=False)
        equity = np.array([100, 120, 110, 90, 85])
        dd = checker.calc_drawdown_pct(equity)
        # Drawdown from 120 to 85 = 35/120 = 29.17%
        assert abs(dd - 0.2917) < 0.01, f"Expected 29.17% drawdown, got {dd:.2%}"
    
    def test_multiple_drawdowns(self):
        """Test multiple drawdowns."""
        checker = CIGateChecker(verbose=False)
        equity = np.array([100, 120, 90, 110, 80, 100])
        dd = checker.calc_drawdown_pct(equity)
        # Maximum drawdown from 120 to 80 = 40/120 = 33.33%
        assert abs(dd - 0.3333) < 0.01, f"Expected 33.33% drawdown, got {dd:.2%}"
    
    def test_negative_start(self):
        """Test equity starting with negative value."""
        checker = CIGateChecker(verbose=False)
        equity = np.array([-10, 0, 100, 90, 110])
        dd = checker.calc_drawdown_pct(equity)
        # Should start from first positive (100), drawdown to 90 = 10%
        assert abs(dd - 0.10) < 0.01, f"Expected 10% drawdown, got {dd:.2%}"
    
    def test_short_equity(self):
        """Test very short equity curve."""
        checker = CIGateChecker(verbose=False)
        equity = np.array([100])
        dd = checker.calc_drawdown_pct(equity)
        assert dd == 0.0, "Single point should have 0% drawdown"
    
    def test_with_nan(self):
        """Test equity with NaN values."""
        checker = CIGateChecker(verbose=False)
        equity = np.array([100, 110, np.nan, 120, 130])
        dd = checker.calc_drawdown_pct(equity)
        assert dd == 0.0, "Equity with NaN should return 0% (invalid)"
    
    @pytest.mark.parametrize("equity,expected_dd", [
        ([100, 100, 100], 0.0),  # Flat
        ([100, 150, 200], 0.0),  # Growth
        ([100, 80, 100], 0.20),  # 20% drawdown
        ([100, 120, 60, 120], 0.50),  # 50% drawdown
    ])
    def test_parametrized_cases(self, equity, expected_dd):
        """Test various equity curves."""
        checker = CIGateChecker(verbose=False)
        dd = checker.calc_drawdown_pct(np.array(equity))
        assert abs(dd - expected_dd) < 0.01, f"Expected {expected_dd:.0%} DD, got {dd:.2%}"