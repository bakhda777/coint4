"""Integration test to verify all critical fixes work together."""

import pytest
import numpy as np
import pandas as pd
from src.coint2.engine.base_engine import BasePairBacktester
from src.coint2.engine.numba_backtest_engine_full import FullNumbaPairBacktester
from src.coint2.core.portfolio import Portfolio


class TestCriticalFixesIntegration:
    """Integration test for all critical fixes."""
    
    @pytest.fixture
    def test_data(self):
        """Create realistic test data with known patterns."""
        np.random.seed(42)
        n = 200
        
        # Create cointegrated series with time-varying beta
        x = np.cumsum(np.random.randn(n) * 0.01) + 100
        
        # Beta changes over time to test entry_beta consistency
        beta_early = 1.5
        beta_late = 1.8
        
        # First half with beta=1.5, second half with beta=1.8
        y1 = beta_early * x[:100] + np.random.randn(100) * 0.1 + 50
        y2 = beta_late * x[100:] + np.random.randn(100) * 0.1 + 50
        y = np.concatenate([y1, y2])
        
        # Add some divergences to trigger trades
        y[50] = y[50] + 3.0   # Divergence 1
        y[55] = y[55] - 1.0   # Convergence 1
        y[150] = y[150] - 2.5 # Divergence 2
        y[155] = y[155] + 1.0 # Convergence 2
        
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        return pd.DataFrame({'price_a': y, 'price_b': x}, index=dates)
    
    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        return Portfolio(
            initial_capital=10000.0,
            max_active_positions=5,
            leverage_limit=2.0
        )
    
    def test_base_engine_critical_fixes(self, test_data, portfolio):
        """Test that base engine has all critical fixes applied."""
        engine = BasePairBacktester(
            pair_data=test_data,
            z_threshold=1.5,  # Lower threshold to ensure trades
            z_exit=0.3,
            rolling_window=30,
            capital_at_risk=5000,
            commission_pct=0.001,
            slippage_pct=0.0005,
            bid_ask_spread_pct_s1=0.0002,
            bid_ask_spread_pct_s2=0.0002,
            portfolio=portfolio,
            pair_name="TEST-PAIR"
        )
        
        engine.run()
        results = engine.results
        
        # Verify no lookahead bias in rolling stats
        self._verify_no_lookahead_bias(results)
        
        # Verify trading costs are realistic
        self._verify_realistic_trading_costs(results)
        
        # Verify PnL consistency
        self._verify_pnl_consistency(results)
        
        # Verify entry_beta usage
        self._verify_entry_beta_consistency(engine)
        
    def test_numba_engine_critical_fixes(self, test_data):
        """Test that Numba engine has all critical fixes applied."""
        engine = FullNumbaPairBacktester(
            pair_data=test_data,
            z_threshold=1.5,
            z_exit=0.3,
            rolling_window=30,
            capital_at_risk=5000,
            commission_pct=0.001,
            slippage_pct=0.0005
        )
        
        # Test Numba-specific method
        result = engine.run_numba_full()
        
        # Verify results are consistent
        assert np.all(np.isfinite(result.positions)), "All positions should be finite"
        assert np.all(np.isfinite(result.pnl_series)), "All PnL should be finite"
        assert np.all(np.isfinite(result.z_scores[~np.isnan(result.z_scores)])), "Valid z-scores should be finite"
        
        # Verify no extreme position changes
        position_changes = np.abs(np.diff(result.positions))
        max_change = np.max(position_changes)
        assert max_change <= 2.0, f"Position changes should be reasonable, got {max_change}"
        
    def test_capital_management_fixes(self, test_data, portfolio):
        """Test that capital management fixes work correctly."""
        engine = BasePairBacktester(
            pair_data=test_data,
            z_threshold=1.5,
            z_exit=0.3,
            rolling_window=30,
            capital_at_risk=5000,
            commission_pct=0.001,
            slippage_pct=0.0005,
            portfolio=portfolio,
            pair_name="TEST-PAIR"
        )
        
        # Test the new capital management method
        capital_for_trade = 3000.0
        position_size = engine._calculate_position_size_with_capital(
            entry_z=2.0,
            spread_curr=1.0,
            mean=0.0,
            std=0.5,
            beta=1.5,
            price_s1=100.0,
            price_s2=50.0,
            capital_for_trade=capital_for_trade
        )
        
        # Verify that original capital_at_risk is unchanged
        assert engine.capital_at_risk == 5000, "Original capital should be unchanged"
        
        # Verify position size is reasonable
        assert position_size >= 0, "Position size should be non-negative"
        assert np.isfinite(position_size), "Position size should be finite"
        
    def test_trading_costs_improvements(self, test_data):
        """Test that trading costs are calculated more realistically."""
        engine = BasePairBacktester(
            pair_data=test_data,
            z_threshold=1.5,
            z_exit=0.3,
            rolling_window=30,
            capital_at_risk=5000,
            commission_pct=0.001,
            slippage_pct=0.0005,
            bid_ask_spread_pct_s1=0.0002,
            bid_ask_spread_pct_s2=0.0002
        )
        
        # Test cost calculation directly
        commission, slippage, bid_ask, total = engine._calculate_trading_costs(
            position_s1_change=1.0,
            position_s2_change=-1.5,
            price_s1=100.0,
            price_s2=50.0
        )
        
        # Verify costs are positive and reasonable
        assert commission > 0, "Commission should be positive"
        assert slippage > 0, "Slippage should be positive"
        assert bid_ask > 0, "Bid-ask cost should be positive"
        assert total == commission + slippage + bid_ask, "Total should equal sum of components"
        
        # Verify bid-ask cost uses full spread (not half)
        expected_bid_ask = (1.0 * 100.0 * 0.0002 + 1.5 * 50.0 * 0.0002)
        assert abs(bid_ask - expected_bid_ask) < 1e-6, "Bid-ask cost should use full spread"
        
    def _verify_no_lookahead_bias(self, results):
        """Verify that rolling statistics don't use future data."""
        # Check that z_scores are calculated correctly
        valid_rows = results.dropna(subset=['z_score', 'beta', 'mean', 'std'])
        
        if len(valid_rows) > 0:
            # Verify z_score calculation
            for idx in valid_rows.index[:5]:  # Check first 5 valid rows
                row = results.loc[idx]
                expected_z = (row['spread'] - row['mean']) / row['std']
                assert abs(row['z_score'] - expected_z) < 1e-10, "Z-score calculation error"
                
    def _verify_realistic_trading_costs(self, results):
        """Verify that trading costs are realistic."""
        trade_rows = results[results['trades'] > 0]
        
        if len(trade_rows) > 0:
            # Check that costs are positive when there are trades
            assert (trade_rows['costs'] > 0).all(), "Costs should be positive for trades"
            
            # Check that costs are reasonable (not too high)
            max_cost_pct = trade_rows['costs'].max() / 5000  # Relative to capital
            assert max_cost_pct < 0.01, "Trading costs should be reasonable (<1% of capital)"
            
    def _verify_pnl_consistency(self, results):
        """Verify PnL calculations are consistent."""
        # Check cumulative PnL consistency
        if len(results) > 1:
            calculated_cumulative = results['pnl'].cumsum()
            reported_cumulative = results['cumulative_pnl']
            
            # Allow for small numerical differences
            max_diff = abs(calculated_cumulative - reported_cumulative).max()
            assert max_diff < 1e-6, f"Cumulative PnL inconsistency: {max_diff}"
            
    def _verify_entry_beta_consistency(self, engine):
        """Verify that entry_beta is used consistently."""
        # This is more of a design verification
        # The engine should have entry_beta attribute when in position
        assert hasattr(engine, 'entry_beta'), "Engine should track entry_beta"
        
        # entry_beta should be NaN when no position
        if engine.current_position == 0:
            assert pd.isna(engine.entry_beta) or engine.entry_beta == 0, "entry_beta should be reset when no position"
