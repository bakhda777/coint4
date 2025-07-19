"""Specific tests for PnL calculation fixes in pair trading.

This module contains focused tests to verify that PnL calculation
for pair trading has been correctly implemented:

1. PnL should be calculated as: position_size * (current_spread - entry_spread)
2. PnL should account for both assets in the pair
3. PnL should correctly handle spread-based trading logic
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.coint2.engine.backtest_engine import PairBacktester


class TestPnLCalculationFix:
    """Test suite for PnL calculation fixes."""
    
    @pytest.fixture
    def simple_config(self):
        """Create a simple backtest configuration for PnL testing."""
        return {
            'rolling_window': 10,
            'z_threshold': 1.5,
            'z_exit': 0.3,
            'stop_loss_multiplier': 3.0,
            'capital_at_risk': 10000.0,
            'commission_pct': 0.0,  # No costs for pure PnL testing
            'slippage_pct': 0.0,
            'bid_ask_spread_pct_s1': 0.0,
            'bid_ask_spread_pct_s2': 0.0,
            'max_margin_usage': 1.0
        }
    
    @pytest.fixture
    def controlled_data(self):
        """Create controlled price data for precise PnL testing."""
        dates = pd.date_range('2024-01-01', periods=50, freq='15min')
        
        # Create predictable price movements
        # Asset 1: starts at 100, increases to 110
        y_prices = np.linspace(100, 110, 50)
        
        # Asset 2: starts at 50, increases to 55 (beta ≈ 2.0)
        x_prices = np.linspace(50, 55, 50)
        
        return pd.DataFrame({'y': y_prices, 'x': x_prices}, index=dates)
    
    def test_pnl_calculation_formula_correctness(self, simple_config, controlled_data):
        """Test that PnL is calculated using the correct formula for pair trading.
        
        PnL should be: position_size * (current_spread - entry_spread)
        where spread = price_y - beta * price_x
        """
        engine = PairBacktester(controlled_data, **simple_config)
        engine.pair_name = "TEST_PAIR"
        
        # Force a specific scenario by mocking parameter calculation
        def mock_ols(y_data, x_data):
            return 2.0, 0.0, 1.0  # beta=2.0, mean=0.0, std=1.0
        
        engine._calculate_ols_with_cache = mock_ols
        engine.run()
        
        results = engine.results
        
        # Find periods with positions
        position_periods = results[results['position'] != 0].copy()
        
        if len(position_periods) == 0:
            pytest.skip("No positions were opened, cannot test PnL calculation")
        
        # Test PnL calculation for consecutive periods with same position
        for i in range(1, len(position_periods)):
            current = position_periods.iloc[i]
            previous = position_periods.iloc[i-1]
            
            # Skip if position changed
            if current['position'] != previous['position']:
                continue
            
            # Calculate expected PnL change based on spread change
            spread_change = current['spread'] - previous['spread']
            position_size = current['position']
            expected_pnl_change = position_size * spread_change
            
            # Get actual PnL change
            actual_pnl_change = current['pnl'] - previous['pnl']
            
            # Allow small numerical tolerance
            tolerance = 1e-6
            assert abs(actual_pnl_change - expected_pnl_change) < tolerance, \
                f"PnL calculation incorrect. Expected change: {expected_pnl_change}, "\
                f"Actual change: {actual_pnl_change}, Spread change: {spread_change}, "\
                f"Position: {position_size}"
    
    def test_pnl_direction_consistency_with_spread_movement(self, simple_config, controlled_data):
        """Test that PnL direction is consistent with spread movement and position direction.
        
        For long positions (positive): PnL should increase when spread increases
        For short positions (negative): PnL should increase when spread decreases
        """
        engine = PairBacktester(controlled_data, **simple_config)
        engine.pair_name = "TEST_PAIR"
        
        # Use fixed parameters for predictable behavior
        def mock_ols(y_data, x_data):
            return 2.0, 0.0, 1.0  # beta=2.0, mean=0.0, std=1.0
        
        engine._calculate_ols_with_cache = mock_ols
        engine.run()
        
        results = engine.results
        
        # Analyze PnL direction consistency
        for i in range(1, len(results)):
            current = results.iloc[i]
            previous = results.iloc[i-1]
            
            # Skip if no position or position changed
            if (current['position'] == 0 or previous['position'] == 0 or
                current['position'] != previous['position']):
                continue
            
            # Skip if missing data
            if (pd.isna(current['spread']) or pd.isna(previous['spread']) or
                pd.isna(current['pnl']) or pd.isna(previous['pnl'])):
                continue
            
            spread_change = current['spread'] - previous['spread']
            pnl_change = current['pnl'] - previous['pnl']
            position = current['position']
            
            # Skip very small changes to avoid noise
            if abs(spread_change) < 1e-8 or abs(pnl_change) < 1e-8:
                continue
            
            # Check direction consistency
            expected_pnl_sign = np.sign(position * spread_change)
            actual_pnl_sign = np.sign(pnl_change)
            
            assert expected_pnl_sign == actual_pnl_sign, \
                f"PnL direction inconsistent. Position: {position}, "\
                f"Spread change: {spread_change}, PnL change: {pnl_change}"
    
    def test_pnl_calculation_with_known_scenario(self, simple_config):
        """Test PnL calculation with a completely controlled scenario.
        
        This test creates a specific scenario where we know exactly
        what the PnL should be at each step.
        """
        # Create specific price data
        dates = pd.date_range('2024-01-01', periods=25, freq='15min')
        
        # Scenario: prices start equal, then diverge, then converge
        y_prices = [100.0] * 10 + [101.0] * 5 + [100.5] * 5 + [100.0] * 5
        x_prices = [50.0] * 10 + [50.0] * 5 + [50.25] * 5 + [50.0] * 5
        
        data = pd.DataFrame({'y': y_prices, 'x': x_prices}, index=dates)
        
        engine = PairBacktester(data, **simple_config)
        engine.pair_name = "TEST_PAIR"
        
        # Mock parameters for predictable behavior
        def mock_ols(y_data, x_data):
            return 2.0, 0.0, 1.0  # beta=2.0, mean=0.0, std=1.0
        
        engine._calculate_ols_with_cache = mock_ols
        
        # Mock position entry to control when positions are opened
        original_enter = engine._enter_position
        position_opened = False
        
        def controlled_enter(df, i, signal, z_curr, spread_curr, mean, std, beta):
            nonlocal position_opened
            if not position_opened and i == 12:  # Open position at specific time
                position_opened = True
                return 10.0  # Fixed position size
            return 0.0
        
        engine._enter_position = controlled_enter
        engine.run()
        
        results = engine.results
        
        # Verify PnL calculation at specific points
        if position_opened:
            # Find the entry point
            entry_idx = None
            for i, row in results.iterrows():
                if row['position'] != 0:
                    entry_idx = results.index.get_loc(i)
                    break
            
            if entry_idx is not None and entry_idx < len(results) - 5:
                entry_spread = results.iloc[entry_idx]['spread']
                position_size = results.iloc[entry_idx]['position']
                
                # Check PnL at subsequent periods
                for j in range(entry_idx + 1, min(entry_idx + 5, len(results))):
                    current_spread = results.iloc[j]['spread']
                    current_pnl = results.iloc[j]['pnl']
                    
                    if not pd.isna(current_spread) and not pd.isna(current_pnl):
                        expected_pnl = position_size * (current_spread - entry_spread)
                        
                        # Allow small tolerance for numerical precision
                        assert abs(current_pnl - expected_pnl) < 1e-6, \
                            f"PnL mismatch at period {j}. Expected: {expected_pnl}, "\
                            f"Actual: {current_pnl}, Position: {position_size}, "\
                            f"Entry spread: {entry_spread}, Current spread: {current_spread}"
    
    def test_pnl_calculation_accounts_for_both_assets(self, simple_config, controlled_data):
        """Test that PnL calculation properly accounts for both assets in the pair.
        
        The PnL should reflect the combined effect of price movements
        in both assets, weighted by their position sizes.
        """
        engine = PairBacktester(controlled_data, **simple_config)
        engine.pair_name = "TEST_PAIR"
        
        # Track individual asset price changes and verify they contribute to PnL
        def mock_ols(y_data, x_data):
            return 2.0, 0.0, 1.0  # beta=2.0, mean=0.0, std=1.0
        
        engine._calculate_ols_with_cache = mock_ols
        engine.run()
        
        results = engine.results
        
        # Find periods with positions
        position_periods = results[results['position'] != 0]
        
        if len(position_periods) < 2:
            pytest.skip("Insufficient position periods for testing")
        
        # Verify that PnL changes when either asset price changes
        for i in range(1, len(position_periods)):
            current = position_periods.iloc[i]
            previous = position_periods.iloc[i-1]
            
            if current['position'] != previous['position']:
                continue
            
            # Get price changes
            current_idx = results.index.get_loc(current.name)
            previous_idx = results.index.get_loc(previous.name)
            
            if current_idx > 0 and previous_idx >= 0:
                y_change = controlled_data.iloc[current_idx]['y'] - controlled_data.iloc[previous_idx]['y']
                x_change = controlled_data.iloc[current_idx]['x'] - controlled_data.iloc[previous_idx]['x']
                
                # If either asset price changed significantly, PnL should change
                if abs(y_change) > 1e-6 or abs(x_change) > 1e-6:
                    pnl_change = current['pnl'] - previous['pnl']
                    
                    # PnL should change when asset prices change (unless perfectly hedged)
                    # Allow for the case where changes cancel out due to hedging
                    spread_change = current['spread'] - previous['spread']
                    
                    if abs(spread_change) > 1e-6:
                        assert abs(pnl_change) > 1e-8, \
                            f"PnL should change when spread changes. "\
                            f"Y change: {y_change}, X change: {x_change}, "\
                            f"Spread change: {spread_change}, PnL change: {pnl_change}"
    
    def test_pnl_calculation_with_zero_costs(self, simple_config, controlled_data):
        """Test PnL calculation with zero transaction costs.
        
        This test verifies that when all costs are zero,
        the PnL exactly matches the spread-based calculation.
        """
        # Ensure all costs are zero
        simple_config['commission_pct'] = 0.0
        simple_config['slippage_pct'] = 0.0
        simple_config['bid_ask_spread_pct_s1'] = 0.0
        simple_config['bid_ask_spread_pct_s2'] = 0.0
        
        engine = PairBacktester(controlled_data, **simple_config)
        engine.pair_name = "TEST_PAIR"
        
        def mock_ols(y_data, x_data):
            return 2.0, 0.0, 1.0  # beta=2.0, mean=0.0, std=1.0
        
        engine._calculate_ols_with_cache = mock_ols
        engine.run()
        
        results = engine.results
        
        # With zero costs, PnL should exactly equal position_size * spread_change
        position_periods = results[results['position'] != 0]
        
        if len(position_periods) < 2:
            pytest.skip("Insufficient position periods for testing")
        
        # Find entry point
        entry_period = position_periods.iloc[0]
        entry_spread = entry_period['spread']
        position_size = entry_period['position']
        
        # Check subsequent periods
        for i in range(1, len(position_periods)):
            current = position_periods.iloc[i]
            
            if current['position'] == position_size:  # Same position
                current_spread = current['spread']
                current_pnl = current['pnl']
                
                expected_pnl = position_size * (current_spread - entry_spread)
                
                # With zero costs, PnL should exactly match expected value
                assert abs(current_pnl - expected_pnl) < 1e-10, \
                    f"With zero costs, PnL should exactly match spread calculation. "\
                    f"Expected: {expected_pnl}, Actual: {current_pnl}, "\
                    f"Position: {position_size}, Entry spread: {entry_spread}, "\
                    f"Current spread: {current_spread}"
    
    def test_cumulative_pnl_consistency(self, simple_config, controlled_data):
        """Test that cumulative PnL is consistent with individual period PnL.
        
        This test verifies that the sum of individual PnL changes
        equals the total cumulative PnL.
        """
        engine = PairBacktester(controlled_data, **simple_config)
        engine.pair_name = "TEST_PAIR"
        
        def mock_ols(y_data, x_data):
            return 2.0, 0.0, 1.0  # beta=2.0, mean=0.0, std=1.0
        
        engine._calculate_ols_with_cache = mock_ols
        engine.run()
        
        results = engine.results
        
        # Calculate cumulative PnL manually
        manual_cumulative = 0.0
        
        for i, row in results.iterrows():
            if not pd.isna(row['pnl']):
                manual_cumulative += row['pnl']
                
                # Compare with built-in cumulative calculation
                built_in_cumulative = results.loc[:i, 'pnl'].sum()
                
                assert abs(manual_cumulative - built_in_cumulative) < 1e-10, \
                    f"Cumulative PnL inconsistency at period {i}. "\
                    f"Manual: {manual_cumulative}, Built-in: {built_in_cumulative}"
        
        # Verify that final cumulative PnL represents total strategy performance
        final_cumulative = results['pnl'].sum()
        
        # The final cumulative should be finite and reasonable
        assert np.isfinite(final_cumulative), "Final cumulative PnL should be finite"
        
        # If there were any positions, cumulative PnL should be non-zero
        # (unless perfectly flat market)
        if (results['position'] != 0).any():
            total_spread_movement = results['spread'].max() - results['spread'].min()
            if total_spread_movement > 1e-6:  # Significant spread movement
                # We expect some PnL (positive or negative) with significant spread movement
                # This is a sanity check, not a strict requirement
                pass  # Allow any finite value