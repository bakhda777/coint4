"""Test to verify that lookahead bias has been fixed in update_rolling_stats method."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from src.coint2.engine.base_engine import BasePairBacktester


class TestLookaheadBiasFix:
    """Test class to verify lookahead bias fix in rolling statistics calculation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'rolling_window': 20,
            'z_threshold': 2.0,
            'z_exit': 0.5,
            'capital_at_risk': 10000,
            'commission_pct': 0.001,
            'slippage_pct': 0.0005,
            'bid_ask_spread_pct_s1': 0.0002,
            'bid_ask_spread_pct_s2': 0.0002
        }
        
        # Create mock data with known pattern
        np.random.seed(42)
        n_points = 100
        
        # Create synthetic cointegrated series
        x = np.cumsum(np.random.randn(n_points)) + 100
        noise = np.random.randn(n_points) * 0.1
        y = 1.5 * x + 10 + noise  # y = 1.5*x + 10 + noise
        
        # Add a known pattern at specific points for testing
        # At point 50, create a temporary divergence
        y[50] = y[50] + 5  # Temporary spike
        
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='15min'),
            'x': x,
            'y': y
        })
        
        self.engine = BasePairBacktester(
            pair_data=self.test_data,
            **self.config
        )
        
    def test_no_lookahead_bias_in_rolling_stats(self):
        """Test that rolling statistics don't use future data."""
        # Initialize DataFrame with required columns
        df = self.test_data.copy()
        df['beta'] = np.nan
        df['mean'] = np.nan
        df['std'] = np.nan
        df['spread'] = np.nan
        df['z_score'] = np.nan
        
        # Test specific point where we know there's a spike
        test_point = 51  # Point after the spike at index 50
        
        # Call update_rolling_stats for test_point
        self.engine.update_rolling_stats(df, test_point)
        
        # The statistics should be calculated for bar test_point (using historical data up to test_point-1)
        target_bar = test_point
        
        # Verify that statistics were calculated
        assert not pd.isna(df['beta'].iloc[target_bar]), "Beta should be calculated"
        assert not pd.isna(df['mean'].iloc[target_bar]), "Mean should be calculated"
        assert not pd.isna(df['std'].iloc[target_bar]), "Std should be calculated"
        assert not pd.isna(df['z_score'].iloc[target_bar]), "Z-score should be calculated"
        
        # Manually calculate what the statistics should be using only historical data
        # FIXED: Use only historical data (exclude current bar)
        start_idx = target_bar - self.config['rolling_window']
        end_idx = target_bar
        
        y_win = df['y'].iloc[start_idx:end_idx]
        x_win = df['x'].iloc[start_idx:end_idx]
        
        # Calculate expected beta using OLS
        X = np.column_stack([np.ones(len(x_win)), x_win])
        coeffs = np.linalg.lstsq(X, y_win, rcond=None)[0]
        expected_beta = coeffs[1]
        
        # Calculate expected spread statistics
        spreads = y_win - expected_beta * x_win
        expected_mean = spreads.mean()
        expected_std = spreads.std()
        
        # Calculate expected z-score for target bar
        target_spread = df['y'].iloc[target_bar] - expected_beta * df['x'].iloc[target_bar]
        expected_z_score = (target_spread - expected_mean) / expected_std
        
        # Verify that calculated values match expected values (within tolerance)
        np.testing.assert_allclose(df['beta'].iloc[target_bar], expected_beta, rtol=1e-10)
        np.testing.assert_allclose(df['mean'].iloc[target_bar], expected_mean, rtol=1e-10)
        np.testing.assert_allclose(df['std'].iloc[target_bar], expected_std, rtol=1e-10)
        np.testing.assert_allclose(df['z_score'].iloc[target_bar], expected_z_score, rtol=1e-10)
        
    def test_no_future_data_contamination(self):
        """Test that statistics at bar i don't use data from bar i+1 or later."""
        # Create modified data where future bars have extreme values
        df = self.test_data.copy()
        df['beta'] = np.nan
        df['mean'] = np.nan
        df['std'] = np.nan
        df['spread'] = np.nan
        df['z_score'] = np.nan
        
        test_point = 50
        
        # Calculate statistics normally
        self.engine.update_rolling_stats(df, test_point)
        normal_beta = df['beta'].iloc[test_point]
        normal_mean = df['mean'].iloc[test_point]
        normal_std = df['std'].iloc[test_point]
        
        # Now modify future data (bars test_point+1 and later) with extreme values
        df_modified = df.copy()
        df_modified.loc[test_point+1:, 'y'] *= 10  # Extreme change in future data
        df_modified.loc[test_point+1:, 'x'] *= 10
        
        # Reset statistics
        df_modified['beta'] = np.nan
        df_modified['mean'] = np.nan
        df_modified['std'] = np.nan
        df_modified['spread'] = np.nan
        df_modified['z_score'] = np.nan
        
        # Calculate statistics again with modified future data
        self.engine.update_rolling_stats(df_modified, test_point)
        modified_beta = df_modified['beta'].iloc[test_point]
        modified_mean = df_modified['mean'].iloc[test_point]
        modified_std = df_modified['std'].iloc[test_point]
        
        # Statistics should be identical because future data shouldn't affect them
        np.testing.assert_allclose(normal_beta, modified_beta, rtol=1e-12,
                                 err_msg="Beta changed when future data was modified - lookahead bias detected!")
        np.testing.assert_allclose(normal_mean, modified_mean, rtol=1e-12,
                                 err_msg="Mean changed when future data was modified - lookahead bias detected!")
        np.testing.assert_allclose(normal_std, modified_std, rtol=1e-12,
                                 err_msg="Std changed when future data was modified - lookahead bias detected!")
        
    def test_rolling_window_boundary_conditions(self):
        """Test that rolling statistics work correctly at window boundaries."""
        df = self.test_data.copy()
        df['beta'] = np.nan
        df['mean'] = np.nan
        df['std'] = np.nan
        df['spread'] = np.nan
        df['z_score'] = np.nan
        
        rolling_window = self.config['rolling_window']
        
        # Test at the minimum point where statistics should be calculated
        min_point = rolling_window  # First point where we can calculate stats
        self.engine.update_rolling_stats(df, min_point)
        
        # Statistics should be calculated for bar min_point
        target_bar = min_point
        assert not pd.isna(df['beta'].iloc[target_bar]), "Beta should be calculated at minimum boundary"
        assert not pd.isna(df['z_score'].iloc[target_bar]), "Z-score should be calculated at minimum boundary"
        
        # Test one point before minimum - should not calculate
        df_early = df.copy()
        df_early['beta'] = np.nan
        df_early['mean'] = np.nan
        df_early['std'] = np.nan
        df_early['spread'] = np.nan
        df_early['z_score'] = np.nan
        
        self.engine.update_rolling_stats(df_early, min_point - 1)
        
        # No statistics should be calculated
        target_bar_early = min_point - 2
        if target_bar_early >= 0:
            assert pd.isna(df_early['beta'].iloc[target_bar_early]), "Beta should not be calculated before minimum boundary"
            assert pd.isna(df_early['z_score'].iloc[target_bar_early]), "Z-score should not be calculated before minimum boundary"
            
    def test_data_window_correctness(self):
        """Test that the correct data window is used for calculations."""
        df = self.test_data.copy()
        df['beta'] = np.nan
        df['mean'] = np.nan
        df['std'] = np.nan
        df['spread'] = np.nan
        df['z_score'] = np.nan
        
        test_point = 30
        rolling_window = self.config['rolling_window']
        
        # Mock the _calculate_ols_with_cache method to capture the data it receives
        original_method = self.engine._calculate_ols_with_cache
        captured_data = {}
        
        def mock_ols(y_win, x_win):
            captured_data['y_win'] = y_win.copy()
            captured_data['x_win'] = x_win.copy()
            return original_method(y_win, x_win)
        
        self.engine._calculate_ols_with_cache = mock_ols
        
        # Call update_rolling_stats
        self.engine.update_rolling_stats(df, test_point)
        
        # Verify the data window used
        target_bar = test_point
        # FIXED: Expected window should exclude current bar
        expected_start = target_bar - rolling_window
        expected_end = target_bar
        
        expected_y = df['y'].iloc[expected_start:expected_end]
        expected_x = df['x'].iloc[expected_start:expected_end]
        
        # Check that the captured data matches expected window
        pd.testing.assert_series_equal(captured_data['y_win'], expected_y, check_names=False)
        pd.testing.assert_series_equal(captured_data['x_win'], expected_x, check_names=False)
        
        # Restore original method
        self.engine._calculate_ols_with_cache = original_method