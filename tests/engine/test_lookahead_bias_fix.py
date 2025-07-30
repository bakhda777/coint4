"""Test to verify that lookahead bias has been fixed in update_rolling_stats method."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from src.coint2.engine.base_engine import BasePairBacktester


class TestLookaheadBiasFix:
    """Test suite to verify that lookahead bias has been properly fixed."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create minimal test data
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        test_data = pd.DataFrame({
            'y': np.random.normal(100, 1, 100),
            'x': np.random.normal(50, 0.5, 100)
        }, index=dates)
        
        self.rolling_window = 20
        self.z_threshold = 2.0
        self.z_exit = 0.5
        
        self.engine = BasePairBacktester(
            pair_data=test_data,
            rolling_window=self.rolling_window,
            z_threshold=self.z_threshold,
            z_exit=self.z_exit,
            commission_pct=0.001,
            slippage_pct=0.0005,
            bid_ask_spread_pct_s1=0.001,
            bid_ask_spread_pct_s2=0.001
        )
        
    def test_update_rolling_stats_uses_only_historical_data(self):
        """Test that update_rolling_stats uses only historical data (no lookahead bias)."""
        # Create test data with known pattern
        n_bars = 50
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')
        
        # Create predictable data where current bar has extreme values
        y_data = np.random.normal(100, 1, n_bars)
        x_data = np.random.normal(50, 0.5, n_bars)
        
        # Make current bar (i=30) have extreme values that would skew statistics
        test_bar = 30
        y_data[test_bar] = 200  # Extreme value
        x_data[test_bar] = 100  # Extreme value
        
        df = pd.DataFrame({
            'timestamp': dates,
            'y': y_data,
            'x': x_data,
            'beta': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'spread': np.nan,
            'z_score': np.nan
        })
        df.set_index('timestamp', inplace=True)
        
        # Calculate statistics for bar test_bar
        self.engine.update_rolling_stats(df, test_bar)
        
        # Verify that statistics were calculated
        assert not pd.isna(df['beta'].iat[test_bar]), "Beta should be calculated"
        assert not pd.isna(df['mean'].iat[test_bar]), "Mean should be calculated"
        assert not pd.isna(df['std'].iat[test_bar]), "Std should be calculated"
        
        # Calculate what the statistics SHOULD be using only historical data
        start_idx = test_bar - self.rolling_window
        end_idx = test_bar  # Excludes current bar
        
        y_historical = df['y'].iloc[start_idx:end_idx]
        x_historical = df['x'].iloc[start_idx:end_idx]
        
        # Calculate expected OLS parameters manually
        X = np.column_stack([np.ones(len(x_historical)), x_historical])
        coeffs = np.linalg.lstsq(X, y_historical, rcond=None)[0]
        expected_beta = coeffs[1]
        
        # Calculate expected spread statistics
        expected_spreads = y_historical - expected_beta * x_historical
        expected_mean = expected_spreads.mean()
        expected_std = expected_spreads.std()
        
        # Verify that calculated statistics match historical-only calculation
        calculated_beta = df['beta'].iat[test_bar]
        calculated_mean = df['mean'].iat[test_bar]
        calculated_std = df['std'].iat[test_bar]
        
        # Allow small numerical differences
        assert abs(calculated_beta - expected_beta) < 1e-10, f"Beta mismatch: {calculated_beta} vs {expected_beta}"
        assert abs(calculated_mean - expected_mean) < 1e-10, f"Mean mismatch: {calculated_mean} vs {expected_mean}"
        assert abs(calculated_std - expected_std) < 1e-10, f"Std mismatch: {calculated_std} vs {expected_std}"
        
    def test_extreme_current_bar_does_not_affect_statistics(self):
        """Test that extreme values in current bar do not affect calculated statistics."""
        # Create stable historical data
        n_bars = 50
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')
        
        # Create stable historical data with linear relationship
        np.random.seed(42)  # For reproducibility
        x_data = np.linspace(50, 60, n_bars) + np.random.normal(0, 0.1, n_bars)
        y_data = 2.0 * x_data + 10 + np.random.normal(0, 0.5, n_bars)
        
        test_bar = 30
        
        # Test with normal current bar
        df_normal = pd.DataFrame({
            'timestamp': dates,
            'y': y_data.copy(),
            'x': x_data.copy(),
            'beta': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'spread': np.nan,
            'z_score': np.nan
        })
        df_normal.set_index('timestamp', inplace=True)
        
        self.engine.update_rolling_stats(df_normal, test_bar)
        normal_beta = df_normal['beta'].iat[test_bar]
        normal_mean = df_normal['mean'].iat[test_bar]
        normal_std = df_normal['std'].iat[test_bar]
        
        # Test with extreme current bar
        df_extreme = df_normal.copy()
        df_extreme['y'].iat[test_bar] = 1000  # Extreme outlier
        df_extreme['x'].iat[test_bar] = 500   # Extreme outlier
        
        # Reset statistics
        for col in ['beta', 'mean', 'std', 'spread', 'z_score']:
            df_extreme[col] = np.nan
            
        self.engine.update_rolling_stats(df_extreme, test_bar)
        extreme_beta = df_extreme['beta'].iat[test_bar]
        extreme_mean = df_extreme['mean'].iat[test_bar]
        extreme_std = df_extreme['std'].iat[test_bar]
        
        # Statistics should be identical because current bar is not used in calculation
        assert abs(normal_beta - extreme_beta) < 1e-10, f"Beta should be identical: {normal_beta} vs {extreme_beta}"
        assert abs(normal_mean - extreme_mean) < 1e-10, f"Mean should be identical: {normal_mean} vs {extreme_mean}"
        assert abs(normal_std - extreme_std) < 1e-10, f"Std should be identical: {normal_std} vs {extreme_std}"
        
    def test_rolling_window_data_range(self):
        """Test that exactly rolling_window bars of historical data are used."""
        n_bars = 50
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')
        
        # Create test data
        y_data = np.random.normal(100, 1, n_bars)
        x_data = np.random.normal(50, 0.5, n_bars)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'y': y_data,
            'x': x_data,
            'beta': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'spread': np.nan,
            'z_score': np.nan
        })
        df.set_index('timestamp', inplace=True)
        
        test_bar = 30
        
        # Mock the _calculate_ols_with_cache method to capture the data it receives
        original_method = self.engine._calculate_ols_with_cache
        captured_data = {}
        
        def mock_ols(y_win, x_win):
            captured_data['y_win'] = y_win.copy()
            captured_data['x_win'] = x_win.copy()
            return original_method(y_win, x_win)
            
        self.engine._calculate_ols_with_cache = mock_ols
        
        # Call update_rolling_stats
        self.engine.update_rolling_stats(df, test_bar)
        
        # Verify the data range used
        expected_start = test_bar - self.rolling_window
        expected_end = test_bar
        
        expected_y = df['y'].iloc[expected_start:expected_end]
        expected_x = df['x'].iloc[expected_start:expected_end]
        
        # Check that exactly the right data was used
        pd.testing.assert_series_equal(captured_data['y_win'], expected_y, check_names=False)
        pd.testing.assert_series_equal(captured_data['x_win'], expected_x, check_names=False)
        
        # Verify length is exactly rolling_window
        assert len(captured_data['y_win']) == self.rolling_window
        assert len(captured_data['x_win']) == self.rolling_window
        
        # Restore original method
        self.engine._calculate_ols_with_cache = original_method
        
    def test_no_lookahead_bias_in_signal_generation(self):
        """Test that signal generation does not use future data."""
        # This test ensures that the z_score used for signal generation
        # is calculated from historical data only
        
        n_bars = 50
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')
        
        # Create data with known pattern
        y_data = np.random.normal(100, 1, n_bars)
        x_data = np.random.normal(50, 0.5, n_bars)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'y': y_data,
            'x': x_data,
            'beta': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'spread': np.nan,
            'z_score': np.nan
        })
        df.set_index('timestamp', inplace=True)
        
        test_bar = 30
        
        # Update statistics (should use only historical data)
        self.engine.update_rolling_stats(df, test_bar)
        
        # Verify that z_score is calculated correctly
        beta = df['beta'].iat[test_bar]
        mean = df['mean'].iat[test_bar]
        std = df['std'].iat[test_bar]
        
        # Current bar's spread
        current_spread = df['y'].iat[test_bar] - beta * df['x'].iat[test_bar]
        expected_z_score = (current_spread - mean) / std
        
        calculated_z_score = df['z_score'].iat[test_bar]
        
        assert abs(calculated_z_score - expected_z_score) < 1e-10, \
            f"Z-score mismatch: {calculated_z_score} vs {expected_z_score}"
            
        # The z_score should be based on statistics calculated from historical data only
        # but applied to current bar's spread - this is correct and not lookahead bias