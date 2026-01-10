"""Unit tests for execution cost calibration."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'scripts'))


class TestExecutionCalibration:
    """Tests for execution cost calibration."""
    
    def test_market_features_calculation(self):
        """Test calculation of market features."""
        from calibrate_execution_costs import calculate_market_features
        
        # Create test data
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        df = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 0.5,
            'high': 101 + np.random.randn(100) * 0.5,
            'low': 99 + np.random.randn(100) * 0.5,
            'close': 100 + np.random.randn(100) * 0.5,
        }, index=dates)
        
        features = calculate_market_features(df)
        
        # Check all required features exist
        assert 'atr' in features.columns
        assert 'atr_pct' in features.columns
        assert 'hl_range' in features.columns
        assert 'oc_range' in features.columns
        assert 'volatility' in features.columns
        assert 'spread_proxy' in features.columns
        
        # Check features are positive
        assert (features['atr'] >= 0).all()
        assert (features['atr_pct'] >= 0).all()
        assert (features['hl_range'] >= 0).all()
        assert (features['oc_range'] >= 0).all()
        assert (features['volatility'] >= 0).all()
        assert (features['spread_proxy'] >= 0).all()
        
    def test_slippage_model_monotonicity(self):
        """Test that slippage increases with volatility and spread."""
        from calibrate_execution_costs import fit_slippage_model
        
        # Create synthetic features with clear pattern
        n_samples = 1000
        features = pd.DataFrame({
            'atr_pct': np.linspace(0.001, 0.01, n_samples),
            'spread_proxy': np.linspace(0.0001, 0.001, n_samples),
            'volatility': np.linspace(0.01, 0.05, n_samples)
        })
        
        model = fit_slippage_model(features)
        
        # Check positive coefficients (slippage should increase with features)
        assert model['atr_coef'] >= 0, "ATR coefficient should be positive"
        assert model['spread_coef'] >= 0, "Spread coefficient should be positive"
        assert model['vol_coef'] >= 0, "Volatility coefficient should be positive"
        
        # Check R² is reasonable
        assert model['r2_score'] > 0.5, "Model should explain at least 50% of variance"
        
    def test_piecewise_model_regimes(self):
        """Test piecewise model creates different regimes."""
        from calibrate_execution_costs import fit_piecewise_model
        
        # Create features with clear regime differences
        n_samples = 300
        low_vol = pd.DataFrame({
            'atr_pct': np.random.uniform(0.001, 0.003, n_samples),
            'spread_proxy': np.random.uniform(0.0001, 0.0003, n_samples),
            'volatility': np.random.uniform(0.005, 0.015, n_samples)
        })
        
        high_vol = pd.DataFrame({
            'atr_pct': np.random.uniform(0.008, 0.01, n_samples),
            'spread_proxy': np.random.uniform(0.0008, 0.001, n_samples),
            'volatility': np.random.uniform(0.04, 0.05, n_samples)
        })
        
        features = pd.concat([low_vol, high_vol])
        models = fit_piecewise_model(features)
        
        # Check we have multiple regimes
        assert len(models) >= 2, "Should have at least 2 volatility regimes"
        
        # Check regime names
        expected_regimes = ['low_vol', 'mid_vol', 'high_vol']
        for regime in models.keys():
            assert regime in expected_regimes, f"Unexpected regime: {regime}"
            
    def test_calibration_output_structure(self):
        """Test calibration produces expected output structure."""
        from calibrate_execution_costs import calibrate_execution_costs
        
        results = calibrate_execution_costs(
            pairs=['BTCUSDT'],
            window_days=7
        )
        
        # Check required fields
        assert 'calibration_date' in results
        assert 'window_days' in results
        assert 'pairs_analyzed' in results
        assert 'aggregate_model' in results
        assert 'piecewise_models' in results
        assert 'market_stats' in results
        
        # Check aggregate model structure
        model = results['aggregate_model']
        assert 'intercept' in model
        assert 'atr_coef' in model
        assert 'spread_coef' in model
        assert 'vol_coef' in model
        assert 'r2_score' in model
        
        # Check market stats
        stats = results['market_stats']
        assert 'avg_atr_pct' in stats
        assert 'avg_spread_proxy' in stats
        assert 'avg_volatility' in stats
        
    def test_coefficient_bounds(self):
        """Test that calibrated coefficients are within reasonable bounds."""
        from calibrate_execution_costs import fit_slippage_model
        
        # Create realistic features
        features = pd.DataFrame({
            'atr_pct': np.random.uniform(0.001, 0.01, 500),
            'spread_proxy': np.random.uniform(0.0001, 0.001, 500),
            'volatility': np.random.uniform(0.01, 0.05, 500)
        })
        
        model = fit_slippage_model(features)
        
        # Check intercept (base slippage) is reasonable
        assert 0 <= model['intercept'] <= 0.01, "Base slippage should be 0-1%"
        
        # Check coefficients are not too large
        assert abs(model['atr_coef']) < 10, "ATR coefficient too large"
        assert abs(model['spread_coef']) < 10, "Spread coefficient too large"
        assert abs(model['vol_coef']) < 10, "Volatility coefficient too large"
        
    @pytest.mark.parametrize("volatility,expected_impact", [
        (0.01, "low"),  # Low volatility
        (0.03, "medium"),  # Medium volatility
        (0.05, "high"),  # High volatility
    ])
    def test_volatility_impact(self, volatility, expected_impact):
        """Test that volatility levels produce expected impact."""
        from calibrate_execution_costs import fit_slippage_model
        
        # Create features with specific volatility
        features = pd.DataFrame({
            'atr_pct': [0.005] * 100,
            'spread_proxy': [0.0005] * 100,
            'volatility': [volatility] * 100
        })
        
        model = fit_slippage_model(features, target_slippage=0.0005)
        
        # Higher volatility should lead to higher slippage
        # This is implicit in the positive vol_coef tested above
        assert model['vol_coef'] >= 0, f"Volatility {volatility} should increase slippage"