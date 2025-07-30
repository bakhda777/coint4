"""Тесты для проверки корректности работы PairBacktester с оптимизациями."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from coint2.engine.base_engine import BasePairBacktester
from coint2.core.data_loader import DataHandler
# Alias for backward compatibility
PairBacktester = BasePairBacktester


class TestBacktestEngineOptimization:
    """Тесты для проверки корректности работы оптимизированного движка бэктестинга."""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные для бэктестинга."""
        np.random.seed(42)
        n_points = 500
        
        # Create cointegrated pair with some trend
        base_series = np.cumsum(np.random.randn(n_points) * 0.01) + 100
        noise1 = np.random.randn(n_points) * 0.5
        noise2 = np.random.randn(n_points) * 0.5
        
        asset1 = base_series + noise1
        asset2 = base_series * 1.2 + 5 + noise2  # Cointegrated with asset1
        
        data = pd.DataFrame({
            'asset1': asset1,
            'asset2': asset2
        })
        
        return data
    
    def test_optimized_backtest_produces_valid_results(self, sample_data):
        """Тест 1: Проверяет, что оптимизированный бэктест дает валидные результаты.
        
        Проверяет, что все оптимизации не нарушают базовую логику
        бэктестинга и производят корректные результаты.
        """
        # Create optimized backtester
        backtester = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=100,
            z_threshold=2.0,
            market_regime_detection=True,
            structural_break_protection=True,
            regime_check_frequency=24,
            use_market_regime_cache=True,
            adf_check_frequency=100,
            cache_cleanup_frequency=200,
            pair_name="TEST_PAIR"
        )
        
        # Run backtest
        backtester.run()
        
        # Get results
        results = backtester.get_performance_metrics()
        
        # Validate results structure
        required_metrics = [
            'total_pnl', 'total_return', 'sharpe_ratio', 'max_drawdown',
            'num_trades', 'win_rate', 'avg_trade_duration'
        ]
        
        for metric in required_metrics:
            assert metric in results, f"Missing required metric: {metric}"
            assert results[metric] is not None, f"Metric {metric} should not be None"
            
        # Validate numeric ranges
        assert isinstance(results['total_pnl'], (int, float)), "PnL should be numeric"
        assert isinstance(results['total_return'], (int, float)), "Return should be numeric"
        assert isinstance(results['num_trades'], int), "Number of trades should be integer"
        assert 0 <= results['win_rate'] <= 1, f"Win rate should be in [0,1], got {results['win_rate']}"
        assert results['max_drawdown'] <= 0, f"Max drawdown should be ≤0, got {results['max_drawdown']}"
        
        # Check that positions were generated
        positions = backtester.results['position'].dropna()
        assert len(positions) > 0, "Should generate some positions"
        assert positions.isin([-1, 0, 1]).all(), "Positions should be -1, 0, or 1"
        
    def test_regime_detection_with_optimization(self, sample_data):
        """Тест 2: Проверяет корректность определения режимов с оптимизацией.
        
        Проверяет, что оптимизация частоты не нарушает логику
        определения рыночных режимов.
        """
        # Create backtester with regime detection
        backtester = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            regime_check_frequency=12,  # Check every 12 bars
            use_market_regime_cache=True,
            hurst_trending_threshold=0.5,
            variance_ratio_trending_min=1.1,
            hurst_window=100,  # Reduced from default 720 to fit test data
            variance_ratio_window=80,  # Reduced from default 480
            pair_name="TEST_REGIME"
        )
        
        # Run backtest
        backtester.run()
        
        # Check regime detection results
        regimes = backtester.market_regime.dropna()
        hurst_values = backtester.hurst_exponents.dropna()
        vr_values = backtester.variance_ratios.dropna()
        
        assert len(regimes) > 0, "Should detect some regimes"
        assert len(hurst_values) > 0, "Should calculate Hurst exponents"
        assert len(vr_values) > 0, "Should calculate variance ratios"
        
        # Check regime values are valid
        valid_regimes = {'trending', 'mean_reverting', 'neutral'}
        assert regimes.isin(valid_regimes).all(), f"Invalid regime values: {regimes.unique()}"
        
        # Check Hurst exponents are in valid range
        assert (hurst_values >= 0).all() and (hurst_values <= 1).all(), "Hurst exponents should be in [0,1]"
        
        # Check variance ratios are positive
        assert (vr_values > 0).all(), "Variance ratios should be positive"
        
        # Check regime consistency with metrics
        for idx in regimes.index:
            if idx in hurst_values.index and idx in vr_values.index:
                regime = regimes[idx]
                hurst = hurst_values[idx]
                vr = vr_values[idx]
                
                if regime == 'trending':
                    # At least one indicator should suggest trending
                    assert hurst > 0.5 or vr > 1.1, f"Trending regime inconsistent: H={hurst:.3f}, VR={vr:.3f}"
                elif regime == 'mean_reverting':
                    # At least one indicator should suggest mean reversion
                    assert hurst < 0.5 or vr < 0.9, f"Mean-reverting regime inconsistent: H={hurst:.3f}, VR={vr:.3f}"
                    
    def test_structural_break_protection_with_optimization(self, sample_data):
        """Тест 3: Проверяет защиту от структурных сдвигов с оптимизацией.
        
        Проверяет, что ленивая оптимизация ADF не нарушает
        логику защиты от структурных сдвигов.
        """
        # Create backtester with structural break protection
        backtester = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=50,
            z_threshold=2.0,
            structural_break_protection=True,
            adf_check_frequency=50,
            use_market_regime_cache=True,
            cointegration_test_frequency=25,
            adf_pvalue_threshold=0.05,
            correlation_window=100,  # Reduced from default 720 to fit test data
            pair_name="TEST_STRUCTURAL"
        )
        
        # Run backtest
        backtester.run()
        
        # Check structural break detection results
        correlations = backtester.rolling_correlations.dropna()
        adf_pvalues = backtester.adf_pvalues.dropna()
        
        assert len(correlations) > 0, "Should calculate rolling correlations"
        
        # Check correlation values are valid
        assert (correlations >= -1).all() and (correlations <= 1).all(), "Correlations should be in [-1,1]"
        
        # Check ADF p-values are valid (if any were calculated)
        if len(adf_pvalues) > 0:
            assert (adf_pvalues >= 0).all() and (adf_pvalues <= 1).all(), "ADF p-values should be in [0,1]"
            
        # Check that positions are adjusted when structural breaks are detected
        positions = backtester.results['position'].dropna()
        if len(adf_pvalues) > 0:
            # Find periods with high p-values (structural breaks)
            break_periods = adf_pvalues[adf_pvalues > 0.05].index
            if len(break_periods) > 0:
                # Positions should be more conservative during break periods
                break_positions = positions.loc[break_periods]
                # Should have some zero positions during breaks
                zero_ratio = (break_positions == 0).mean()
                assert zero_ratio > 0.1, f"Should have more zero positions during breaks, got {zero_ratio:.2%}"
                
    def test_cache_effectiveness(self, sample_data):
        """Тест 4: Проверяет эффективность кэширования.
        
        Проверяет, что кэш действительно ускоряет вычисления
        и дает те же результаты.
        """
        # Test without cache
        backtester_no_cache = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            use_market_regime_cache=False,
            regime_check_frequency=1,  # Check every bar for fair comparison
            pair_name="TEST_NO_CACHE"
        )
        
        # Test with cache
        backtester_with_cache = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            use_market_regime_cache=True,
            regime_check_frequency=1,  # Check every bar for fair comparison
            cache_cleanup_frequency=100,
            pair_name="TEST_WITH_CACHE"
        )
        
        # Run both backtests
        backtester_no_cache.run()
        backtester_with_cache.run()
        
        # Get results
        results_no_cache = backtester_no_cache.get_performance_metrics()
        results_with_cache = backtester_with_cache.get_performance_metrics()
        
        # Results should be very similar (allowing for small numerical differences)
        pnl_diff = abs(results_with_cache['total_pnl'] - results_no_cache['total_pnl'])
        if abs(results_no_cache['total_pnl']) > 0.01:  # Avoid division by zero
            pnl_diff_pct = pnl_diff / abs(results_no_cache['total_pnl'])
            assert pnl_diff_pct < 0.05, f"PnL difference too large: {pnl_diff_pct:.2%}"
            
        # Number of trades should be similar
        trade_diff = abs(results_with_cache['num_trades'] - results_no_cache['num_trades'])
        assert trade_diff <= 2, f"Trade count difference too large: {trade_diff}"
        
    def test_frequency_optimization_consistency(self, sample_data):
        """Тест 5: Проверяет консистентность при оптимизации частоты.
        
        Проверяет, что снижение частоты проверок не приводит
        к кардинально разным результатам.
        """
        # High frequency (baseline)
        backtester_high_freq = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            regime_check_frequency=1,  # Every bar
            use_market_regime_cache=True,
            pair_name="TEST_HIGH_FREQ"
        )
        
        # Low frequency (optimized)
        backtester_low_freq = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            regime_check_frequency=24,  # Every 24 bars (6 hours for 15-min data)
            use_market_regime_cache=True,
            pair_name="TEST_LOW_FREQ"
        )
        
        # Run both backtests
        backtester_high_freq.run()
        backtester_low_freq.run()
        
        # Get results
        results_high = backtester_high_freq.get_performance_metrics()
        results_low = backtester_low_freq.get_performance_metrics()
        
        # Results should be reasonably similar
        # (allowing for larger differences due to different regime detection frequency)
        if abs(results_high['total_pnl']) > 0.01:
            pnl_diff_pct = abs(results_low['total_pnl'] - results_high['total_pnl']) / abs(results_high['total_pnl'])
            assert pnl_diff_pct < 0.3, f"PnL difference too large with frequency optimization: {pnl_diff_pct:.2%}"
            
        # Sharpe ratio should be reasonably similar
        if abs(results_high['sharpe_ratio']) > 0.1:
            sharpe_diff_pct = abs(results_low['sharpe_ratio'] - results_high['sharpe_ratio']) / abs(results_high['sharpe_ratio'])
            assert sharpe_diff_pct < 0.5, f"Sharpe ratio difference too large: {sharpe_diff_pct:.2%}"
            
    def test_lazy_adf_effectiveness(self, sample_data):
        """Тест 6: Проверяет эффективность ленивого ADF.
        
        Проверяет, что ленивый ADF действительно снижает
        количество вычислений без потери качества.
        """
        # Create data with varying correlation
        modified_data = sample_data.copy()
        # Add some periods with different correlation structure
        n_points = len(modified_data)
        mid_point = n_points // 2
        
        # Second half has different relationship
        modified_data.iloc[mid_point:, 1] = modified_data.iloc[mid_point:, 1] * 0.5 + np.random.randn(n_points - mid_point) * 2
        
        # Mock ADF test to count calls
        with patch.object(BasePairBacktester, '_perform_adf_test') as mock_adf:
            mock_adf.return_value = 0.02  # Good p-value
            
            # Create backtester with lazy ADF
            backtester = BasePairBacktester(
                pair_data=modified_data,
                rolling_window=50,
                z_threshold=2.0,
                structural_break_protection=True,
                adf_check_frequency=25,  # Check every 25 bars
                use_market_regime_cache=True,
                cointegration_test_frequency=25,
                pair_name="TEST_LAZY_ADF"
            )
            
            # Run backtest
            backtester.run()
            
            # Check that ADF was called less frequently than possible
            total_possible_calls = (len(modified_data) - 50) // 25  # Rough estimate
            actual_calls = mock_adf.call_count
            
            print(f"Lazy ADF: {actual_calls} calls out of {total_possible_calls} possible")
            
            # Should have made some calls but not all possible
            assert actual_calls > 0, "Should make some ADF calls"
            assert actual_calls <= total_possible_calls, "Should not exceed maximum possible calls"
            
            # Get results to ensure backtest completed successfully
            results = backtester.get_performance_metrics()
            assert 'total_pnl' in results, "Should produce valid results"
            
    def test_parameter_edge_cases(self, sample_data):
        """Тест 7: Проверяет граничные случаи параметров оптимизации.
        
        Проверяет поведение системы при экстремальных значениях
        параметров оптимизации.
        """
        # Test with very high frequency (should work like no optimization)
        backtester_high = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            regime_check_frequency=1,  # Every bar
            adf_check_frequency=1,     # Every bar
            use_market_regime_cache=True,
            pair_name="TEST_HIGH_FREQ_EDGE"
        )
        
        # Test with very low frequency
        backtester_low = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            regime_check_frequency=100,  # Very infrequent
            adf_check_frequency=200,     # Very infrequent
            use_market_regime_cache=True,
            pair_name="TEST_LOW_FREQ_EDGE"
        )
        
        # Both should run without errors
        backtester_high.run()
        backtester_low.run()
        
        # Both should produce valid results
        results_high = backtester_high.get_performance_metrics()
        results_low = backtester_low.get_performance_metrics()
        
        assert 'total_pnl' in results_high, "High frequency should produce results"
        assert 'total_pnl' in results_low, "Low frequency should produce results"
        
        # High frequency should have more regime detections
        regimes_high = backtester_high.market_regime.dropna()
        regimes_low = backtester_low.market_regime.dropna()
        
        assert len(regimes_high) >= len(regimes_low), "High frequency should have more regime detections"
        
    def test_optimization_with_real_config(self, sample_data):
        """Тест 8: Проверяет работу с реальными параметрами конфигурации.
        
        Проверяет, что оптимизации работают с параметрами,
        аналогичными тем, что используются в main_2024.yaml.
        """
        # Use parameters similar to main_2024.yaml
        backtester = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=100,
            z_threshold=2.0,
            market_regime_detection=True,
            structural_break_protection=True,
            # Optimization parameters from config
            regime_check_frequency=48,      # 12 hours for 15-min data
            use_market_regime_cache=True,
            adf_check_frequency=5376,       # ~3 weeks for 15-min data
            cache_cleanup_frequency=1000,
            # Market regime parameters
            hurst_window=100,
            variance_ratio_window=50,
            hurst_trending_threshold=0.5,
            variance_ratio_trending_min=1.1,
            # Structural break parameters
            correlation_window=720,         # 180 hours for 15-min data
            cointegration_test_frequency=96, # 24 hours for 15-min data
            adf_pvalue_threshold=0.05,
            pair_name="TEST_REAL_CONFIG"
        )
        
        # Run backtest
        backtester.run()
        
        # Get results
        results = backtester.get_performance_metrics()
        
        # Should produce valid results
        assert 'total_pnl' in results, "Should produce PnL"
        assert 'sharpe_ratio' in results, "Should calculate Sharpe ratio"
        assert 'max_drawdown' in results, "Should calculate max drawdown"
        assert 'num_trades' in results, "Should count trades"
        
        # Check that optimization features were used
        assert backtester.market_regime_cache is not None, "Should use cache"
        assert len(backtester.market_regime.dropna()) > 0, "Should detect regimes"
        
        # Verify regime check frequency was respected
        regime_indices = backtester.market_regime.dropna().index
        if len(regime_indices) > 1:
            # Check that regime checks are spaced according to frequency
            regime_gaps = np.diff(regime_indices)
            # Most gaps should be around the frequency (allowing some variation)
            frequent_gaps = regime_gaps[regime_gaps <= backtester.regime_check_frequency * 1.5]
            assert len(frequent_gaps) > len(regime_gaps) * 0.7, "Most regime checks should respect frequency"
            
    def test_memory_usage_optimization(self, sample_data):
        """Тест 9: Проверяет оптимизацию использования памяти.
        
        Проверяет, что кэш не приводит к чрезмерному
        потреблению памяти.
        """
        # Create larger dataset
        large_data = pd.concat([sample_data] * 4, ignore_index=True)  # 4x larger
        
        # Create backtester with aggressive caching
        backtester = BasePairBacktester(
            pair_data=large_data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            use_market_regime_cache=True,
            regime_check_frequency=10,
            cache_cleanup_frequency=100,  # Frequent cleanup
            pair_name="TEST_MEMORY"
        )
        
        # Run backtest
        backtester.run()
        
        # Check cache size after completion
        cache = backtester.market_regime_cache
        
        # Count total cache entries
        total_hurst_entries = sum(len(asset_cache) for asset_cache in cache.hurst_cache.values())
        total_vr_entries = sum(len(asset_cache) for asset_cache in cache.vr_cache.values())
        total_corr_entries = sum(len(pair_cache) for pair_cache in cache.corr_cache.values())
        
        total_entries = total_hurst_entries + total_vr_entries + total_corr_entries
        
        # Cache should not grow unbounded
        max_reasonable_entries = len(large_data)  # Very lenient heuristic
        assert total_entries < max_reasonable_entries, f"Cache too large: {total_entries} entries (max: {max_reasonable_entries})"
        
        print(f"Memory test: {total_entries} total cache entries for {len(large_data)} data points")
        
    def test_error_handling_with_optimizations(self, sample_data):
        """Тест 10: Проверяет обработку ошибок при включенных оптимизациях.
        
        Проверяет, что оптимизации не нарушают обработку
        ошибочных ситуаций.
        """
        # Test with insufficient data
        small_data = sample_data.head(30)  # Very small dataset
        
        backtester = BasePairBacktester(
            pair_data=small_data,
            rolling_window=10,  # Much smaller than data size
            z_threshold=2.0,
            market_regime_detection=True,
            use_market_regime_cache=True,
            regime_check_frequency=10,
            pair_name="TEST_ERROR_HANDLING"
        )
        
        # Should handle gracefully without crashing
        try:
            backtester.run()
            results = backtester.get_performance_metrics()
            # Should produce some results even with limited data
            assert 'total_pnl' in results, "Should handle insufficient data gracefully"
        except Exception as e:
            # If it raises an exception, it should be informative
            assert "insufficient" in str(e).lower() or "data" in str(e).lower(), f"Unexpected error: {e}"
            
        # Test with NaN data
        nan_data = sample_data.copy()
        nan_data.iloc[100:110, 0] = np.nan  # Introduce NaN values
        
        backtester_nan = BasePairBacktester(
            pair_data=nan_data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            use_market_regime_cache=True,
            pair_name="TEST_NAN_HANDLING"
        )
        
        # Should handle NaN values gracefully
        backtester_nan.run()
        results_nan = backtester_nan.get_performance_metrics()
        assert 'total_pnl' in results_nan, "Should handle NaN values gracefully"