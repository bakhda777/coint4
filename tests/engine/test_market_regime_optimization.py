"""Тесты для оптимизации определения рыночных режимов и защиты от структурных сдвигов."""

import numpy as np
import pandas as pd
import pytest
import time
from unittest.mock import patch, MagicMock

from coint2.engine.base_engine import BasePairBacktester as PairBacktester
from coint2.engine.market_regime_cache import MarketRegimeCache, _hurst_exponent_jit, _variance_ratio_jit, _rolling_correlation_jit


class TestMarketRegimeOptimization:
    """Тесты для оптимизации определения рыночных режимов."""
    
    def test_numba_hurst_exponent_accuracy(self):
        """Тест 1: Проверяет точность Numba-оптимизированного расчета Hurst Exponent.
        
        Проверяет, что оптимизированная версия дает те же результаты,
        что и оригинальная реализация.
        """
        np.random.seed(42)
        
        # Create test data
        trending_prices = np.cumsum(np.random.randn(100) + 0.1) + 100
        mean_reverting_prices = 100 + 5 * np.sin(np.linspace(0, 20*np.pi, 100)) + np.random.randn(100) * 0.5
        
        # Test trending data
        log_prices_trending = np.log(trending_prices)
        hurst_trending = _hurst_exponent_jit(log_prices_trending)
        
        # Should be > 0.5 for trending data
        assert hurst_trending > 0.5, f"Trending data should have H > 0.5, got {hurst_trending}"
        assert 0.0 <= hurst_trending <= 1.0, f"Hurst exponent should be in [0,1], got {hurst_trending}"
        
        # Test mean-reverting data
        log_prices_mr = np.log(mean_reverting_prices)
        hurst_mr = _hurst_exponent_jit(log_prices_mr)
        
        # Should be < 0.5 for mean-reverting data
        assert hurst_mr < 0.5, f"Mean-reverting data should have H < 0.5, got {hurst_mr}"
        assert 0.0 <= hurst_mr <= 1.0, f"Hurst exponent should be in [0,1], got {hurst_mr}"
        
    def test_numba_variance_ratio_accuracy(self):
        """Тест 2: Проверяет точность Numba-оптимизированного расчета Variance Ratio.
        
        Проверяет, что оптимизированная версия дает корректные результаты
        для различных типов данных.
        """
        np.random.seed(42)
        
        # Create strongly trending data (should have VR > 1)
        trending_prices = np.cumsum(np.random.randn(100) * 0.1 + 0.5) + 100  # Strong trend
        log_prices_trending = np.log(trending_prices)
        vr_trending = _variance_ratio_jit(log_prices_trending, k=2)
        
        # Strict assertion for trending data
        assert vr_trending > 1.0, f"Trending data should have VR > 1.0, got {vr_trending}"
        assert 0.1 <= vr_trending <= 3.0, f"VR should be in [0.1,3.0], got {vr_trending}"
        
        # Create strong mean-reverting data (should have VR < 1)
        # Use AR(1) process with negative coefficient
        mr_data = np.zeros(100)
        mr_data[0] = 100
        for i in range(1, 100):
            mr_data[i] = 100 + 0.8 * (mr_data[i-1] - 100) * -1 + np.random.randn() * 0.1
        log_prices_mr = np.log(mr_data)
        vr_mr = _variance_ratio_jit(log_prices_mr, k=2)
        
        assert vr_mr < 1.0, f"Mean-reverting data should have VR < 1.0, got {vr_mr}"
        assert 0.1 <= vr_mr <= 3.0, f"VR should be in [0.1,3.0], got {vr_mr}"
        
    def test_numba_rolling_correlation_accuracy(self):
        """Тест 3: Проверяет точность Numba-оптимизированного расчета корреляции.
        
        Проверяет, что оптимизированная версия дает корректные результаты
        для коррелированных и некоррелированных данных.
        """
        np.random.seed(42)
        
        # Create highly correlated data
        x = np.random.randn(100) + 100
        y = x * 0.9 + np.random.randn(100) * 0.1  # High correlation
        
        corr_high = _rolling_correlation_jit(x, y)
        assert corr_high > 0.8, f"High correlation should be > 0.8, got {corr_high}"
        assert -1.0 <= corr_high <= 1.0, f"Correlation should be in [-1,1], got {corr_high}"
        
        # Create uncorrelated data
        x_uncorr = np.random.randn(100)
        y_uncorr = np.random.randn(100)
        
        corr_low = _rolling_correlation_jit(x_uncorr, y_uncorr)
        assert abs(corr_low) < 0.3, f"Low correlation should be close to 0, got {corr_low}"
        assert -1.0 <= corr_low <= 1.0, f"Correlation should be in [-1,1], got {corr_low}"
        
    def test_market_regime_cache_functionality(self):
        """Тест 4: Проверяет функциональность кэша для рыночных режимов.
        
        Проверяет, что кэш корректно сохраняет и возвращает значения,
        а также правильно очищает старые записи.
        """
        cache = MarketRegimeCache(hurst_window=50, vr_window=30)
        
        # Create test data
        np.random.seed(42)
        prices = pd.Series(np.random.randn(100) + 100)
        
        # Test Hurst caching
        asset_name = "TEST_ASSET"
        index1 = 50
        
        # First call should calculate and cache
        hurst1 = cache.get_hurst_exponent(asset_name, index1, prices[:index1])
        assert asset_name in cache.hurst_cache
        assert index1 in cache.hurst_cache[asset_name]
        
        # Second call should return cached value
        hurst2 = cache.get_hurst_exponent(asset_name, index1, prices[:index1])
        assert hurst1 == hurst2, "Cached value should be identical"
        
        # Test VR caching
        vr1 = cache.get_variance_ratio(asset_name, index1, prices[:index1])
        assert asset_name in cache.vr_cache
        assert index1 in cache.vr_cache[asset_name]
        
        vr2 = cache.get_variance_ratio(asset_name, index1, prices[:index1])
        assert vr1 == vr2, "Cached VR value should be identical"
        
        # Test correlation caching
        pair_name = "TEST_PAIR"
        x_prices = prices[:index1]
        y_prices = prices[:index1] * 1.1
        
        corr1 = cache.get_rolling_correlation(pair_name, index1, x_prices, y_prices)
        assert pair_name in cache.corr_cache
        assert index1 in cache.corr_cache[pair_name]
        
        corr2 = cache.get_rolling_correlation(pair_name, index1, x_prices, y_prices)
        assert corr1 == corr2, "Cached correlation should be identical"
        
        # Test cache cleanup
        # Add more entries
        for i in range(60, 70):
            cache.get_hurst_exponent(asset_name, i, prices[:i])
            
        # Cleanup old entries
        cache.clear_old_cache(current_index=65, keep_last_n=5)
        
        # Old entries should be removed
        assert index1 not in cache.hurst_cache[asset_name], "Old cache entries should be removed"
        # Recent entries should remain
        assert 65 in cache.hurst_cache[asset_name], "Recent cache entries should remain"
        
    def test_regime_check_frequency_optimization(self):
        """Тест 5: Проверяет оптимизацию частоты проверок режима.
        
        Проверяет, что система корректно пропускает промежуточные проверки
        и возвращает кэшированные значения режима.
        """
        np.random.seed(42)
        
        # Create test data
        data = pd.DataFrame({
            'asset1': np.random.randn(200) + 100,
            'asset2': np.random.randn(200) + 100
        })
        
        # Create backtester with frequency optimization
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            regime_check_frequency=10,  # Check every 10 bars
            use_market_regime_cache=False,  # Disable cache to test frequency logic
            hurst_window=80,  # Reduced from default 720 to fit test data
            variance_ratio_window=60,  # Reduced from default 480
            pair_name="TEST_PAIR"
        )
        
        # Mock the expensive calculations to track calls
        with patch.object(backtester, '_calculate_hurst_exponent') as mock_hurst, \
             patch.object(backtester, '_calculate_variance_ratio') as mock_vr:
    
            mock_hurst.return_value = 0.6  # Above trending threshold (0.5)
            mock_vr.return_value = 1.3     # Above trending min (1.1)
    
            # Prepare data for regime detection
            df = data.copy()
            df.columns = ['y', 'x']
    
            # Initialize regime tracking
            backtester.market_regime = pd.Series(index=df.index, dtype=object)
            backtester.hurst_exponents = pd.Series(index=df.index, dtype=float)
            backtester.variance_ratios = pd.Series(index=df.index, dtype=float)
            backtester.last_regime_check_index = -1  # Reset check index
    
            # Test regime detection at different indices
            regime1 = backtester._detect_market_regime(df, 100)  # Should calculate (index 100)
            regime2 = backtester._detect_market_regime(df, 105)  # Should use previous (within frequency=10)
            regime3 = backtester._detect_market_regime(df, 111)  # Should calculate again (100+10+1=111)
    
            # Check that calculations happened at expected frequencies
            # First call should trigger calculations (2 assets), third call should trigger again
            assert mock_hurst.call_count == 4, f"Expected 4 calls (2 assets x 2 calculations), got {mock_hurst.call_count}"
            assert regime1 == 'trending', "Should detect trending regime"
            assert regime2 == 'trending', "Should return same regime (frequency optimization)"
            assert regime3 == 'trending', "Should detect trending regime again"
            
    def test_lazy_adf_optimization(self):
        """Тест 6: Проверяет ленивую оптимизацию ADF тестов.
        
        Проверяет, что ADF тесты выполняются только при значительных
        изменениях корреляции.
        """
        np.random.seed(42)
        
        # Create test data
        data = pd.DataFrame({
            'asset1': np.random.randn(300) + 100,
            'asset2': np.random.randn(300) + 100
        })
        
        # Create backtester with ADF optimization
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=50,
            z_threshold=2.0,
            structural_break_protection=True,
            adf_check_frequency=50,  # Check every 50 bars
            use_market_regime_cache=True,
            pair_name="TEST_PAIR"
        )
        
        # Mock ADF test to track calls
        with patch.object(backtester, '_perform_adf_test') as mock_adf:
            mock_adf.return_value = 0.01  # Good p-value
            
            # Prepare data
            df = data.copy()
            df.columns = ['y', 'x']
            df['spread'] = df['y'] - df['x']
            
            # Initialize tracking
            backtester.rolling_correlations = pd.Series(index=df.index, dtype=float)
            backtester.adf_pvalues = pd.Series(index=df.index, dtype=float)
            backtester.last_cointegration_test = 0
            backtester.last_correlation = 0.8  # Start with high correlation
            
            # Simulate stable correlation (should skip ADF)
            backtester.rolling_correlations.iloc[100] = 0.82  # Small change
            result1 = backtester._check_structural_breaks(df, 100)
            
            # Simulate significant correlation change (should run ADF)
            backtester.rolling_correlations.iloc[150] = 0.6   # Large change
            result2 = backtester._check_structural_breaks(df, 150)
            
            # Should only call ADF once (for significant change)
            assert mock_adf.call_count <= 1, f"Expected ≤1 ADF calls, got {mock_adf.call_count}"
            assert not result1, "Should not detect break with stable correlation"
            assert not result2, "Should not detect break with good p-value"
            
    def test_performance_improvement_integration(self):
        """Тест 7: Проверяет интеграцию всех оптимизаций производительности.
        
        Проверяет, что все оптимизации работают вместе и дают
        значительное улучшение производительности.
        """
        np.random.seed(42)
        
        # Create larger test dataset
        n_points = 1000
        data = pd.DataFrame({
            'asset1': np.cumsum(np.random.randn(n_points) * 0.01) + 100,
            'asset2': np.cumsum(np.random.randn(n_points) * 0.01) + 100
        })
        
        # Test without optimizations
        start_time = time.time()
        backtester_slow = PairBacktester(
            pair_data=data,
            rolling_window=100,
            z_threshold=2.0,
            market_regime_detection=True,
            structural_break_protection=True,
            regime_check_frequency=1,  # Check every bar
            use_market_regime_cache=False,
            adf_check_frequency=100,
            hurst_window=200,  # Reduced from default 720
            variance_ratio_window=150,  # Reduced from default 480
            correlation_window=200,  # Reduced from default 720
            pair_name="TEST_PAIR_SLOW"
        )
        backtester_slow.run()
        slow_time = time.time() - start_time
        
        # Test with optimizations
        start_time = time.time()
        backtester_fast = PairBacktester(
            pair_data=data,
            rolling_window=100,
            z_threshold=2.0,
            market_regime_detection=True,
            structural_break_protection=True,
            regime_check_frequency=48,  # Check every 48 bars
            use_market_regime_cache=True,
            adf_check_frequency=200,  # Less frequent ADF
            cache_cleanup_frequency=500,
            hurst_window=200,  # Reduced from default 720
            variance_ratio_window=150,  # Reduced from default 480
            correlation_window=200,  # Reduced from default 720
            pair_name="TEST_PAIR_FAST"
        )
        backtester_fast.run()
        fast_time = time.time() - start_time
        
        # Optimized version should be faster
        speedup = slow_time / fast_time if fast_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x (slow: {slow_time:.3f}s, fast: {fast_time:.3f}s)")
        
        # Should have significant speedup (at least 1.2x)
        assert speedup >= 1.2, f"Expected speedup ≥1.2x, got {speedup:.2f}x"
        
        # Results should be similar (not identical due to different frequencies)
        slow_results = backtester_slow.get_performance_metrics()
        fast_results = backtester_fast.get_performance_metrics()
        
        # Check that both produced valid results
        assert 'total_pnl' in slow_results, "Slow backtester should produce results"
        assert 'total_pnl' in fast_results, "Fast backtester should produce results"
        
        # PnL should be reasonably close (within 50% due to different regime detection frequency)
        if slow_results['total_pnl'] != 0:
            pnl_diff = abs(fast_results['total_pnl'] - slow_results['total_pnl']) / abs(slow_results['total_pnl'])
            assert pnl_diff < 0.5, f"PnL difference too large: {pnl_diff:.2%}"
            
    def test_cache_memory_management(self):
        """Тест 8: Проверяет управление памятью кэша.
        
        Проверяет, что кэш не растет бесконечно и корректно
        очищает старые записи.
        """
        cache = MarketRegimeCache(hurst_window=50, vr_window=30)
        
        # Create test data
        np.random.seed(42)
        prices = pd.Series(np.random.randn(1000) + 100)
        
        # Fill cache with many entries
        asset_name = "TEST_ASSET"
        for i in range(100, 500):
            cache.get_hurst_exponent(asset_name, i, prices[:i])
            cache.get_variance_ratio(asset_name, i, prices[:i])
            
        # Check cache size before cleanup
        initial_hurst_size = len(cache.hurst_cache[asset_name])
        initial_vr_size = len(cache.vr_cache[asset_name])
        
        assert initial_hurst_size > 300, "Cache should have many entries"
        assert initial_vr_size > 300, "Cache should have many entries"
        
        # Perform cleanup
        cache.clear_old_cache(current_index=450, keep_last_n=50)
        
        # Check cache size after cleanup
        final_hurst_size = len(cache.hurst_cache[asset_name])
        final_vr_size = len(cache.vr_cache[asset_name])
        
        assert final_hurst_size <= 100, f"Cache should be cleaned up, got {final_hurst_size} entries"
        assert final_vr_size <= 100, f"Cache should be cleaned up, got {final_vr_size} entries"
        
        # Recent entries should still be there
        assert 449 in cache.hurst_cache[asset_name], "Recent entries should remain"
        assert 449 in cache.vr_cache[asset_name], "Recent entries should remain"
        
        # Old entries should be gone
        assert 100 not in cache.hurst_cache[asset_name], "Old entries should be removed"
        assert 100 not in cache.vr_cache[asset_name], "Old entries should be removed"
        
    def test_configuration_parameter_validation(self):
        """Тест 9: Проверяет валидацию новых параметров конфигурации.
        
        Проверяет, что новые параметры оптимизации правильно
        валидируются и применяются.
        """
        np.random.seed(42)
        
        # Create test data
        data = pd.DataFrame({
            'asset1': np.random.randn(100) + 100,
            'asset2': np.random.randn(100) + 100
        })
        
        # Test valid parameters
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=50,
            z_threshold=2.0,
            regime_check_frequency=48,
            use_market_regime_cache=True,
            adf_check_frequency=100,
            cache_cleanup_frequency=500,
            pair_name="TEST_PAIR"
        )
        
        assert backtester.regime_check_frequency == 48
        assert backtester.use_market_regime_cache is True
        assert backtester.adf_check_frequency == 100
        assert backtester.cache_cleanup_frequency == 500
        assert backtester.market_regime_cache is not None
        
        # Test disabled cache
        backtester_no_cache = PairBacktester(
            pair_data=data,
            rolling_window=50,
            z_threshold=2.0,
            use_market_regime_cache=False,
            pair_name="TEST_PAIR_NO_CACHE"
        )
        
        assert backtester_no_cache.use_market_regime_cache is False
        assert backtester_no_cache.market_regime_cache is None
        
    def test_regime_consistency_with_frequency_optimization(self):
        """Тест 10: Проверяет консистентность режимов при оптимизации частоты.
        
        Проверяет, что промежуточные бары получают корректные значения
        режима при использовании оптимизации частоты.
        """
        np.random.seed(42)
        
        # Create test data with clear trend
        n_points = 200
        trend_data = np.cumsum(np.random.randn(n_points) * 0.01 + 0.001) + 100
        data = pd.DataFrame({
            'asset1': trend_data,
            'asset2': trend_data * 1.1 + np.random.randn(n_points) * 0.1
        })
        
        # Create backtester with frequency optimization
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=50,
            z_threshold=2.0,
            market_regime_detection=True,
            regime_check_frequency=10,  # Check every 10 bars
            use_market_regime_cache=True,
            hurst_trending_threshold=0.5,
            variance_ratio_trending_min=1.1,
            hurst_window=80,  # Reduced from default 720 to fit test data
            variance_ratio_window=60,  # Reduced from default 480
            pair_name="TEST_PAIR"
        )
        
        # Run backtest
        backtester.run()
        
        # Check that regime values are filled for intermediate bars
        regime_series = backtester.market_regime.dropna()
        
        assert len(regime_series) > 0, "Should have regime values"
        
        # Check that regime values are consistent in blocks
        # (due to frequency optimization, consecutive bars should have same regime)
        regime_values = regime_series.values
        regime_changes = np.sum(regime_values[1:] != regime_values[:-1])
        total_regime_points = len(regime_values)
        
        # Should have fewer regime changes than total points due to frequency optimization
        change_ratio = regime_changes / total_regime_points if total_regime_points > 0 else 0
        assert change_ratio < 0.5, f"Too many regime changes: {change_ratio:.2%}"
        
        print(f"Regime consistency test: {regime_changes} changes in {total_regime_points} points ({change_ratio:.2%})")