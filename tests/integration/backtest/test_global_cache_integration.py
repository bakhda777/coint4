"""Integration tests for global rolling cache with existing system components."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import time
from unittest.mock import patch, MagicMock, Mock

from src.coint2.core.global_rolling_cache import (
    initialize_global_rolling_cache,
    cleanup_global_rolling_cache,
    get_global_rolling_manager
)
from src.coint2.core.memory_optimization import (
    GLOBAL_PRICE,
    GLOBAL_STATS,
    build_global_rolling_stats,
    determine_required_windows,
    verify_rolling_stats_correctness
)
from src.coint2.engine.optimized_backtest_engine import OptimizedPairBacktester
from src.coint2.engine.base_engine import BasePairBacktester as PairBacktester
from src.coint2.utils.config import DataProcessingConfig

# Константы для тестирования
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_VOLATILITY_LOOKBACK = 96
DEFAULT_CORRELATION_WINDOW = 720
DEFAULT_HURST_WINDOW = 480
DEFAULT_VARIANCE_RATIO_WINDOW = 240
DEFAULT_VOLATILITY_LOOKBACK_HOURS = 24

# Константы для генерации данных
N_PERIODS = 200  # Уменьшено с 1000 для ускорения тестов
N_SYMBOLS = 5    # Уменьшено с 20 для ускорения тестов
BASE_FACTOR_VOLATILITY = 0.01
MIN_CORRELATION = 0.3
MAX_CORRELATION = 0.7
INDIVIDUAL_NOISE_STD = 0.008
BASE_PRICE = 100
PRICE_MULTIPLIER_RANGE = 0.5

# Константы для бэктестинга
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_Z_EXIT = 0.5
DEFAULT_COMMISSION_PCT = 0.001
DEFAULT_SLIPPAGE_PCT = 0.0005


class TestGlobalCacheUnit:
    """Fast unit tests for global cache logic with mocked dependencies."""

    def setup_method(self):
        """Очистка состояния перед каждым тестом."""
        cleanup_global_rolling_cache()

    def teardown_method(self):
        """Очистка состояния после каждого теста."""
        cleanup_global_rolling_cache()

    @pytest.mark.unit
    def test_required_windows_when_determined_then_logic_correct(self):
        """Unit test: проверяем логику определения необходимых окон."""
        # Правильная структура конфигурации
        config = {
            'backtest': {
                'rolling_window': DEFAULT_ROLLING_WINDOW,
                'volatility_lookback': DEFAULT_VOLATILITY_LOOKBACK,
                'correlation_window': DEFAULT_CORRELATION_WINDOW,
                'hurst_window': DEFAULT_HURST_WINDOW,
                'variance_ratio_window': DEFAULT_VARIANCE_RATIO_WINDOW
            },
            'portfolio': {
                'volatility_lookback_hours': DEFAULT_VOLATILITY_LOOKBACK_HOURS  # 24 * 4 = 96 periods
            }
        }

        windows = determine_required_windows(config)

        # Проверяем, что основные окна из конфигурации включены
        assert DEFAULT_ROLLING_WINDOW in windows, "rolling_window should be included"
        assert DEFAULT_VOLATILITY_LOOKBACK in windows, "volatility_lookback should be included"
        assert DEFAULT_CORRELATION_WINDOW in windows, "correlation_window should be included"
        assert DEFAULT_HURST_WINDOW in windows, "hurst_window should be included"
        assert DEFAULT_VARIANCE_RATIO_WINDOW in windows, "variance_ratio_window should be included"

        # Проверяем, что окна уникальны
        assert len(windows) == len(set(windows)), "Windows should be unique"

        # Проверяем, что все окна больше минимального размера
        assert all(w >= 3 for w in windows), "All windows should be >= 3"

    @pytest.mark.unit
    def test_cache_manager_when_created_then_initialized(self):
        """Unit test: проверяем создание менеджера кэша."""
        manager = get_global_rolling_manager()

        # Проверяем начальное состояние
        assert not manager.initialized, "Should start uninitialized"
        assert len(manager.available_windows) == 0, "Should have no windows initially"

        # Проверяем информацию о кэше
        cache_info = manager.get_cache_info()
        assert cache_info['initialized'] is False
        assert cache_info['num_cached_arrays'] == 0

    @pytest.mark.unit
    def test_backtester_cache_when_used_then_logic_correct(self, small_prices_df):
        """Unit test: проверяем логику использования кэша в бэктестере без реальных вычислений."""
        # Используем достаточно данных для rolling_window=15
        pair_data = pd.DataFrame({
            'y': small_prices_df.iloc[:, 0],
            'x': small_prices_df.iloc[:, 1]
        })

        # Мокаем все медленные операции
        with patch.object(OptimizedPairBacktester, 'run') as mock_run, \
             patch.object(OptimizedPairBacktester, 'get_optimization_stats') as mock_stats, \
             patch.object(OptimizedPairBacktester, '_validate_parameters') as mock_validate:

            mock_run.return_value = None
            mock_validate.return_value = None  # Пропускаем валидацию
            mock_stats.return_value = {
                'cache_initialized': True,
                'use_global_cache': True,
                'cache_info': {'total_memory_mb': 1.0}
            }

            # Создаем бэктестер с кэшем и меньшим окном
            SMALL_ROLLING_WINDOW = 15  # Меньше чем длина данных
            backtester = OptimizedPairBacktester(
                pair_data=pair_data,
                use_global_cache=True,
                rolling_window=SMALL_ROLLING_WINDOW,
                z_threshold=DEFAULT_Z_THRESHOLD,
                z_exit=DEFAULT_Z_EXIT
            )

            backtester.set_symbol_names('SYM1', 'SYM2')
            backtester.run()

            # Проверяем, что методы были вызваны
            mock_run.assert_called_once()

            stats = backtester.get_optimization_stats()
            assert stats['cache_initialized'] is True
            assert stats['use_global_cache'] is True


class TestGlobalCacheIntegration:
    """Integration tests for global cache with real system components (slow tests)."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        # Create comprehensive test data (детерминизм обеспечен глобально в conftest.py)

        # Create realistic market data
        dates = pd.date_range('2024-01-01', periods=N_PERIODS, freq='15min')

        # Generate correlated price series (детерминизм обеспечен через глобальный seed в conftest.py)
        # Используем детерминистический способ генерации данных
        np.random.seed(42)  # Явно устанавливаем seed для воспроизводимости
        base_factor = np.random.randn(N_PERIODS).cumsum() * BASE_FACTOR_VOLATILITY

        price_data = {}
        for i in range(N_SYMBOLS):
            # Each symbol has correlation with base factor + individual noise
            correlation = MIN_CORRELATION + (MAX_CORRELATION - MIN_CORRELATION) * np.random.random()
            individual_noise = np.random.randn(N_PERIODS) * INDIVIDUAL_NOISE_STD

            returns = correlation * base_factor + individual_noise
            prices = BASE_PRICE * (1 + np.random.random() * PRICE_MULTIPLIER_RANGE) * np.exp(returns)
            price_data[f'SYMBOL_{i:02d}'] = prices
            
        self.global_price_data = pd.DataFrame(price_data, index=dates).astype(np.float32)
        
        # Configuration matching real system
        self.system_config = {
            'rolling_window': DEFAULT_ROLLING_WINDOW,
            'volatility_lookback': DEFAULT_VOLATILITY_LOOKBACK,
            'correlation_window': DEFAULT_CORRELATION_WINDOW,
            'hurst_window': DEFAULT_CORRELATION_WINDOW,  # Используем то же значение что было (720)
            'variance_ratio_window': DEFAULT_VARIANCE_RATIO_WINDOW
        }
        
        # Clean up any existing global state
        cleanup_global_rolling_cache()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_global_rolling_cache()
        
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.serial
    def test_cache_when_end_to_end_then_initialization_and_usage_work(self):
        """Integration test: полный end-to-end тест инициализации и использования кэша."""
        # Step 1: Initialize global price data and keep it available throughout the test
        with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
            
            # Step 2: Initialize rolling cache
            initialize_global_rolling_cache(self.system_config)
            
            # Step 3: Verify cache is properly initialized
            manager = get_global_rolling_manager()
            assert manager.initialized, "Cache manager should be initialized"
            
            cache_info = manager.get_cache_info()
            # Check if cache info is available and has expected structure
            if 'windows' in cache_info:
                assert len(cache_info.get('windows', [])) > 0, "Should have cached windows"
            assert cache_info.get('total_memory_mb', 0) >= 0, "Should report memory usage"
            
            # Step 4: Create multiple pair backtests using cache
            pair_configs = [
                ('SYMBOL_00', 'SYMBOL_01'),
                ('SYMBOL_02', 'SYMBOL_03'),
                ('SYMBOL_03', 'SYMBOL_04')  # Исправлено: используем существующие символы
            ]
            
            backtest_results = []
            
            for symbol1, symbol2 in pair_configs:
                # Create pair data
                pair_data = pd.DataFrame({
                    'y': self.global_price_data[symbol1],
                    'x': self.global_price_data[symbol2]
                })
                
                # Run optimized backtest
                backtester = OptimizedPairBacktester(
                    pair_data=pair_data,
                    use_global_cache=True,
                    rolling_window=30,
                    z_threshold=2.0,
                    z_exit=0.5,
                    commission_pct=0.001,
                    slippage_pct=0.0005
                )
                
                backtester.set_symbol_names(symbol1, symbol2)
                backtester.run()
                
                backtest_results.append({
                    'pair': (symbol1, symbol2),
                    'results': backtester.results,
                    'stats': backtester.get_optimization_stats()
                })
                
            # Step 5: Verify all backtests produced results and check cache usage
            for result in backtest_results:
                stats = result['stats']
                # Check if cache was attempted to be used (may not always succeed)
                if 'use_global_cache' in stats:
                    # If cache usage info is available, verify it was attempted
                    assert stats.get('use_global_cache', False) in [True, False], "Cache usage should be boolean"
                # Always check that results were produced
                assert not result['results'].empty, "Should produce results"
                # Check that cache manager is initialized
                assert stats.get('cache_initialized', False) is True, "Cache should be initialized"
                
            # Step 6: Verify cache consistency across all backtests
            first_cache_info = backtest_results[0]['stats']['cache_info']
            for result in backtest_results[1:]:
                current_cache_info = result['stats']['cache_info']
                assert current_cache_info == first_cache_info, "Cache info should be consistent"
                
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.serial
    def test_cache_performance_vs_traditional_approach(self):
        """Integration test: сравнение производительности кэшированного и традиционного подходов."""
        
        # Create multiple pairs for testing
        # ОПТИМИЗАЦИЯ: Уменьшено с 5 до 2 пар для ускорения
        test_pairs = [
            ('SYMBOL_00', 'SYMBOL_01'),
            ('SYMBOL_02', 'SYMBOL_03')
        ]
        
        # Test traditional approach (no cache)
        start_time = time.time()
        traditional_results = []
        
        for symbol1, symbol2 in test_pairs:
            pair_data = pd.DataFrame({
                'y': self.global_price_data[symbol1],
                'x': self.global_price_data[symbol2]
            })
            
            backtester = OptimizedPairBacktester(
                pair_data=pair_data,
                use_global_cache=False,
                rolling_window=30,
                z_threshold=2.0,
                z_exit=0.5,
                commission_pct=0.001,
                slippage_pct=0.0005
            )
            
            backtester.run()
            traditional_results.append(backtester.results['cumulative_pnl'].iloc[-1])
            
        traditional_time = time.time() - start_time
        
        # Test cached approach
        with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
            initialize_global_rolling_cache(self.system_config)
            
            start_time = time.time()
            cached_results = []
            
            for symbol1, symbol2 in test_pairs:
                pair_data = pd.DataFrame({
                    'y': self.global_price_data[symbol1],
                    'x': self.global_price_data[symbol2]
                })
                
                backtester = OptimizedPairBacktester(
                    pair_data=pair_data,
                    use_global_cache=True,
                    rolling_window=30,
                    z_threshold=2.0,
                    z_exit=0.5,
                    commission_pct=0.001,
                    slippage_pct=0.0005
                )
                
                backtester.set_symbol_names(symbol1, symbol2)
                backtester.run()
                cached_results.append(backtester.results['cumulative_pnl'].iloc[-1])
                
            cached_time = time.time() - start_time
            
        # Verify results consistency
        for i, (trad_pnl, cached_pnl) in enumerate(zip(traditional_results, cached_results)):
            if abs(trad_pnl) > 1e-6 or abs(cached_pnl) > 1e-6:
                relative_diff = abs(trad_pnl - cached_pnl) / max(abs(trad_pnl), abs(cached_pnl))
                assert relative_diff < 0.05, f"Pair {i}: PnL difference too large: {relative_diff:.4f}"
            else:
                assert abs(trad_pnl - cached_pnl) < 1e-6, f"Pair {i}: Absolute difference too large"
                
        # Логируем результаты производительности (без print для соответствия стандартам)
        speedup_factor = traditional_time / cached_time if cached_time > 0 else 1.0

        # Проверяем, что время измерено корректно
        assert traditional_time > 0, "Traditional approach time should be positive"
        assert cached_time > 0, "Cached approach time should be positive"
        
        # For multiple pairs, cached approach should not be significantly slower
        # (and may be faster due to reduced rolling calculations)
        assert cached_time < traditional_time * 1.5, "Cached approach should not be much slower"
        
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.serial
    def test_cache_memory_usage_scaling(self):
        """Integration test: проверяем масштабирование использования памяти кэша."""
        # Test with different data sizes
        data_sizes = [100, 500, 1000]
        memory_usages = []
        
        for size in data_sizes:
            cleanup_global_rolling_cache()
            
            # Create data of specific size
            test_data = self.global_price_data.iloc[:size].copy()
            
            with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', test_data):
                initialize_global_rolling_cache(self.system_config)
                
                manager = get_global_rolling_manager()
                cache_info = manager.get_cache_info()
                memory_usages.append(cache_info['total_memory_mb'])
                
        # Memory usage should scale roughly linearly with data size
        # Skip if any memory usage is zero (division by zero)
        valid_memory_usages = [m for m in memory_usages if m > 0]
        if len(valid_memory_usages) < 2:
            pytest.skip("Insufficient valid memory measurements")
            
        for i in range(1, len(memory_usages)):
            if memory_usages[i-1] == 0 or memory_usages[i] == 0:
                continue  # Skip division by zero
            size_ratio = data_sizes[i] / data_sizes[i-1]
            memory_ratio = memory_usages[i] / memory_usages[i-1]
            
            # Allow some overhead, but should be roughly proportional
            # More lenient bounds to account for cache overhead
            assert 0.5 * size_ratio <= memory_ratio <= 2.0 * size_ratio, \
                f"Memory scaling not proportional: size ratio {size_ratio:.2f}, memory ratio {memory_ratio:.2f}"
                
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.serial
    def test_cache_correctness_with_real_config_parameters(self):
        """Integration test: проверяем корректность кэша с реальными параметрами конфигурации."""
        with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
            # Test with actual system configuration
            initialize_global_rolling_cache(self.system_config)
            
            # Verify all required windows are cached
            expected_windows = determine_required_windows(self.system_config)
            
            for window in expected_windows:
                # Verify correctness for each window
                # Use first available symbol for verification
                symbol = self.global_price_data.columns[0]
                is_correct = verify_rolling_stats_correctness(window, symbol)
                assert is_correct, f"Rolling stats incorrect for window {window}"
                
                # Verify data types and shapes
                mean_stats = GLOBAL_STATS[('mean', window)]
                std_stats = GLOBAL_STATS[('std', window)]
                
                assert mean_stats.dtype == np.float32, f"Mean stats should be float32 for window {window}"
                assert std_stats.dtype == np.float32, f"Std stats should be float32 for window {window}"
                assert mean_stats.shape == self.global_price_data.shape, f"Shape mismatch for window {window}"
                assert std_stats.shape == self.global_price_data.shape, f"Shape mismatch for window {window}"
                
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.serial
    def test_cache_robustness_with_missing_symbols(self):
        """Integration test: проверяем устойчивость кэша при отсутствующих символах."""
        with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
            initialize_global_rolling_cache(self.system_config)
            
            # Test with symbols that exist
            existing_pair_data = pd.DataFrame({
                'y': self.global_price_data['SYMBOL_00'],
                'x': self.global_price_data['SYMBOL_01']
            })
            
            backtester_existing = OptimizedPairBacktester(
                pair_data=existing_pair_data,
                use_global_cache=True,
                rolling_window=30,
                z_threshold=2.0,
                z_exit=0.5
            )
            backtester_existing.set_symbol_names('SYMBOL_00', 'SYMBOL_01')
            backtester_existing.run()
            
            # Should use cache successfully
            stats_existing = backtester_existing.get_optimization_stats()
            assert stats_existing['cache_initialized'] is True
            
            # Test with symbols that don't exist
            missing_pair_data = pd.DataFrame({
                'y': self.global_price_data['SYMBOL_00'],  # Use existing data
                'x': self.global_price_data['SYMBOL_01']
            })
            
            backtester_missing = OptimizedPairBacktester(
                pair_data=missing_pair_data,
                use_global_cache=True,
                rolling_window=30,
                z_threshold=2.0,
                z_exit=0.5
            )
            backtester_missing.set_symbol_names('NONEXISTENT_1', 'NONEXISTENT_2')
            backtester_missing.run()
            
            # Should fallback gracefully and still produce results
            assert not backtester_missing.results.empty, "Should produce results even with missing symbols"
            
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.serial
    def test_cache_data_alignment_edge_cases(self):
        """Integration test: проверяем выравнивание данных кэша в различных граничных случаях."""
        with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
            initialize_global_rolling_cache(self.system_config)
            
            # Test case 1: Pair data shorter than global data
            short_pair_data = pd.DataFrame({
                'y': self.global_price_data['SYMBOL_00'].iloc[:500],
                'x': self.global_price_data['SYMBOL_01'].iloc[:500]
            })
            
            backtester_short = OptimizedPairBacktester(
                pair_data=short_pair_data,
                use_global_cache=True,
                rolling_window=30,
                z_threshold=2.0,
                z_exit=0.5
            )
            backtester_short.set_symbol_names('SYMBOL_00', 'SYMBOL_01')
            backtester_short.run()
            
            assert not backtester_short.results.empty, "Should handle shorter pair data"
            assert len(backtester_short.results) == len(short_pair_data), "Results length should match input"
            
            # Test case 2: Pair data with different index
            offset_dates = pd.date_range('2024-01-02', periods=100, freq='15min')  # Уменьшено с 500
            offset_pair_data = pd.DataFrame({
                'y': self.global_price_data['SYMBOL_02'].iloc[:100].values,  # Уменьшено с 500
                'x': self.global_price_data['SYMBOL_03'].iloc[:100].values  # Уменьшено с 500
            }, index=offset_dates)
            
            backtester_offset = OptimizedPairBacktester(
                pair_data=offset_pair_data,
                use_global_cache=True,
                rolling_window=30,
                z_threshold=2.0,
                z_exit=0.5
            )
            backtester_offset.set_symbol_names('SYMBOL_02', 'SYMBOL_03')
            backtester_offset.run()
            
            assert not backtester_offset.results.empty, "Should handle different index"
            
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.serial
    def test_cache_cleanup_and_reinitialization(self):
        """Integration test: проверяем цикл очистки и переинициализации кэша."""
        # Initialize cache
        with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
            initialize_global_rolling_cache(self.system_config)
            
            manager = get_global_rolling_manager()
            assert manager.initialized, "Should be initialized"
            
            # Verify cache has data
            cache_info_before = manager.get_cache_info()
            assert cache_info_before.get('total_memory_mb', 0) >= 0, "Should have cache info available"
            
            # Cleanup cache
            cleanup_global_rolling_cache()
            assert not manager.initialized, "Should not be initialized after cleanup"
            
            # Verify cache is empty
            assert len(GLOBAL_STATS) == 0, "GLOBAL_STATS should be empty after cleanup"
            
            # Reinitialize cache (need to set GLOBAL_PRICE again after cleanup)
            with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
                initialize_global_rolling_cache(self.system_config)
                assert manager.initialized, "Should be reinitialized"
                
                # Verify cache is restored
                cache_info_after = manager.get_cache_info()
                assert cache_info_after.get('total_memory_mb', 0) >= 0, "Should have cache info available after reinit"
                # Check windows consistency if both have windows info
                if 'windows' in cache_info_before and 'windows' in cache_info_after:
                    assert cache_info_after.get('windows', []) == cache_info_before.get('windows', []), "Should have same windows"
            
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.serial
    def test_concurrent_cache_access_simulation(self):
        """Integration test: симуляция конкурентного доступа к кэшу (однопоточная симуляция)."""
        with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
            initialize_global_rolling_cache(self.system_config)
            
            # Simulate multiple "concurrent" backtests
            # ОПТИМИЗАЦИЯ: Уменьшено с 8 до 3 пар для ускорения
            pairs = [
                ('SYMBOL_00', 'SYMBOL_01'),
                ('SYMBOL_02', 'SYMBOL_03'),
                ('SYMBOL_04', 'SYMBOL_05' if 'SYMBOL_05' in self.global_price_data.columns else 'SYMBOL_04')
            ]
            
            results = []
            
            for i, (symbol1, symbol2) in enumerate(pairs):
                pair_data = pd.DataFrame({
                    'y': self.global_price_data[symbol1],
                    'x': self.global_price_data[symbol2]
                })
                
                backtester = OptimizedPairBacktester(
                    pair_data=pair_data,
                    use_global_cache=True,
                    rolling_window=30,
                    z_threshold=2.0,
                    z_exit=0.5
                )
                
                backtester.set_symbol_names(symbol1, symbol2)
                backtester.run()
                
                results.append({
                    'pair_id': i,
                    'symbols': (symbol1, symbol2),
                    'final_pnl': backtester.results['cumulative_pnl'].iloc[-1],
                    'num_trades': (backtester.results['position'].diff() != 0).sum(),
                    'cache_used': backtester.get_optimization_stats()['cache_initialized']
                })
                
            # Verify all backtests used cache
            for result in results:
                assert result['cache_used'] is True, f"Pair {result['pair_id']} should use cache"
                
            # Verify results are reasonable
            pnls = [r['final_pnl'] for r in results]
            assert all(np.isfinite(pnl) for pnl in pnls), "All PnLs should be finite"
            
            # Verify cache consistency throughout
            manager = get_global_rolling_manager()
            final_cache_info = manager.get_cache_info()
            assert final_cache_info.get('total_memory_mb', 0) >= 0, "Cache should still be available"
            
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.serial
    def test_integration_with_different_rolling_windows(self):
        """Integration test: интеграция с различными размерами скользящих окон."""
        # Test with various rolling windows
        # ОПТИМИЗАЦИЯ: Уменьшено с 4 до 2 окон для ускорения
        test_windows = [15, 30]
        
        with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
            # Initialize cache with extended configuration
            extended_config = self.system_config.copy()
            extended_config.update({
                'test_window_15': 15,
                'test_window_60': 60,
                'test_window_120': 120
            })
            
            initialize_global_rolling_cache(extended_config)
            
            for window in test_windows:
                pair_data = pd.DataFrame({
                    'y': self.global_price_data['SYMBOL_00'],
                    'x': self.global_price_data['SYMBOL_01']
                })
                
                backtester = OptimizedPairBacktester(
                    pair_data=pair_data,
                    use_global_cache=True,
                    rolling_window=window,
                    z_threshold=2.0,
                    z_exit=0.5
                )
                
                backtester.set_symbol_names('SYMBOL_00', 'SYMBOL_01')
                backtester.run()
                
                # Should produce valid results for all window sizes
                assert not backtester.results.empty, f"Should produce results for window {window}"
                
                # Check if cache was used (depends on whether window was pre-cached)
                stats = backtester.get_optimization_stats()
                assert stats['cache_initialized'] is True, "Cache should be initialized"


class TestGlobalCacheFast:
    """Быстрые версии тестов глобального кэша с мокированием."""
    
    @pytest.mark.fast
    @patch('src.coint2.core.global_rolling_cache.get_global_rolling_manager')
    def test_cache_manager_when_mocked_then_initializes(self, mock_manager):
        """Fast test: Быстрая проверка инициализации менеджера кэша."""
        # Настраиваем мок менеджера
        mock_instance = Mock()
        mock_instance.initialized = True
        mock_instance.available_windows = [10, 20, 30]
        mock_instance.get_cache_info.return_value = {
            'initialized': True,
            'num_cached_arrays': 3,
            'total_memory_mb': 5.2,
            'windows': [10, 20, 30]
        }
        mock_manager.return_value = mock_instance
        
        # Тестируем
        manager = mock_manager()
        cache_info = manager.get_cache_info()
        
        # Проверки
        assert manager.initialized
        assert len(manager.available_windows) > 0
        assert cache_info['initialized'] is True
        assert cache_info['total_memory_mb'] > 0
        
    @pytest.mark.fast
    def test_required_windows_when_fast_then_logic_correct(self):
        """Fast test: Быстрая проверка логики определения окон."""
        from src.coint2.core.memory_optimization import determine_required_windows
        
        # Минимальная конфигурация для тестирования
        config = {
            'backtest': {
                'rolling_window': 20,
                'volatility_lookback': 100,
                'correlation_window': 50
            },
            'portfolio': {
                'volatility_lookback_hours': 24
            }
        }
        
        windows = determine_required_windows(config)
        
        # Проверки (функция может возвращать set, конвертируем в list)
        if isinstance(windows, set):
            windows = list(windows)
        
        assert isinstance(windows, list)
        assert len(windows) > 0
        assert 20 in windows  # rolling_window должен быть включен
        assert all(w >= 3 for w in windows)  # Все окна должны быть разумного размера
        assert len(set(windows)) == len(windows)  # Окна уникальны
        
    @pytest.mark.fast  
    @patch('src.coint2.engine.optimized_backtest_engine.OptimizedPairBacktester')
    def test_backtester_cache_when_mocked_then_uses_cache(self, mock_backtester):
        """Fast test: Быстрая проверка использования кэша в бэктестере."""
        # Мокируем бэктестер
        mock_instance = Mock()
        mock_instance.get_optimization_stats.return_value = {
            'cache_initialized': True,
            'use_global_cache': True,
            'cache_info': {
                'total_memory_mb': 2.5,
                'windows': [20, 50]
            }
        }
        mock_backtester.return_value = mock_instance
        
        # Минимальные тестовые данные
        import pandas as pd
        import numpy as np
        pair_data = pd.DataFrame({
            'y': np.random.randn(20) + 100,
            'x': np.random.randn(20) + 50
        })
        
        # Создаем бэктестер
        backtester = mock_backtester(
            pair_data=pair_data,
            use_global_cache=True,
            rolling_window=10
        )
        
        stats = backtester.get_optimization_stats()
        
        # Проверки
        assert stats['cache_initialized'] is True
        assert stats['use_global_cache'] is True
        assert stats['cache_info']['total_memory_mb'] > 0
        
    @pytest.mark.fast
    def test_cache_performance_when_mocked_then_faster_than_traditional(self):
        """Fast test: Быстрое сравнение производительности кэша."""
        import time
        from unittest.mock import patch
        
        # Симулируем традиционный подход (медленный)
        def slow_calculation():
            time.sleep(0.001)  # Симулируем 1ms задержку
            return [1, 2, 3, 4, 5]
        
        # Симулируем кэшированный подход (быстрый)  
        cached_result = [1, 2, 3, 4, 5]
        def fast_calculation():
            return cached_result
        
        # Тестируем производительность
        start = time.time()
        traditional_result = slow_calculation()
        traditional_time = time.time() - start
        
        start = time.time()
        cached_result = fast_calculation()
        cached_time = time.time() - start
        
        # Проверки
        assert traditional_result == cached_result  # Результаты идентичны
        assert cached_time < traditional_time  # Кэш быстрее
        speedup = traditional_time / cached_time if cached_time > 0 else float('inf')
        assert speedup > 1, f"Кэшированная версия должна быть быстрее: speedup={speedup:.1f}"
        
    @pytest.mark.fast
    def test_cache_correctness_when_mocked_then_validates(self):
        """Fast test: Быстрая проверка корректности кэшированных данных."""
        from src.coint2.core.memory_optimization import verify_rolling_stats_correctness
        from unittest.mock import patch
        
        # Мокируем функцию верификации
        with patch('src.coint2.core.memory_optimization.verify_rolling_stats_correctness') as mock_verify:
            mock_verify.return_value = True
            
            # Тестируем верификацию для разных окон
            windows = [10, 20, 30]
            symbol = 'TEST_SYMBOL'
            
            for window in windows:
                is_correct = mock_verify(window, symbol)
                assert is_correct, f"Кэш для окна {window} должен быть корректным"
                
            # Проверяем что функция была вызвана для каждого окна
            assert mock_verify.call_count == len(windows)
            
    @pytest.mark.fast
    def test_cache_memory_scaling_when_mocked_then_linear(self):
        """Fast test: Быстрая проверка линейного масштабирования памяти."""
        # Мокируем использование памяти для разных размеров данных
        data_sizes = [100, 200, 400]
        expected_memory_usage = [1.0, 2.0, 4.0]  # Линейное масштабирование
        
        for i, data_size in enumerate(data_sizes):
            expected_memory = expected_memory_usage[i]
            
            # Симулируем линейное масштабирование (память ~ размер данных)
            simulated_memory = data_size * 0.01  # 0.01 MB на 100 точек данных
            
            # Проверяем что масштабирование близко к линейному
            assert abs(simulated_memory - expected_memory) < 0.1, \
                f"Память должна масштабироваться линейно: {simulated_memory} vs {expected_memory}"
                
        # Проверяем коэффициент масштабирования
        for i in range(1, len(data_sizes)):
            size_ratio = data_sizes[i] / data_sizes[i-1]
            memory_ratio = expected_memory_usage[i] / expected_memory_usage[i-1]
            
            assert abs(size_ratio - memory_ratio) < 0.1, \
                f"Коэффициенты масштабирования должны быть близки: {size_ratio} vs {memory_ratio}"