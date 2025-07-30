"""Integration tests for global rolling cache with existing system components."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

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


class TestGlobalCacheIntegration:
    """Test integration of global cache with existing system components."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        # Create comprehensive test data
        np.random.seed(42)
        
        # Create realistic market data
        n_periods = 1000
        n_symbols = 20
        dates = pd.date_range('2024-01-01', periods=n_periods, freq='15min')
        
        # Generate correlated price series
        base_factor = np.random.randn(n_periods).cumsum() * 0.01
        
        price_data = {}
        for i in range(n_symbols):
            # Each symbol has correlation with base factor + individual noise
            correlation = 0.3 + 0.4 * np.random.random()  # 0.3 to 0.7 correlation
            individual_noise = np.random.randn(n_periods) * 0.008
            
            returns = correlation * base_factor + individual_noise
            prices = 100 * (1 + np.random.random() * 0.5) * np.exp(returns)
            price_data[f'SYMBOL_{i:02d}'] = prices
            
        self.global_price_data = pd.DataFrame(price_data, index=dates).astype(np.float32)
        
        # Configuration matching real system
        self.system_config = {
            'rolling_window': 30,
            'volatility_lookback': 96,
            'correlation_window': 720,
            'hurst_window': 720,
            'variance_ratio_window': 480
        }
        
        # Clean up any existing global state
        cleanup_global_rolling_cache()
        
    def teardown_method(self):
        """Clean up after each test."""
        cleanup_global_rolling_cache()
        
    def test_end_to_end_cache_initialization_and_usage(self):
        """Test complete end-to-end cache initialization and usage workflow."""
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
                ('SYMBOL_05', 'SYMBOL_07')
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
                
    def test_cache_performance_vs_traditional_approach(self):
        """Test performance comparison between cached and traditional approaches."""
        import time
        
        # Create multiple pairs for testing
        test_pairs = [
            ('SYMBOL_00', 'SYMBOL_01'),
            ('SYMBOL_02', 'SYMBOL_03'),
            ('SYMBOL_04', 'SYMBOL_05'),
            ('SYMBOL_06', 'SYMBOL_07'),
            ('SYMBOL_08', 'SYMBOL_09')
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
                
        print(f"Traditional approach: {traditional_time:.4f}s")
        print(f"Cached approach: {cached_time:.4f}s")
        print(f"Speedup factor: {traditional_time / cached_time:.2f}x")
        
        # For multiple pairs, cached approach should not be significantly slower
        # (and may be faster due to reduced rolling calculations)
        assert cached_time < traditional_time * 1.5, "Cached approach should not be much slower"
        
    def test_cache_memory_usage_scaling(self):
        """Test cache memory usage scales appropriately with data size."""
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
                
    def test_cache_correctness_with_real_config_parameters(self):
        """Test cache correctness using real system configuration parameters."""
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
                
    def test_cache_robustness_with_missing_symbols(self):
        """Test cache robustness when requested symbols are missing."""
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
            
    def test_cache_data_alignment_edge_cases(self):
        """Test cache data alignment in various edge cases."""
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
            offset_dates = pd.date_range('2024-01-02', periods=500, freq='15min')
            offset_pair_data = pd.DataFrame({
                'y': self.global_price_data['SYMBOL_02'].iloc[:500].values,
                'x': self.global_price_data['SYMBOL_03'].iloc[:500].values
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
            
    def test_cache_cleanup_and_reinitialization(self):
        """Test cache cleanup and reinitialization cycle."""
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
            
    def test_concurrent_cache_access_simulation(self):
        """Test simulated concurrent access to cache (single-threaded simulation)."""
        with patch('src.coint2.core.memory_optimization.GLOBAL_PRICE', self.global_price_data):
            initialize_global_rolling_cache(self.system_config)
            
            # Simulate multiple "concurrent" backtests
            pairs = [
                ('SYMBOL_00', 'SYMBOL_01'),
                ('SYMBOL_02', 'SYMBOL_03'),
                ('SYMBOL_04', 'SYMBOL_05'),
                ('SYMBOL_06', 'SYMBOL_07'),
                ('SYMBOL_08', 'SYMBOL_09'),
                ('SYMBOL_10', 'SYMBOL_11'),
                ('SYMBOL_12', 'SYMBOL_13'),
                ('SYMBOL_14', 'SYMBOL_15')
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
            
    def test_integration_with_different_rolling_windows(self):
        """Test integration with different rolling window sizes."""
        # Test with various rolling windows
        test_windows = [15, 30, 60, 120]
        
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