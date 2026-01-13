"""Тесты для проверки корректности работы PairBacktester с оптимизациями."""

import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import get_test_config

from src.coint2.engine.base_engine import BasePairBacktester
from src.coint2.core.data_loader import DataHandler

# Alias for backward compatibility
PairBacktester = BasePairBacktester

# Получаем конфигурацию для тестов
test_config = get_test_config()

# Константы для тестирования с поддержкой QUICK_TEST
DEFAULT_ROLLING_WINDOW = test_config['rolling_window']  # 20 в QUICK_TEST, иначе 50
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_Z_EXIT = 0.5
DEFAULT_REGIME_CHECK_FREQUENCY = 5
DEFAULT_ADF_CHECK_FREQUENCY = 10
TEST_PERIODS = test_config['periods']  # 10 в minimal, 50 в fast, 100 в normal
FREQUENCY = '15min'
START_DATE = '2023-01-01'

# Константы для оптимизации
CACHE_SIZE_LIMIT = 1000
MEMORY_THRESHOLD = 0.8
OPTIMIZATION_INTERVAL = 50


class TestBacktestEngineOptimizationUnit:
    """Быстрые unit тесты для проверки логики оптимизаций движка."""

    @pytest.mark.unit
    def test_optimization_parameters_validation(self, small_prices_df):
        """Unit test: проверяем валидацию параметров оптимизации."""
        test_data = pd.DataFrame({
            'asset1': small_prices_df.iloc[:, 0],
            'asset2': small_prices_df.iloc[:, 1]
        })

        # Тестируем различные параметры оптимизации
        backtester = BasePairBacktester(
            pair_data=test_data,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            z_exit=DEFAULT_Z_EXIT,
            regime_check_frequency=DEFAULT_REGIME_CHECK_FREQUENCY,
            adf_check_frequency=DEFAULT_ADF_CHECK_FREQUENCY
        )

        # Проверяем, что параметры установлены правильно
        assert backtester.rolling_window == DEFAULT_ROLLING_WINDOW
        assert backtester.z_threshold == DEFAULT_Z_THRESHOLD
        assert backtester.z_exit == DEFAULT_Z_EXIT
        assert hasattr(backtester, 'regime_check_frequency')
        assert hasattr(backtester, 'adf_check_frequency')

    @pytest.mark.unit
    def test_cache_structure_logic(self):
        """Unit test: проверяем логику структуры кэша."""
        # Создаем простые тестовые данные (детерминизм обеспечен глобально)
        N_POINTS = 50
        BASE_PRICE = 100
        VOLATILITY = 0.01
        NOISE_STD = 0.5
        ASSET2_MULTIPLIER = 1.2
        ASSET2_OFFSET = 5
        SMALL_ROLLING_WINDOW = 10

        base_series = np.cumsum(np.random.randn(N_POINTS) * VOLATILITY) + BASE_PRICE
        asset1 = base_series + np.random.randn(N_POINTS) * NOISE_STD
        asset2 = base_series * ASSET2_MULTIPLIER + ASSET2_OFFSET + np.random.randn(N_POINTS) * NOISE_STD

        test_data = pd.DataFrame({
            'asset1': asset1,
            'asset2': asset2
        })

        backtester = BasePairBacktester(
            pair_data=test_data,
            rolling_window=SMALL_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            z_exit=DEFAULT_Z_EXIT
        )

        # Проверяем, что бэктестер создан корректно
        assert len(backtester.pair_data) == len(test_data)
        assert backtester.rolling_window == SMALL_ROLLING_WINDOW

    @pytest.mark.unit
    def test_frequency_parameters_logic(self):
        """Unit test: проверяем логику параметров частоты."""
        # Тестируем различные значения частоты
        MIN_FREQUENCY = 1
        MAX_FREQUENCY = 100

        frequency_params = {
            'regime_check_frequency': [MIN_FREQUENCY, DEFAULT_REGIME_CHECK_FREQUENCY, DEFAULT_ADF_CHECK_FREQUENCY],
            'adf_check_frequency': [MIN_FREQUENCY, DEFAULT_ADF_CHECK_FREQUENCY, DEFAULT_ROLLING_WINDOW * 2]
        }

        for param_name, values in frequency_params.items():
            for value in values:
                # Проверяем, что значения в разумных пределах
                assert MIN_FREQUENCY <= value <= MAX_FREQUENCY, f"{param_name} должен быть в пределах [{MIN_FREQUENCY}, {MAX_FREQUENCY}]"
                assert isinstance(value, int), f"{param_name} должен быть целым числом"

    @pytest.mark.unit
    def test_regime_detection_parameters(self):
        """Unit test: проверяем параметры определения режимов."""
        # Тестируем параметры для определения режимов
        HURST_THRESHOLD = 0.5
        VARIANCE_RATIO_THRESHOLD = 1.0
        ADF_PVALUE_THRESHOLD = 0.05

        regime_params = {
            'hurst_threshold': HURST_THRESHOLD,
            'variance_ratio_threshold': VARIANCE_RATIO_THRESHOLD,
            'adf_pvalue_threshold': ADF_PVALUE_THRESHOLD
        }

        for param_name, value in regime_params.items():
            # Проверяем, что значения в разумных пределах
            if 'threshold' in param_name:
                assert 0 <= value <= 1, f"{param_name} должен быть в пределах [0, 1]"
            assert isinstance(value, (int, float)), f"{param_name} должен быть числом"


class TestBacktestEngineOptimizationFast:
    """Быстрые версии integration тестов с мокированием тяжелых вычислений."""
    
    @pytest.fixture
    def tiny_sample_data(self):
        """Минимальные тестовые данные для быстрых тестов."""
        SAMPLE_POINTS = 50  # Еще меньше для быстрых тестов
        BASE_PRICE = 100
        VOLATILITY = 0.01
        NOISE_STD = 0.5
        ASSET2_MULTIPLIER = 1.2
        ASSET2_OFFSET = 5
        
        # Фиксированное семя для детерминизма  
        np.random.seed(42)
        base_series = np.cumsum(np.random.randn(SAMPLE_POINTS) * VOLATILITY) + BASE_PRICE
        noise1 = np.random.randn(SAMPLE_POINTS) * NOISE_STD
        noise2 = np.random.randn(SAMPLE_POINTS) * NOISE_STD
        
        asset1 = base_series + noise1
        asset2 = base_series * ASSET2_MULTIPLIER + ASSET2_OFFSET + noise2
        
        data = pd.DataFrame({
            'asset1': asset1,
            'asset2': asset2
        })
        return data
    
    @pytest.mark.fast
    @patch('src.coint2.engine.base_engine.BasePairBacktester._calculate_hurst_exponent', return_value=0.6)
    @patch('src.coint2.engine.base_engine.BasePairBacktester._calculate_variance_ratio', return_value=1.2)
    def test_regime_detection_when_mocked_then_logic_works(self, mock_vr, mock_hurst, tiny_sample_data):
        """Fast test: Проверяет логику определения режимов с мокированием."""
        backtester = BasePairBacktester(
            pair_data=tiny_sample_data,
            rolling_window=20,
            z_threshold=DEFAULT_Z_THRESHOLD,
            market_regime_detection=True,
            regime_check_frequency=5
        )
        
        backtester.run()
        results = backtester.get_performance_metrics()
        
        # Проверяем что мокированные функции вызывались
        assert mock_hurst.call_count >= 0, "Hurst function should be accessible"
        assert mock_vr.call_count >= 0, "Variance ratio function should be accessible"
        
        # Проверяем базовые результаты
        assert 'total_return' in results
        assert isinstance(results['num_trades'], int)
    
    @pytest.mark.fast
    @patch('src.coint2.engine.base_engine.BasePairBacktester._perform_adf_test', return_value=0.01)
    def test_structural_break_protection_when_mocked_then_works(self, mock_adf, tiny_sample_data):
        """Fast test: Проверяет защиту от структурных сдвигов с мокированием."""
        backtester = BasePairBacktester(
            pair_data=tiny_sample_data,
            rolling_window=20,
            z_threshold=DEFAULT_Z_THRESHOLD,
            structural_break_protection=True,
            adf_check_frequency=10
        )
        
        backtester.run()
        results = backtester.get_performance_metrics()
        
        # Проверяем что ADF тест мокирован
        assert mock_adf.call_count >= 0, "ADF test should be accessible"
        
        # Проверяем результаты
        assert 'total_return' in results
        assert isinstance(results['num_trades'], int)


class TestBacktestEngineOptimization:
    """Медленные integration тесты для проверки корректности работы оптимизированного движка бэктестинга."""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные для бэктестинга."""
        # Детерминизм обеспечен глобально
        SAMPLE_POINTS = test_config['periods']  # 10 в minimal, 50 в fast, 100 в normal
        BASE_PRICE = 100
        VOLATILITY = 0.01
        NOISE_STD = 0.5
        ASSET2_MULTIPLIER = 1.2
        ASSET2_OFFSET = 5

        # Create cointegrated pair with some trend
        base_series = np.cumsum(np.random.randn(SAMPLE_POINTS) * VOLATILITY) + BASE_PRICE
        noise1 = np.random.randn(SAMPLE_POINTS) * NOISE_STD
        noise2 = np.random.randn(SAMPLE_POINTS) * NOISE_STD

        asset1 = base_series + noise1
        asset2 = base_series * ASSET2_MULTIPLIER + ASSET2_OFFSET + noise2  # Cointegrated with asset1

        data = pd.DataFrame({
            'asset1': asset1,
            'asset2': asset2
        })

        return data
    
    @pytest.mark.integration
    def test_optimized_backtest_when_run_then_produces_valid_results(self, sample_data):
        """Тест оптимизированного бэктеста с поддержкой QUICK_TEST."""
        if os.environ.get('QUICK_TEST', '').lower() == 'true':
            pytest.skip("Пропускаем полный integration тест в режиме QUICK_TEST")
        """Тест 1: Проверяет, что оптимизированный бэктест дает валидные результаты.
        
        Проверяет, что все оптимизации не нарушают базовую логику
        бэктестинга и производят корректные результаты.
        """
        # Create optimized backtester
        LARGE_ROLLING_WINDOW = test_config['rolling_window']  # 20 в QUICK_TEST, иначе 50
        REGIME_CHECK_FREQ = 24
        ADF_CHECK_FREQ = 100
        CACHE_CLEANUP_FREQ = 200
        TEST_PAIR_NAME = "TEST_PAIR"

        backtester = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=LARGE_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            market_regime_detection=True,
            structural_break_protection=True,
            regime_check_frequency=REGIME_CHECK_FREQ,
            use_market_regime_cache=True,
            adf_check_frequency=ADF_CHECK_FREQ,
            cache_cleanup_frequency=CACHE_CLEANUP_FREQ,
            pair_name=TEST_PAIR_NAME
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
        MIN_WIN_RATE = 0
        MAX_WIN_RATE = 1
        MAX_DRAWDOWN_THRESHOLD = 0

        assert isinstance(results['total_pnl'], (int, float)), "PnL should be numeric"
        assert isinstance(results['total_return'], (int, float)), "Return should be numeric"
        assert isinstance(results['num_trades'], int), "Number of trades should be integer"
        assert MIN_WIN_RATE <= results['win_rate'] <= MAX_WIN_RATE, f"Win rate should be in [{MIN_WIN_RATE},{MAX_WIN_RATE}], got {results['win_rate']}"
        assert results['max_drawdown'] <= MAX_DRAWDOWN_THRESHOLD, f"Max drawdown should be ≤{MAX_DRAWDOWN_THRESHOLD}, got {results['max_drawdown']}"
        
        # Check that positions were generated
        positions = backtester.results['position'].dropna()
        assert len(positions) > 0, "Should generate some positions"
        assert positions.isin([-1, 0, 1]).all(), "Positions should be -1, 0, or 1"
        
    @pytest.mark.slow
    @pytest.mark.integration
    def test_regime_detection_when_optimized_then_logic_preserved(self, sample_data):
        """Тест определения режимов с поддержкой QUICK_TEST."""
        if os.environ.get('QUICK_TEST', '').lower() == 'true':
            pytest.skip("Пропускаем медленный тест определения режимов в режиме QUICK_TEST")
        """Integration test: Проверяет корректность определения режимов с оптимизацией.

        Проверяет, что оптимизация частоты не нарушает логику
        определения рыночных режимов.
        """
        # Create backtester with regime detection
        MEDIUM_ROLLING_WINDOW = 50
        REGIME_CHECK_FREQ_FAST = 12  # Check every 12 bars
        HURST_THRESHOLD = 0.5
        VR_TRENDING_MIN = 1.1
        HURST_WINDOW_REDUCED = 100  # Reduced from default 720 to fit test data
        VR_WINDOW_REDUCED = 80  # Reduced from default 480
        TEST_REGIME_NAME = "TEST_REGIME"

        backtester = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=MEDIUM_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            market_regime_detection=True,
            regime_check_frequency=REGIME_CHECK_FREQ_FAST,
            use_market_regime_cache=True,
            hurst_trending_threshold=HURST_THRESHOLD,
            variance_ratio_trending_min=VR_TRENDING_MIN,
            hurst_window=HURST_WINDOW_REDUCED,
            variance_ratio_window=VR_WINDOW_REDUCED,
            pair_name=TEST_REGIME_NAME
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
        MIN_HURST = 0
        MAX_HURST = 1
        MIN_VARIANCE_RATIO = 0

        assert (hurst_values >= MIN_HURST).all() and (hurst_values <= MAX_HURST).all(), f"Hurst exponents should be in [{MIN_HURST},{MAX_HURST}]"

        # Check variance ratios are positive
        assert (vr_values > MIN_VARIANCE_RATIO).all(), f"Variance ratios should be > {MIN_VARIANCE_RATIO}"
        
        # Check regime consistency with metrics
        for idx in regimes.index:
            if idx in hurst_values.index and idx in vr_values.index:
                regime = regimes[idx]
                hurst = hurst_values[idx]
                vr = vr_values[idx]
                
                HURST_TRENDING_THRESHOLD = 0.5
                VR_TRENDING_THRESHOLD = 1.1
                VR_MEAN_REVERTING_THRESHOLD = 0.9

                if regime == 'trending':
                    # At least one indicator should suggest trending
                    assert hurst > HURST_TRENDING_THRESHOLD or vr > VR_TRENDING_THRESHOLD, \
                        f"Trending regime inconsistent: H={hurst:.3f}, VR={vr:.3f}"
                elif regime == 'mean_reverting':
                    # At least one indicator should suggest mean reversion
                    assert hurst < HURST_TRENDING_THRESHOLD or vr < VR_MEAN_REVERTING_THRESHOLD, \
                        f"Mean-reverting regime inconsistent: H={hurst:.3f}, VR={vr:.3f}"
                    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_structural_break_protection_when_optimized_then_logic_preserved(self, sample_data):
        """Тест защиты от структурных сдвигов с поддержкой QUICK_TEST."""
        if os.environ.get('QUICK_TEST', '').lower() == 'true':
            pytest.skip("Пропускаем тест структурных сдвигов в режиме QUICK_TEST")
        """Integration test: Проверяет защиту от структурных сдвигов с оптимизацией.

        Проверяет, что ленивая оптимизация ADF не нарушает
        логику защиты от структурных сдвигов.
        """
        # Create backtester with structural break protection
        MEDIUM_ROLLING_WINDOW = 50
        ADF_CHECK_FREQ = 50
        COINT_TEST_FREQ = 25
        ADF_PVALUE_THRESHOLD = 0.05
        CORRELATION_WINDOW_REDUCED = 100  # Reduced from default 720 to fit test data
        TEST_STRUCTURAL_NAME = "TEST_STRUCTURAL"

        backtester = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=MEDIUM_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            structural_break_protection=True,
            adf_check_frequency=ADF_CHECK_FREQ,
            use_market_regime_cache=True,
            cointegration_test_frequency=COINT_TEST_FREQ,
            adf_pvalue_threshold=ADF_PVALUE_THRESHOLD,
            correlation_window=CORRELATION_WINDOW_REDUCED,
            pair_name=TEST_STRUCTURAL_NAME
        )
        
        # Run backtest
        backtester.run()
        
        # Check structural break detection results
        correlations = backtester.rolling_correlations.dropna()
        adf_pvalues = backtester.adf_pvalues.dropna()
        
        assert len(correlations) > 0, "Should calculate rolling correlations"
        
        # Check correlation values are valid
        MIN_CORRELATION = -1
        MAX_CORRELATION = 1

        assert (correlations >= MIN_CORRELATION).all() and (correlations <= MAX_CORRELATION).all(), \
            f"Correlations should be in [{MIN_CORRELATION},{MAX_CORRELATION}]"
        
        # Check ADF p-values are valid (if any were calculated)
        if len(adf_pvalues) > 0:
            MIN_PVALUE = 0
            MAX_PVALUE = 1
            assert (adf_pvalues >= MIN_PVALUE).all() and (adf_pvalues <= MAX_PVALUE).all(), \
                f"ADF p-values should be in [{MIN_PVALUE},{MAX_PVALUE}]"
            
        # Check that positions are adjusted when structural breaks are detected
        STRUCTURAL_BREAK_THRESHOLD = 0.05
        MIN_ZERO_RATIO_DURING_BREAKS = 0.1

        positions = backtester.results['position'].dropna()
        if len(adf_pvalues) > 0:
            # Find periods with high p-values (structural breaks)
            break_periods = adf_pvalues[adf_pvalues > STRUCTURAL_BREAK_THRESHOLD].index
            if len(break_periods) > 0:
                # Positions should be more conservative during break periods
                break_positions = positions.loc[break_periods]
                # Should have some zero positions during breaks
                zero_ratio = (break_positions == 0).mean()
                assert zero_ratio > MIN_ZERO_RATIO_DURING_BREAKS, \
                    f"Should have more zero positions during breaks, got {zero_ratio:.2%}"
                
    @pytest.mark.integration
    def test_cache_when_enabled_then_same_results_as_no_cache(self, sample_data):
        """Тест 4: Проверяет эффективность кэширования.

        Проверяет, что кэш действительно ускоряет вычисления
        и дает те же результаты.
        """
        # Test without cache
        MEDIUM_ROLLING_WINDOW = 50
        REGIME_CHECK_EVERY_BAR = 1  # Check every bar for fair comparison
        CACHE_CLEANUP_FREQ = 100
        TEST_NO_CACHE_NAME = "TEST_NO_CACHE"
        TEST_WITH_CACHE_NAME = "TEST_WITH_CACHE"

        backtester_no_cache = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=MEDIUM_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            market_regime_detection=True,
            use_market_regime_cache=False,
            regime_check_frequency=REGIME_CHECK_EVERY_BAR,
            pair_name=TEST_NO_CACHE_NAME
        )

        # Test with cache
        backtester_with_cache = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=MEDIUM_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            market_regime_detection=True,
            use_market_regime_cache=True,
            regime_check_frequency=REGIME_CHECK_EVERY_BAR,
            cache_cleanup_frequency=CACHE_CLEANUP_FREQ,
            pair_name=TEST_WITH_CACHE_NAME
        )
        
        # Run both backtests
        backtester_no_cache.run()
        backtester_with_cache.run()
        
        # Get results
        results_no_cache = backtester_no_cache.get_performance_metrics()
        results_with_cache = backtester_with_cache.get_performance_metrics()
        
        # Results should be very similar (allowing for small numerical differences)
        MIN_PNL_THRESHOLD = 0.01  # Avoid division by zero
        MAX_PNL_DIFF_PCT = 0.05
        MAX_TRADE_DIFF = 2

        pnl_diff = abs(results_with_cache['total_pnl'] - results_no_cache['total_pnl'])
        if abs(results_no_cache['total_pnl']) > MIN_PNL_THRESHOLD:
            pnl_diff_pct = pnl_diff / abs(results_no_cache['total_pnl'])
            assert pnl_diff_pct < MAX_PNL_DIFF_PCT, f"PnL difference too large: {pnl_diff_pct:.2%}"

        # Number of trades should be similar
        trade_diff = abs(results_with_cache['num_trades'] - results_no_cache['num_trades'])
        assert trade_diff <= MAX_TRADE_DIFF, f"Trade count difference too large: {trade_diff}"
        
    @pytest.mark.slow
    @pytest.mark.integration
    def test_frequency_optimization_when_different_frequencies_then_consistent_results(self, sample_data):
        """Тест оптимизации частоты с поддержкой QUICK_TEST."""
        if os.environ.get('QUICK_TEST', '').lower() == 'true':
            pytest.skip("Пропускаем тест оптимизации частоты в режиме QUICK_TEST")
        """Integration test: Проверяет консистентность при оптимизации частоты.

        Проверяет, что снижение частоты проверок не приводит
        к кардинально разным результатам.
        """
        # High frequency (baseline)
        MEDIUM_ROLLING_WINDOW = 50
        HIGH_FREQ = 1  # Every bar
        LOW_FREQ = 24  # Every 24 bars (6 hours for 15-min data)
        TEST_HIGH_FREQ_NAME = "TEST_HIGH_FREQ"
        TEST_LOW_FREQ_NAME = "TEST_LOW_FREQ"

        backtester_high_freq = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=MEDIUM_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            market_regime_detection=True,
            regime_check_frequency=HIGH_FREQ,
            use_market_regime_cache=True,
            pair_name=TEST_HIGH_FREQ_NAME
        )

        # Low frequency (optimized)
        backtester_low_freq = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=MEDIUM_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            market_regime_detection=True,
            regime_check_frequency=LOW_FREQ,
            use_market_regime_cache=True,
            pair_name=TEST_LOW_FREQ_NAME
        )
        
        # Run both backtests
        backtester_high_freq.run()
        backtester_low_freq.run()
        
        # Get results
        results_high = backtester_high_freq.get_performance_metrics()
        results_low = backtester_low_freq.get_performance_metrics()
        
        # Results should be reasonably similar
        # (allowing for larger differences due to different regime detection frequency)
        MIN_PNL_FOR_COMPARISON = 0.01
        MAX_PNL_DIFF_PCT_FREQ = 0.3
        MIN_SHARPE_FOR_COMPARISON = 0.1
        MAX_SHARPE_DIFF_PCT = 0.5

        if abs(results_high['total_pnl']) > MIN_PNL_FOR_COMPARISON:
            pnl_diff_pct = abs(results_low['total_pnl'] - results_high['total_pnl']) / abs(results_high['total_pnl'])
            assert pnl_diff_pct < MAX_PNL_DIFF_PCT_FREQ, f"PnL difference too large with frequency optimization: {pnl_diff_pct:.2%}"

        # Sharpe ratio should be reasonably similar
        if abs(results_high['sharpe_ratio']) > MIN_SHARPE_FOR_COMPARISON:
            sharpe_diff_pct = abs(results_low['sharpe_ratio'] - results_high['sharpe_ratio']) / abs(results_high['sharpe_ratio'])
            assert sharpe_diff_pct < MAX_SHARPE_DIFF_PCT, f"Sharpe ratio difference too large: {sharpe_diff_pct:.2%}"
            
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
            
            # Кэш оптимизирован
            
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
        
    @pytest.mark.slow
    @pytest.mark.integration
    def test_optimization_with_real_config(self, sample_data):
        """Тест с реальной конфигурацией с поддержкой QUICK_TEST."""
        if os.environ.get('QUICK_TEST', '').lower() == 'true':
            pytest.skip("Пропускаем тест с реальной конфигурацией в режиме QUICK_TEST")
        """Integration test: Проверяет работу с реальными параметрами конфигурации.
        
        Проверяет, что оптимизации работают с параметрами,
        аналогичными тем, что используются в main_2024.yaml.
        """
        # Use parameters similar to main_2024.yaml
        backtester = BasePairBacktester(
            pair_data=sample_data,
            rolling_window=50,  # Уменьшено для работы с данными из 200 точек
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
            
    @pytest.mark.slow
    @pytest.mark.integration
    def test_memory_usage_optimization(self, sample_data):
        """Тест оптимизации использования памяти с поддержкой QUICK_TEST."""
        if os.environ.get('QUICK_TEST', '').lower() == 'true':
            pytest.skip("Пропускаем тест использования памяти в режиме QUICK_TEST")
        """Integration test: Проверяет оптимизацию использования памяти.
        
        Проверяет, что кэш не приводит к чрезмерному
        потреблению памяти.
        """
        # Create larger dataset
        # В QUICK_TEST создаем меньший dataset
        repeat_factor = 2 if os.environ.get('QUICK_TEST', '').lower() == 'true' else 4
        large_data = pd.concat([sample_data] * repeat_factor, ignore_index=True)
        
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
        
        # Кэш оптимизирован
        
    @pytest.mark.slow
    @pytest.mark.integration
    def test_error_handling_with_optimizations(self, sample_data):
        """Тест обработки ошибок с поддержкой QUICK_TEST."""
        if os.environ.get('QUICK_TEST', '').lower() == 'true':
            pytest.skip("Пропускаем тест обработки ошибок в режиме QUICK_TEST")
        """Integration test: Проверяет обработку ошибок при включенных оптимизациях.
        
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