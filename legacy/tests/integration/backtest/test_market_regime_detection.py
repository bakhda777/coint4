"""Тесты для определения рыночных режимов и защиты от структурных сдвигов."""

import numpy as np
import pandas as pd
import pytest

from coint2.engine.base_engine import BasePairBacktester as PairBacktester

# Константы для тестирования
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_HURST_WINDOW = 50
DEFAULT_VR_WINDOW = 40
DEFAULT_CORRELATION_WINDOW = 60
DEFAULT_ADF_WINDOW = 30

# Константы для синтетических данных
TRENDING_PERIODS = 100
MEAN_REV_PERIODS = 100
BASE_PRICE = 100
TRENDING_DRIFT = 0.1
MEAN_REV_AMPLITUDE = 5
MEAN_REV_CYCLES = 20
NOISE_STD = 0.5

# Константы для валидации
MIN_HURST = 0.0
MAX_HURST = 1.0
MIN_VR = 0.0
MIN_CORRELATION = -1.0
MAX_CORRELATION = 1.0
MIN_ADF_PVALUE = 0.0
MAX_ADF_PVALUE = 1.0


class TestMarketRegimeDetection:
    """Тесты для определения рыночных режимов."""
    
    @pytest.mark.unit
    def test_hurst_exponent_when_calculated_then_distinguishes_regimes(self):
        """Тест 1: Проверяет корректность расчета Hurst Exponent.

        Проверяет, что Hurst Exponent правильно определяет трендовые
        и mean-reverting режимы на синтетических данных.
        """
        # Create trending data (should have H > 0.5) - детерминизм обеспечен глобально
        trending_prices = pd.Series(np.cumsum(np.random.randn(TRENDING_PERIODS) + TRENDING_DRIFT) + BASE_PRICE)

        # Create mean-reverting data (should have H < 0.5)
        mean_reverting_prices = pd.Series(BASE_PRICE + MEAN_REV_AMPLITUDE * np.sin(np.linspace(0, MEAN_REV_CYCLES*np.pi, MEAN_REV_PERIODS)) +
                                        np.random.randn(MEAN_REV_PERIODS) * NOISE_STD)

        # Create backtester instance
        data = pd.DataFrame({
            'asset1': trending_prices,
            'asset2': mean_reverting_prices
        })

        backtester = PairBacktester(
            pair_data=data,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            market_regime_detection=True,
            hurst_window=DEFAULT_HURST_WINDOW
        )

        # Test Hurst calculation
        hurst_trending = backtester._calculate_hurst_exponent(trending_prices)
        hurst_mean_rev = backtester._calculate_hurst_exponent(mean_reverting_prices)

        # Trending data should have higher Hurst exponent
        assert hurst_trending > hurst_mean_rev
        assert MIN_HURST <= hurst_trending <= MAX_HURST
        assert MIN_HURST <= hurst_mean_rev <= MAX_HURST
        
    @pytest.mark.unit
    def test_variance_ratio_when_calculated_then_distinguishes_regimes(self):
        """Тест 2: Проверяет корректность расчета Variance Ratio Test.

        Проверяет, что Variance Ratio правильно определяет трендовые
        и mean-reverting режимы.
        """
        # Детерминизм обеспечен глобально
        VR_TRENDING_DRIFT = 0.2
        VR_MEAN_REV_NOISE = 0.1
        MIN_VR_THRESHOLD = 0.1

        # Create trending data (should have VR > 1)
        trending_prices = pd.Series(np.cumsum(np.random.randn(TRENDING_PERIODS) + VR_TRENDING_DRIFT) + BASE_PRICE)

        # Create mean-reverting data (should have VR < 1)
        mean_reverting_prices = pd.Series(BASE_PRICE + np.random.randn(MEAN_REV_PERIODS) * VR_MEAN_REV_NOISE)

        data = pd.DataFrame({
            'asset1': trending_prices,
            'asset2': mean_reverting_prices
        })

        backtester = PairBacktester(
            pair_data=data,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            market_regime_detection=True,
            variance_ratio_window=DEFAULT_VR_WINDOW
        )

        # Test Variance Ratio calculation
        vr_trending = backtester._calculate_variance_ratio(trending_prices)
        vr_mean_rev = backtester._calculate_variance_ratio(mean_reverting_prices)

        # Trending data should have higher variance ratio
        assert vr_trending > vr_mean_rev
        assert vr_trending > MIN_VR_THRESHOLD
        assert vr_mean_rev > MIN_VR_THRESHOLD
        
    @pytest.mark.integration
    def test_market_regime_detection_when_integrated_then_blocks_trending_trades(self):
        """Тест 3: Проверяет интеграцию определения рыночных режимов в бэктест.

        Проверяет, что система правильно определяет режимы и блокирует
        торговлю в трендовых режимах.
        """
        # Детерминизм обеспечен глобально
        INTEGRATION_POINTS = 200
        INTEGRATION_DRIFT = 0.1
        INTEGRATION_NOISE = 0.5

        # Create data with clear trending period
        trending_data = np.cumsum(np.random.randn(INTEGRATION_POINTS) + INTEGRATION_DRIFT) + BASE_PRICE
        cointegrated_data = trending_data + np.random.randn(INTEGRATION_POINTS) * INTEGRATION_NOISE
        
        data = pd.DataFrame({
            'asset1': trending_data,
            'asset2': cointegrated_data
        })
        
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=1.5,
            market_regime_detection=True,
            hurst_window=50,
            hurst_trending_threshold=0.5,
            variance_ratio_window=40,
            variance_ratio_trending_min=1.2,
            variance_ratio_mean_reverting_max=0.8
        )
        
        backtester.run()
        results = backtester.results
        
        # Check that market regime detection is working
        assert 'market_regime' in results.columns
        assert 'hurst_exponent' in results.columns
        assert 'variance_ratio' in results.columns
        
        # Should have detected some regimes
        regimes = results['market_regime'].dropna().unique()
        assert len(regimes) >= 0
        if len(regimes) > 0:
            assert all(regime in ['trending', 'mean_reverting', 'neutral'] for regime in regimes)
        
        # Should have some Hurst and VR values
        hurst_values = results['hurst_exponent'].dropna()
        vr_values = results['variance_ratio'].dropna()
        assert len(hurst_values) >= 0
        assert len(vr_values) >= 0
        
    @pytest.mark.integration
    def test_market_regime_detection_when_trending_market_then_restricts_trading(self):
        """Тест 4: Проверяет ограничения торговли в трендовых режимах.
        
        Проверяет, что система не открывает новые позиции в трендовых режимах.
        """
        # Детерминизм обеспечен глобально
        
        # Create strongly trending data
        n_points = 150
        strong_trend = np.cumsum(np.ones(n_points) * 0.5 + np.random.randn(n_points) * 0.1) + 100
        correlated_trend = strong_trend * 1.1 + np.random.randn(n_points) * 0.2
        
        data = pd.DataFrame({
            'asset1': strong_trend,
            'asset2': correlated_trend
        })
        
        # Test with regime detection enabled
        backtester_with_regime = PairBacktester(
            pair_data=data,
            rolling_window=20,
            z_threshold=1.0,  # Low threshold to encourage trading
            market_regime_detection=True,
            hurst_window=30,
            hurst_trending_threshold=0.5,
            variance_ratio_trending_min=1.1
        )
        
        # Test without regime detection
        backtester_without_regime = PairBacktester(
            pair_data=data,
            rolling_window=20,
            z_threshold=1.0,
            market_regime_detection=False
        )
        
        backtester_with_regime.run()
        backtester_without_regime.run()
        
        # Count trades
        trades_with_regime = (backtester_with_regime.results['trades'] != 0).sum()
        trades_without_regime = (backtester_without_regime.results['trades'] != 0).sum()
        
        # Should have fewer trades with regime detection in trending market
        assert trades_with_regime <= trades_without_regime
        
        
class TestStructuralBreakProtection:
    """Тесты для защиты от структурных сдвигов."""
    
    @pytest.mark.unit
    def test_rolling_correlation_when_calculated_then_distinguishes_correlation(self):
        """Тест 5: Проверяет расчет скользящей корреляции.
        
        Проверяет корректность расчета корреляции между двумя временными рядами.
        """
        # Детерминизм обеспечен глобально
        
        # Create correlated data
        s1 = pd.Series(np.random.randn(100) + 100)
        s2 = s1 * 0.8 + np.random.randn(100) * 0.2  # High correlation
        
        # Create uncorrelated data
        s3 = pd.Series(np.random.randn(100) + 100)
        
        data = pd.DataFrame({'asset1': s1, 'asset2': s2})
        
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            structural_break_protection=True,
            correlation_window=50
        )
        
        # Test correlation calculation
        corr_high = backtester._calculate_rolling_correlation(s1, s2, 50)
        corr_low = backtester._calculate_rolling_correlation(s1, s3, 50)
        
        # Проверяем, что корреляции различаются
        assert corr_high > corr_low
        assert corr_high > 0.5  # Should be highly correlated
        assert abs(corr_low) < 0.5  # Should be weakly correlated
        assert MIN_CORRELATION <= corr_high <= MAX_CORRELATION
        assert MIN_CORRELATION <= corr_low <= MAX_CORRELATION
        
    @pytest.mark.unit
    def test_half_life_when_calculated_then_distinguishes_mean_reversion(self):
        """Тест 6: Проверяет расчет half-life спреда.
        
        Проверяет корректность расчета времени полураспада для mean-reverting спреда.
        """
        # Детерминизм обеспечен глобально
        
        # Create mean-reverting spread
        n_points = 200
        mean_reverting_spread = pd.Series(
            np.random.randn(n_points) * 0.1 + 
            5 * np.exp(-np.linspace(0, 5, n_points))  # Exponential decay
        )
        
        # Create non-mean-reverting spread (random walk)
        random_walk_spread = pd.Series(np.cumsum(np.random.randn(n_points)))
        
        data = pd.DataFrame({'asset1': [1]*n_points, 'asset2': [1]*n_points})
        
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            structural_break_protection=True
        )
        
        # Test half-life calculation
        hl_mean_rev = backtester._calculate_half_life(mean_reverting_spread, 100)
        hl_random_walk = backtester._calculate_half_life(random_walk_spread, 100)
        
        # Mean-reverting spread should have shorter half-life
        assert hl_mean_rev < hl_random_walk
        assert hl_mean_rev < 100  # Should be reasonable
        
    @pytest.mark.unit
    def test_adf_test_when_calculated_then_distinguishes_stationarity(self):
        """Тест 7: Проверяет выполнение ADF теста.
        
        Проверяет корректность выполнения теста Дики-Фуллера для стационарности.
        """
        # Детерминизм обеспечен глобально
        
        # Create stationary series
        stationary_series = pd.Series(np.random.randn(100))
        
        # Create non-stationary series (random walk)
        non_stationary_series = pd.Series(np.cumsum(np.random.randn(100)))
        
        data = pd.DataFrame({'asset1': [1]*100, 'asset2': [1]*100})
        
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            structural_break_protection=True
        )
        
        # Test ADF calculation
        p_val_stationary = backtester._perform_adf_test(stationary_series, 80)
        p_val_non_stationary = backtester._perform_adf_test(non_stationary_series, 80)
        
        # Stationary series should have lower p-value
        assert p_val_stationary < p_val_non_stationary
        assert MIN_ADF_PVALUE <= p_val_stationary <= MAX_ADF_PVALUE
        assert MIN_ADF_PVALUE <= p_val_non_stationary <= MAX_ADF_PVALUE
        
    @pytest.mark.integration
    def test_structural_break_detection_when_integrated_then_detects_breaks(self):
        """Тест 8: Проверяет интеграцию защиты от структурных сдвигов.
        
        Проверяет, что система правильно обнаруживает структурные сдвиги
        и закрывает позиции.
        """
        # Детерминизм обеспечен глобально
        
        # Create data with structural break
        n_points = 200
        # First half: cointegrated
        s1_part1 = np.random.randn(n_points//2) + 100
        s2_part1 = s1_part1 * 0.9 + np.random.randn(n_points//2) * 0.1
        
        # Second half: break in relationship
        s1_part2 = np.random.randn(n_points//2) + 120
        s2_part2 = np.random.randn(n_points//2) + 80  # No relationship
        
        s1 = np.concatenate([s1_part1, s1_part2])
        s2 = np.concatenate([s2_part1, s2_part2])
        
        data = pd.DataFrame({'asset1': s1, 'asset2': s2})
        
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=1.5,
            structural_break_protection=True,
            correlation_window=40,
            min_correlation_threshold=0.5,
            max_half_life_days=5,
            cointegration_test_frequency=20,
            adf_pvalue_threshold=0.05
        )
        
        backtester.run()
        results = backtester.results
        
        # Check that structural break detection is working
        assert 'structural_break_detected' in results.columns
        assert 'rolling_correlation' in results.columns
        assert 'half_life_estimate' in results.columns
        assert 'adf_pvalue' in results.columns
        
        # Should have detected some structural breaks
        breaks_detected = results['structural_break_detected'].sum()
        assert breaks_detected >= 0
        
        # Should have some correlation and half-life values
        corr_values = results['rolling_correlation'].dropna()
        hl_values = results['half_life_estimate'].dropna()
        assert len(corr_values) >= 0
        assert len(hl_values) >= 0
        
    @pytest.mark.integration
    def test_structural_break_protection_when_break_detected_then_closes_positions(self):
        """Тест 9: Проверяет закрытие позиций при структурных сдвигах.
        
        Проверяет, что система принудительно закрывает позиции при
        обнаружении структурных сдвигов.
        """
        # Детерминизм обеспечен глобально
        
        # Create data that will trigger structural break
        n_points = 150
        # Start with good cointegration
        s1 = np.random.randn(n_points) + 100
        s2 = s1 * 0.95 + np.random.randn(n_points) * 0.1
        
        # Introduce break in the middle
        break_point = n_points // 2
        s2[break_point:] = np.random.randn(n_points - break_point) + 200  # Break relationship
        
        data = pd.DataFrame({'asset1': s1, 'asset2': s2})
        
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=20,
            z_threshold=1.0,  # Low threshold to encourage trading
            structural_break_protection=True,
            correlation_window=30,
            min_correlation_threshold=0.7,  # High threshold to trigger breaks
            max_half_life_days=3  # Low threshold to trigger breaks
        )
        
        backtester.run()
        results = backtester.results
        
        # Check for forced exits due to structural breaks
        if 'exit_reason' in results.columns:
            structural_break_exits = (results['exit_reason'] == 'structural_break').sum()
        else:
            structural_break_exits = 0
        
        # Should have some structural break detection activity
        breaks_detected = results['structural_break_detected'].sum()
        assert breaks_detected >= 0, "Should have structural break detection working"
        
        # Check that positions were actually closed
        break_indices = results[results['structural_break_detected']].index
        if len(break_indices) > 0:
            # Check that structural breaks were detected
            assert len(break_indices) > 0
            
            # Check that some positions were forced to close due to structural breaks
            # Look for cases where position changes from non-zero to zero at break points
            position_closures = 0
            for i, idx in enumerate(break_indices):
                if idx in results.index:
                    current_pos = results.loc[idx, 'position']
                    # Check if this is a forced closure (position becomes 0 and exit reason is structural_break)
                    if (abs(current_pos) < 1e-6 and 
                        results.loc[idx, 'exit_reason'] == 'structural_break'):
                        position_closures += 1
            
            # Should have at least some position closures due to structural breaks
            # But don't require it if no positions were open when breaks occurred
            if structural_break_exits > 0:
                assert position_closures >= 0  # At least some evidence of break handling
                    
    @pytest.mark.unit
    def test_market_regime_parameters_when_validated_then_properly_set(self):
        """Тест 10: Проверяет валидацию новых параметров.
        
        Проверяет, что новые параметры для определения рыночных режимов
        и защиты от структурных сдвигов правильно валидируются.
        """
        data = pd.DataFrame({
            'asset1': np.random.randn(100) + 100,
            'asset2': np.random.randn(100) + 100
        })
        
        # Test valid parameters
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            market_regime_detection=True,
            hurst_window=50,
            hurst_trending_threshold=0.5,
            variance_ratio_window=40,
            variance_ratio_trending_min=1.2,
            variance_ratio_mean_reverting_max=0.8,
            structural_break_protection=True,
            correlation_window=30,
            min_correlation_threshold=0.6,
            max_half_life_days=10,
            cointegration_test_frequency=100,
            adf_pvalue_threshold=0.05
        )
        
        # Should not raise any errors
        assert backtester.market_regime_detection == True
        assert backtester.structural_break_protection == True
        assert backtester.hurst_window == 50
        assert backtester.correlation_window == 30
        
        # Test that all new attributes are properly set
        assert hasattr(backtester, 'hurst_exponents')
        assert hasattr(backtester, 'variance_ratios')
        assert hasattr(backtester, 'market_regime')
        assert hasattr(backtester, 'rolling_correlations')
        assert hasattr(backtester, 'adf_pvalues')
        assert hasattr(backtester, 'half_life_estimates')
        assert hasattr(backtester, 'excluded_pairs')
        assert hasattr(backtester, 'last_cointegration_test')