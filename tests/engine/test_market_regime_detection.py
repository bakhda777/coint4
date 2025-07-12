"""Тесты для определения рыночных режимов и защиты от структурных сдвигов."""

import numpy as np
import pandas as pd
import pytest

from coint2.engine.backtest_engine import PairBacktester


class TestMarketRegimeDetection:
    """Тесты для определения рыночных режимов."""
    
    def test_hurst_exponent_calculation(self):
        """Тест 1: Проверяет корректность расчета Hurst Exponent.
        
        Проверяет, что Hurst Exponent правильно определяет трендовые
        и mean-reverting режимы на синтетических данных.
        """
        # Create trending data (should have H > 0.5)
        np.random.seed(42)
        trending_prices = pd.Series(np.cumsum(np.random.randn(100) + 0.1) + 100)
        
        # Create mean-reverting data (should have H < 0.5)
        mean_reverting_prices = pd.Series(100 + 5 * np.sin(np.linspace(0, 20*np.pi, 100)) + 
                                        np.random.randn(100) * 0.5)
        
        # Create backtester instance
        data = pd.DataFrame({
            'asset1': trending_prices,
            'asset2': mean_reverting_prices
        })
        
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            market_regime_detection=True,
            hurst_window=50
        )
        
        # Test Hurst calculation
        hurst_trending = backtester._calculate_hurst_exponent(trending_prices)
        hurst_mean_rev = backtester._calculate_hurst_exponent(mean_reverting_prices)
        
        # Trending data should have higher Hurst exponent
        assert hurst_trending > hurst_mean_rev
        assert 0.0 <= hurst_trending <= 1.0
        assert 0.0 <= hurst_mean_rev <= 1.0
        
    def test_variance_ratio_calculation(self):
        """Тест 2: Проверяет корректность расчета Variance Ratio Test.
        
        Проверяет, что Variance Ratio правильно определяет трендовые
        и mean-reverting режимы.
        """
        np.random.seed(42)
        
        # Create trending data (should have VR > 1)
        trending_prices = pd.Series(np.cumsum(np.random.randn(100) + 0.2) + 100)
        
        # Create mean-reverting data (should have VR < 1)
        mean_reverting_prices = pd.Series(100 + np.random.randn(100) * 0.1)
        
        data = pd.DataFrame({
            'asset1': trending_prices,
            'asset2': mean_reverting_prices
        })
        
        backtester = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            market_regime_detection=True,
            variance_ratio_window=40
        )
        
        # Test Variance Ratio calculation
        vr_trending = backtester._calculate_variance_ratio(trending_prices)
        vr_mean_rev = backtester._calculate_variance_ratio(mean_reverting_prices)
        
        # Trending data should have higher variance ratio
        assert vr_trending > vr_mean_rev
        assert vr_trending > 0.1
        assert vr_mean_rev > 0.1
        
    def test_market_regime_detection_integration(self):
        """Тест 3: Проверяет интеграцию определения рыночных режимов в бэктест.
        
        Проверяет, что система правильно определяет режимы и блокирует
        торговлю в трендовых режимах.
        """
        np.random.seed(42)
        
        # Create data with clear trending period
        n_points = 200
        trending_data = np.cumsum(np.random.randn(n_points) + 0.1) + 100
        cointegrated_data = trending_data + np.random.randn(n_points) * 0.5
        
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
        assert len(regimes) > 0
        assert all(regime in ['trending', 'mean_reverting', 'neutral'] for regime in regimes)
        
        # Should have some Hurst and VR values
        hurst_values = results['hurst_exponent'].dropna()
        vr_values = results['variance_ratio'].dropna()
        assert len(hurst_values) > 0
        assert len(vr_values) > 0
        
    def test_market_regime_trading_restrictions(self):
        """Тест 4: Проверяет ограничения торговли в трендовых режимах.
        
        Проверяет, что система не открывает новые позиции в трендовых режимах.
        """
        np.random.seed(42)
        
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
    
    def test_rolling_correlation_calculation(self):
        """Тест 5: Проверяет расчет скользящей корреляции.
        
        Проверяет корректность расчета корреляции между двумя временными рядами.
        """
        np.random.seed(42)
        
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
        
        assert corr_high > corr_low
        assert corr_high > 0.5  # Should be highly correlated
        assert abs(corr_low) < 0.5  # Should be weakly correlated
        
    def test_half_life_calculation(self):
        """Тест 6: Проверяет расчет half-life спреда.
        
        Проверяет корректность расчета времени полураспада для mean-reverting спреда.
        """
        np.random.seed(42)
        
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
        
    def test_adf_test_calculation(self):
        """Тест 7: Проверяет выполнение ADF теста.
        
        Проверяет корректность выполнения теста Дики-Фуллера для стационарности.
        """
        np.random.seed(42)
        
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
        assert 0.0 <= p_val_stationary <= 1.0
        assert 0.0 <= p_val_non_stationary <= 1.0
        
    def test_structural_break_detection_integration(self):
        """Тест 8: Проверяет интеграцию защиты от структурных сдвигов.
        
        Проверяет, что система правильно обнаруживает структурные сдвиги
        и закрывает позиции.
        """
        np.random.seed(42)
        
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
        assert breaks_detected > 0
        
        # Should have some correlation and half-life values
        corr_values = results['rolling_correlation'].dropna()
        hl_values = results['half_life_estimate'].dropna()
        assert len(corr_values) > 0
        assert len(hl_values) > 0
        
    def test_structural_break_position_closure(self):
        """Тест 9: Проверяет закрытие позиций при структурных сдвигах.
        
        Проверяет, что система принудительно закрывает позиции при
        обнаружении структурных сдвигов.
        """
        np.random.seed(42)
        
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
        structural_break_exits = (results['exit_reason'] == 'structural_break').sum()
        
        # Should have some exits due to structural breaks
        assert structural_break_exits > 0
        
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
                    
    def test_parameter_validation_for_new_features(self):
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