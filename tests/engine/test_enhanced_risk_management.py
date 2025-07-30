"""Тесты для улучшенного управления рисками в парной торговле."""

import numpy as np
import pandas as pd
import pytest

from src.coint2.engine.base_engine import BasePairBacktester as PairBacktester


class TestEnhancedRiskManagement:
    """Тесты для проверки новых функций управления рисками."""

    def test_kelly_sizing_calculation(self):
        """Тест 1: Проверяет корректность расчета Kelly criterion.
        
        Проверяет, что Kelly sizing правильно рассчитывается на основе
        исторических доходностей и ограничивается максимальной долей.
        """
        # Создаем тестовые данные
        np.random.seed(42)
        n_periods = 200
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='15min')
        
        # Создаем коинтегрированные ряды с известными параметрами
        base_price = 100.0
        noise_std = 0.5
        
        s1_prices = base_price + np.cumsum(np.random.normal(0, noise_std, n_periods))
        s2_prices = 0.8 * s1_prices + np.random.normal(0, noise_std * 0.5, n_periods)
        
        data = pd.DataFrame({
            'S1': s1_prices,
            'S2': s2_prices
        }, index=dates)
        
        # Создаем бэктестер с Kelly sizing
        bt = PairBacktester(
            data,
            rolling_window=30,
            z_threshold=2.0,
            use_kelly_sizing=True,
            max_kelly_fraction=0.25,
            volatility_lookback=50
        )
        
        # Тестируем расчет Kelly fraction с достаточным количеством данных
        returns = pd.Series([0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.01, 0.025, -0.008, 0.012, -0.015, 0.018])
        kelly_fraction = bt._calculate_kelly_fraction(returns)
        
        # Kelly fraction должен быть между 0 и max_kelly_fraction
        assert 0 <= kelly_fraction <= bt.max_kelly_fraction
        assert isinstance(kelly_fraction, float)
        
    def test_adaptive_threshold_calculation(self):
        """Тест 2: Проверяет адаптивные пороги входа.
        
        Проверяет, что пороги адаптируются к текущей волатильности рынка
        и становятся более консервативными в периоды высокой волатильности.
        """
        np.random.seed(42)
        n_periods = 150
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='15min')
        
        # Создаем данные с изменяющейся волатильностью
        low_vol_period = np.random.normal(0, 0.5, 75)
        high_vol_period = np.random.normal(0, 2.0, 75)
        z_scores = pd.Series(np.concatenate([low_vol_period, high_vol_period]), index=dates)
        
        data = pd.DataFrame({
            'S1': np.random.randn(n_periods) + 100,
            'S2': np.random.randn(n_periods) + 100
        }, index=dates)
        
        bt = PairBacktester(
            data,
            rolling_window=30,
            z_threshold=2.0,
            adaptive_thresholds=True,
            volatility_lookback=50,
            max_var_multiplier=3.0
        )
        
        # Тестируем адаптивный порог в период низкой волатильности
        low_vol_threshold = bt._calculate_adaptive_threshold(z_scores[:75], 0.5)
        
        # Тестируем адаптивный порог в период высокой волатильности
        high_vol_threshold = bt._calculate_adaptive_threshold(z_scores, 2.0)
        
        # Порог должен быть выше в период высокой волатильности
        assert high_vol_threshold >= low_vol_threshold
        assert high_vol_threshold <= bt.z_threshold * bt.max_var_multiplier
        
    def test_var_position_sizing(self):
        """Тест 3: Проверяет VaR-based позиционирование.
        
        Проверяет, что размер позиции корректно рассчитывается на основе
        Value at Risk для ограничения максимальных потерь.
        """
        np.random.seed(42)
        
        # Создаем исторические доходности с известным распределением
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))  # 2% волатильность
        
        data = pd.DataFrame({
            'S1': np.random.randn(50) + 100,
            'S2': np.random.randn(50) + 100
        })
        
        bt = PairBacktester(
            data,
            rolling_window=20,
            z_threshold=2.0,
            var_confidence=0.05
        )
        
        # Тестируем VaR position sizing
        var_multiplier = bt._calculate_var_position_size(returns)
        
        # VaR multiplier должен быть положительным и разумным
        assert var_multiplier > 0
        assert var_multiplier <= 1.0  # Не должен превышать базовый размер
        
    def test_enhanced_risk_management_integration(self):
        """Тест 4: Интеграционный тест улучшенного управления рисками.
        
        Проверяет, что все компоненты управления рисками работают вместе
        и не нарушают основную логику бэктеста.
        """
        np.random.seed(42)
        n_periods = 100
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='15min')
        
        # Создаем коинтегрированные данные
        s1_prices = 100 + np.cumsum(np.random.normal(0, 0.5, n_periods))
        s2_prices = 0.9 * s1_prices + np.random.normal(0, 0.3, n_periods)
        
        data = pd.DataFrame({
            'S1': s1_prices,
            'S2': s2_prices
        }, index=dates)
        
        # Создаем бэктестер с полным набором улучшений
        bt = PairBacktester(
            data,
            rolling_window=20,
            z_threshold=2.0,
            z_exit=0.5,
            commission_pct=0.001,
            slippage_pct=0.0005,
            use_kelly_sizing=True,
            max_kelly_fraction=0.25,
            volatility_lookback=30,
            adaptive_thresholds=True,
            var_confidence=0.05,
            max_var_multiplier=2.5
        )
        
        # Запускаем бэктест
        bt.run()
        results = bt.get_results()
        
        # Проверяем, что результаты корректны
        assert 'pnl' in results
        assert 'position' in results
        assert 'trades' in results
        assert len(results['pnl']) == len(data)
        
        # Проверяем, что PnL является числовым
        assert pd.api.types.is_numeric_dtype(results['pnl'])
        
        # Проверяем, что позиции разумны
        positions = results['position']
        assert positions.abs().max() < 1000  # Разумный максимальный размер позиции
        
    def test_parameter_validation_enhanced(self):
        """Тест 5: Проверяет валидацию новых параметров.
        
        Проверяет, что новые параметры управления рисками корректно
        валидируются при создании бэктестера.
        """
        data = pd.DataFrame({
            'S1': [100, 101, 102],
            'S2': [100, 101, 102]
        })
        
        # Тест некорректного max_kelly_fraction
        with pytest.raises(ValueError):
            PairBacktester(
                data,
                rolling_window=2,
                z_threshold=2.0,
                max_kelly_fraction=1.5  # > 1.0
            )
        
        # Тест некорректного volatility_lookback
        with pytest.raises(ValueError):
            PairBacktester(
                data,
                rolling_window=2,
                z_threshold=2.0,
                volatility_lookback=5  # < 10
            )
        
        # Тест некорректного var_confidence
        with pytest.raises(ValueError):
            PairBacktester(
                data,
                rolling_window=2,
                z_threshold=2.0,
                var_confidence=1.5  # >= 1.0
            )
        
        # Тест некорректного max_var_multiplier
        with pytest.raises(ValueError):
            PairBacktester(
                data,
                rolling_window=2,
                z_threshold=2.0,
                max_var_multiplier=0.5  # <= 1.0
            )
    
    def test_kelly_sizing_with_insufficient_data(self):
        """Тест 6: Проверяет поведение Kelly sizing при недостатке данных.
        
        Проверяет, что Kelly sizing корректно обрабатывает случаи
        с недостаточным количеством исторических данных.
        """
        data = pd.DataFrame({
            'S1': np.random.randn(20) + 100,
            'S2': np.random.randn(20) + 100
        })
        
        bt = PairBacktester(
            data,
            rolling_window=10,
            z_threshold=2.0,
            use_kelly_sizing=True,
            max_kelly_fraction=0.25
        )
        
        # Тестируем с недостаточными данными
        insufficient_returns = pd.Series([0.01, -0.005])  # Только 2 значения
        kelly_fraction = bt._calculate_kelly_fraction(insufficient_returns)
        
        # Должен вернуть базовое значение capital_at_risk
        assert kelly_fraction == bt.capital_at_risk
        
    def test_adaptive_thresholds_disabled(self):
        """Тест 7: Проверяет поведение при отключенных адаптивных порогах.
        
        Проверяет, что при adaptive_thresholds=False используются
        статические пороги независимо от волатильности.
        """
        np.random.seed(42)
        z_scores = pd.Series(np.random.normal(0, 2.0, 100))
        
        data = pd.DataFrame({
            'S1': np.random.randn(50) + 100,
            'S2': np.random.randn(50) + 100
        })
        
        bt = PairBacktester(
            data,
            rolling_window=20,
            z_threshold=2.5,
            adaptive_thresholds=False  # Отключены
        )
        
        # Порог должен оставаться постоянным
        threshold = bt._calculate_adaptive_threshold(z_scores, 1.0)
        assert threshold == bt.z_threshold
        
        threshold_high_vol = bt._calculate_adaptive_threshold(z_scores, 5.0)
        assert threshold_high_vol == bt.z_threshold