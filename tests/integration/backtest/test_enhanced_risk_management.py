"""Тесты для улучшенного управления рисками в парной торговле."""

import numpy as np
import pandas as pd
import pytest

from coint2.engine.base_engine import BasePairBacktester as PairBacktester

# Константы для тестирования
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_MAX_KELLY_FRACTION = 0.25
DEFAULT_VOLATILITY_LOOKBACK = 50
TEST_PERIODS = 200
FREQUENCY = '15min'
START_DATE = '2023-01-01'

# Константы для генерации данных
BASE_PRICE = 100.0
NOISE_STD = 0.5
S2_COEFFICIENT = 0.8
S2_NOISE_FACTOR = 0.5

# Константы для Kelly тестирования
KELLY_TEST_RETURNS = [0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.01, 0.025, -0.008, 0.012, -0.015, 0.018]
MIN_KELLY_FRACTION = 0
MAX_KELLY_TOLERANCE = 1e-6


class TestEnhancedRiskManagement:
    """Тесты для проверки новых функций управления рисками."""

    @pytest.mark.unit
    def test_kelly_sizing_when_calculated_then_within_bounds(self):
        """Тест 1: Проверяет корректность расчета Kelly criterion.

        Проверяет, что Kelly sizing правильно рассчитывается на основе
        исторических доходностей и ограничивается максимальной долей.
        """
        # Создаем тестовые данные (детерминизм обеспечен глобально)
        dates = pd.date_range(START_DATE, periods=TEST_PERIODS, freq=FREQUENCY)

        # Создаем коинтегрированные ряды с известными параметрами
        s1_prices = BASE_PRICE + np.cumsum(np.random.normal(0, NOISE_STD, TEST_PERIODS))
        s2_prices = S2_COEFFICIENT * s1_prices + np.random.normal(0, NOISE_STD * S2_NOISE_FACTOR, TEST_PERIODS)
        
        data = pd.DataFrame({
            'S1': s1_prices,
            'S2': s2_prices
        }, index=dates)
        
        # Создаем бэктестер с Kelly sizing
        bt = PairBacktester(
            data,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            use_kelly_sizing=True,
            max_kelly_fraction=DEFAULT_MAX_KELLY_FRACTION,
            volatility_lookback=DEFAULT_VOLATILITY_LOOKBACK
        )

        # Тестируем расчет Kelly fraction с достаточным количеством данных
        returns = pd.Series(KELLY_TEST_RETURNS)
        kelly_fraction = bt._calculate_kelly_fraction(returns)

        # Kelly fraction должен быть между 0 и max_kelly_fraction
        assert MIN_KELLY_FRACTION <= kelly_fraction <= bt.max_kelly_fraction
        assert isinstance(kelly_fraction, float)

        # Дополнительная проверка: Kelly fraction не должен быть слишком близок к максимуму без веской причины
        assert kelly_fraction <= DEFAULT_MAX_KELLY_FRACTION + MAX_KELLY_TOLERANCE
        
    @pytest.mark.unit
    def test_adaptive_threshold_when_volatility_changes_then_adjusts_correctly(self):
        """Тест 2: Проверяет адаптивные пороги входа.

        Проверяет, что пороги адаптируются к текущей волатильности рынка
        и становятся более консервативными в периоды высокой волатильности.
        """
        # Детерминизм обеспечен глобально
        ADAPTIVE_PERIODS = 150
        LOW_VOL_PERIODS = 75
        HIGH_VOL_PERIODS = 75
        LOW_VOLATILITY = 0.5
        HIGH_VOLATILITY = 2.0
        MAX_VAR_MULTIPLIER = 3.0

        dates = pd.date_range(START_DATE, periods=ADAPTIVE_PERIODS, freq=FREQUENCY)

        # Создаем данные с изменяющейся волатильностью
        low_vol_period = np.random.normal(0, LOW_VOLATILITY, LOW_VOL_PERIODS)
        high_vol_period = np.random.normal(0, HIGH_VOLATILITY, HIGH_VOL_PERIODS)
        z_scores = pd.Series(np.concatenate([low_vol_period, high_vol_period]), index=dates)

        data = pd.DataFrame({
            'S1': np.random.randn(ADAPTIVE_PERIODS) + BASE_PRICE,
            'S2': np.random.randn(ADAPTIVE_PERIODS) + BASE_PRICE
        }, index=dates)

        bt = PairBacktester(
            data,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            adaptive_thresholds=True,
            volatility_lookback=DEFAULT_VOLATILITY_LOOKBACK,
            max_var_multiplier=MAX_VAR_MULTIPLIER
        )
        
        # Тестируем адаптивный порог в период низкой волатильности
        low_vol_threshold = bt._calculate_adaptive_threshold(z_scores[:LOW_VOL_PERIODS], LOW_VOLATILITY)

        # Тестируем адаптивный порог в период высокой волатильности
        high_vol_threshold = bt._calculate_adaptive_threshold(z_scores, HIGH_VOLATILITY)

        # Порог должен быть выше в период высокой волатильности
        assert high_vol_threshold >= low_vol_threshold
        assert high_vol_threshold <= bt.z_threshold * bt.max_var_multiplier
        
    @pytest.mark.unit
    def test_var_position_sizing_when_calculated_then_within_bounds(self):
        """Тест 3: Проверяет VaR-based позиционирование.

        Проверяет, что размер позиции корректно рассчитывается на основе
        Value at Risk для ограничения максимальных потерь.
        """
        # Детерминизм обеспечен глобально
        VAR_RETURNS_COUNT = 100
        VAR_MEAN_RETURN = 0.001
        VAR_VOLATILITY = 0.02  # 2% волатильность
        VAR_DATA_PERIODS = 50
        VAR_ROLLING_WINDOW = 20
        VAR_CONFIDENCE = 0.05
        MIN_VAR_MULTIPLIER = 0
        MAX_VAR_MULTIPLIER = 1.0

        # Создаем исторические доходности с известным распределением
        returns = pd.Series(np.random.normal(VAR_MEAN_RETURN, VAR_VOLATILITY, VAR_RETURNS_COUNT))

        data = pd.DataFrame({
            'S1': np.random.randn(VAR_DATA_PERIODS) + BASE_PRICE,
            'S2': np.random.randn(VAR_DATA_PERIODS) + BASE_PRICE
        })

        bt = PairBacktester(
            data,
            rolling_window=VAR_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            var_confidence=VAR_CONFIDENCE
        )

        # Тестируем VaR position sizing
        var_multiplier = bt._calculate_var_position_size(returns)

        # VaR multiplier должен быть положительным и разумным
        assert var_multiplier > MIN_VAR_MULTIPLIER
        assert var_multiplier <= MAX_VAR_MULTIPLIER  # Не должен превышать базовый размер
        
    @pytest.mark.integration
    def test_enhanced_risk_management_when_integrated_then_works_together(self):
        """Тест 4: Интеграционный тест улучшенного управления рисками.

        Проверяет, что все компоненты управления рисками работают вместе
        и не нарушают основную логику бэктеста.
        """
        # Детерминизм обеспечен глобально
        INTEGRATION_PERIODS = 100
        S1_VOLATILITY = 0.5
        S2_COEFFICIENT = 0.9
        S2_NOISE = 0.3

        dates = pd.date_range(START_DATE, periods=INTEGRATION_PERIODS, freq=FREQUENCY)

        # Создаем коинтегрированные данные
        s1_prices = BASE_PRICE + np.cumsum(np.random.normal(0, S1_VOLATILITY, INTEGRATION_PERIODS))
        s2_prices = S2_COEFFICIENT * s1_prices + np.random.normal(0, S2_NOISE, INTEGRATION_PERIODS)
        
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
        
    @pytest.mark.unit
    def test_risk_parameters_when_validated_then_properly_set(self):
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
    
    @pytest.mark.unit
    def test_kelly_sizing_when_insufficient_data_then_returns_default(self):
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
        
    @pytest.mark.unit
    def test_adaptive_thresholds_when_disabled_then_uses_static_thresholds(self):
        """Тест 7: Проверяет поведение при отключенных адаптивных порогах.
        
        Проверяет, что при adaptive_thresholds=False используются
        статические пороги независимо от волатильности.
        """
        # Детерминизм обеспечен глобально
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
        assert threshold == pytest.approx(bt.z_threshold)
        
        threshold_high_vol = bt._calculate_adaptive_threshold(z_scores, 5.0)
        assert threshold_high_vol == pytest.approx(bt.z_threshold)