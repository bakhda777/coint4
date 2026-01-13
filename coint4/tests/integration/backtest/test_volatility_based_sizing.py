"""Тесты для динамического позиционирования на основе волатильности."""

import numpy as np
import pandas as pd
import pytest

from coint2.engine.base_engine import BasePairBacktester as PairBacktester

# Константы для тестирования
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_VOLATILITY_LOOKBACK_HOURS = 24
DEFAULT_MIN_POSITION_SIZE_PCT = 0.005
DEFAULT_MAX_POSITION_SIZE_PCT = 0.02
DEFAULT_VOLATILITY_ADJUSTMENT_FACTOR = 2.0

# Константы для генерации данных
TEST_PERIODS = 200
FREQUENCY = '15min'
START_DATE = '2024-01-01'
BASE_PRICE = 100
LOW_VOL_PERIODS = 100
HIGH_VOL_PERIODS = 100
LOW_VOLATILITY = 0.5
HIGH_VOLATILITY = 2.0
ASSET2_COEFFICIENT = 0.8

# Константы для валидации
MIN_MULTIPLIER = 0.1
MAX_MULTIPLIER = 5.0
MIN_POSITION_SIZE = 0.001
MAX_POSITION_SIZE = 0.05


class TestVolatilityBasedSizing:
    """Тесты для проверки динамического позиционирования на основе волатильности."""

    @pytest.mark.unit
    def test_volatility_multiplier_when_calculated_then_within_bounds(self):
        """Тест 1: Проверяет расчет множителя волатильности.

        Проверяет, что множитель корректно рассчитывается на основе
        текущей и исторической волатильности.
        """
        # Создаем данные с разной волатильностью (детерминизм обеспечен глобально)
        dates = pd.date_range(START_DATE, periods=TEST_PERIODS, freq=FREQUENCY)

        # Первая половина - низкая волатильность
        low_vol_data = np.random.normal(BASE_PRICE, LOW_VOLATILITY, LOW_VOL_PERIODS)
        # Вторая половина - высокая волатильность
        high_vol_data = np.random.normal(BASE_PRICE, HIGH_VOLATILITY, HIGH_VOL_PERIODS)

        data = pd.DataFrame({
            'asset1': np.concatenate([low_vol_data, high_vol_data]),
            'asset2': np.concatenate([low_vol_data * ASSET2_COEFFICIENT, high_vol_data * ASSET2_COEFFICIENT])
        }, index=dates)

        # Создаем бэктестер с включенным динамическим позиционированием
        bt = PairBacktester(
            pair_data=data,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            volatility_based_sizing=True,
            volatility_lookback_hours=DEFAULT_VOLATILITY_LOOKBACK_HOURS,
            min_position_size_pct=DEFAULT_MIN_POSITION_SIZE_PCT,
            max_position_size_pct=DEFAULT_MAX_POSITION_SIZE_PCT,
            volatility_adjustment_factor=DEFAULT_VOLATILITY_ADJUSTMENT_FACTOR
        )

        # Тестируем расчет множителя в период высокой волатильности
        multiplier = bt._calculate_volatility_multiplier()

        # Множитель должен быть в разумных пределах
        assert MIN_MULTIPLIER <= multiplier <= MAX_MULTIPLIER, f"Множитель {multiplier} вне разумных пределов [{MIN_MULTIPLIER}, {MAX_MULTIPLIER}]"
        
        # Проверяем, что метод не падает с ошибкой
        assert isinstance(multiplier, float)
        assert not np.isnan(multiplier)
        assert not np.isinf(multiplier)

    @pytest.mark.integration
    def test_volatility_based_position_sizing_when_integrated_then_adjusts_correctly(self):
        """Тест 2: Проверяет интеграцию динамического позиционирования в расчет размера позиции.

        Проверяет, что размер позиции корректно корректируется на основе волатильности.
        """
        # Детерминизм обеспечен глобально
        INTEGRATION_PERIODS = 150
        PRICE_VOLATILITY = 0.01
        Y_COEFFICIENT = 1.5
        Y_NOISE = 0.5
        CAPITAL_AT_RISK = 10000.0  # Увеличиваем капитал для прохождения min_notional

        dates = pd.date_range(START_DATE, periods=INTEGRATION_PERIODS, freq=FREQUENCY)

        # Создаем коинтегрированную пару с трендом
        x = np.cumsum(np.random.normal(0, PRICE_VOLATILITY, INTEGRATION_PERIODS)) + BASE_PRICE
        y = Y_COEFFICIENT * x + np.random.normal(0, Y_NOISE, INTEGRATION_PERIODS)

        data = pd.DataFrame({
            'asset1': y,
            'asset2': x
        }, index=dates)

        # Тест с выключенным динамическим позиционированием
        bt_static = PairBacktester(
            pair_data=data,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            capital_at_risk=CAPITAL_AT_RISK,
            volatility_based_sizing=False
        )

        # Тест с включенным динамическим позиционированием
        bt_dynamic = PairBacktester(
            pair_data=data,
            rolling_window=DEFAULT_ROLLING_WINDOW,
            z_threshold=DEFAULT_Z_THRESHOLD,
            capital_at_risk=CAPITAL_AT_RISK,
            volatility_based_sizing=True,
            volatility_lookback_hours=DEFAULT_VOLATILITY_LOOKBACK_HOURS,
            min_position_size_pct=DEFAULT_MIN_POSITION_SIZE_PCT,
            max_position_size_pct=DEFAULT_MAX_POSITION_SIZE_PCT,
            volatility_adjustment_factor=DEFAULT_VOLATILITY_ADJUSTMENT_FACTOR
        )
        
        # Рассчитываем размер позиции для тестового случая
        entry_z = 2.5
        spread_curr = 5.0
        mean = 3.0
        std = 1.0
        beta = 1.5
        price_s1 = 150.0
        price_s2 = 100.0
        
        size_static = bt_static._calculate_position_size(
            entry_z, spread_curr, mean, std, beta, price_s1, price_s2
        )
        
        size_dynamic = bt_dynamic._calculate_position_size(
            entry_z, spread_curr, mean, std, beta, price_s1, price_s2
        )
        
        # Размеры должны быть положительными
        assert size_static > 0, "Статический размер позиции должен быть положительным"
        assert size_dynamic > 0, "Динамический размер позиции должен быть положительным"
        
        # Размеры могут отличаться (в зависимости от волатильности)
        # Но оба должны быть разумными
        assert size_static < 1000, "Статический размер позиции слишком большой"
        assert size_dynamic < 1000, "Динамический размер позиции слишком большой"

    @pytest.mark.unit
    def test_volatility_multiplier_when_bounded_then_respects_limits(self):
        """Тест 3: Проверяет ограничения множителя волатильности.
        
        Проверяет, что множитель остается в заданных границах.
        """
        # Детерминизм обеспечен глобально
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        
        # Создаем данные с экстремальной волатильностью
        extreme_vol_data = np.random.normal(100, 10.0, 100)  # Очень высокая волатильность
        
        data = pd.DataFrame({
            'asset1': extreme_vol_data,
            'asset2': extreme_vol_data * 0.9
        }, index=dates)
        
        bt = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            volatility_based_sizing=True,
            volatility_lookback_hours=24,
            min_position_size_pct=0.005,  # 0.5%
            max_position_size_pct=0.02,   # 2%
            volatility_adjustment_factor=2.0
        )
        
        multiplier = bt._calculate_volatility_multiplier()
        
        # Множитель должен быть ограничен заданными пределами
        # min_multiplier = 0.005 / 0.01 = 0.5
        # max_multiplier = 0.02 / 0.01 = 2.0
        assert 0.5 <= multiplier <= 2.0, f"Множитель {multiplier} вне заданных границ [0.5, 2.0]"

    @pytest.mark.unit
    def test_volatility_multiplier_when_insufficient_data_then_returns_default(self):
        """Тест 4: Проверяет обработку случаев с недостаточными данными.
        
        Проверяет, что метод корректно обрабатывает случаи с малым количеством данных.
        """
        # Создаем очень мало данных
        dates = pd.date_range('2024-01-01', periods=10, freq='15min')
        data = pd.DataFrame({
            'asset1': np.random.randn(10) + 100,
            'asset2': np.random.randn(10) + 50
        }, index=dates)
        
        bt = PairBacktester(
            pair_data=data,
            rolling_window=5,
            z_threshold=2.0,
            volatility_based_sizing=True,
            volatility_lookback_hours=24  # Требует 96 периодов, но у нас только 10
        )
        
        # Должен вернуть дефолтный множитель 1.0
        multiplier = bt._calculate_volatility_multiplier()
        assert multiplier == pytest.approx(1.0), f"При недостаточных данных должен возвращаться множитель 1.0, получен {multiplier}"

    @pytest.mark.unit
    def test_volatility_multiplier_when_zero_volatility_then_returns_default(self):
        """Тест 5: Проверяет обработку случаев с нулевой волатильностью.
        
        Проверяет, что метод корректно обрабатывает константные данные.
        """
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        
        # Создаем константные данные (нулевая волатильность)
        data = pd.DataFrame({
            'asset1': np.full(100, 100.0),  # Константа
            'asset2': np.full(100, 50.0)    # Константа
        }, index=dates)
        
        bt = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            volatility_based_sizing=True,
            volatility_lookback_hours=24
        )
        
        # Должен вернуть дефолтный множитель 1.0
        multiplier = bt._calculate_volatility_multiplier()
        assert multiplier == pytest.approx(1.0), f"При нулевой волатильности должен возвращаться множитель 1.0, получен {multiplier}"

    @pytest.mark.unit
    def test_volatility_parameters_when_validated_then_properly_set(self):
        """Тест 6: Проверяет валидацию параметров конфигурации.
        
        Проверяет, что параметры динамического позиционирования корректно передаются и используются.
        """
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        data = pd.DataFrame({
            'asset1': np.random.randn(100) + 100,
            'asset2': np.random.randn(100) + 50
        }, index=dates)
        
        # Тестируем различные параметры
        bt = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            volatility_based_sizing=True,
            volatility_lookback_hours=12,  # Нестандартное значение
            min_position_size_pct=0.001,   # Очень маленький минимум
            max_position_size_pct=0.05,    # Большой максимум
            volatility_adjustment_factor=1.5  # Нестандартный фактор
        )
        
        # Проверяем, что параметры сохранились
        assert bt.volatility_based_sizing == True
        assert bt.volatility_lookback_hours == 12
        assert bt.min_position_size_pct == pytest.approx(0.001)
        assert bt.max_position_size_pct == pytest.approx(0.05)
        assert bt.volatility_adjustment_factor == pytest.approx(1.5)
        
        # Проверяем, что метод работает с нестандартными параметрами
        multiplier = bt._calculate_volatility_multiplier()
        assert isinstance(multiplier, float)
        assert not np.isnan(multiplier)
        assert not np.isinf(multiplier)