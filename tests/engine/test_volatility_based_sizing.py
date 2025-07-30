"""Тесты для динамического позиционирования на основе волатильности."""

import numpy as np
import pandas as pd
import pytest

from src.coint2.engine.base_engine import BasePairBacktester as PairBacktester


class TestVolatilityBasedSizing:
    """Тесты для проверки динамического позиционирования на основе волатильности."""

    def test_volatility_multiplier_calculation(self):
        """Тест 1: Проверяет расчет множителя волатильности.
        
        Проверяет, что множитель корректно рассчитывается на основе
        текущей и исторической волатильности.
        """
        # Создаем данные с разной волатильностью
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='15T')
        
        # Первая половина - низкая волатильность
        low_vol_data = np.random.normal(100, 0.5, 100)
        # Вторая половина - высокая волатильность
        high_vol_data = np.random.normal(100, 2.0, 100)
        
        data = pd.DataFrame({
            'asset1': np.concatenate([low_vol_data, high_vol_data]),
            'asset2': np.concatenate([low_vol_data * 0.8, high_vol_data * 0.8])
        }, index=dates)
        
        # Создаем бэктестер с включенным динамическим позиционированием
        bt = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            volatility_based_sizing=True,
            volatility_lookback_hours=24,
            min_position_size_pct=0.005,
            max_position_size_pct=0.02,
            volatility_adjustment_factor=2.0
        )
        
        # Тестируем расчет множителя в период высокой волатильности
        multiplier = bt._calculate_volatility_multiplier()
        
        # Множитель должен быть в разумных пределах
        assert 0.1 <= multiplier <= 5.0, f"Множитель {multiplier} вне разумных пределов"
        
        # Проверяем, что метод не падает с ошибкой
        assert isinstance(multiplier, float)
        assert not np.isnan(multiplier)
        assert not np.isinf(multiplier)

    def test_volatility_based_position_sizing_integration(self):
        """Тест 2: Проверяет интеграцию динамического позиционирования в расчет размера позиции.
        
        Проверяет, что размер позиции корректно корректируется на основе волатильности.
        """
        np.random.seed(123)
        dates = pd.date_range('2024-01-01', periods=150, freq='15T')
        
        # Создаем коинтегрированную пару с трендом
        x = np.cumsum(np.random.normal(0, 0.01, 150)) + 100
        y = 1.5 * x + np.random.normal(0, 0.5, 150)
        
        data = pd.DataFrame({
            'asset1': y,
            'asset2': x
        }, index=dates)
        
        # Тест с выключенным динамическим позиционированием
        bt_static = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            capital_at_risk=10000.0,  # Увеличиваем капитал для прохождения min_notional
            volatility_based_sizing=False
        )
        
        # Тест с включенным динамическим позиционированием
        bt_dynamic = PairBacktester(
            pair_data=data,
            rolling_window=30,
            z_threshold=2.0,
            capital_at_risk=10000.0,  # Увеличиваем капитал для прохождения min_notional
            volatility_based_sizing=True,
            volatility_lookback_hours=24,
            min_position_size_pct=0.005,
            max_position_size_pct=0.02,
            volatility_adjustment_factor=2.0
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

    def test_volatility_multiplier_bounds(self):
        """Тест 3: Проверяет ограничения множителя волатильности.
        
        Проверяет, что множитель остается в заданных границах.
        """
        np.random.seed(456)
        dates = pd.date_range('2024-01-01', periods=100, freq='15T')
        
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

    def test_insufficient_data_handling(self):
        """Тест 4: Проверяет обработку случаев с недостаточными данными.
        
        Проверяет, что метод корректно обрабатывает случаи с малым количеством данных.
        """
        # Создаем очень мало данных
        dates = pd.date_range('2024-01-01', periods=10, freq='15T')
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
        assert multiplier == 1.0, f"При недостаточных данных должен возвращаться множитель 1.0, получен {multiplier}"

    def test_zero_volatility_handling(self):
        """Тест 5: Проверяет обработку случаев с нулевой волатильностью.
        
        Проверяет, что метод корректно обрабатывает константные данные.
        """
        dates = pd.date_range('2024-01-01', periods=100, freq='15T')
        
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
        assert multiplier == 1.0, f"При нулевой волатильности должен возвращаться множитель 1.0, получен {multiplier}"

    def test_config_parameter_validation(self):
        """Тест 6: Проверяет валидацию параметров конфигурации.
        
        Проверяет, что параметры динамического позиционирования корректно передаются и используются.
        """
        dates = pd.date_range('2024-01-01', periods=100, freq='15T')
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
        assert bt.min_position_size_pct == 0.001
        assert bt.max_position_size_pct == 0.05
        assert bt.volatility_adjustment_factor == 1.5
        
        # Проверяем, что метод работает с нестандартными параметрами
        multiplier = bt._calculate_volatility_multiplier()
        assert isinstance(multiplier, float)
        assert not np.isnan(multiplier)
        assert not np.isinf(multiplier)