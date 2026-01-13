"""Тесты для проверки увеличения максимального количества позиций до 15."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from coint2.utils.config import AppConfig, PortfolioConfig, BacktestConfig, PairSelectionConfig, WalkForwardConfig
from coint2.core.portfolio import Portfolio

# Константы для тестирования
DEFAULT_INITIAL_CAPITAL = 10000.0
MAX_POSITIONS_NEW = 15
MAX_POSITIONS_OLD = 5
DEFAULT_POSITION_SIZE = 100.0
DEFAULT_ENTRY_PRICE = 50.0
TEST_DATE = '2024-01-01'

# Константы для конфигурации
DEFAULT_RISK_PER_POSITION = 0.01
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_COINT_PVALUE = 0.05
DEFAULT_SSD_TOP_N = 10
DEFAULT_MIN_HALF_LIFE = 1
DEFAULT_MAX_HALF_LIFE = 30
DEFAULT_MIN_MEAN_CROSSINGS = 12
DEFAULT_TIMEFRAME = "1D"
DEFAULT_STOP_LOSS_MULTIPLIER = 3.0
DEFAULT_COMMISSION_PCT = 0.001
DEFAULT_SLIPPAGE_PCT = 0.0005
DEFAULT_ANNUALIZING_FACTOR = 252


class TestMaxPositionsIncrease:
    """Тесты для проверки увеличения максимального количества позиций."""

    @pytest.mark.unit
    def test_portfolio_when_max_positions_15_then_handles_correctly(self):
        """Тест 1: Проверяет, что портфель может содержать до 15 позиций.

        Проверяет, что Portfolio класс корректно обрабатывает
        увеличенное количество максимальных позиций.
        """
        # Создаем портфель с 15 максимальными позициями
        portfolio = Portfolio(
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            max_active_positions=MAX_POSITIONS_NEW
        )

        # Проверяем начальное состояние
        assert portfolio.max_active_positions == MAX_POSITIONS_NEW
        assert portfolio.can_open_position() == True
        assert len(portfolio.active_positions) == 0

        # Добавляем 15 позиций
        for i in range(MAX_POSITIONS_NEW):
            pair_name = f"pair_{i}"
            portfolio.active_positions[pair_name] = {
                'entry_date': pd.Timestamp(TEST_DATE),
                'position_size': DEFAULT_POSITION_SIZE,
                'entry_price': DEFAULT_ENTRY_PRICE
            }

        # Проверяем, что все 15 позиций добавлены
        assert len(portfolio.active_positions) == MAX_POSITIONS_NEW
        assert portfolio.can_open_position() == False  # Больше нельзя открывать

        # Закрываем одну позицию
        POSITIONS_AFTER_CLOSE = MAX_POSITIONS_NEW - 1
        del portfolio.active_positions['pair_0']
        assert len(portfolio.active_positions) == POSITIONS_AFTER_CLOSE
        assert portfolio.can_open_position() == True  # Теперь можно открыть еще одну

    @pytest.mark.unit
    def test_config_when_max_positions_15_then_validates_correctly(self):
        """Тест 2: Проверяет валидацию максимального количества позиций в конфигурации.

        Проверяет, что конфигурация корректно принимает значение 15
        для максимального количества позиций.
        """
        # Константы для конфигурации портфеля
        MAX_MARGIN_USAGE = 0.5
        VOLATILITY_LOOKBACK_HOURS = 24
        MIN_POSITION_SIZE_PCT = 0.005
        MAX_POSITION_SIZE_PCT = 0.02
        VOLATILITY_ADJUSTMENT_FACTOR = 2.0

        # Создаем конфигурацию портфеля с 15 позициями
        portfolio_config = PortfolioConfig(
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            risk_per_position_pct=DEFAULT_RISK_PER_POSITION,
            max_active_positions=MAX_POSITIONS_NEW,
            max_margin_usage=MAX_MARGIN_USAGE,
            volatility_based_sizing=True,
            volatility_lookback_hours=VOLATILITY_LOOKBACK_HOURS,
            min_position_size_pct=MIN_POSITION_SIZE_PCT,
            max_position_size_pct=MAX_POSITION_SIZE_PCT,
            volatility_adjustment_factor=VOLATILITY_ADJUSTMENT_FACTOR
        )

        # Проверяем, что конфигурация создалась корректно
        assert portfolio_config.max_active_positions == MAX_POSITIONS_NEW
        assert portfolio_config.volatility_based_sizing == True
        assert portfolio_config.min_position_size_pct == MIN_POSITION_SIZE_PCT
        assert portfolio_config.max_position_size_pct == MAX_POSITION_SIZE_PCT

    @pytest.mark.unit
    def test_diversification_when_15_positions_then_reduces_risk(self):
        """Тест 3: Проверяет преимущества диверсификации с 15 позициями.

        Симулирует портфель с 15 позициями и проверяет,
        что риск портфеля снижается по сравнению с меньшим количеством позиций.
        """
        # Детерминизм обеспечен глобально

        # Симулируем доходности для 15 различных пар
        SIMULATION_PERIODS = 100
        CORRELATION_LEVEL = 0.3  # 30% корреляция

        # Создаем корреляционную матрицу (умеренная корреляция между парами)
        correlation_matrix = np.full((MAX_POSITIONS_NEW, MAX_POSITIONS_NEW), CORRELATION_LEVEL)
        np.fill_diagonal(correlation_matrix, 1.0)

        # Генерируем коррелированные доходности
        returns = np.random.multivariate_normal(
            mean=np.zeros(MAX_POSITIONS_NEW),
            cov=correlation_matrix,
            size=SIMULATION_PERIODS
        )

        # Рассчитываем волатильность портфеля с равными весами
        equal_weights = np.ones(MAX_POSITIONS_NEW) / MAX_POSITIONS_NEW
        portfolio_returns = returns @ equal_weights
        portfolio_volatility = np.std(portfolio_returns)
        
        # Рассчитываем среднюю волатильность отдельных активов
        individual_volatilities = np.std(returns, axis=0)
        avg_individual_volatility = np.mean(individual_volatilities)
        
        # Волатильность портфеля должна быть меньше средней волатильности активов
        diversification_ratio = portfolio_volatility / avg_individual_volatility
        
        # При 30% корреляции и 15 активах ожидаем значительное снижение риска
        assert diversification_ratio < 0.7, f"Диверсификация недостаточна: {diversification_ratio:.3f}"
        assert diversification_ratio > 0.2, f"Диверсификация слишком сильная: {diversification_ratio:.3f}"

    def test_position_sizing_with_15_positions(self):
        """Тест 4: Проверяет корректность расчета размера позиций при 15 активных позициях.
        
        Проверяет, что при 15 позициях размер каждой позиции
        корректно рассчитывается с учетом общего капитала.
        """
        initial_capital = 10000.0
        risk_per_position = 0.01  # 1% на позицию
        max_positions = 15
        
        portfolio = Portfolio(
            initial_capital=initial_capital,
            max_active_positions=max_positions
        )
        
        # Рассчитываем капитал под риск для одной позиции
        capital_per_position = portfolio.calculate_position_risk_capital(risk_per_position)

        # ИСПРАВЛЕНО: Новая логика использует max(risk_capital, capital_per_pair)
        risk_capital = initial_capital * risk_per_position  # 100.0
        capital_per_pair = initial_capital / max_positions  # 666.67
        expected_capital_per_position = max(risk_capital, capital_per_pair)  # 666.67

        assert abs(capital_per_position - expected_capital_per_position) < 0.01
        assert abs(capital_per_position - 666.67) < 0.01  # Новая логика
        
        # При 15 позициях общий риск составит 15%
        total_risk_pct = max_positions * risk_per_position
        assert total_risk_pct == 0.15  # 15%
        
        # Это разумный уровень риска для криптовалютного портфеля
        assert total_risk_pct <= 0.20  # Не более 20%

    def test_memory_efficiency_with_15_positions(self):
        """Тест 5: Проверяет эффективность использования памяти при 15 позициях.
        
        Проверяет, что система может эффективно обрабатывать
        15 одновременных позиций без чрезмерного потребления памяти.
        """
        import sys
        
        # Создаем портфель с 15 позициями
        portfolio = Portfolio(
            initial_capital=10000.0,
            max_active_positions=15
        )
        
        # Измеряем размер объекта портфеля
        portfolio_size = sys.getsizeof(portfolio)
        
        # Добавляем 15 позиций с реалистичными данными
        for i in range(15):
            pair_name = f"BTC-ETH_{i}"
            portfolio.active_positions[pair_name] = {
                'entry_date': pd.Timestamp('2024-01-01'),
                'position_size': 100.0 + i * 10,
                'entry_price': 50.0 + i,
                'stop_loss': 45.0 + i,
                'take_profit': 55.0 + i,
                'current_pnl': np.random.uniform(-50, 50)
            }
        
        # Измеряем размер после добавления позиций
        portfolio_with_positions_size = sys.getsizeof(portfolio)
        positions_dict_size = sys.getsizeof(portfolio.active_positions)
        
        # Проверяем, что размер разумный (не более нескольких KB)
        assert portfolio_with_positions_size < 10000, f"Портфель слишком большой: {portfolio_with_positions_size} байт"
        assert positions_dict_size < 5000, f"Словарь позиций слишком большой: {positions_dict_size} байт"
        
        # Проверяем, что все позиции сохранились
        assert len(portfolio.active_positions) == 15

    def test_risk_management_with_15_positions(self):
        """Тест 6: Проверяет управление рисками при 15 позициях.
        
        Проверяет, что система риск-менеджмента корректно работает
        при увеличенном количестве позиций.
        """
        np.random.seed(123)
        
        initial_capital = 10000.0
        risk_per_position = 0.01
        max_positions = 15
        
        portfolio = Portfolio(
            initial_capital=initial_capital,
            max_active_positions=max_positions
        )
        
        # Симулируем открытие 15 позиций с разными уровнями риска
        total_risk_capital = 0.0
        
        for i in range(max_positions):
            # Каждая позиция рискует 1% капитала
            position_risk = portfolio.calculate_position_risk_capital(risk_per_position)
            total_risk_capital += position_risk
            
            # Добавляем позицию в портфель
            pair_name = f"pair_{i}"
            portfolio.active_positions[pair_name] = {
                'risk_capital': position_risk,
                'entry_date': pd.Timestamp('2024-01-01'),
                'position_size': position_risk / 50.0  # Условный размер позиции
            }
        
        # ИСПРАВЛЕНО: Общий капитал под риском с новой логикой
        # Новая логика: 15 * 666.67 = 10000 (весь капитал)
        assert abs(total_risk_capital - 10000.0) < 0.01

        # Это составляет 100% от общего капитала (новая логика равномерного распределения)
        risk_percentage = total_risk_capital / initial_capital
        assert abs(risk_percentage - 1.0) < 0.01  # 100%
        
        # ИСПРАВЛЕНО: Новая логика использует весь капитал при равномерном распределении
        # Это нормально для системы равномерного распределения капитала
        assert risk_percentage <= 1.0  # Не более 100%
        assert risk_percentage >= 0.10  # Не менее 10% для эффективности
        
        # Проверяем, что нельзя открыть 16-ю позицию
        assert not portfolio.can_open_position()
        assert len(portfolio.active_positions) == 15

    def test_performance_scaling_with_15_positions(self):
        """Тест 7: Проверяет масштабируемость производительности при 15 позициях.
        
        Проверяет, что операции с портфелем выполняются быстро
        даже при максимальном количестве позиций.
        """
        import time
        
        portfolio = Portfolio(
            initial_capital=10000.0,
            max_active_positions=15
        )
        
        # Измеряем время добавления 15 позиций
        start_time = time.time()
        
        for i in range(15):
            pair_name = f"high_freq_pair_{i}"
            portfolio.active_positions[pair_name] = {
                'entry_date': pd.Timestamp('2024-01-01'),
                'position_size': 100.0,
                'entry_price': 50.0,
                'current_price': 51.0,
                'pnl': 100.0
            }
        
        add_time = time.time() - start_time
        
        # Измеряем время проверки возможности открытия позиции
        start_time = time.time()
        
        for _ in range(1000):  # 1000 проверок
            can_open = portfolio.can_open_position()
        
        check_time = time.time() - start_time
        
        # Операции должны выполняться быстро
        assert add_time < 0.1, f"Добавление 15 позиций заняло слишком много времени: {add_time:.3f}s"
        assert check_time < 0.1, f"1000 проверок заняли слишком много времени: {check_time:.3f}s"
        
        # Проверяем корректность состояния
        assert len(portfolio.active_positions) == 15
        assert not portfolio.can_open_position()