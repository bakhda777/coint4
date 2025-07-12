"""Тесты для проверки увеличения максимального количества позиций до 15."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.coint2.utils.config import AppConfig, PortfolioConfig, BacktestConfig, PairSelectionConfig, WalkForwardConfig
from src.coint2.core.portfolio import Portfolio


class TestMaxPositionsIncrease:
    """Тесты для проверки увеличения максимального количества позиций."""

    def test_portfolio_max_positions_15(self):
        """Тест 1: Проверяет, что портфель может содержать до 15 позиций.
        
        Проверяет, что Portfolio класс корректно обрабатывает
        увеличенное количество максимальных позиций.
        """
        # Создаем портфель с 15 максимальными позициями
        portfolio = Portfolio(
            initial_capital=10000.0,
            max_active_positions=15
        )
        
        # Проверяем начальное состояние
        assert portfolio.max_active_positions == 15
        assert portfolio.can_open_position() == True
        assert len(portfolio.active_positions) == 0
        
        # Добавляем 15 позиций
        for i in range(15):
            pair_name = f"pair_{i}"
            portfolio.active_positions[pair_name] = {
                'entry_date': pd.Timestamp('2024-01-01'),
                'position_size': 100.0,
                'entry_price': 50.0
            }
        
        # Проверяем, что все 15 позиций добавлены
        assert len(portfolio.active_positions) == 15
        assert portfolio.can_open_position() == False  # Больше нельзя открывать
        
        # Закрываем одну позицию
        del portfolio.active_positions['pair_0']
        assert len(portfolio.active_positions) == 14
        assert portfolio.can_open_position() == True  # Теперь можно открыть еще одну

    def test_config_max_positions_validation(self):
        """Тест 2: Проверяет валидацию максимального количества позиций в конфигурации.
        
        Проверяет, что конфигурация корректно принимает значение 15
        для максимального количества позиций.
        """
        # Создаем конфигурацию портфеля с 15 позициями
        portfolio_config = PortfolioConfig(
            initial_capital=10000.0,
            risk_per_position_pct=0.01,
            max_active_positions=15,
            max_margin_usage=0.5,
            volatility_based_sizing=True,
            volatility_lookback_hours=24,
            min_position_size_pct=0.005,
            max_position_size_pct=0.02,
            volatility_adjustment_factor=2.0
        )
        
        # Проверяем, что конфигурация создалась корректно
        assert portfolio_config.max_active_positions == 15
        assert portfolio_config.volatility_based_sizing == True
        assert portfolio_config.min_position_size_pct == 0.005
        assert portfolio_config.max_position_size_pct == 0.02

    def test_diversification_benefits_with_15_positions(self):
        """Тест 3: Проверяет преимущества диверсификации с 15 позициями.
        
        Симулирует портфель с 15 позициями и проверяет,
        что риск портфеля снижается по сравнению с меньшим количеством позиций.
        """
        np.random.seed(42)
        
        # Симулируем доходности для 15 различных пар
        n_periods = 100
        n_pairs = 15
        
        # Создаем корреляционную матрицу (умеренная корреляция между парами)
        correlation_matrix = np.full((n_pairs, n_pairs), 0.3)  # 30% корреляция
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Генерируем коррелированные доходности
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_pairs),
            cov=correlation_matrix,
            size=n_periods
        )
        
        # Рассчитываем волатильность портфеля с равными весами
        equal_weights = np.ones(n_pairs) / n_pairs
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
        expected_capital_per_position = initial_capital * risk_per_position
        
        assert capital_per_position == expected_capital_per_position
        assert capital_per_position == 100.0  # 1% от 10000
        
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
        
        # Общий капитал под риском
        assert total_risk_capital == 1500.0  # 15 * 100 = 1500
        
        # Это составляет 15% от общего капитала
        risk_percentage = total_risk_capital / initial_capital
        assert risk_percentage == 0.15
        
        # Проверяем, что это разумный уровень риска
        assert risk_percentage <= 0.20  # Не более 20%
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