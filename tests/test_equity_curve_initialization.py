import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from coint2.core.portfolio import Portfolio


class TestEquityCurveInitialization:
    """Тесты для проверки корректной инициализации equity кривой с первой тестовой даты."""

    def test_portfolio_equity_initialization_removes_artificial_date(self):
        """Проверяет, что инициализация equity кривой удаляет искусственную дату 1970-01-01."""
        portfolio = Portfolio(initial_capital=100000, max_active_positions=10)
        
        # Проверяем начальное состояние
        assert portfolio.equity_curve.empty
        assert not portfolio.equity_initialized
        
        # Инициализируем с первой тестовой даты
        first_test_date = pd.Timestamp('2024-01-15 09:30:00')
        portfolio.initialize_equity_curve(first_test_date)
        
        # Проверяем корректную инициализацию
        assert portfolio.equity_initialized
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve.index[0] == first_test_date
        assert portfolio.equity_curve.iloc[0] == 100000
        assert portfolio.first_test_start == first_test_date
        
        # Проверяем, что нет искусственной даты 1970-01-01
        artificial_date = pd.Timestamp('1970-01-01')
        assert artificial_date not in portfolio.equity_curve.index

    def test_portfolio_equity_initialization_only_once(self):
        """Проверяет, что инициализация equity кривой происходит только один раз."""
        portfolio = Portfolio(initial_capital=100000, max_active_positions=10)
        
        first_test_date = pd.Timestamp('2024-01-15 09:30:00')
        second_test_date = pd.Timestamp('2024-01-16 09:30:00')
        
        # Первая инициализация
        portfolio.initialize_equity_curve(first_test_date)
        assert portfolio.equity_initialized
        assert portfolio.first_test_start == first_test_date
        
        # Попытка повторной инициализации не должна изменить дату
        portfolio.initialize_equity_curve(second_test_date)
        assert portfolio.first_test_start == first_test_date  # Не изменилась
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve.index[0] == first_test_date

    def test_record_daily_pnl_initializes_if_needed(self):
        """Проверяет, что record_daily_pnl инициализирует equity кривую если она не была инициализирована."""
        portfolio = Portfolio(initial_capital=100000, max_active_positions=10)
        
        # Записываем PnL без предварительной инициализации
        test_date = pd.Timestamp('2024-01-15 09:30:00')
        daily_pnl = 1500.0
        
        portfolio.record_daily_pnl(test_date, daily_pnl)
        
        # Проверяем автоматическую инициализацию
        assert portfolio.equity_initialized
        assert portfolio.first_test_start == test_date
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve.iloc[0] == 100000 + daily_pnl

    def test_equity_curve_progression_after_initialization(self):
        """Проверяет корректное обновление equity кривой после инициализации."""
        portfolio = Portfolio(initial_capital=100000, max_active_positions=10)
        
        # Инициализируем
        first_test_date = pd.Timestamp('2024-01-15 09:30:00')
        portfolio.initialize_equity_curve(first_test_date)
        
        # Добавляем несколько дней PnL
        dates_and_pnl = [
            (pd.Timestamp('2024-01-16 09:30:00'), 1000.0),
            (pd.Timestamp('2024-01-17 09:30:00'), -500.0),
            (pd.Timestamp('2024-01-18 09:30:00'), 2000.0),
        ]
        
        expected_equity = [100000, 101000, 100500, 102500]
        
        for i, (date, pnl) in enumerate(dates_and_pnl):
            portfolio.record_daily_pnl(date, pnl)
            assert portfolio.equity_curve.iloc[i + 1] == expected_equity[i + 1]
        
        # Проверяем финальное состояние
        assert len(portfolio.equity_curve) == 4
        assert portfolio.get_current_equity() == 102500
        
        # Проверяем, что все даты корректные (нет искусственных)
        for date in portfolio.equity_curve.index:
            assert date.year >= 2024

    def test_equity_curve_no_artificial_dates_in_sequence(self):
        """Проверяет, что в последовательности обновлений equity кривой нет искусственных дат."""
        portfolio = Portfolio(initial_capital=100000, max_active_positions=10)
        
        # Последовательность реальных дат
        dates = [
            pd.Timestamp('2024-01-15 09:30:00'),
            pd.Timestamp('2024-01-16 09:30:00'),
            pd.Timestamp('2024-01-17 09:30:00'),
        ]
        
        pnl_values = [-200, 1000]
        
        # Инициализируем с первой даты
        portfolio.initialize_equity_curve(dates[0])
        
        # Записываем PnL для последующих дат
        for date, pnl in zip(dates[1:], pnl_values):
            portfolio.record_daily_pnl(date, pnl)
        
        # Проверяем, что все даты реальные
        for date in portfolio.equity_curve.index:
            assert date.year >= 2024
            assert date not in [pd.Timestamp('1970-01-01'), pd.Timestamp('1900-01-01')]
        
        # Проверяем корректность значений: начальный капитал + накопленный PnL
        expected_values = [100000, 99800, 100800]  # 100000, 100000-200, 100000-200+1000
        for i, expected in enumerate(expected_values):
            assert abs(portfolio.equity_curve.iloc[i] - expected) < 0.01

    def test_get_current_equity_before_initialization(self):
        """Проверяет, что get_current_equity возвращает initial_capital до инициализации."""
        portfolio = Portfolio(initial_capital=50000, max_active_positions=5)
        
        # До инициализации должен возвращать initial_capital
        assert portfolio.get_current_equity() == 50000
        assert portfolio.equity_curve.empty
        
        # После инициализации должен возвращать значение из кривой
        first_test_date = pd.Timestamp('2024-01-15 09:30:00')
        portfolio.initialize_equity_curve(first_test_date)
        assert portfolio.get_current_equity() == 50000
        assert not portfolio.equity_curve.empty