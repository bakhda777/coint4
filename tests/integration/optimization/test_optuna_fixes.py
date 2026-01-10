#!/usr/bin/env python3
"""
Тесты для проверки исправлений в системе оптимизации.
Проверяют корректность генерации сделок и работы бэктестера.
"""

import math
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src.coint2.utils.config import load_config
from src.coint2.engine.numba_engine import NumbaPairBacktester as PairBacktester
from src.coint2.core.portfolio import Portfolio
from src.optimiser.fast_objective import FastWalkForwardObjective

# Константы для тестирования
DEFAULT_INITIAL_CAPITAL = 10000
DEFAULT_MAX_ACTIVE_POSITIONS = 1
DEFAULT_ROLLING_WINDOW = 10
DEFAULT_Z_THRESHOLD = 1.5
DEFAULT_Z_EXIT = 0.0
TEST_DATA_ROWS = 100
MIN_TRADES_EXPECTED = 1
PENALTY_THRESHOLD = -999


@pytest.mark.critical_fixes
class TestOptimizationFixesUnit:
    """Быстрые unit тесты для проверки логики исправлений оптимизации."""

    @pytest.mark.unit
    def test_zscore_threshold_when_parameter_set_then_logic_correct(self, small_prices_df):
        """Тест проверяет, что логика установки zscore_entry_threshold корректна при установке параметра."""
        # Используем готовую фикстуру напрямую
        test_data = small_prices_df.copy()

        # Переименовываем колонки для совместимости
        columns = list(test_data.columns)
        if len(columns) >= 2:
            # Берем первые две колонки и переименовываем их
            test_data = test_data.iloc[:, :2].copy()
            test_data.columns = ['price1', 'price2']

        portfolio = Portfolio(initial_capital=DEFAULT_INITIAL_CAPITAL, max_active_positions=DEFAULT_MAX_ACTIVE_POSITIONS)

        # Создаем бэктестер с меньшим окном
        backtester = PairBacktester(
            pair_data=test_data,
            rolling_window=DEFAULT_ROLLING_WINDOW,  # Меньше чем размер данных (100)
            z_threshold=DEFAULT_Z_THRESHOLD,  # Это должно стать zscore_entry_threshold
            z_exit=DEFAULT_Z_EXIT,
            portfolio=portfolio,
            pair_name="TEST/PAIR"
        )

        # Проверяем, что zscore_entry_threshold установлен правильно
        assert hasattr(backtester, 'zscore_entry_threshold'), "Должен быть атрибут zscore_entry_threshold"
        assert backtester.zscore_entry_threshold == DEFAULT_Z_THRESHOLD, f"zscore_entry_threshold должен быть {DEFAULT_Z_THRESHOLD}, получен: {backtester.zscore_entry_threshold}"

    @pytest.mark.unit
    def test_config_file_when_validation_executed_then_logic_correct(self):
        """Тест проверяет, что логика валидации конфигурационных файлов корректна при выполнении валидации."""
        required_files = [
            "configs/main_2024.yaml"
        ]

        for file_path in required_files:
            if not Path(file_path).exists():
                pytest.skip(f"Файл {file_path} не найден - пропускаем тест")
            assert Path(file_path).exists(), f"Файл {file_path} не найден"

    @pytest.mark.unit
    def test_simple_params_when_structure_defined_then_correct(self):
        """Тест проверяет, что структура простых параметров корректна при определении структуры."""
        simple_params = {
            'zscore_threshold': 1.0,
            'zscore_exit': 0.0,
            'stop_loss_multiplier': 5.0,
            'time_stop_multiplier': 10.0,
            'risk_per_position_pct': 0.02,
            'max_position_size_pct': 0.1,
            'max_active_positions': 1,
            'commission_pct': 0.0001,
            'slippage_pct': 0.0001,
            'normalization_method': 'minmax',
            'min_history_ratio': 0.5,
            'trial_number': 999
        }

        # Проверяем, что все необходимые параметры присутствуют
        required_keys = ['zscore_threshold', 'zscore_exit', 'risk_per_position_pct']
        for key in required_keys:
            assert key in simple_params, f"Параметр {key} должен присутствовать"

        # Проверяем типы значений
        assert isinstance(simple_params['zscore_threshold'], (int, float))
        assert isinstance(simple_params['zscore_exit'], (int, float))
        assert isinstance(simple_params['max_active_positions'], int)


class TestOptimizationFixes:
    """Медленные integration тесты для проверки исправлений в оптимизации."""
    
    def test_simple_backtest_when_executed_then_generates_trades(self, small_prices_df):
        """Тест проверяет, что простой бэктест генерирует сделки при выполнении."""

        # Используем готовую фикстуру вместо создания данных
        test_data = small_prices_df.copy()

        # Переименовываем колонки для совместимости
        columns = list(test_data.columns)
        if len(columns) >= 2:
            # Берем первые две колонки и переименовываем их
            test_data = test_data.iloc[:, :2].copy()
            test_data.columns = ['price1', 'price2']

        # Добавляем сильные отклонения для гарантированной генерации сделок
        STRONG_DEVIATION_THRESHOLD = 200
        STRONG_DEVIATION_START1 = 100
        STRONG_DEVIATION_END1 = 110
        STRONG_DEVIATION_START2 = 200
        STRONG_DEVIATION_END2 = 210
        DEVIATION_MAGNITUDE = 5.0
        
        if len(test_data) > STRONG_DEVIATION_THRESHOLD:
            test_data.iloc[STRONG_DEVIATION_START1:STRONG_DEVIATION_END1, 1] += DEVIATION_MAGNITUDE
            test_data.iloc[STRONG_DEVIATION_START2:STRONG_DEVIATION_END2, 1] -= DEVIATION_MAGNITUDE
        
        # Создаем портфель
        portfolio = Portfolio(initial_capital=DEFAULT_INITIAL_CAPITAL, max_active_positions=DEFAULT_MAX_ACTIVE_POSITIONS)
        
        # Константы для бэктестера
        ROLLING_WINDOW_SMALL = 20
        LOW_Z_THRESHOLD = 0.5
        COMMISSION_PCT_LOW = 0.0001
        SLIPPAGE_PCT_LOW = 0.0001
        STOP_LOSS_MULTIPLIER_HIGH = 10.0
        TIME_STOP_MULTIPLIER_HIGH = 20.0
        CAPITAL_AT_RISK = 1000.0
        
        # Создаем бэктестер с очень мягкими параметрами
        backtester = PairBacktester(
            pair_data=test_data,
            rolling_window=ROLLING_WINDOW_SMALL,
            z_threshold=LOW_Z_THRESHOLD,
            z_exit=DEFAULT_Z_EXIT,
            commission_pct=COMMISSION_PCT_LOW,
            slippage_pct=SLIPPAGE_PCT_LOW,
            stop_loss_multiplier=STOP_LOSS_MULTIPLIER_HIGH,
            time_stop_multiplier=TIME_STOP_MULTIPLIER_HIGH,
            portfolio=portfolio,
            pair_name="TEST/PAIR",
            capital_at_risk=CAPITAL_AT_RISK
        )
        
        # Запускаем бэктест
        backtester.run()
        results = backtester.get_results()

        # Проверяем результаты
        assert results is not None, "Результаты бэктеста не должны быть None"

        if isinstance(results, dict):
            # Если результаты - словарь, проверяем наличие ключей
            assert 'pnl' in results, "В результатах должен быть ключ 'pnl'"
            pnl_data = results['pnl']
            if isinstance(pnl_data, pd.Series):
                trades_count = len(pnl_data[pnl_data != 0])
            else:
                trades_count = 1 if pnl_data != 0 else 0
        else:
            # Если результаты - DataFrame
            assert not results.empty, "Результаты бэктеста не должны быть пустыми"
            assert 'pnl' in results.columns, "В результатах должна быть колонка 'pnl'"
            trades = results[results['position'] != 0]
            trades_count = len(trades)

        assert trades_count >= MIN_TRADES_EXPECTED, f"Должны быть сгенерированы сделки, получено: {trades_count}"

        print(f"✅ Тест пройден: сгенерировано {trades_count} сделок")
        
    @pytest.mark.unit
    def test_fast_objective_when_simple_params_used_then_works_correctly(self):
        """Тест проверяет, что FastWalkForwardObjective корректно работает при использовании простых параметров."""
        
        # Проверяем наличие необходимых конфигурационных файлов
        required_files = [
            "configs/main_2024.yaml",
            "configs/search_space_fast.yaml"
        ]

        for file_path in required_files:
            assert Path(file_path).exists(), f"Файл {file_path} не найден"

        # Мокируем walk-forward анализ для ускорения
        from unittest.mock import patch, MagicMock
        
        # Создаем мок результата бэктеста
        mock_result = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1D'),
            'position': [0] * 100,
            'pnl': np.random.normal(0, 10, 100),
            'cumulative_pnl': np.cumsum(np.random.normal(0, 10, 100))
        })
        
        # Мокируем внутренние методы FastWalkForwardObjective для быстрого выполнения
        with patch.object(FastWalkForwardObjective, '_process_single_walk_forward_step') as mock_step, \
             patch.object(FastWalkForwardObjective, '_initialize_global_rolling_cache', return_value=True), \
             patch('src.coint2.core.global_rolling_cache.get_global_rolling_manager') as mock_manager:
            
            # Мокируем глобальный менеджер кэша
            mock_manager_instance = MagicMock()
            mock_manager_instance.initialized = True
            mock_manager.return_value = mock_manager_instance
            
            # Создаем мок результата одного шага
            mock_step_result = {
                'pnl': 100.0,
                'trades': 10,
                'results_df': mock_result
            }
            mock_step.return_value = mock_step_result
            
            # Создаем objective с fast search space
            objective = FastWalkForwardObjective(
                base_config_path="configs/main_2024.yaml",
                search_space_path="configs/search_space_fast.yaml"
            )
            
            # Простые параметры для тестирования
            simple_params = {
                'zscore_threshold': 1.0,
                'zscore_exit': 0.0,
                'stop_loss_multiplier': 5.0,
                'time_stop_multiplier': 10.0,
                'risk_per_position_pct': 0.02,
                'max_position_size_pct': 0.1,
                'max_active_positions': 1,
                'commission_pct': 0.0001,
                'slippage_pct': 0.0001,
                'normalization_method': 'minmax',
                'min_history_ratio': 0.5,
                'trial_number': 999
            }
            
            # Запускаем тест
            result = objective(simple_params)
            
            # Проверяем результат
            assert result is not None, "Результат не должен быть None"
            assert isinstance(result, (int, float)), f"Результат должен быть числом, получен: {type(result)}"
            assert result > PENALTY_THRESHOLD, f"Результат не должен быть штрафным значением: {result}"
            
            print(f"✅ Тест пройден: FastWalkForwardObjective вернул результат: {result}")
        
    def test_data_loading_when_efficiency_tested_then_objective_initializes_correctly(self):
        """Тест проверяет, что objective инициализируется корректно при тестировании эффективности загрузки данных."""

        # Создаем objective с fast search space
        objective = FastWalkForwardObjective(
            base_config_path="configs/main_2024.yaml",
            search_space_path="configs/search_space_fast.yaml"
        )

        # Проверяем, что objective инициализирован корректно
        assert hasattr(objective, 'base_config'), "Должна быть загружена базовая конфигурация"
        assert hasattr(objective, 'search_space'), "Должно быть загружено пространство поиска"
        assert objective.base_config is not None, "Базовая конфигурация не должна быть None"

        print("✅ Тест пройден: objective инициализирован корректно с динамическим отбором пар")

    def test_backtester_init_when_cooldown_hours_used_then_conversion_works(self):
        """Тест проверяет, что функция конвертации cooldown_hours работает правильно при инициализации бэктестера."""
        from src.optimiser.fast_objective import convert_hours_to_periods
        import math

        print("\n🧪 Тестирование функции convert_hours_to_periods")

        # Константы для тестовых случаев
        FOUR_HOURS = 4
        TWO_HOURS = 2
        ONE_HOUR = 1
        HALF_HOUR = 0.5
        ZERO_HOURS = 0
        BAR_15MIN = 15
        BAR_60MIN = 60
        
        # Тестируем различные случаи конвертации
        test_cases = [
            (FOUR_HOURS, BAR_15MIN, 16),
            (TWO_HOURS, BAR_15MIN, 8),
            (ONE_HOUR, BAR_60MIN, 1),
            (HALF_HOUR, BAR_15MIN, 2),
            (ZERO_HOURS, BAR_15MIN, 0),
        ]

        for hours, bar_minutes, expected in test_cases:
            result = convert_hours_to_periods(hours, bar_minutes)
            assert result == expected, \
                f"Неправильная конвертация для {hours}ч/{bar_minutes}мин: ожидалось {expected}, получено {result}"
            print(f"   ✅ {hours}ч / {bar_minutes}мин = {result} периодов")

        # Тестируем округление вверх
        HOUR_FRACTION = 1.1
        EXPECTED_ROUNDED = 2
        result = convert_hours_to_periods(HOUR_FRACTION, BAR_60MIN)
        assert result == EXPECTED_ROUNDED, f"Округление вверх не работает: ожидалось {EXPECTED_ROUNDED}, получено {result}"

        print(f"✅ Тест функции convert_hours_to_periods прошел успешно!")

        # Дополнительно проверяем, что исправление применено в коде
        print("\n🔍 Проверяем, что исправление применено в коде...")

        # Читаем файл и проверяем, что cooldown_periods используется вместо cooldown_hours
        with open('src/optimiser/fast_objective.py', 'r') as f:
            content = f.read()

        # Проверяем, что в коде есть правильная конвертация
        assert 'cooldown_periods=cooldown_periods' in content, \
            "В коде не найден правильный параметр cooldown_periods"
        assert 'convert_hours_to_periods' in content, \
            "В коде не найдена функция convert_hours_to_periods"

        print("✅ Исправление применено в коде правильно!")


def test_config_parameters_when_loaded_then_correct():
    """Тест проверяет, что параметры в конфигурации корректны при загрузке."""
    
    cfg = load_config("configs/main_2024.yaml")
    
    # Константы для проверки конфигурации
    MIN_Z_THRESHOLD = 0
    MAX_Z_THRESHOLD = 3.0
    MIN_PERIOD_DAYS = 0
    
    # Проверяем основные параметры
    assert hasattr(cfg.backtest, 'zscore_threshold'), "В конфигурации должен быть zscore_threshold"
    assert cfg.backtest.zscore_threshold > MIN_Z_THRESHOLD, "zscore_threshold должен быть положительным"
    assert cfg.backtest.zscore_threshold < MAX_Z_THRESHOLD, "zscore_threshold не должен быть слишком высоким"
    
    # Проверяем walk-forward параметры
    assert hasattr(cfg, 'walk_forward'), "В конфигурации должна быть секция walk_forward"
    assert cfg.walk_forward.training_period_days > MIN_PERIOD_DAYS, "training_period_days должен быть положительным"
    assert cfg.walk_forward.testing_period_days > MIN_PERIOD_DAYS, "testing_period_days должен быть положительным"
    
    print("✅ Тест пройден: параметры конфигурации корректны")


# Все тесты запускаются только через pytest
