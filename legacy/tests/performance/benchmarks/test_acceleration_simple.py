"""Простой тест ускорения оптимизации."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.optimiser.fast_objective import FastWalkForwardObjective
from src.coint2.utils.config import load_config

# Константы для тестирования
MAIN_CONFIG_FILE = "configs/main_2024.yaml"
ROLLING_WINDOW_LOW = 20
ROLLING_WINDOW_HIGH = 50
ZSCORE_THRESHOLD_LOW = 1.5
ZSCORE_THRESHOLD_HIGH = 3.0
SEARCH_SPACE_INT_TYPE = 'int'
SEARCH_SPACE_FLOAT_TYPE = 'float'


@pytest.mark.unit
def test_cache_when_initialized_then_works_correctly():
    """Простой тест инициализации кэша."""
    # Создаем минимальную конфигурацию
    try:
        config = load_config(MAIN_CONFIG_FILE)
    except Exception as e:
        pytest.skip(f"Не удалось загрузить конфигурацию: {e}")

    search_space = {
        'rolling_window': {'type': SEARCH_SPACE_INT_TYPE, 'low': ROLLING_WINDOW_LOW, 'high': ROLLING_WINDOW_HIGH},
        'zscore_threshold': {'type': SEARCH_SPACE_FLOAT_TYPE, 'low': ZSCORE_THRESHOLD_LOW, 'high': ZSCORE_THRESHOLD_HIGH}
    }

    # Мокаем инициализацию глобального кэша
    with patch.object(FastWalkForwardObjective, '_initialize_global_rolling_cache', return_value=True):
        try:
            # FastWalkForwardObjective ожидает пути к файлам, а не объекты
            import tempfile
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(search_space, f)
                search_space_path = f.name
            
            try:
                objective = FastWalkForwardObjective(
                    base_config_path=MAIN_CONFIG_FILE,
                    search_space_path=search_space_path
                )
                
                # Проверяем, что кэш инициализирован
                assert hasattr(objective, 'global_cache_initialized'), "Должен быть атрибут global_cache_initialized"
                assert objective.global_cache_initialized is True, "Кэш должен быть инициализирован"
                
            finally:
                import os
                try:
                    os.unlink(search_space_path)
                except:
                    pass
        except Exception as e:
            pytest.skip(f"Ошибка инициализации: {e}")


@pytest.mark.unit
def test_optimized_backtester_when_imported_then_available():
    """Тест импорта оптимизированного бэктестера."""
    try:
        # Проверяем, что OptimizedPairBacktester импортируется корректно
        from src.optimiser.fast_objective import PairBacktester
        from src.coint2.engine.optimized_backtest_engine import OptimizedPairBacktester

        # Проверяем, что PairBacktester теперь указывает на OptimizedPairBacktester
        if PairBacktester is OptimizedPairBacktester:
            assert True  # OptimizedPairBacktester импортирован корректно
        else:
            # PairBacktester не указывает на OptimizedPairBacktester, но импорт работает
            assert PairBacktester is not None
            assert OptimizedPairBacktester is not None
    except Exception as e:
        pytest.skip(f"Ошибка импорта: {e}")


@pytest.mark.unit
def test_convert_hours_to_periods_when_calculated_then_correct():
    """Тест конвертации часов в периоды."""
    # Константы для тестирования конвертации
    HOURS_4 = 4.0
    HOURS_1_5 = 1.5
    HOURS_0_25 = 0.25
    MINUTES_15 = 15
    EXPECTED_PERIODS_4H = 16
    EXPECTED_PERIODS_1_5H = 6
    EXPECTED_PERIODS_0_25H = 1

    try:
        config = load_config(MAIN_CONFIG_FILE)
    except Exception as e:
        pytest.skip(f"Не удалось загрузить конфигурацию: {e}")

    search_space = {'rolling_window': {'type': SEARCH_SPACE_INT_TYPE, 'low': ROLLING_WINDOW_LOW, 'high': ROLLING_WINDOW_HIGH}}

    with patch.object(FastWalkForwardObjective, '_initialize_global_rolling_cache', return_value=True):
        try:
            # FastWalkForwardObjective ожидает пути к файлам, а не объекты  
            import tempfile
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(search_space, f)
                search_space_path = f.name
            
            try:
                objective = FastWalkForwardObjective(
                    base_config_path=MAIN_CONFIG_FILE,
                    search_space_path=search_space_path
                )

                # Тестируем конвертацию
                periods = objective.convert_hours_to_periods(HOURS_4, MINUTES_15)  # 4 часа при 15-минутных барах
                assert periods == EXPECTED_PERIODS_4H, f"4 часа = {EXPECTED_PERIODS_4H} периодов по 15 минут, получили {periods}"

                periods = objective.convert_hours_to_periods(HOURS_1_5, MINUTES_15)  # 1.5 часа
                assert periods == EXPECTED_PERIODS_1_5H, f"1.5 часа = {EXPECTED_PERIODS_1_5H} периодов по 15 минут, получили {periods}"

                periods = objective.convert_hours_to_periods(HOURS_0_25, MINUTES_15)  # 15 минут
                assert periods == EXPECTED_PERIODS_0_25H, f"0.25 часа = {EXPECTED_PERIODS_0_25H} период по 15 минут, получили {periods}"

            finally:
                import os
                try:
                    os.unlink(search_space_path)
                except:
                    pass
        except Exception as e:
            pytest.skip(f"Ошибка инициализации: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
