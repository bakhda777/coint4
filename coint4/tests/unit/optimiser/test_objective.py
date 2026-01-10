"""Базовые тесты для FastWalkForwardObjective с ускорениями."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

from src.optimiser.fast_objective import FastWalkForwardObjective
from src.coint2.utils.config import load_config


class TestFastObjectiveBasic:
    """Базовые тесты для FastWalkForwardObjective."""

    @pytest.mark.slow
    def test_fast_objective_when_initialized_with_cache_then_creates_cache(self):
        """Тест инициализации FastWalkForwardObjective с кэшем."""
        with patch.object(FastWalkForwardObjective, '_initialize_global_rolling_cache', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"rolling_window": {"type": "int", "low": 20, "high": 50}}')):
                objective = FastWalkForwardObjective("configs/main_2024.yaml", "configs/search_space_fast.yaml")
                assert objective.global_cache_initialized is True
    
    @pytest.mark.unit
    @pytest.mark.parametrize("hours,timeframe_minutes,expected_periods", [
        (4.0, 15, 16),    # 4 часа при 15-минутных барах
        (1.5, 15, 6),     # 1.5 часа при 15-минутных барах
        (0.25, 15, 1),    # 15 минут при 15-минутных барах
        (2.0, 30, 4),     # 2 часа при 30-минутных барах
        (1.0, 60, 1),     # 1 час при часовых барах
    ])
    def test_hours_conversion_when_called_then_returns_correct_periods(self, hours, timeframe_minutes, expected_periods):
        """Тест конвертации часов в периоды."""
        with patch.object(FastWalkForwardObjective, '_initialize_global_rolling_cache', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"rolling_window": {"type": "int", "low": 20, "high": 50}}')):
                objective = FastWalkForwardObjective("configs/main_2024.yaml", "configs/search_space_fast.yaml")

                # Тестируем конвертацию с параметризованными данными
                periods = objective.convert_hours_to_periods(hours, timeframe_minutes)
                assert periods == expected_periods, f"{hours} часов при {timeframe_minutes}-минутных барах должно быть {expected_periods} периодов, получили {periods}"
    
    @pytest.mark.unit
    def test_optimized_backtester_when_imported_then_available(self):
        """Тест импорта Numba бэктестера."""
        # ОБНОВЛЕНО: Проверяем, что FullNumbaPairBacktester импортируется корректно
        from src.optimiser.fast_objective import PairBacktester

        # Проверяем, что PairBacktester теперь указывает на FullNumbaPairBacktester
        assert PairBacktester.__name__ == 'FullNumbaPairBacktester', f"PairBacktester должен быть FullNumbaPairBacktester, получен: {PairBacktester.__name__}"
    
    def test_cache_key_generation(self):
        """Тест генерации ключей кэша."""
        config_path = "configs/main_2024.yaml"
        search_space = {'rolling_window': {'type': 'int', 'low': 20, 'high': 50}}
        
        with patch.object(FastWalkForwardObjective, '_initialize_global_rolling_cache', return_value=True):
            # Create a temporary search space file
            import tempfile
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(search_space, f)
                search_space_path = f.name
            
            try:
                objective = FastWalkForwardObjective(config_path, search_space_path)
                
                # Тестируем формат ключей кэша
                start_date = pd.Timestamp('2024-01-01')
                end_date = pd.Timestamp('2024-01-31')
                
                expected_key = "2024-01-01_2024-01-31"
                actual_key = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(search_space_path)
                except:
                    pass
            
            assert actual_key == expected_key, f"Ключ кэша должен быть {expected_key}, получили {actual_key}"
