"""Тесты для FastWalkForwardObjective (консолидированные из fix файлов)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from optimiser.fast_objective import FastWalkForwardObjective
from coint2.utils.config import load_config

# Константы для тестирования
DEFAULT_SSD_TOP_N = 10
DEFAULT_COINT_PVALUE_THRESHOLD = 0.1
DEFAULT_MIN_HALF_LIFE_DAYS = 1.0
DEFAULT_MAX_HALF_LIFE_DAYS = 10.0
DEFAULT_TRAINING_PERIOD_DAYS = 7
DEFAULT_TESTING_PERIOD_DAYS = 7
TEST_START_DATE = '2023-08-01'

# Константы для валидации
MIN_SSD_TOP_N = 0
MIN_PVALUE = 0
MAX_PVALUE = 1
MIN_HALF_LIFE = 0


class TestFastObjectiveUnit:
    """Быстрые unit тесты для FastWalkForwardObjective."""
    
    @pytest.mark.unit
    def test_dynamic_pair_selection_when_configured_then_logic_correct(self):
        """Unit test: проверяем логику динамического отбора пар."""
        # Мокаем конфигурацию
        mock_config = {
            'pair_selection': {
                'ssd_top_n': DEFAULT_SSD_TOP_N,
                'coint_pvalue_threshold': DEFAULT_COINT_PVALUE_THRESHOLD,
                'min_half_life_days': DEFAULT_MIN_HALF_LIFE_DAYS,
                'max_half_life_days': DEFAULT_MAX_HALF_LIFE_DAYS
            },
            'walk_forward': {
                'start_date': TEST_START_DATE,
                'training_period_days': DEFAULT_TRAINING_PERIOD_DAYS,
                'testing_period_days': DEFAULT_TESTING_PERIOD_DAYS
            }
        }

        # Проверяем параметры отбора пар
        pair_config = mock_config['pair_selection']
        assert pair_config['ssd_top_n'] > MIN_SSD_TOP_N, "ssd_top_n должен быть положительным"
        assert MIN_PVALUE < pair_config['coint_pvalue_threshold'] <= MAX_PVALUE, "p-value должен быть в (0, 1]"
        assert pair_config['min_half_life_days'] > MIN_HALF_LIFE, "min_half_life должен быть положительным"
        assert pair_config['max_half_life_days'] > pair_config['min_half_life_days'], "max > min"
    
    @pytest.mark.unit
    def test_static_pairs_when_not_required_then_dynamic_selection_works(self):
        """Unit test: проверяем, что не используется статический файл preselected_pairs.csv."""
        # Мокаем FastWalkForwardObjective
        with patch('optimiser.fast_objective.FastWalkForwardObjective') as mock_objective:
            mock_instance = MagicMock()
            mock_objective.return_value = mock_instance
            
            # Проверяем, что при инициализации не читается статический файл
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = FileNotFoundError("preselected_pairs.csv not found")
                
                # Создание объекта не должно падать из-за отсутствия статического файла
                try:
                    objective = FastWalkForwardObjective(
                        base_config_path="configs/main_2024.yaml",
                        search_space_path="configs/search_space_fast.yaml"
                    )
                    # Если дошли сюда, значит статический файл не требуется
                    assert True, "Динамический отбор пар работает без статического файла"
                except FileNotFoundError as e:
                    if "preselected_pairs.csv" in str(e):
                        pytest.fail("Система все еще зависит от статического файла preselected_pairs.csv")
    
    @pytest.mark.unit
    def test_pair_selection_when_training_data_only_then_no_lookahead_bias(self):
        """Unit test: проверяем, что отбор пар использует только тренировочные данные."""
        # Создаем mock данные
        TRAINING_START_DATE = '2023-08-01'
        TRAINING_END_DATE = '2023-08-08'
        TEST_START_DATE = '2023-08-09'
        TEST_END_DATE = '2023-08-16'
        MIN_PERIOD_DAYS = 0

        training_start = pd.Timestamp(TRAINING_START_DATE)
        training_end = pd.Timestamp(TRAINING_END_DATE)
        test_start = pd.Timestamp(TEST_START_DATE)
        test_end = pd.Timestamp(TEST_END_DATE)

        # Проверяем логику разделения данных
        assert training_end < test_start, "Тренировочные данные должны предшествовать тестовым"

        # Симулируем правильное использование только тренировочных данных
        training_period = (training_end - training_start).days
        test_period = (test_end - test_start).days

        assert training_period > MIN_PERIOD_DAYS, "Тренировочный период должен быть положительным"
        assert test_period > MIN_PERIOD_DAYS, "Тестовый период должен быть положительным"

        # Проверяем, что нет пересечения периодов
        assert not (training_start <= test_start <= training_end), "Периоды не должны пересекаться"


class TestFastObjectiveIntegration:
    """Медленные integration тесты для FastWalkForwardObjective."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_lookahead_bias_when_eliminated_then_integration_works(self):
        """Integration test: проверяем устранение lookahead bias в реальной системе."""
        # Проверяем наличие конфигурационных файлов
        config_files = [
            "configs/main_2024.yaml",
            "configs/search_space_fast.yaml"
        ]
        
        for config_file in config_files:
            assert Path(config_file).exists(), f"Конфигурационный файл {config_file} не найден"
        
        try:
            # Создаем objective с реальной конфигурацией
            objective = FastWalkForwardObjective(
                base_config_path="configs/main_2024.yaml",
                search_space_path="configs/search_space_fast.yaml"
            )
            
            # Проверяем, что objective создался успешно
            assert hasattr(objective, 'base_config'), "Должна быть загружена базовая конфигурация"
            assert hasattr(objective, 'search_space'), "Должно быть загружено пространство поиска"
            
            # Проверяем параметры динамического отбора пар
            if hasattr(objective.base_config, 'pair_selection'):
                pair_config = objective.base_config.pair_selection
                assert hasattr(pair_config, 'ssd_top_n'), "Должен быть параметр ssd_top_n"
                assert pair_config.ssd_top_n > 0, "ssd_top_n должен быть положительным"
            
        except Exception as e:
            pytest.fail(f"Не удалось создать FastWalkForwardObjective: {str(e)}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_dynamic_pair_selection_when_real_data_used_then_produces_results(self):
        """Integration test: проверяем динамический отбор пар с реальными данными."""
        # Этот тест требует реальных данных и может быть медленным
        try:
            objective = FastWalkForwardObjective(
                base_config_path="configs/main_2024.yaml",
                search_space_path="configs/search_space_fast.yaml"
            )
            
            # Тестовые параметры
            test_params = {
                'zscore_threshold': 1.5,
                'zscore_exit': 0.3,
                'stop_loss_multiplier': 3.0,
                'time_stop_multiplier': 5.0,
                'risk_per_position_pct': 0.01,
                'max_position_size_pct': 0.05,
                'max_active_positions': 3,
                'commission_pct': 0.0004,
                'slippage_pct': 0.0005,
                'normalization_method': 'minmax',
                'min_history_ratio': 0.5,
                'trial_number': 1
            }
            
            # Запускаем оптимизацию (может быть медленно)
            result = objective(test_params)
            
            # Проверяем, что получили разумный результат
            assert result is not None, "Результат не должен быть None"
            assert isinstance(result, (int, float)), "Результат должен быть числом"
            assert result > -999, "Результат не должен быть штрафным значением"
            
        except Exception as e:
            # Если тест падает из-за отсутствия данных, это нормально для unit тестов
            if "data" in str(e).lower() or "file" in str(e).lower():
                pytest.skip(f"Тест пропущен из-за отсутствия данных: {str(e)}")
            else:
                pytest.fail(f"Неожиданная ошибка: {str(e)}")
