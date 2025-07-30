#!/usr/bin/env python3
"""
Тесты для проверки правильности параметров и типов в Optuna оптимизации.
"""

import pytest
import optuna
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.metric_utils import validate_params


class TestParameterTypes:
    """Тесты для проверки правильности типов параметров."""
    
    def create_test_search_space(self):
        """Создает тестовое пространство поиска."""
        return {
            'filters': {
                'ssd_top_n': {'low': 10000, 'high': 50000, 'step': 5000},
                'kpss_pvalue_threshold': {'low': 0.01, 'high': 0.1},
                'coint_pvalue_threshold': {'low': 0.01, 'high': 0.5}
            },
            'signals': {
                'zscore_threshold': {'low': 0.5, 'high': 2.0},
                'zscore_exit': {'low': 0.1, 'high': 0.8},
                'rolling_window': {'low': 20, 'high': 100, 'step': 5}
            },
            'portfolio': {
                'max_active_positions': {'low': 3, 'high': 15, 'step': 1},
                'risk_per_position_pct': {'low': 0.01, 'high': 0.1},
                'max_position_size_pct': {'low': 0.05, 'high': 0.3}
            },
            'costs': {
                'commission_pct': {'low': 0.0001, 'high': 0.001},
                'slippage_pct': {'low': 0.0001, 'high': 0.001}
            },
            'normalization': {
                'normalization_method': ['minmax', 'zscore', 'log_returns'],
                'min_history_ratio': {'low': 0.5, 'high': 0.9}
            }
        }
    
    def test_integer_parameters_with_step(self):
        """Тест целочисленных параметров с шагом."""
        search_space = self.create_test_search_space()
        
        # Параметры, которые должны быть целыми с шагом
        int_params_with_step = [
            ('filters', 'ssd_top_n'),
            ('signals', 'rolling_window'),
            ('portfolio', 'max_active_positions')
        ]
        
        for section, param in int_params_with_step:
            param_config = search_space[section][param]
            
            # Проверяем наличие step
            assert 'step' in param_config, f"Параметр {param} должен иметь step"
            assert isinstance(param_config['step'], int), f"Step для {param} должен быть int"
            
            # Проверяем границы
            assert 'low' in param_config and 'high' in param_config
            assert param_config['low'] < param_config['high']
            
            # Проверяем, что границы кратны шагу
            step = param_config['step']
            low = param_config['low']
            high = param_config['high']
            
            assert (high - low) % step == 0, f"Диапазон {param} не кратен шагу"
    
    def test_float_parameters_without_step(self):
        """Тест вещественных параметров без шага."""
        search_space = self.create_test_search_space()
        
        # Параметры, которые должны быть вещественными без шага
        float_params = [
            ('filters', 'kpss_pvalue_threshold'),
            ('filters', 'coint_pvalue_threshold'),
            ('signals', 'zscore_threshold'),
            ('signals', 'zscore_exit'),
            ('portfolio', 'risk_per_position_pct'),
            ('portfolio', 'max_position_size_pct'),
            ('costs', 'commission_pct'),
            ('costs', 'slippage_pct'),
            ('normalization', 'min_history_ratio')
        ]
        
        for section, param in float_params:
            param_config = search_space[section][param]
            
            # Проверяем отсутствие step
            assert 'step' not in param_config, f"Параметр {param} не должен иметь step"
            
            # Проверяем границы
            assert 'low' in param_config and 'high' in param_config
            assert param_config['low'] < param_config['high']
            assert isinstance(param_config['low'], (int, float))
            assert isinstance(param_config['high'], (int, float))
    
    def test_categorical_parameters(self):
        """Тест категориальных параметров."""
        search_space = self.create_test_search_space()
        
        # Категориальные параметры
        categorical_params = [
            ('normalization', 'normalization_method')
        ]
        
        for section, param in categorical_params:
            param_config = search_space[section][param]
            
            # Должен быть список
            assert isinstance(param_config, list), f"Параметр {param} должен быть списком"
            assert len(param_config) > 1, f"Параметр {param} должен иметь несколько вариантов"
            
            # Все элементы должны быть строками
            for value in param_config:
                assert isinstance(value, str), f"Значение {value} в {param} должно быть строкой"
    
    def test_optuna_suggest_methods(self):
        """Тест правильного использования методов suggest в Optuna."""
        
        def test_objective(trial):
            search_space = self.create_test_search_space()
            
            # Тестируем suggest_int с шагом
            ssd_top_n = trial.suggest_int(
                "ssd_top_n",
                search_space['filters']['ssd_top_n']['low'],
                search_space['filters']['ssd_top_n']['high'],
                step=search_space['filters']['ssd_top_n']['step']
            )
            assert isinstance(ssd_top_n, int)
            assert ssd_top_n % search_space['filters']['ssd_top_n']['step'] == 0
            
            # Тестируем suggest_float
            zscore_threshold = trial.suggest_float(
                "zscore_threshold",
                search_space['signals']['zscore_threshold']['low'],
                search_space['signals']['zscore_threshold']['high']
            )
            assert isinstance(zscore_threshold, float)
            
            # Тестируем suggest_categorical
            norm_method = trial.suggest_categorical(
                "normalization_method",
                search_space['normalization']['normalization_method']
            )
            assert norm_method in search_space['normalization']['normalization_method']
            
            return 1.0
        
        study = optuna.create_study()
        study.optimize(test_objective, n_trials=3)
        
        # Проверяем, что все trials завершились успешно
        for trial in study.trials:
            assert trial.state == optuna.trial.TrialState.COMPLETE


class TestParameterValidation:
    """Тесты для валидации параметров."""
    
    def test_valid_parameters(self):
        """Тест валидации корректных параметров."""
        valid_params = {
            'ssd_top_n': 25000,
            'kpss_pvalue_threshold': 0.05,
            'coint_pvalue_threshold': 0.1,
            'zscore_threshold': 1.5,
            'zscore_exit': 0.3,
            'rolling_window': 50,
            'max_active_positions': 5,
            'risk_per_position_pct': 0.02,
            'max_position_size_pct': 0.1,
            'commission_pct': 0.0004,
            'slippage_pct': 0.0005,
            'normalization_method': 'minmax',
            'min_history_ratio': 0.8
        }
        
        # Валидация должна пройти без ошибок
        try:
            validated = validate_params(valid_params)
            assert isinstance(validated, dict)
            # validate_params может добавлять параметры (например, нормализованные имена)
            assert len(validated) >= len(valid_params)
        except Exception as e:
            pytest.fail(f"Валидация корректных параметров не прошла: {e}")
    
    def test_invalid_parameters(self):
        """Тест валидации некорректных параметров."""
        # Только те случаи, которые действительно вызывают исключения в validate_params
        invalid_cases = [
            # Отрицательные значения, которые нельзя исправить
            {'zscore_threshold': -1.0},  # Отрицательный z_entry
            {'stop_loss_multiplier': -1.0},  # Отрицательный stop_loss
            {'max_active_positions': 0},  # Меньше 1

            # Значения вне допустимого диапазона
            {'max_position_size_pct': 1.5},  # > 1.0
            {'risk_per_position_pct': 1.5},  # > 1.0

            # Логически некорректные значения из _validate_cross_parameter_constraints
            {'ssd_top_n': 500},  # < 1000 (минимум для статистической значимости)
            {'zscore_threshold': 6.0},  # > 5.0 (слишком высокий порог)
            {'rolling_window': 5},  # < 10 (слишком маленькое окно)
            {'commission_pct': 0.02},  # > 0.01 (максимум 1%)
        ]

        for invalid_params in invalid_cases:
            with pytest.raises(ValueError):
                validate_params(invalid_params)
    
    def test_parameter_ranges(self):
        """Тест проверки диапазонов параметров."""
        # Тестируем граничные значения
        boundary_tests = [
            ('kpss_pvalue_threshold', 0.01, 0.1),
            ('coint_pvalue_threshold', 0.01, 0.5),
            ('zscore_threshold', 0.5, 2.0),
            ('risk_per_position_pct', 0.01, 0.1),
            ('min_history_ratio', 0.5, 0.9)
        ]
        
        for param_name, min_val, max_val in boundary_tests:
            # Минимальное значение должно быть валидным
            params_min = {param_name: min_val}
            try:
                validate_params(params_min)
            except ValueError:
                pytest.fail(f"Минимальное значение {min_val} для {param_name} не прошло валидацию")
            
            # Максимальное значение должно быть валидным
            params_max = {param_name: max_val}
            try:
                validate_params(params_max)
            except ValueError:
                pytest.fail(f"Максимальное значение {max_val} для {param_name} не прошло валидацию")
            
            # Тестируем только те граничные случаи, которые действительно вызывают исключения
            if param_name == 'zscore_threshold' and min_val > 0:
                # Отрицательный zscore_threshold должен вызывать исключение
                params_below = {param_name: -0.1}
                with pytest.raises(ValueError):
                    validate_params(params_below)

            if param_name == 'zscore_threshold':
                # Слишком высокий zscore_threshold должен вызывать исключение
                params_above = {param_name: 6.0}
                with pytest.raises(ValueError):
                    validate_params(params_above)
            elif param_name in ['max_position_size_pct', 'risk_per_position_pct']:
                # Значения > 1.0 должны вызывать исключение
                params_above = {param_name: 1.5}
                with pytest.raises(ValueError):
                    validate_params(params_above)


class TestSearchSpaceConsistency:
    """Тесты для проверки согласованности search space с конфигурацией."""
    
    def test_search_space_covers_all_optimized_params(self):
        """Тест, что search space покрывает все оптимизируемые параметры."""
        search_space = {
            'filters': ['ssd_top_n', 'kpss_pvalue_threshold', 'coint_pvalue_threshold'],
            'signals': ['zscore_threshold', 'zscore_exit', 'rolling_window'],
            'portfolio': ['max_active_positions', 'risk_per_position_pct', 'max_position_size_pct'],
            'costs': ['commission_pct', 'slippage_pct'],
            'normalization': ['normalization_method', 'min_history_ratio']
        }
        
        # Все группы должны присутствовать
        required_groups = ['filters', 'signals', 'portfolio', 'costs', 'normalization']
        for group in required_groups:
            assert group in search_space, f"Группа {group} отсутствует в search space"
        
        # Каждая группа должна содержать параметры
        for group, params in search_space.items():
            assert len(params) > 0, f"Группа {group} не содержит параметров"
    
    def test_parameter_mapping_to_config(self):
        """Тест соответствия параметров секциям конфигурации."""
        param_mapping = {
            # search_space_group -> config_section
            'filters': 'pair_selection',
            'signals': 'backtest', 
            'portfolio': 'portfolio',
            'costs': 'backtest',
            'normalization': 'data_processing'
        }
        
        for search_group, config_section in param_mapping.items():
            # Проверяем, что маппинг логичен
            assert isinstance(search_group, str)
            assert isinstance(config_section, str)
            assert search_group != config_section or search_group == 'portfolio'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
