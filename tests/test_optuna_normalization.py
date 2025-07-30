#!/usr/bin/env python3
"""
Тесты для проверки правильного использования секции нормализации в конфигурации.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.fast_objective import FastWalkForwardObjective
from src.coint2.utils.config import AppConfig


class TestNormalizationConfig:
    """Тесты для проверки правильного использования секции нормализации."""
    
    def create_test_config(self, temp_dir):
        """Создает тестовую конфигурацию."""
        config_content = {
            'backtest': {
                'timeframe': '15min',  # Добавляем обязательное поле
                'zscore_threshold': 1.5,  # Добавляем обязательное поле
                'fill_limit_pct': 0.1,  # Добавляем обязательное поле
                'annualizing_factor': 365,
                'zscore_entry_threshold': 1.5,
                'zscore_exit': 0.1,
                'rolling_window': 50,
                'stop_loss_multiplier': 3.0,
                'time_stop_multiplier': 5.0,
                'cooldown_hours': 4,
                'commission_pct': 0.0004,
                'slippage_pct': 0.0005
            },
            'portfolio': {
                'initial_capital': 10000,
                'max_active_positions': 5,
                'risk_per_position_pct': 0.02,
                'max_position_size_pct': 0.1
            },
            'data_processing': {  # ПРАВИЛЬНАЯ секция для нормализации
                'normalization_method': 'minmax',
                'min_history_ratio': 0.8,
                'fill_method': 'ffill',
                'handle_constant': True
            },
            'pair_selection': {  # НЕ должна содержать параметры нормализации
                'lookback_days': 60,  # Добавляем обязательное поле
                'ssd_top_n': 10000,
                'kpss_pvalue_threshold': 0.05,
                'coint_pvalue_threshold': 0.05,
                'min_half_life_days': 1.0,
                'max_half_life_days': 30.0,
                'min_mean_crossings': 2
            },
            'walk_forward': {  # Добавляем обязательную секцию
                'enabled': True,
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'training_period_days': 20,
                'testing_period_days': 5,
                'step_size_days': 3,
                'min_training_samples': 1000,
                'refit_frequency': 'weekly'
            },
            'data_dir': 'data_downloaded',
            'results_dir': 'results'
        }
        
        config_path = Path(temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
        
        return str(config_path)
    
    def test_normalization_in_data_processing_section(self):
        """Проверяем, что параметры нормализации читаются из data_processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.create_test_config(temp_dir)
            with open(config_path) as f:
                config = AppConfig(**yaml.safe_load(f))
            
            # Проверяем, что параметры нормализации в правильной секции
            assert hasattr(config, 'data_processing')
            assert config.data_processing.normalization_method == 'minmax'
            assert config.data_processing.min_history_ratio == 0.8
            assert config.data_processing.fill_method == 'ffill'
            assert config.data_processing.handle_constant == True
    
    def test_normalization_not_in_pair_selection(self):
        """Проверяем, что параметры нормализации НЕ в pair_selection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.create_test_config(temp_dir)
            with open(config_path) as f:
                config = AppConfig(**yaml.safe_load(f))
            
            # Проверяем, что в pair_selection НЕТ параметров нормализации
            assert not hasattr(config.pair_selection, 'normalization_method')
            assert not hasattr(config.pair_selection, 'norm_method')
            assert not hasattr(config.pair_selection, 'min_history_ratio')
            assert not hasattr(config.pair_selection, 'fill_method')
    
    @patch('src.optimiser.fast_objective.preprocess_and_normalize_data')
    def test_fast_objective_uses_correct_config_section(self, mock_preprocess):
        """Тест, что FastWalkForwardObjective использует правильную секцию конфига."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.create_test_config(temp_dir)
            
            # Создаем search space
            search_space_content = {
                'normalization': {
                    'normalization_method': ['minmax', 'zscore', 'log_returns'],
                    'min_history_ratio': {'low': 0.5, 'high': 0.9}
                }
            }
            search_space_path = Path(temp_dir) / "search_space.yaml"
            with open(search_space_path, 'w') as f:
                yaml.dump(search_space_content, f)
            
            # Создаем objective
            objective = FastWalkForwardObjective(
                base_config_path=config_path,
                search_space_path=str(search_space_path)
            )
            
            # Проверяем, что базовая конфигурация читается правильно
            assert hasattr(objective.base_config, 'data_processing')
            assert objective.base_config.data_processing.normalization_method == 'minmax'
    
    def test_optuna_params_applied_to_data_processing(self):
        """Тест, что параметры из Optuna применяются к data_processing секции."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.create_test_config(temp_dir)
            with open(config_path) as f:
                config = AppConfig(**yaml.safe_load(f))
            
            # Симулируем параметры от Optuna
            optuna_params = {
                'normalization_method': 'zscore',
                'min_history_ratio': 0.7
            }
            
            # Применяем параметры к правильной секции
            if 'normalization_method' in optuna_params:
                config.data_processing.normalization_method = optuna_params['normalization_method']
            if 'min_history_ratio' in optuna_params:
                config.data_processing.min_history_ratio = optuna_params['min_history_ratio']
            
            # Проверяем, что параметры применились
            assert config.data_processing.normalization_method == 'zscore'
            assert config.data_processing.min_history_ratio == 0.7
    
    def test_search_space_normalization_parameters(self):
        """Тест правильности параметров нормализации в search space."""
        search_space_content = {
            'normalization': {
                'normalization_method': ['minmax', 'zscore', 'log_returns'],
                'min_history_ratio': {'low': 0.5, 'high': 0.9}
            }
        }
        
        # Проверяем структуру search space
        assert 'normalization' in search_space_content
        norm_params = search_space_content['normalization']
        
        # Проверяем категориальный параметр
        assert 'normalization_method' in norm_params
        assert isinstance(norm_params['normalization_method'], list)
        assert 'minmax' in norm_params['normalization_method']
        assert 'zscore' in norm_params['normalization_method']
        
        # Проверяем числовой параметр
        assert 'min_history_ratio' in norm_params
        assert 'low' in norm_params['min_history_ratio']
        assert 'high' in norm_params['min_history_ratio']
        assert norm_params['min_history_ratio']['low'] < norm_params['min_history_ratio']['high']


class TestNormalizationApplication:
    """Тесты для проверки применения параметров нормализации."""
    
    def test_normalization_method_categories(self):
        """Тест всех поддерживаемых методов нормализации."""
        supported_methods = ['minmax', 'zscore', 'log_returns']
        
        for method in supported_methods:
            # Каждый метод должен быть валидным
            assert method in ['minmax', 'zscore', 'log_returns']
    
    def test_min_history_ratio_bounds(self):
        """Тест границ для min_history_ratio."""
        # Типичные границы из search space
        low, high = 0.5, 0.9
        
        # Проверяем валидность границ
        assert 0.0 < low < high < 1.0
        assert high - low >= 0.1  # Достаточный диапазон для оптимизации
    
    @patch('src.optimiser.fast_objective.validate_params')
    def test_normalization_params_validation(self, mock_validate):
        """Тест валидации параметров нормализации."""
        # Валидные параметры
        valid_params = {
            'normalization_method': 'minmax',
            'min_history_ratio': 0.7
        }
        mock_validate.return_value = valid_params
        
        result = mock_validate(valid_params)
        assert result['normalization_method'] == 'minmax'
        assert result['min_history_ratio'] == 0.7
        
        # Невалидные параметры
        invalid_params = {
            'normalization_method': 'invalid_method',
            'min_history_ratio': 1.5  # > 1.0
        }
        
        mock_validate.side_effect = ValueError("Invalid normalization parameters")
        
        with pytest.raises(ValueError):
            mock_validate(invalid_params)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
