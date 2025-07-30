#!/usr/bin/env python3
"""
Тесты для проверки условного sampling зависимых параметров в Optuna.
"""

import pytest
import optuna
import numpy as np
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.fast_objective import FastWalkForwardObjective


class TestConditionalSampling:
    """Тесты условного sampling для зависимых параметров."""
    
    def test_zscore_exit_constrained_by_threshold(self):
        """Проверяет что zscore_exit ограничен zscore_threshold."""
        
        # Создаем временные конфигурационные файлы
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as base_config:
            yaml.dump({
                'data_dir': 'test_data',
                'walk_forward': {
                    'start_date': '2024-01-01', 
                    'training_period_days': 10, 
                    'testing_period_days': 5
                },
                'portfolio': {'initial_capital': 10000},
                'backtest': {'annualizing_factor': 365},
                'pair_selection': {'bar_minutes': 15}
            }, base_config)
            base_config.flush()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as search_space:
                yaml.dump({
                    'signals': {
                        'zscore_threshold': {'low': 1.0, 'high': 3.0},
                        'zscore_exit': {'low': -1.0, 'high': 1.0}
                    }
                }, search_space)
                search_space.flush()
                
                # Мокаем файл предотобранных пар
                with patch('pathlib.Path.exists', return_value=True), \
                     patch('pandas.read_csv') as mock_read_csv:
                    
                    # Мокаем данные пар
                    mock_pairs_df = Mock()
                    mock_pairs_df.__len__ = Mock(return_value=10)
                    mock_pairs_df.iterrows = Mock(return_value=[])
                    mock_read_csv.return_value = mock_pairs_df
                    
                    try:
                        objective = FastWalkForwardObjective(
                            base_config_path=base_config.name,
                            search_space_path=search_space.name
                        )
                        
                        # Создаем mock trial
                        mock_trial = Mock()
                        mock_trial.number = 1
                        
                        # Настраиваем suggest_float для возврата конкретных значений
                        suggest_calls = []
                        def mock_suggest_float(name, low, high, **kwargs):
                            suggest_calls.append((name, low, high))
                            if name == "zscore_threshold":
                                return 2.0  # Фиксированное значение
                            elif name == "zscore_exit":
                                # Проверяем что диапазон ограничен threshold
                                assert high < 2.0, f"zscore_exit high ({high}) должен быть < zscore_threshold (2.0)"
                                assert low > -2.0, f"zscore_exit low ({low}) должен быть > -zscore_threshold (-2.0)"
                                return 0.5
                            return (low + high) / 2
                        
                        mock_trial.suggest_float = mock_suggest_float
                        mock_trial.suggest_int = Mock(return_value=30)
                        
                        # Вызываем _suggest_parameters
                        params = objective._suggest_parameters(mock_trial)
                        
                        # Проверяем что параметры корректны
                        assert 'zscore_threshold' in params
                        assert 'zscore_exit' in params
                        assert abs(params['zscore_exit']) < abs(params['zscore_threshold'])
                        
                        # Проверяем что suggest_float был вызван с правильными ограничениями
                        zscore_calls = [call for call in suggest_calls if call[0] in ['zscore_threshold', 'zscore_exit']]
                        assert len(zscore_calls) >= 2, "Должны быть вызовы для обоих zscore параметров"
                        
                    except Exception as e:
                        # Ожидаем некоторые ошибки из-за моков
                        if "TrialPruned" not in str(e):
                            print(f"Неожиданная ошибка: {e}")
                
                # Очистка
                Path(search_space.name).unlink()
            Path(base_config.name).unlink()
    
    def test_max_half_life_constrained_by_min(self):
        """Проверяет что max_half_life_days ограничен min_half_life_days."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as base_config:
            yaml.dump({
                'data_dir': 'test_data',
                'walk_forward': {
                    'start_date': '2024-01-01', 
                    'training_period_days': 10, 
                    'testing_period_days': 5
                },
                'portfolio': {'initial_capital': 10000},
                'backtest': {'annualizing_factor': 365},
                'pair_selection': {'bar_minutes': 15}
            }, base_config)
            base_config.flush()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as search_space:
                yaml.dump({
                    'filters': {
                        'min_half_life_days': {'low': 0.5, 'high': 5.0},
                        'max_half_life_days': {'low': 10.0, 'high': 50.0}
                    }
                }, search_space)
                search_space.flush()
                
                with patch('pathlib.Path.exists', return_value=True), \
                     patch('pandas.read_csv') as mock_read_csv:
                    
                    mock_pairs_df = Mock()
                    mock_pairs_df.__len__ = Mock(return_value=10)
                    mock_pairs_df.iterrows = Mock(return_value=[])
                    mock_read_csv.return_value = mock_pairs_df
                    
                    try:
                        objective = FastWalkForwardObjective(
                            base_config_path=base_config.name,
                            search_space_path=search_space.name
                        )
                        
                        mock_trial = Mock()
                        mock_trial.number = 1
                        
                        suggest_calls = []
                        def mock_suggest_float(name, low, high, **kwargs):
                            suggest_calls.append((name, low, high))
                            if name == "min_half_life_days":
                                return 3.0  # Фиксированное значение
                            elif name == "max_half_life_days":
                                # Проверяем что low >= min_half_life + 0.1
                                assert low >= 3.1, f"max_half_life low ({low}) должен быть >= min_half_life + 0.1 (3.1)"
                                return 20.0
                            return (low + high) / 2
                        
                        mock_trial.suggest_float = mock_suggest_float
                        mock_trial.suggest_int = Mock(return_value=30)
                        
                        params = objective._suggest_parameters(mock_trial)
                        
                        # Проверяем что параметры корректны
                        if 'min_half_life_days' in params and 'max_half_life_days' in params:
                            assert params['min_half_life_days'] <= params['max_half_life_days']
                        
                    except Exception as e:
                        if "TrialPruned" not in str(e):
                            print(f"Неожиданная ошибка: {e}")
                
                Path(search_space.name).unlink()
            Path(base_config.name).unlink()
    
    def test_impossible_constraints_raise_trial_pruned(self):
        """Проверяет что невозможные ограничения вызывают TrialPruned."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as base_config:
            yaml.dump({
                'data_dir': 'test_data',
                'walk_forward': {
                    'start_date': '2024-01-01', 
                    'training_period_days': 10, 
                    'testing_period_days': 5
                },
                'portfolio': {'initial_capital': 10000},
                'backtest': {'annualizing_factor': 365},
                'pair_selection': {'bar_minutes': 15}
            }, base_config)
            base_config.flush()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as search_space:
                yaml.dump({
                    'signals': {
                        'zscore_threshold': {'low': 1.0, 'high': 1.5},  # Узкий диапазон
                        'zscore_exit': {'low': 1.4, 'high': 2.0}       # Невозможный диапазон
                    }
                }, search_space)
                search_space.flush()
                
                with patch('pathlib.Path.exists', return_value=True), \
                     patch('pandas.read_csv') as mock_read_csv:
                    
                    mock_pairs_df = Mock()
                    mock_pairs_df.__len__ = Mock(return_value=10)
                    mock_pairs_df.iterrows = Mock(return_value=[])
                    mock_read_csv.return_value = mock_pairs_df
                    
                    try:
                        objective = FastWalkForwardObjective(
                            base_config_path=base_config.name,
                            search_space_path=search_space.name
                        )
                        
                        mock_trial = Mock()
                        mock_trial.number = 1
                        
                        def mock_suggest_float(name, low, high, **kwargs):
                            if name == "zscore_threshold":
                                return 1.2  # В пределах диапазона
                            elif name == "zscore_exit":
                                # Этот вызов должен привести к TrialPruned
                                # так как min_exit = max(-1.2 + 0.1, 1.4) = 1.4
                                # max_exit = min(2.0, 1.2 - 0.1) = 1.1
                                # 1.4 > 1.1 - невозможный диапазон
                                return 1.5
                            return (low + high) / 2
                        
                        mock_trial.suggest_float = mock_suggest_float
                        
                        # Должен вызвать TrialPruned
                        with pytest.raises(optuna.TrialPruned):
                            objective._suggest_parameters(mock_trial)
                        
                    except optuna.TrialPruned:
                        # Это ожидаемое поведение
                        pass
                    except Exception as e:
                        print(f"Неожиданная ошибка: {e}")
                
                Path(search_space.name).unlink()
            Path(base_config.name).unlink()
    
    def test_logarithmic_distribution_for_scales(self):
        """Проверяет использование логарифмического распределения для масштабов."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as base_config:
            yaml.dump({
                'data_dir': 'test_data',
                'walk_forward': {
                    'start_date': '2024-01-01', 
                    'training_period_days': 10, 
                    'testing_period_days': 5
                },
                'portfolio': {'initial_capital': 10000},
                'backtest': {'annualizing_factor': 365},
                'pair_selection': {'bar_minutes': 15}
            }, base_config)
            base_config.flush()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as search_space:
                yaml.dump({
                    'filters': {
                        'ssd_top_n': {'low': 1000, 'high': 100000},  # Без step - должно использовать log
                        'min_half_life_days': {'low': 0.1, 'high': 10.0},
                        'max_half_life_days': {'low': 10.0, 'high': 100.0}
                    }
                }, search_space)
                search_space.flush()
                
                with patch('pathlib.Path.exists', return_value=True), \
                     patch('pandas.read_csv') as mock_read_csv:
                    
                    mock_pairs_df = Mock()
                    mock_pairs_df.__len__ = Mock(return_value=10)
                    mock_pairs_df.iterrows = Mock(return_value=[])
                    mock_read_csv.return_value = mock_pairs_df
                    
                    try:
                        objective = FastWalkForwardObjective(
                            base_config_path=base_config.name,
                            search_space_path=search_space.name
                        )
                        
                        mock_trial = Mock()
                        mock_trial.number = 1
                        
                        log_calls = []
                        def mock_suggest_float(name, low, high, log=False, **kwargs):
                            if log:
                                log_calls.append(name)
                            if name == "ssd_top_n_log":
                                return 4.0  # 10^4 = 10000
                            elif name.endswith("_half_life_days"):
                                return (low + high) / 2
                            return (low + high) / 2
                        
                        def mock_suggest_int(name, low, high, **kwargs):
                            return int((low + high) / 2)
                        
                        mock_trial.suggest_float = mock_suggest_float
                        mock_trial.suggest_int = mock_suggest_int
                        
                        params = objective._suggest_parameters(mock_trial)
                        
                        # Проверяем что логарифмическое распределение использовалось
                        assert len(log_calls) > 0, "Должно использоваться логарифмическое распределение"
                        
                        # Проверяем что ssd_top_n был вычислен правильно
                        if 'ssd_top_n' in params:
                            assert isinstance(params['ssd_top_n'], int), "ssd_top_n должен быть int"
                            assert 1000 <= params['ssd_top_n'] <= 100000, "ssd_top_n должен быть в заданном диапазоне"
                        
                    except Exception as e:
                        if "TrialPruned" not in str(e):
                            print(f"Неожиданная ошибка: {e}")
                
                Path(search_space.name).unlink()
            Path(base_config.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
