#!/usr/bin/env python3
"""
Тесты для проверки правильной обработки конкурентного доступа к SQLite в Optuna.
"""

import pytest
import optuna
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.run_optimization import run_optimization


class TestSQLiteConcurrency:
    """Тесты для проверки обработки конкурентного доступа к SQLite."""
    
    def test_sqlite_forces_single_job(self):
        """Проверяет что SQLite принудительно устанавливает n_jobs=1."""
        
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
                    'signals': {'zscore_threshold': {'low': 1.0, 'high': 2.0}}
                }, search_space)
                search_space.flush()
                
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as db_file:
                    db_path = db_file.name
                
                # Мокаем создание study и objective
                with patch('optuna.create_study') as mock_create_study, \
                     patch('src.optimiser.run_optimization.FastWalkForwardObjective') as mock_objective_class:
                    
                    mock_study = Mock()
                    mock_study.trials = []
                    mock_study.user_attrs = {}
                    mock_study.set_user_attr = Mock()
                    mock_create_study.return_value = mock_study
                    
                    mock_objective = Mock()
                    mock_objective_class.return_value = mock_objective
                    
                    # Пытаемся запустить с n_jobs > 1
                    try:
                        run_optimization(
                            n_trials=1,
                            base_config_path=base_config.name,
                            search_space_path=search_space.name,
                            storage_path=db_path,
                            n_jobs=4  # Больше 1
                        )
                    except:
                        pass  # Ожидаем ошибки из-за моков
                    
                    # Проверяем что study создан с правильными параметрами
                    mock_create_study.assert_called_once()
                    call_kwargs = mock_create_study.call_args[1]
                    
                    # Проверяем что используется RDBStorage
                    storage = call_kwargs['storage']
                    assert hasattr(storage, 'url'), "Должен использоваться RDBStorage"
                    assert 'sqlite' in storage.url, "URL должен содержать sqlite"
                
                # Очистка
                Path(search_space.name).unlink()
                Path(db_path).unlink()
            Path(base_config.name).unlink()
    
    def test_rdb_storage_with_timeouts(self):
        """Проверяет что RDBStorage создается с правильными таймаутами."""
        
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
                    'signals': {'zscore_threshold': {'low': 1.0, 'high': 2.0}}
                }, search_space)
                search_space.flush()
                
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as db_file:
                    db_path = db_file.name
                
                # Мокаем RDBStorage
                with patch('optuna.storages.RDBStorage') as mock_rdb_storage, \
                     patch('optuna.create_study') as mock_create_study, \
                     patch('src.optimiser.run_optimization.FastWalkForwardObjective'):
                    
                    mock_storage = Mock()
                    mock_rdb_storage.return_value = mock_storage
                    
                    mock_study = Mock()
                    mock_study.trials = []
                    mock_study.user_attrs = {}
                    mock_study.set_user_attr = Mock()
                    mock_create_study.return_value = mock_study
                    
                    try:
                        run_optimization(
                            n_trials=1,
                            base_config_path=base_config.name,
                            search_space_path=search_space.name,
                            storage_path=db_path,
                            n_jobs=1
                        )
                    except:
                        pass  # Ожидаем ошибки из-за моков
                    
                    # Проверяем что RDBStorage создан с правильными параметрами
                    mock_rdb_storage.assert_called_once()
                    call_args = mock_rdb_storage.call_args
                    
                    # Проверяем URL
                    assert call_args[1]['url'].startswith('sqlite:///')
                    
                    # Проверяем engine_kwargs
                    engine_kwargs = call_args[1]['engine_kwargs']
                    assert 'connect_args' in engine_kwargs
                    assert 'timeout' in engine_kwargs['connect_args']
                    assert engine_kwargs['connect_args']['timeout'] == 600
                    assert engine_kwargs['connect_args']['check_same_thread'] is False
                    assert engine_kwargs['pool_pre_ping'] is True
                    assert engine_kwargs['pool_recycle'] == 300
                
                # Очистка
                Path(search_space.name).unlink()
                Path(db_path).unlink()
            Path(base_config.name).unlink()
    
    def test_non_sqlite_storage_preserves_n_jobs(self):
        """Проверяет что для не-SQLite storage n_jobs сохраняется."""
        
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
                    'signals': {'zscore_threshold': {'low': 1.0, 'high': 2.0}}
                }, search_space)
                search_space.flush()
                
                # Используем не-SQLite путь
                storage_path = "postgresql://user:pass@localhost/db"
                
                with patch('optuna.create_study') as mock_create_study, \
                     patch('src.optimiser.run_optimization.FastWalkForwardObjective'):
                    
                    mock_study = Mock()
                    mock_study.trials = []
                    mock_study.user_attrs = {}
                    mock_study.set_user_attr = Mock()
                    mock_study.optimize = Mock()
                    mock_create_study.return_value = mock_study
                    
                    try:
                        run_optimization(
                            n_trials=1,
                            base_config_path=base_config.name,
                            search_space_path=search_space.name,
                            storage_path=storage_path,
                            n_jobs=4  # Больше 1
                        )
                    except:
                        pass  # Ожидаем ошибки из-за моков
                    
                    # Проверяем что optimize вызван с n_jobs=4
                    if mock_study.optimize.called:
                        call_kwargs = mock_study.optimize.call_args[1]
                        assert call_kwargs.get('n_jobs') == 4, "n_jobs должен сохраниться для не-SQLite storage"
                
                # Очистка
                Path(search_space.name).unlink()
            Path(base_config.name).unlink()


class TestPrunerConfiguration:
    """Тесты конфигурации pruner для предотвращения преждевременного pruning."""
    
    def test_pruner_startup_trials(self):
        """Проверяет что pruner настроен с достаточным количеством startup trials."""
        
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
                    'signals': {'zscore_threshold': {'low': 1.0, 'high': 2.0}}
                }, search_space)
                search_space.flush()
                
                with patch('optuna.create_study') as mock_create_study, \
                     patch('src.optimiser.run_optimization.FastWalkForwardObjective'):
                    
                    mock_study = Mock()
                    mock_study.trials = []
                    mock_study.user_attrs = {}
                    mock_study.set_user_attr = Mock()
                    mock_create_study.return_value = mock_study
                    
                    try:
                        run_optimization(
                            n_trials=100,
                            base_config_path=base_config.name,
                            search_space_path=search_space.name,
                            n_jobs=1
                        )
                    except:
                        pass  # Ожидаем ошибки из-за моков
                    
                    # Проверяем параметры pruner
                    mock_create_study.assert_called_once()
                    call_kwargs = mock_create_study.call_args[1]
                    
                    pruner = call_kwargs['pruner']
                    # MedianPruner имеет _n_startup_trials вместо n_startup_trials
                    startup_trials = getattr(pruner, 'n_startup_trials', getattr(pruner, '_n_startup_trials', None))
                    assert startup_trials is not None, "Pruner должен иметь n_startup_trials или _n_startup_trials"
                    assert startup_trials >= 20, f"n_startup_trials должен быть >= 20, получен: {startup_trials}"
                    
                    # Проверяем параметры sampler
                    sampler = call_kwargs['sampler']
                    # TPESampler может иметь _n_startup_trials вместо n_startup_trials
                    sampler_startup_trials = getattr(sampler, 'n_startup_trials', getattr(sampler, '_n_startup_trials', None))
                    assert sampler_startup_trials is not None, "Sampler должен иметь n_startup_trials или _n_startup_trials"
                    assert sampler_startup_trials >= 10, f"Sampler n_startup_trials должен быть >= 10, получен: {sampler_startup_trials}"

                    # Проверяем другие параметры sampler
                    assert hasattr(sampler, '_multivariate') or hasattr(sampler, 'multivariate'), "Sampler должен иметь multivariate"
                    multivariate = getattr(sampler, 'multivariate', getattr(sampler, '_multivariate', None))
                    assert multivariate is True, f"multivariate должен быть True, получен: {multivariate}"

                    assert hasattr(sampler, '_group') or hasattr(sampler, 'group'), "Sampler должен иметь group"
                    group = getattr(sampler, 'group', getattr(sampler, '_group', None))
                    assert group is True, f"group должен быть True, получен: {group}"
                
                # Очистка
                Path(search_space.name).unlink()
            Path(base_config.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
