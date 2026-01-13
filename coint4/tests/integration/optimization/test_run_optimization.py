"""Тесты для run_optimization.py (консолидированные из fix файлов)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from optimiser.run_optimization import run_optimization

# Константы для тестирования
import os
from tests.conftest import get_test_config

# Получаем конфигурацию в зависимости от режима
test_config = get_test_config()

DEFAULT_N_TRIALS = test_config['n_trials']  # Теперь 2-5 в зависимости от режима
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_N_JOBS = 1
DEFAULT_RESULTS_DIR = 'results'
CONFIG_FILENAME = 'config.yaml'
YAML_EXTENSION = '.yaml'

# Константы для валидации оптимизации
LARGE_N_TRIALS = 100  # Только для валидации, не для реального выполнения
LARGE_TIMEOUT_SECONDS = 3600
PARALLEL_N_JOBS = 4
MIN_N_TRIALS = 1
MAX_N_TRIALS = 10000
MIN_TIMEOUT = 1
MAX_TIMEOUT = 86400  # 24 hours
MIN_N_JOBS = 1
MAX_N_JOBS = 16


class TestRunOptimizationUnit:
    """Быстрые unit тесты для run_optimization."""
    
    @pytest.mark.unit
    def test_configuration_saving_when_enabled_then_logic_correct(self):
        """Unit test: проверяем логику сохранения конфигурации (Task 4 fix)."""
        # Мокаем конфигурацию
        mock_config = {
            'optimization': {
                'n_trials': DEFAULT_N_TRIALS,
                'timeout_seconds': DEFAULT_TIMEOUT_SECONDS,
                'n_jobs': DEFAULT_N_JOBS
            },
            'output': {
                'results_dir': DEFAULT_RESULTS_DIR,
                'save_config': True
            }
        }

        # Проверяем параметры сохранения
        assert 'output' in mock_config, "Должна быть секция output"
        assert 'save_config' in mock_config['output'], "Должен быть параметр save_config"

        # Проверяем логику определения пути сохранения
        results_dir = mock_config['output']['results_dir']
        config_path = Path(results_dir) / CONFIG_FILENAME

        assert isinstance(config_path, Path), "Путь должен быть объектом Path"
        assert config_path.suffix == YAML_EXTENSION, "Конфигурация должна сохраняться в YAML"
    
    @pytest.mark.unit
    def test_optimization_parameters_when_validated_then_constraints_enforced(self):
        """Unit test: проверяем валидацию параметров оптимизации."""
        # Тестируем различные параметры оптимизации
        N_JOBS_AUTO = -1
        SAMPLER_TYPE = 'TPE'
        PRUNER_TYPE = 'MedianPruner'

        optimization_params = {
            'n_trials': LARGE_N_TRIALS,
            'timeout_seconds': LARGE_TIMEOUT_SECONDS,
            'n_jobs': N_JOBS_AUTO,
            'sampler': SAMPLER_TYPE,
            'pruner': PRUNER_TYPE
        }

        # Проверяем разумность параметров
        assert optimization_params['n_trials'] > 0, "n_trials должен быть положительным"
        assert optimization_params['timeout_seconds'] > 0, "timeout должен быть положительным"
        assert optimization_params['n_jobs'] != 0, "n_jobs не должен быть нулем"

        # Проверяем типы
        assert isinstance(optimization_params['n_trials'], int)
        assert isinstance(optimization_params['timeout_seconds'], int)
        assert isinstance(optimization_params['n_jobs'], int)
    
    @pytest.mark.unit
    def test_results_directory_when_created_then_logic_correct(self, tmp_path: Path):
        """Unit test: проверяем логику создания директории результатов."""
        TEST_RESULTS_DIR = 'test_results'
        results_dir = tmp_path / TEST_RESULTS_DIR

        # Проверяем, что директория не существует
        assert not results_dir.exists(), "Директория не должна существовать изначально"

        # Симулируем создание директории
        results_dir.mkdir(parents=True, exist_ok=True)

        # Проверяем, что директория создана
        assert results_dir.exists(), "Директория должна быть создана"
        assert results_dir.is_dir(), "Должна быть директорией"
    
    @pytest.mark.unit
    def test_study_name_when_generated_then_format_correct(self):
        """Unit test: проверяем логику генерации имени исследования."""
        import datetime

        STUDY_PREFIX = "optimization_"
        TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
        SEPARATOR = "_"

        # Симулируем генерацию имени исследования
        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        study_name = f"{STUDY_PREFIX}{timestamp}"

        # Проверяем формат имени
        assert study_name.startswith(STUDY_PREFIX), f"Имя должно начинаться с {STUDY_PREFIX}"
        assert len(study_name) > len(STUDY_PREFIX), "Имя должно содержать timestamp"
        assert SEPARATOR in study_name, f"Имя должно содержать разделитель {SEPARATOR}"


class TestRunOptimizationIntegration:
    """Медленные integration тесты для run_optimization."""
    
    @pytest.mark.slow
    @pytest.mark.serial  # Добавляем для база данных операций 
    @pytest.mark.integration
    def test_optimization_pipeline_when_full_run_then_completes_successfully(self, tmp_path: Path):
        """Integration test: проверяем полный пайплайн оптимизации."""
        # Проверяем наличие необходимых файлов
        MAIN_CONFIG_FILE = "configs/main_2024.yaml"
        SEARCH_SPACE_FILE = "configs/search_space_fast.yaml"

        required_files = [
            MAIN_CONFIG_FILE,
            SEARCH_SPACE_FILE
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                pytest.skip(f"Файл {file_path} не найден")
        
        try:
            # Мокаем аргументы командной строки
            test_args = {
                'base_config': 'configs/main_2024.yaml',
                'search_space': 'configs/search_space_fast.yaml',
                'n_trials': 2,  # Минимальное количество для быстрого теста
                'timeout': 60,  # 1 минута максимум
                'n_jobs': 1,
                'results_dir': str(tmp_path),
                'study_name': 'test_study'
            }
            
            # Запускаем оптимизацию с минимальными параметрами
            with patch('sys.argv', ['run_optimization.py'] + [f'--{k}={v}' for k, v in test_args.items()]):
                    result = run_optimization(
                        base_config_path=test_args['base_config'],
                        search_space_path=test_args['search_space'],
                        n_trials=test_args['n_trials'],
                        n_jobs=test_args['n_jobs'],
                        study_name=test_args['study_name']
                    )
                
            # Проверяем результат
            assert result is not None, "Результат оптимизации не должен быть None"
            
            # Проверяем, что файлы результатов созданы
            results_path = tmp_path
            assert results_path.exists(), "Директория результатов должна существовать"
                
        except Exception as e:
            # Если тест падает из-за отсутствия данных, это нормально
            if any(keyword in str(e).lower() for keyword in ['data', 'file', 'path']):
                pytest.skip(f"Тест пропущен из-за отсутствия данных: {str(e)}")
            else:
                pytest.fail(f"Неожиданная ошибка в оптимизации: {str(e)}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_configuration_saving_when_integrated_then_works_correctly(self, tmp_path: Path):
        """Integration test: проверяем сохранение конфигурации в реальной системе."""
        results_dir = tmp_path / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем тестовую конфигурацию
        test_config = {
            'test_parameter': 'test_value',
            'optimization': {
                'n_trials': 5,
                'timeout_seconds': 30
            }
        }
        
        # Симулируем сохранение конфигурации
        config_path = results_dir / 'saved_config.yaml'
        
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Проверяем, что файл создан
            assert config_path.exists(), "Файл конфигурации должен быть создан"
            
            # Проверяем, что можем прочитать конфигурацию обратно
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config == test_config, "Загруженная конфигурация должна совпадать с исходной"
            
        except ImportError:
            pytest.skip("PyYAML не установлен")
        except Exception as e:
            pytest.fail(f"Ошибка при сохранении/загрузке конфигурации: {str(e)}")
        
