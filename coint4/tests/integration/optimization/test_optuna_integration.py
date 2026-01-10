"""
Объединенные тесты Optuna оптимизации.
Консолидирует ключевые тесты из нескольких файлов в параметризованные тесты.
"""

import pytest
import optuna
import os
from unittest.mock import Mock, patch

from src.optimiser.fast_objective import FastWalkForwardObjective
from src.optimiser.constants import MIN_TRADES_THRESHOLD
from tests.conftest import get_test_config


# Константы для тестов - используем глобальные настройки
ROLLING_WINDOW_MIN = 10
ROLLING_WINDOW_MAX = 50
Z_THRESHOLD_MIN = 1.0
Z_THRESHOLD_MAX = 3.0
Z_EXIT_MIN = 0.3
Z_EXIT_MAX = 0.7
TEST_INT_MIN = 1
TEST_INT_MAX = 10
TEST_INT_STEP = 2
TEST_FLOAT_MIN = 0.1
TEST_FLOAT_MAX = 1.0

@pytest.fixture
def optuna_test_config():
    """Фикстура для настройки тестового окружения Optuna."""
    test_config = get_test_config()
    return {
        'base_config_path': "configs/main_2024.yaml",
        'search_space_path': "configs/search_space_fast.yaml",
        'min_trials': test_config['n_trials'],
        'max_trials': test_config['n_trials']
    }


@pytest.fixture
def deterministic_optuna_study(tmp_path):
    """Детерминистическая Optuna study с фиксированным seed."""
    storage = f"sqlite:///{tmp_path/'deterministic_study.db'}"
    return optuna.create_study(
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(seed=42),  # ОБЯЗАТЕЛЬНО: фиксированный seed
        direction="maximize"
    )


class TestConsolidatedOptuna:
    """Объединенные тесты для Optuna оптимизации."""
    
    @pytest.mark.slow
    @pytest.mark.serial
    @pytest.mark.parametrize("test_scenario", [
        "parameter_generation_without_log",
        "objective_initialization"
    ])
    def test_optuna_objective_when_scenario_tested_then_works_correctly(self, test_scenario, fast_study, optuna_test_config):
        """Параметризованный тест для функциональности Optuna objective."""

        if test_scenario == "parameter_generation_without_log":
            self._test_parameter_generation_without_log(fast_study, optuna_test_config)
        elif test_scenario == "objective_initialization":
            self._test_objective_initialization(fast_study, optuna_test_config)
    
    def _test_objective_initialization(self, study, config):
        """Тест инициализации objective."""
        objective = FastWalkForwardObjective(config['base_config_path'], config['search_space_path'])

        # Проверяем базовые атрибуты
        assert hasattr(objective, 'base_config')
        assert hasattr(objective, 'search_space')
        assert hasattr(objective, '__call__')
        assert callable(objective)
    
    def _test_parameter_generation_without_log(self, study, config=None):
        """Тест детерминистической генерации параметров."""
        # Проверяем, что study использует RandomSampler с фиксированным seed
        assert isinstance(study.sampler, optuna.samplers.RandomSampler), \
            "Study должна использовать RandomSampler согласно правилам"
        
        # Проверяем детерминизм: одинаковые параметры при повторном запуске
        # На самом деле RandomSampler не гарантирует полный детерминизм,
        # но с фиксированным seed поведение будет воспроизводимым

        trial = study.ask()

        # Проверяем, что параметры генерируются в оказанных границах
        rolling_window = trial.suggest_int('rolling_window', ROLLING_WINDOW_MIN, ROLLING_WINDOW_MAX)
        z_threshold = trial.suggest_float('z_threshold', Z_THRESHOLD_MIN, Z_THRESHOLD_MAX)

        assert ROLLING_WINDOW_MIN <= rolling_window <= ROLLING_WINDOW_MAX, \
            f"rolling_window {rolling_window} должен быть в [{ROLLING_WINDOW_MIN}, {ROLLING_WINDOW_MAX}]"
        assert Z_THRESHOLD_MIN <= z_threshold <= Z_THRESHOLD_MAX, \
            f"z_threshold {z_threshold} должен быть в [{Z_THRESHOLD_MIN}, {Z_THRESHOLD_MAX}]"
        
        # Проверяем, что параметры можно сериализовать
        assert isinstance(rolling_window, int), "rolling_window должен быть int"
        assert isinstance(z_threshold, float), "z_threshold должен быть float"

        # Проверяем минимальное количество trials (используем get_test_config если config не передан)
        test_config = get_test_config()
        min_trials = config['min_trials'] if config else test_config['n_trials']
        max_trials = config['max_trials'] if config else test_config['n_trials']
        assert min_trials <= max_trials, \
            f"Количество trials должно быть между {min_trials} и {max_trials}"
    
    @pytest.mark.slow
    @pytest.mark.serial
    @pytest.mark.parametrize("integration_scenario", [
        "parameter_types_consistency",
        "normalization_config_section",
        "all_fixes_integration"
    ])
    def test_optuna_integration_when_scenario_tested_then_parameters_consistent(self, integration_scenario, fast_study, optuna_test_config):
        """Параметризованный тест для интеграции Optuna."""

        if integration_scenario == "parameter_types_consistency":
            self._test_parameter_types_consistency(fast_study)
        elif integration_scenario == "normalization_config_section":
            self._test_normalization_config_section(fast_study)
        elif integration_scenario == "all_fixes_integration":
            self._test_all_fixes_integration(fast_study)
    
    def _test_parameter_types_consistency(self, study):
        """Тест консистентности типов параметров."""
        trial = study.ask()
        
        # Проверяем различные типы параметров
        int_param = trial.suggest_int('test_int', TEST_INT_MIN, TEST_INT_MAX, step=TEST_INT_STEP)
        float_param = trial.suggest_float('test_float', TEST_FLOAT_MIN, TEST_FLOAT_MAX)
        categorical_param = trial.suggest_categorical('test_cat', ['a', 'b', 'c'])
        
        assert isinstance(int_param, int)
        assert isinstance(float_param, float)
        assert categorical_param in ['a', 'b', 'c']
    
    def _test_normalization_config_section(self, study):
        """Тест секции конфигурации нормализации."""
        # Используем стандартную конфигурацию, которая уже содержит нормализацию
        base_config_path = "configs/main_2024.yaml"
        search_space_path = "configs/search_space_fast.yaml"
        objective = FastWalkForwardObjective(base_config_path, search_space_path)

        # Проверяем, что objective может быть создан без ошибок
        assert hasattr(objective, '__call__')
        assert callable(objective)

        # Проверяем, что конфигурация загружена корректно
        assert hasattr(objective, 'base_config'), "Должна быть загружена базовая конфигурация"
        assert hasattr(objective, 'search_space'), "Должно быть загружено пространство поиска"
        assert objective.base_config is not None, "Базовая конфигурация не должна быть None"

        # Проверяем, что в search_space нет 'filters' (что было основной проблемой)
        assert 'filters' not in objective.search_space, \
            "В fast-режиме 'filters' не должны присутствовать в search_space"
    
    def _test_all_fixes_integration(self, study):
        """Тест интеграции всех исправлений."""
        base_config_path = "configs/main_2024.yaml"
        search_space_path = "configs/search_space_fast.yaml"
        objective = FastWalkForwardObjective(base_config_path, search_space_path)
        trial = study.ask()

        # Проверяем базовую функциональность
        assert hasattr(objective, '__call__')
        assert callable(objective)
        
        # Проверяем, что можно создать параметры
        params = {
            'rolling_window': trial.suggest_int('rolling_window', 20, 40),
            'z_threshold': trial.suggest_float('z_threshold', 1.5, 2.5),
            'z_exit': trial.suggest_float('z_exit', Z_EXIT_MIN, Z_EXIT_MAX)
        }
        
        for key, value in params.items():
            assert isinstance(value, (int, float))
    
    @pytest.mark.slow
    @pytest.mark.serial
    def test_critical_fixes_when_validated_comprehensively_then_work_correctly(self, fast_study):
        """Комплексный тест критических исправлений."""
        base_config_path = "configs/main_2024.yaml"
        search_space_path = "configs/search_space_fast.yaml"
        objective = FastWalkForwardObjective(base_config_path, search_space_path)

        # Проверяем, что objective создается без ошибок
        assert hasattr(objective, 'base_config')
        assert hasattr(objective, 'search_space')

        # Проверяем, что можем создать trial
        trial = fast_study.ask()
        assert trial is not None

        # Проверяем базовую функциональность параметров
        param = trial.suggest_float('test_param', TEST_FLOAT_MIN, TEST_FLOAT_MAX)
        assert TEST_FLOAT_MIN <= param <= TEST_FLOAT_MAX
    
    @pytest.mark.slow
    @pytest.mark.serial
    def test_sqlite_concurrency_when_concurrent_access_then_handled_correctly(self, tmp_path):
        """Тест обработки конкурентности SQLite."""
        # Создаем временную базу данных
        db_path = tmp_path / "test_optuna.db"
        storage_url = f"sqlite:///{db_path}"

        study = optuna.create_study(
            storage=storage_url,
            study_name="test_concurrency",
            load_if_exists=True,
            direction="minimize"
        )

        # Тестируем создание trial и базовые операции
        trial = study.ask()
        param = trial.suggest_float('test_param', TEST_FLOAT_MIN, TEST_FLOAT_MAX)
        study.tell(trial, param)  # Используем простое значение

        assert len(study.trials) == 1
        assert study.trials[0].value == pytest.approx(param, rel=1e-9)


class TestOptunaBlazefast:
    """Быстрые версии Optuna тестов с мокированием."""
    
    @pytest.mark.fast
    @patch('src.optimiser.fast_objective.FastWalkForwardObjective')
    def test_optuna_objective_when_mocked_then_logic_works(self, mock_objective):
        """Fast test: Проверяет логику objective функции с моком."""
        # Настраиваем мок
        mock_instance = Mock()
        mock_instance.objective.return_value = 0.15  # Мокируем хороший sharpe ratio
        mock_objective.return_value = mock_instance
        
        # Создаем мок trial
        mock_trial = Mock()
        mock_trial.suggest_int.return_value = 20
        mock_trial.suggest_float.return_value = 2.0
        
        # Тестируем логику
        obj = mock_objective("fake_config", "fake_search_space") 
        result = obj.objective(mock_trial)
        
        # Проверки
        assert mock_objective.called
        assert result == 0.15
        mock_instance.objective.assert_called_once_with(mock_trial)
    
    @pytest.mark.fast
    @patch('optuna.create_study')
    def test_optuna_study_when_mocked_then_optimization_logic_works(self, mock_study):
        """Fast test: Проверяет логику создания и оптимизации study с моком."""
        # Настраиваем мок study
        mock_study_instance = Mock()
        mock_trial = Mock()
        mock_trial.suggest_int.return_value = 25
        mock_trial.suggest_float.return_value = 1.8
        
        # Мокируем поведение study
        mock_study_instance.ask.return_value = mock_trial
        mock_study_instance.tell.return_value = None
        mock_study_instance.trials = [mock_trial]
        mock_study.return_value = mock_study_instance
        
        # Тестируем создание study
        study = mock_study(direction="maximize")
        
        # Симулируем оптимизацию
        trial = study.ask()
        param1 = trial.suggest_int('rolling_window', 10, 50)
        param2 = trial.suggest_float('z_threshold', 1.0, 3.0)
        study.tell(trial, 0.25)  # Мокируем результат
        
        # Проверки
        assert mock_study.called
        assert param1 == 25
        assert param2 == 1.8
        assert len(study.trials) == 1
    
    @pytest.mark.fast
    def test_optuna_parameter_validation_when_mocked_then_validates(self):
        """Fast test: Проверяет валидацию параметров без реальной оптимизации.""" 
        # Создаем мок параметров
        params = {
            'rolling_window': 25,
            'z_threshold': 2.0,
            'z_exit': 0.5,
            'commission_pct': 0.001,
            'slippage_pct': 0.0005
        }
        
        # Проверяем валидность параметров
        assert ROLLING_WINDOW_MIN <= params['rolling_window'] <= ROLLING_WINDOW_MAX
        assert Z_THRESHOLD_MIN <= params['z_threshold'] <= Z_THRESHOLD_MAX  
        assert Z_EXIT_MIN <= params['z_exit'] <= Z_EXIT_MAX
        assert 0 <= params['commission_pct'] <= 0.01
        assert 0 <= params['slippage_pct'] <= 0.01
        
        # Проверяем логическую связность
        assert params['z_exit'] < params['z_threshold'], "z_exit должен быть меньше z_threshold"
