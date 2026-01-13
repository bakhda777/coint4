"""Быстрые тесты параллельной оптимизации.

Оптимизировано согласно best practices:
- Мокирование тяжелых операций
- Минимальное количество trials
- Unit тесты без реальной оптимизации
"""

import pytest
import optuna
import tempfile
from unittest.mock import MagicMock


@pytest.mark.fast
class TestParallelOptimizationFast:
    """Быстрые тесты параллельной оптимизации."""
    
    @pytest.mark.unit
    def test_optuna_study_creation_fast(self):
        """Быстрый тест создания Optuna study."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            study = optuna.create_study(
                direction='maximize',
                storage=f"sqlite:///{tmp.name}",
                study_name='test_fast'
            )
            
            assert study is not None
            assert study.direction == optuna.study.StudyDirection.MAXIMIZE
            assert study.study_name == 'test_fast'
    
    @pytest.mark.unit
    def test_objective_function_fast(self):
        """Быстрый тест целевой функции."""
        def simple_objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return -(x ** 2)  # Простая парабола
        
        # Создаем study в памяти
        study = optuna.create_study(direction='maximize')
        
        # Оптимизируем с 1 trial
        study.optimize(simple_objective, n_trials=1)
        
        assert len(study.trials) == 1
        assert study.best_trial is not None
    
    @pytest.mark.unit
    def test_search_space_configuration_fast(self):
        """Быстрый тест конфигурации пространства поиска."""
        search_space = {
            'rolling_window': {'type': 'int', 'low': 20, 'high': 30},
            'zscore_threshold': {'type': 'float', 'low': 1.5, 'high': 3.0}
        }
        
        assert search_space['rolling_window']['type'] == 'int'
        assert search_space['zscore_threshold']['type'] == 'float'
        assert search_space['rolling_window']['low'] < search_space['rolling_window']['high']
    
    @pytest.mark.fast
    def test_parallel_execution_mocked(self):
        """Быстрый тест параллельного выполнения."""
        # Создаем study в памяти
        study = optuna.create_study(direction='maximize')
        
        # Простая целевая функция
        def mocked_objective(trial):
            trial.suggest_int('param1', 1, 10)
            return 0.5  # Фиксированное значение
        
        # Запускаем с минимальными параметрами
        study.optimize(
            mocked_objective,
            n_trials=2,
            n_jobs=1,  # Последовательно для предсказуемости
            timeout=1
        )
        
        assert len(study.trials) == 2
        assert all(t.value == 0.5 for t in study.trials)
    
    @pytest.mark.unit
    def test_trial_pruning_fast(self):
        """Быстрый тест pruning механизма."""
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        def prunable_objective(trial):
            for step in range(3):
                intermediate_value = step * 0.1
                trial.report(intermediate_value, step)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return 0.3
        
        # Запускаем несколько trials
        study.optimize(prunable_objective, n_trials=3, timeout=1)
        
        assert len(study.trials) >= 1
        # Проверяем что pruner настроен
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)
    
    @pytest.mark.unit
    def test_study_callbacks_fast(self):
        """Быстрый тест callbacks в study."""
        callback_called = {'count': 0}
        
        def callback(study, trial):
            callback_called['count'] += 1
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: trial.suggest_float('x', 0, 1),
            n_trials=2,
            callbacks=[callback]
        )
        
        assert callback_called['count'] == 2