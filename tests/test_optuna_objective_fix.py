#!/usr/bin/env python3
"""
Тест для проверки работы FastWalkForwardObjective после исправлений.
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

from src.optimiser.fast_objective import FastWalkForwardObjective, PENALTY
from src.optimiser.metric_utils import validate_params


class TestOptunaObjectiveFix:
    """Тесты для проверки исправленной objective функции."""
    
    def test_objective_with_real_trial(self):
        """Тестирует objective с реальным Optuna trial."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем реальный study
        study = optuna.create_study(direction='maximize')
        
        # Создаем objective
        objective = FastWalkForwardObjective(
            str(config_path),
            str(search_space_path)
        )
        
        # Мокаем бэктест для быстрого тестирования
        with patch.object(objective, '_run_fast_backtest') as mock_backtest:
            mock_backtest.return_value = {
                'sharpe_ratio_abs': 1.5,
                'total_trades': 100,
                'max_drawdown': 0.12,
                'total_pnl': 1500.0,
                'total_return_pct': 0.15,
                'win_rate': 0.55,
                'avg_trade_size': 500.0,
                'avg_hold_time': 24.0
            }
            
            # Функция для тестирования одного trial
            def test_objective_func(trial):
                try:
                    result = objective(trial)
                    print(f"Trial {trial.number} результат: {result}")
                    
                    # Проверяем что результат не PENALTY
                    assert result != PENALTY, f"Trial вернул PENALTY: {result}"
                    assert isinstance(result, (int, float)), f"Результат должен быть числом: {type(result)}"
                    assert not np.isnan(result), "Результат не должен быть NaN"
                    assert not np.isinf(result), "Результат не должен быть бесконечностью"
                    
                    return result
                    
                except optuna.TrialPruned as e:
                    print(f"Trial {trial.number} был pruned: {e}")
                    raise
                except Exception as e:
                    print(f"Trial {trial.number} ошибка: {e}")
                    raise
            
            # Запускаем несколько trials
            study.optimize(test_objective_func, n_trials=3)
            
            # Проверяем результаты
            assert len(study.trials) == 3, f"Должно быть 3 trials, получено: {len(study.trials)}"
            
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            assert len(completed_trials) > 0, "Должен быть хотя бы один завершенный trial"
            
            for trial in completed_trials:
                assert trial.value != PENALTY, f"Trial {trial.number} вернул PENALTY: {trial.value}"
                print(f"✓ Trial {trial.number}: {trial.value}")
    
    def test_parameter_generation_without_log(self):
        """Тестирует генерацию параметров без использования log=True."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем mock trial
        mock_trial = Mock()
        mock_trial.number = 1
        
        # Настраиваем mock для suggest методов
        def mock_suggest_float(name, low, high, **kwargs):
            # Проверяем что log=True не передается
            assert 'log' not in kwargs, f"Параметр log не должен использоваться в suggest_float для {name}"
            return (low + high) / 2
        
        def mock_suggest_int(name, low, high, step=1):
            return int((low + high) / 2)
        
        def mock_suggest_categorical(name, choices):
            return choices[0]
        
        mock_trial.suggest_float = mock_suggest_float
        mock_trial.suggest_int = mock_suggest_int
        mock_trial.suggest_categorical = mock_suggest_categorical
        
        # Создаем objective
        objective = FastWalkForwardObjective(
            str(config_path),
            str(search_space_path)
        )
        
        # Генерируем параметры
        params = objective._suggest_parameters(mock_trial)
        print(f"Сгенерированные параметры: {params}")
        
        # Проверяем что параметры валидны
        validated_params = validate_params(params)
        print(f"Валидированные параметры: {validated_params}")
        
        # Проверяем ключевые параметры
        assert 'zscore_threshold' in validated_params
        assert 'zscore_exit' in validated_params
        assert 'risk_per_position_pct' in validated_params
        assert 'max_active_positions' in validated_params
        
        # Проверяем логические ограничения
        assert validated_params['zscore_threshold'] > validated_params['zscore_exit']
        assert 0 < validated_params['risk_per_position_pct'] <= 1.0
        assert validated_params['max_active_positions'] > 0
    
    def test_objective_with_direct_parameters(self):
        """Тестирует objective с прямой передачей параметров."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем objective
        objective = FastWalkForwardObjective(
            str(config_path),
            str(search_space_path)
        )
        
        # Тестовые параметры
        test_params = {
            'zscore_threshold': 1.8,
            'zscore_exit': 0.2,
            'rolling_window': 40,
            'stop_loss_multiplier': 3.0,
            'time_stop_multiplier': 2.0,
            'risk_per_position_pct': 0.025,
            'max_position_size_pct': 0.08,
            'max_active_positions': 12,
            'commission_pct': 0.0005,
            'slippage_pct': 0.0006,
            'normalization_method': 'minmax',
            'min_history_ratio': 0.6,
            'cooldown_hours': 3
        }
        
        # Мокаем бэктест
        with patch.object(objective, '_run_fast_backtest') as mock_backtest:
            mock_backtest.return_value = {
                'sharpe_ratio_abs': 1.2,
                'total_trades': 80,
                'max_drawdown': 0.15,
                'total_pnl': 1200.0,
                'total_return_pct': 0.12,
                'win_rate': 0.52,
                'avg_trade_size': 400.0,
                'avg_hold_time': 20.0
            }
            
            # Выполняем objective
            result = objective(test_params)
            
            print(f"Результат objective с прямыми параметрами: {result}")
            
            # Проверяем результат
            assert result != PENALTY, f"Objective вернул PENALTY: {result}"
            assert isinstance(result, (int, float)), f"Результат должен быть числом: {type(result)}"
            assert not np.isnan(result), "Результат не должен быть NaN"
            assert not np.isinf(result), "Результат не должен быть бесконечностью"
            
            print(f"✓ Objective работает корректно с прямыми параметрами: {result}")
    
    def test_error_handling_in_objective(self):
        """Тестирует обработку ошибок в objective."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем objective
        objective = FastWalkForwardObjective(
            str(config_path),
            str(search_space_path)
        )
        
        # Тест 1: Невалидные параметры
        invalid_params = {
            'zscore_threshold': -1.0,  # Невалидный
            'zscore_exit': 0.0,
            'risk_per_position_pct': 0.02
        }
        
        result = objective(invalid_params)
        assert result == PENALTY, f"Невалидные параметры должны возвращать PENALTY, получен: {result}"
        print("✓ Невалидные параметры корректно обрабатываются")
        
        # Тест 2: Ошибка в бэктесте
        valid_params = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.0,
            'risk_per_position_pct': 0.02,
            'max_active_positions': 10
        }
        
        with patch.object(objective, '_run_fast_backtest') as mock_backtest:
            mock_backtest.side_effect = RuntimeError("Ошибка в бэктесте")
            
            result = objective(valid_params)
            assert result == PENALTY, f"Ошибка в бэктесте должна возвращать PENALTY, получен: {result}"
            print("✓ Ошибки в бэктесте корректно обрабатываются")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
