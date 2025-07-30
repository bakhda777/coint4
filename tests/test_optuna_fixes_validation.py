#!/usr/bin/env python3
"""
Валидация исправлений в Optuna оптимизации.
Проверяет что основные проблемы устранены.
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


class TestOptunaFixesValidation:
    """Валидация исправлений в Optuna оптимизации."""
    
    def test_log_parameter_fix(self):
        """Проверяет что параметр log=True больше не используется."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем mock trial который будет отслеживать вызовы suggest_float
        mock_trial = Mock()
        mock_trial.number = 1
        
        suggest_float_calls = []
        
        def mock_suggest_float(name, low, high, **kwargs):
            suggest_float_calls.append((name, low, high, kwargs))
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
        
        # Проверяем что все вызовы suggest_float были без log=True
        print(f"Вызовы suggest_float: {suggest_float_calls}")
        
        for name, low, high, kwargs in suggest_float_calls:
            assert 'log' not in kwargs, f"suggest_float для {name} использует запрещенный параметр log"
        
        print(f"✓ Все {len(suggest_float_calls)} вызовов suggest_float корректны")
        
        # Проверяем что параметры валидны
        validated_params = validate_params(params)
        assert len(validated_params) > 0, "Должны быть сгенерированы валидные параметры"
        
        print("✓ Исправление log=True работает корректно")
    
    def test_objective_returns_valid_values(self):
        """Проверяет что objective возвращает валидные значения вместо PENALTY."""
        
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
        
        # Мокаем бэктест для возврата реалистичных результатов
        with patch.object(objective, '_run_fast_backtest') as mock_backtest:
            # Тест 1: Хорошие результаты
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
            
            result1 = objective(test_params)
            print(f"Результат с хорошими метриками: {result1}")
            
            assert result1 != PENALTY, f"Хорошие метрики не должны возвращать PENALTY: {result1}"
            assert isinstance(result1, (int, float)), f"Результат должен быть числом: {type(result1)}"
            assert not np.isnan(result1), "Результат не должен быть NaN"
            assert not np.isinf(result1), "Результат не должен быть бесконечностью"
            assert result1 > 0, f"Результат должен быть положительным: {result1}"
            
            # Тест 2: Плохие результаты (но не критичные)
            mock_backtest.return_value = {
                'sharpe_ratio_abs': 0.5,
                'total_trades': 50,
                'max_drawdown': 0.20,
                'total_pnl': 200.0,
                'total_return_pct': 0.02,
                'win_rate': 0.45,
                'avg_trade_size': 100.0,
                'avg_hold_time': 12.0
            }
            
            result2 = objective(test_params)
            print(f"Результат с плохими метриками: {result2}")
            
            assert result2 != PENALTY, f"Плохие метрики не должны возвращать PENALTY: {result2}"
            assert isinstance(result2, (int, float)), f"Результат должен быть числом: {type(result2)}"
            assert not np.isnan(result2), "Результат не должен быть NaN"
            assert not np.isinf(result2), "Результат не должен быть бесконечностью"
            
            # Тест 3: Очень плохие результаты (должны получить штраф)
            mock_backtest.return_value = {
                'sharpe_ratio_abs': -2.0,
                'total_trades': 10,
                'max_drawdown': 0.80,
                'total_pnl': -5000.0,
                'total_return_pct': -0.50,
                'win_rate': 0.20,
                'avg_trade_size': 50.0,
                'avg_hold_time': 6.0
            }
            
            result3 = objective(test_params)
            print(f"Результат с очень плохими метриками: {result3}")
            
            # Очень плохие результаты могут получить штраф, но не обязательно PENALTY
            assert isinstance(result3, (int, float)), f"Результат должен быть числом: {type(result3)}"
            assert not np.isnan(result3), "Результат не должен быть NaN"
            assert not np.isinf(result3), "Результат не должен быть бесконечностью"
        
        print("✓ Objective возвращает валидные значения")
    
    def test_error_handling_improvements(self):
        """Проверяет улучшенную обработку ошибок."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем objective
        objective = FastWalkForwardObjective(
            str(config_path),
            str(search_space_path)
        )
        
        # Тест 1: Невалидные параметры должны возвращать PENALTY
        invalid_params = {
            'zscore_threshold': -1.0,  # Невалидный
            'zscore_exit': 0.0,
            'risk_per_position_pct': 0.02
        }
        
        result = objective(invalid_params)
        assert result == PENALTY, f"Невалидные параметры должны возвращать PENALTY, получен: {result}"
        print("✓ Невалидные параметры корректно обрабатываются")
        
        # Тест 2: Ошибка в бэктесте должна возвращать PENALTY
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
        
        print("✓ Обработка ошибок работает корректно")
    
    def test_optuna_trial_with_mock(self):
        """Тестирует работу с реальным Optuna trial и мокнутым бэктестом."""
        
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
                'sharpe_ratio_abs': 1.2,
                'total_trades': 80,
                'max_drawdown': 0.15,
                'total_pnl': 1200.0,
                'total_return_pct': 0.12,
                'win_rate': 0.52,
                'avg_trade_size': 400.0,
                'avg_hold_time': 20.0
            }
            
            # Функция для тестирования одного trial
            def test_objective_func(trial):
                result = objective(trial)
                print(f"Trial {trial.number} результат: {result}")
                
                # Проверяем что результат не PENALTY
                assert result != PENALTY, f"Trial вернул PENALTY: {result}"
                assert isinstance(result, (int, float)), f"Результат должен быть числом: {type(result)}"
                assert not np.isnan(result), "Результат не должен быть NaN"
                assert not np.isinf(result), "Результат не должен быть бесконечностью"
                
                return result
            
            # Запускаем один trial
            study.optimize(test_objective_func, n_trials=1)
            
            # Проверяем результат
            assert len(study.trials) == 1, f"Должен быть 1 trial, получено: {len(study.trials)}"
            
            trial = study.trials[0]
            assert trial.state == optuna.trial.TrialState.COMPLETE, f"Trial должен быть завершен: {trial.state}"
            assert trial.value != PENALTY, f"Trial вернул PENALTY: {trial.value}"
            
            print(f"✓ Trial выполнен успешно: {trial.value}")
        
        print("✓ Работа с Optuna trial исправлена")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
