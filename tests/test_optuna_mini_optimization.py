#!/usr/bin/env python3
"""
Мини-тест оптимизации для проверки что PENALTY больше не возвращается.
"""

import pytest
import optuna
import numpy as np
from pathlib import Path
from unittest.mock import patch
import sys

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.fast_objective import FastWalkForwardObjective, PENALTY


class TestOptunaMiniOptimization:
    """Мини-тест оптимизации."""
    
    def test_mini_optimization_no_penalty(self):
        """Проверяет что мини-оптимизация не возвращает PENALTY."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем study
        study = optuna.create_study(direction='maximize')
        
        # Создаем objective
        objective = FastWalkForwardObjective(str(config_path), str(search_space_path))
        
        # Мокаем бэктест для возврата разных результатов
        mock_results = [
            {
                'sharpe_ratio_abs': 1.5,
                'total_trades': 100,
                'max_drawdown': 0.12,
                'total_pnl': 1500.0,
                'total_return_pct': 0.15,
                'win_rate': 0.55,
                'avg_trade_size': 500.0,
                'avg_hold_time': 24.0
            },
            {
                'sharpe_ratio_abs': 0.8,
                'total_trades': 60,
                'max_drawdown': 0.18,
                'total_pnl': 800.0,
                'total_return_pct': 0.08,
                'win_rate': 0.48,
                'avg_trade_size': 300.0,
                'avg_hold_time': 18.0
            },
            {
                'sharpe_ratio_abs': 1.2,
                'total_trades': 80,
                'max_drawdown': 0.15,
                'total_pnl': 1200.0,
                'total_return_pct': 0.12,
                'win_rate': 0.52,
                'avg_trade_size': 400.0,
                'avg_hold_time': 20.0
            }
        ]
        
        trial_count = 0
        
        def mock_backtest_func(*args, **kwargs):
            nonlocal trial_count
            result = mock_results[trial_count % len(mock_results)]
            trial_count += 1
            return result
        
        with patch.object(objective, '_run_fast_backtest', side_effect=mock_backtest_func):
            
            # Функция для тестирования
            def test_objective_func(trial):
                result = objective(trial)
                print(f"Trial {trial.number} результат: {result}")
                
                # КРИТИЧЕСКАЯ ПРОВЕРКА: результат НЕ должен быть PENALTY
                if result == PENALTY:
                    print(f"❌ Trial {trial.number} вернул PENALTY!")
                    print(f"   Параметры: {trial.params}")
                    if hasattr(trial, 'user_attrs'):
                        print(f"   Атрибуты: {trial.user_attrs}")
                    pytest.fail(f"Trial {trial.number} вернул PENALTY вместо валидного значения")
                
                # Проверяем что результат валиден
                assert isinstance(result, (int, float)), f"Результат должен быть числом: {type(result)}"
                assert not np.isnan(result), f"Результат не должен быть NaN: {result}"
                assert not np.isinf(result), f"Результат не должен быть бесконечностью: {result}"
                
                print(f"✅ Trial {trial.number}: {result}")
                return result
            
            # Запускаем мини-оптимизацию
            print("🚀 Запуск мини-оптимизации (3 trials)...")
            study.optimize(test_objective_func, n_trials=3)
            
            # Проверяем результаты
            print(f"\n📊 Результаты мини-оптимизации:")
            print(f"   Всего trials: {len(study.trials)}")
            
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            
            print(f"   Завершенных: {len(completed_trials)}")
            print(f"   Pruned: {len(pruned_trials)}")
            print(f"   Неудачных: {len(failed_trials)}")
            
            # Проверяем что есть завершенные trials
            assert len(completed_trials) > 0, f"Должен быть хотя бы один завершенный trial, получено: {len(completed_trials)}"
            
            # Проверяем значения завершенных trials
            for trial in completed_trials:
                assert trial.value != PENALTY, f"Trial {trial.number} имеет значение PENALTY: {trial.value}"
                assert isinstance(trial.value, (int, float)), f"Trial {trial.number} значение не число: {type(trial.value)}"
                print(f"   ✅ Trial {trial.number}: {trial.value}")
            
            # Находим лучший результат
            if completed_trials:
                best_trial = study.best_trial
                print(f"\n🏆 Лучший результат: {best_trial.value}")
                print(f"   Параметры: {best_trial.params}")
                
                assert best_trial.value != PENALTY, f"Лучший результат не должен быть PENALTY: {best_trial.value}"
                assert best_trial.value > 0, f"Лучший результат должен быть положительным: {best_trial.value}"
            
            print("\n✅ МИНИ-ОПТИМИЗАЦИЯ ПРОШЛА УСПЕШНО!")
            print("✅ Все trials возвращают валидные значения вместо PENALTY")
            print("✅ Optuna оптимизация работает корректно")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
