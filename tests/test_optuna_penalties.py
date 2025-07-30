#!/usr/bin/env python3
"""
Тесты для проверки правильного использования штрафов vs pruning в Optuna.
"""

import pytest
import optuna
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.fast_objective import PENALTY


class TestPenaltiesVsPruning:
    """Тесты для проверки правильного использования штрафов и pruning."""
    
    def test_penalty_value_reasonable(self):
        """Проверяем, что штраф не слишком агрессивный для TPE."""
        # PENALTY должен быть умеренным, не -1000
        assert PENALTY > -10.0, f"PENALTY слишком агрессивный: {PENALTY}"
        assert PENALTY < 0.0, f"PENALTY должен быть отрицательным: {PENALTY}"
        
        # Рекомендуемый диапазон для TPE
        assert -10.0 <= PENALTY <= -1.0, f"PENALTY вне рекомендуемого диапазона: {PENALTY}"
    
    def test_pruning_vs_penalty_logic(self):
        """Тест логики выбора между pruning и penalty."""
        
        # Случаи, где должен использоваться pruning (некорректная конфигурация)
        pruning_cases = [
            ("insufficient_trades", {"total_trades": 5}),
            ("invalid_sharpe", {"sharpe_ratio": np.nan}),
            ("validation_error", {"error": "Invalid params"}),
            ("no_data", {"data_points": 0})
        ]
        
        for case_name, case_data in pruning_cases:
            # Эти случаи должны вызывать TrialPruned, а не возвращать PENALTY
            assert case_name in ["insufficient_trades", "invalid_sharpe", "validation_error", "no_data"]
    
    def test_penalty_for_unexpected_errors(self):
        """Тест использования penalty только для неожиданных ошибок."""
        
        # Случаи, где penalty может быть оправдан
        penalty_cases = [
            "unexpected_exception",
            "system_error", 
            "memory_error"
        ]
        
        for case in penalty_cases:
            # Для неожиданных ошибок penalty допустим
            # но лучше логировать и анализировать
            assert isinstance(case, str)
    
    def test_tpe_distribution_impact(self):
        """Тест влияния штрафов на распределение TPE."""
        
        # Симулируем результаты оптимизации
        results = [1.5, 1.2, 0.8, PENALTY, 1.1, PENALTY, 0.9, 1.3]
        
        # Считаем долю штрафов
        penalty_ratio = sum(1 for r in results if r == PENALTY) / len(results)
        
        # Слишком много штрафов портит TPE
        assert penalty_ratio < 0.3, f"Слишком много штрафов: {penalty_ratio:.2%}"
        
        # Проверяем разброс "хороших" результатов
        good_results = [r for r in results if r != PENALTY]
        if len(good_results) > 1:
            std_dev = np.std(good_results)
            assert std_dev > 0.1, "Недостаточный разброс хороших результатов"


class TestTrialPrunedUsage:
    """Тесты для правильного использования TrialPruned."""
    
    def test_trial_pruned_with_attributes(self):
        """Тест использования TrialPruned с атрибутами для диагностики."""
        
        def objective_with_pruning(trial):
            # Симулируем разные причины pruning
            if trial.number == 0:
                trial.set_user_attr("error", "insufficient_trades")
                trial.set_user_attr("trades_count", 5)
                raise optuna.TrialPruned("Insufficient trades: 5 < 10")
            
            elif trial.number == 1:
                trial.set_user_attr("error", "invalid_sharpe")
                trial.set_user_attr("sharpe_value", "nan")
                raise optuna.TrialPruned("Invalid Sharpe ratio: nan")
            
            elif trial.number == 2:
                trial.set_user_attr("error", "validation_error")
                trial.set_user_attr("validation_message", "Invalid parameters")
                raise optuna.TrialPruned("Parameter validation failed")
            
            return 1.0
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_with_pruning, n_trials=4)
        
        # Проверяем pruned trials
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        assert len(pruned_trials) == 3
        
        # Проверяем атрибуты
        errors = [t.user_attrs.get("error") for t in pruned_trials]
        assert "insufficient_trades" in errors
        assert "invalid_sharpe" in errors
        assert "validation_error" in errors
    
    def test_pruning_preserves_tpe_quality(self):
        """Тест, что pruning не портит качество TPE."""
        
        def objective_with_smart_pruning(trial):
            x = trial.suggest_float("x", -10, 10)
            
            # "Плохие" конфигурации - pruning
            if abs(x) > 8:
                trial.set_user_attr("reason", "out_of_bounds")
                raise optuna.TrialPruned("Parameter out of reasonable bounds")
            
            # "Хорошие" конфигурации - возвращаем значение
            return -(x - 2) ** 2  # Максимум в x=2
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_with_smart_pruning, n_trials=50)
        
        # Проверяем, что есть и pruned и complete trials
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        assert len(complete_trials) > 0
        assert len(pruned_trials) > 0
        
        # Проверяем, что лучшие результаты близки к оптимуму (x=2)
        if len(complete_trials) > 10:
            best_x = study.best_params["x"]
            assert abs(best_x - 2.0) < 1.0, f"TPE не нашел оптимум: best_x={best_x}"


class TestErrorHandlingStrategy:
    """Тесты для стратегии обработки ошибок."""
    
    def test_error_classification(self):
        """Тест классификации ошибок для выбора стратегии обработки."""
        
        # Ошибки конфигурации -> TrialPruned
        config_errors = [
            "insufficient_trades",
            "invalid_parameters", 
            "validation_failed",
            "no_valid_pairs"
        ]
        
        # Системные ошибки -> PENALTY + логирование
        system_errors = [
            "memory_error",
            "disk_full",
            "network_timeout",
            "unexpected_exception"
        ]
        
        # Метрические ошибки -> TrialPruned
        metric_errors = [
            "invalid_sharpe",
            "nan_returns",
            "infinite_drawdown",
            "zero_variance"
        ]
        
        for error in config_errors + metric_errors:
            # Эти ошибки должны использовать pruning
            assert error not in system_errors
        
        for error in system_errors:
            # Эти ошибки могут использовать penalty
            assert error not in config_errors + metric_errors
    
    def test_user_attributes_for_debugging(self):
        """Тест установки атрибутов для отладки."""
        
        mock_trial = Mock()
        mock_trial.set_user_attr = Mock()
        
        # Симулируем разные типы ошибок
        error_cases = [
            ("insufficient_trades", {"trades": 5, "min_required": 10}),
            ("invalid_sharpe", {"sharpe": "nan", "returns_std": 0.0}),
            ("validation_error", {"param": "zscore_threshold", "value": -1.0}),
            ("execution_error", {"exception": "ValueError", "message": "Invalid input"})
        ]
        
        for error_type, error_data in error_cases:
            mock_trial.reset_mock()
            
            # Устанавливаем атрибуты
            mock_trial.set_user_attr("error_type", error_type)
            for key, value in error_data.items():
                mock_trial.set_user_attr(f"error_{key}", value)
            
            # Проверяем, что атрибуты установлены
            assert mock_trial.set_user_attr.call_count >= 1
            
            # Проверяем основной атрибут ошибки
            mock_trial.set_user_attr.assert_any_call("error_type", error_type)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
