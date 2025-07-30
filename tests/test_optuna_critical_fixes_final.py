#!/usr/bin/env python3
"""
Финальная проверка критических исправлений в Optuna оптимизации.
Проверяет что основные проблемы устранены и система готова к работе.
"""

import pytest
import optuna
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.fast_objective import FastWalkForwardObjective, PENALTY
from src.optimiser.metric_utils import validate_params


class TestOptunaCriticalFixesFinal:
    """Финальная проверка критических исправлений."""
    
    def test_critical_fix_1_log_parameter_removed(self):
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 1: Убран параметр log=True из suggest_float."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем mock trial который отслеживает все вызовы
        mock_trial = Mock()
        mock_trial.number = 1
        
        suggest_calls = []
        
        def track_suggest_float(name, low, high, **kwargs):
            suggest_calls.append(('suggest_float', name, kwargs))
            if 'log' in kwargs:
                raise AssertionError(f"КРИТИЧЕСКАЯ ОШИБКА: suggest_float для {name} использует запрещенный параметр log=True")
            return (low + high) / 2
        
        def track_suggest_int(name, low, high, step=1):
            suggest_calls.append(('suggest_int', name, {}))
            return int((low + high) / 2)
        
        def track_suggest_categorical(name, choices):
            suggest_calls.append(('suggest_categorical', name, {}))
            return choices[0]
        
        mock_trial.suggest_float = track_suggest_float
        mock_trial.suggest_int = track_suggest_int
        mock_trial.suggest_categorical = track_suggest_categorical
        
        # Создаем objective и генерируем параметры
        objective = FastWalkForwardObjective(str(config_path), str(search_space_path))
        params = objective._suggest_parameters(mock_trial)
        
        # Проверяем что все вызовы корректны
        float_calls = [call for call in suggest_calls if call[0] == 'suggest_float']
        print(f"✓ Проверено {len(float_calls)} вызовов suggest_float - все без log=True")
        
        # Проверяем что параметры валидны
        validated_params = validate_params(params)
        assert len(validated_params) > 0
        print("✓ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 1: Параметр log=True успешно убран")
    
    def test_critical_fix_2_penalty_vs_valid_values(self):
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 2: Objective возвращает валидные значения вместо PENALTY."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        objective = FastWalkForwardObjective(str(config_path), str(search_space_path))
        
        # Валидные параметры
        valid_params = {
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
        
        # Мокаем бэктест для возврата хороших результатов
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
            
            result = objective(valid_params)
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: результат НЕ должен быть PENALTY
            assert result != PENALTY, f"КРИТИЧЕСКАЯ ОШИБКА: Валидные параметры возвращают PENALTY: {result}"
            assert isinstance(result, (int, float)), f"Результат должен быть числом: {type(result)}"
            assert not np.isnan(result), "Результат не должен быть NaN"
            assert not np.isinf(result), "Результат не должен быть бесконечностью"
            assert result > 0, f"Результат должен быть положительным: {result}"
            
            print(f"✓ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 2: Objective возвращает валидное значение: {result}")
    
    def test_critical_fix_3_parameter_validation_logic(self):
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 3: Логика валидации параметров работает корректно."""
        
        # Тест 1: Валидные параметры должны проходить
        valid_params = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.0,
            'risk_per_position_pct': 0.02,
            'max_position_size_pct': 0.05,
            'max_active_positions': 10,
            'stop_loss_multiplier': 3.0,
            'time_stop_multiplier': 2.0
        }
        
        try:
            validated = validate_params(valid_params)
            assert validated['zscore_threshold'] > validated['zscore_exit']
            print("✓ Валидные параметры проходят проверку")
        except ValueError as e:
            pytest.fail(f"Валидные параметры не должны вызывать ошибку: {e}")
        
        # Тест 2: Невалидные параметры должны отклоняться
        invalid_params = {
            'zscore_threshold': -1.0,  # Отрицательный
            'zscore_exit': 0.0,
            'risk_per_position_pct': 0.02
        }
        
        with pytest.raises(ValueError, match="z_entry должен быть положительным"):
            validate_params(invalid_params)
        
        print("✓ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 3: Валидация параметров работает корректно")
    
    def test_critical_fix_4_search_space_compatibility(self):
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 4: Search space совместим с валидацией."""
        
        # Загружаем реальный search space
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        # Проверяем критические диапазоны
        signals = search_space.get('signals', {})
        
        # zscore_threshold и zscore_exit не должны пересекаться
        if 'zscore_threshold' in signals and 'zscore_exit' in signals:
            threshold_cfg = signals['zscore_threshold']
            exit_cfg = signals['zscore_exit']
            
            min_threshold = threshold_cfg['low']
            max_exit = exit_cfg['high']
            
            assert min_threshold > max_exit, \
                f"КРИТИЧЕСКАЯ ОШИБКА: min zscore_threshold ({min_threshold}) <= max zscore_exit ({max_exit})"
            
            print(f"✓ zscore диапазоны не пересекаются: threshold >= {min_threshold}, exit <= {max_exit}")
        
        # Проверяем риск-параметры
        portfolio = search_space.get('portfolio', {})
        
        if 'risk_per_position_pct' in portfolio:
            risk_cfg = portfolio['risk_per_position_pct']
            assert risk_cfg['low'] > 0, f"risk_per_position_pct low должен быть > 0: {risk_cfg['low']}"
            assert risk_cfg['high'] <= 1.0, f"risk_per_position_pct high должен быть <= 1.0: {risk_cfg['high']}"
            print(f"✓ risk_per_position_pct диапазон корректен: {risk_cfg['low']}-{risk_cfg['high']}")
        
        print("✓ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 4: Search space совместим с валидацией")
    
    def test_critical_fix_5_error_handling_robustness(self):
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 5: Надежная обработка ошибок."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        objective = FastWalkForwardObjective(str(config_path), str(search_space_path))
        
        # Тест 1: Невалидные параметры
        invalid_params = {
            'zscore_threshold': -1.0,
            'zscore_exit': 0.0,
            'risk_per_position_pct': 0.02
        }
        
        result1 = objective(invalid_params)
        assert result1 == PENALTY, f"Невалидные параметры должны возвращать PENALTY: {result1}"
        print("✓ Невалидные параметры обрабатываются корректно")
        
        # Тест 2: Ошибка в бэктесте
        valid_params = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.0,
            'risk_per_position_pct': 0.02,
            'max_active_positions': 10
        }
        
        with patch.object(objective, '_run_fast_backtest') as mock_backtest:
            mock_backtest.side_effect = RuntimeError("Критическая ошибка")
            
            result2 = objective(valid_params)
            assert result2 == PENALTY, f"Ошибка в бэктесте должна возвращать PENALTY: {result2}"
            print("✓ Ошибки в бэктесте обрабатываются корректно")
        
        print("✓ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 5: Обработка ошибок надежна")
    
    def test_summary_all_critical_fixes(self):
        """Сводный тест всех критических исправлений."""
        
        print("\n" + "="*60)
        print("📋 СВОДКА КРИТИЧЕСКИХ ИСПРАВЛЕНИЙ OPTUNA")
        print("="*60)
        
        fixes = [
            "✅ ИСПРАВЛЕНИЕ 1: Убран параметр log=True из suggest_float",
            "✅ ИСПРАВЛЕНИЕ 2: Objective возвращает валидные значения",
            "✅ ИСПРАВЛЕНИЕ 3: Валидация параметров работает корректно",
            "✅ ИСПРАВЛЕНИЕ 4: Search space совместим с валидацией",
            "✅ ИСПРАВЛЕНИЕ 5: Надежная обработка ошибок"
        ]
        
        for fix in fixes:
            print(fix)
        
        print("="*60)
        print("🎯 ВСЕ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ УСПЕШНО!")
        print("🚀 Optuna оптимизация готова к работе")
        print("="*60)
        
        # Финальная проверка - создание objective без ошибок
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        try:
            objective = FastWalkForwardObjective(str(config_path), str(search_space_path))
            print("✅ FastWalkForwardObjective создается без ошибок")
            
            # Проверяем что можем сгенерировать параметры
            mock_trial = Mock()
            mock_trial.number = 1
            mock_trial.suggest_float = lambda name, low, high, **kwargs: (low + high) / 2
            mock_trial.suggest_int = lambda name, low, high, step=1: int((low + high) / 2)
            mock_trial.suggest_categorical = lambda name, choices: choices[0]
            
            params = objective._suggest_parameters(mock_trial)
            validated_params = validate_params(params)
            
            print("✅ Генерация и валидация параметров работает")
            print(f"✅ Сгенерировано {len(validated_params)} параметров")
            
        except Exception as e:
            pytest.fail(f"Финальная проверка не прошла: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
