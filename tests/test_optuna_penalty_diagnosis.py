#!/usr/bin/env python3
"""
Диагностика проблемы с PENALTY (-5.0) в Optuna оптимизации.
Выявляет причины почему все trials возвращают штраф вместо реальных значений.
"""

import pytest
import optuna
import numpy as np
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import traceback

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.fast_objective import FastWalkForwardObjective, PENALTY
from src.optimiser.metric_utils import validate_params, normalize_params


class TestPenaltyDiagnosis:
    """Диагностика проблемы с PENALTY в оптимизации."""
    
    def test_parameter_validation_with_real_search_space(self):
        """Проверяет валидацию параметров с реальным search_space."""
        
        # Загружаем реальный search space
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        print(f"Загружен search_space: {search_space}")
        
        # Создаем mock trial для генерации параметров
        mock_trial = Mock()
        mock_trial.number = 1
        
        # Настраиваем mock для suggest методов
        def mock_suggest_float(name, low, high):
            # Возвращаем среднее значение диапазона
            return (low + high) / 2
        
        def mock_suggest_int(name, low, high, step=1):
            # Возвращаем среднее значение диапазона
            return int((low + high) / 2)
        
        def mock_suggest_categorical(name, choices):
            # Возвращаем первый вариант
            return choices[0]
        
        mock_trial.suggest_float = mock_suggest_float
        mock_trial.suggest_int = mock_suggest_int
        mock_trial.suggest_categorical = mock_suggest_categorical
        
        # Создаем objective
        config_path = project_root / "configs" / "main_2024.yaml"
        
        try:
            objective = FastWalkForwardObjective(
                str(config_path),
                str(search_space_path)
            )
            
            # Генерируем параметры
            params = objective._suggest_parameters(mock_trial)
            print(f"Сгенерированные параметры: {params}")
            
            # Проверяем валидацию
            try:
                validated_params = validate_params(params)
                print(f"Валидация прошла успешно: {validated_params}")
                
                # Если валидация прошла, проблема не в параметрах
                assert True, "Параметры валидны - проблема не в валидации"
                
            except ValueError as e:
                print(f"ОШИБКА ВАЛИДАЦИИ: {e}")
                print(f"Проблемные параметры: {params}")
                
                # Анализируем какие именно параметры вызывают ошибку
                for key, value in params.items():
                    try:
                        single_param = {key: value}
                        validate_params(single_param)
                        print(f"  ✓ {key}: {value} - OK")
                    except ValueError as param_error:
                        print(f"  ✗ {key}: {value} - ОШИБКА: {param_error}")
                
                pytest.fail(f"Валидация параметров не прошла: {e}")
                
        except Exception as e:
            print(f"ОШИБКА СОЗДАНИЯ OBJECTIVE: {e}")
            traceback.print_exc()
            pytest.fail(f"Не удалось создать objective: {e}")
    
    def test_objective_execution_with_mocked_data(self):
        """Тестирует выполнение objective с мокнутыми данными."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем простые тестовые параметры
        test_params = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.0,
            'rolling_window': 30,
            'stop_loss_multiplier': 3.0,
            'time_stop_multiplier': 2.0,
            'risk_per_position_pct': 0.02,
            'max_position_size_pct': 0.05,
            'max_active_positions': 10,
            'commission_pct': 0.0004,
            'slippage_pct': 0.0005,
            'normalization_method': 'minmax',
            'min_history_ratio': 0.5,
            'cooldown_hours': 4
        }
        
        print(f"Тестовые параметры: {test_params}")
        
        try:
            # Проверяем валидацию
            validated_params = validate_params(test_params)
            print(f"Валидация прошла: {validated_params}")
            
            # Создаем objective
            objective = FastWalkForwardObjective(
                str(config_path),
                str(search_space_path)
            )
            
            # Мокаем загрузку данных
            with patch.object(objective, '_load_data') as mock_load:
                mock_load.return_value = None
                
                # Мокаем бэктест
                with patch.object(objective, '_run_backtest') as mock_backtest:
                    # Возвращаем реалистичные метрики
                    mock_backtest.return_value = {
                        'sharpe_ratio_abs': 1.5,
                        'total_trades': 100,
                        'max_drawdown': 0.15,
                        'total_pnl': 1500.0,
                        'total_return_pct': 0.15,
                        'win_rate': 0.55,
                        'avg_trade_size': 500.0,
                        'avg_hold_time': 24.0
                    }
                    
                    # Выполняем objective
                    result = objective(test_params)
                    
                    print(f"Результат objective: {result}")
                    
                    # Проверяем что результат не PENALTY
                    assert result != PENALTY, f"Objective вернул PENALTY ({PENALTY}) вместо реального значения"
                    assert isinstance(result, (int, float)), f"Результат должен быть числом, получен: {type(result)}"
                    assert not np.isnan(result), "Результат не должен быть NaN"
                    assert not np.isinf(result), "Результат не должен быть бесконечностью"
                    
                    print(f"✓ Objective работает корректно, результат: {result}")
                    
        except Exception as e:
            print(f"ОШИБКА В OBJECTIVE: {e}")
            traceback.print_exc()
            pytest.fail(f"Objective завершился с ошибкой: {e}")
    
    def test_real_optuna_trial_simulation(self):
        """Симулирует реальный Optuna trial для диагностики."""
        
        config_path = project_root / "configs" / "main_2024.yaml"
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        # Создаем реальный study
        study = optuna.create_study(direction='maximize')
        
        # Создаем objective
        try:
            objective = FastWalkForwardObjective(
                str(config_path),
                str(search_space_path)
            )
            
            # Мокаем данные для быстрого тестирования
            with patch.object(objective, '_load_data'):
                with patch.object(objective, '_run_backtest') as mock_backtest:
                    mock_backtest.return_value = {
                        'sharpe_ratio_abs': 1.2,
                        'total_trades': 80,
                        'max_drawdown': 0.12,
                        'total_pnl': 1200.0,
                        'total_return_pct': 0.12,
                        'win_rate': 0.52,
                        'avg_trade_size': 400.0,
                        'avg_hold_time': 20.0
                    }
                    
                    # Запускаем один trial
                    def test_objective(trial):
                        try:
                            result = objective(trial)
                            print(f"Trial {trial.number} результат: {result}")
                            
                            # Логируем атрибуты trial
                            if hasattr(trial, 'user_attrs'):
                                print(f"Trial {trial.number} атрибуты: {trial.user_attrs}")
                            
                            return result
                            
                        except optuna.TrialPruned as e:
                            print(f"Trial {trial.number} был pruned: {e}")
                            raise
                        except Exception as e:
                            print(f"Trial {trial.number} ошибка: {e}")
                            traceback.print_exc()
                            raise
                    
                    # Выполняем trial
                    study.optimize(test_objective, n_trials=1)
                    
                    # Проверяем результат
                    if len(study.trials) > 0:
                        trial = study.trials[0]
                        print(f"Trial состояние: {trial.state}")
                        print(f"Trial значение: {trial.value}")
                        print(f"Trial параметры: {trial.params}")
                        
                        if trial.state == optuna.trial.TrialState.COMPLETE:
                            assert trial.value != PENALTY, f"Trial вернул PENALTY: {trial.value}"
                            print(f"✓ Trial выполнен успешно: {trial.value}")
                        else:
                            pytest.fail(f"Trial не завершился успешно: {trial.state}")
                    else:
                        pytest.fail("Не было выполнено ни одного trial")
                        
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
            traceback.print_exc()
            pytest.fail(f"Критическая ошибка в симуляции trial: {e}")
    
    def test_search_space_parameter_ranges(self):
        """Проверяет что диапазоны параметров в search_space разумны."""
        
        search_space_path = project_root / "configs" / "search_space.yaml"
        
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        
        print(f"Анализ search_space: {search_space}")
        
        # Проверяем критические параметры
        issues = []
        
        # 1. zscore_threshold
        if 'signals' in search_space and 'zscore_threshold' in search_space['signals']:
            zscore_config = search_space['signals']['zscore_threshold']
            if zscore_config['low'] <= 0:
                issues.append(f"zscore_threshold low <= 0: {zscore_config['low']}")
            if zscore_config['high'] <= zscore_config['low']:
                issues.append(f"zscore_threshold high <= low: {zscore_config}")
        
        # 2. zscore_exit
        if 'signals' in search_space and 'zscore_exit' in search_space['signals']:
            zscore_exit_config = search_space['signals']['zscore_exit']
            zscore_threshold_config = search_space['signals']['zscore_threshold']
            if zscore_exit_config['high'] >= zscore_threshold_config['low']:
                issues.append(f"zscore_exit high >= zscore_threshold low: exit={zscore_exit_config}, threshold={zscore_threshold_config}")
        
        # 3. risk_per_position_pct
        if 'portfolio' in search_space and 'risk_per_position_pct' in search_space['portfolio']:
            risk_config = search_space['portfolio']['risk_per_position_pct']
            if risk_config['low'] <= 0:
                issues.append(f"risk_per_position_pct low <= 0: {risk_config['low']}")
            if risk_config['high'] > 1.0:
                issues.append(f"risk_per_position_pct high > 100%: {risk_config['high']}")
        
        # 4. max_active_positions
        if 'portfolio' in search_space and 'max_active_positions' in search_space['portfolio']:
            positions_config = search_space['portfolio']['max_active_positions']
            if positions_config['low'] <= 0:
                issues.append(f"max_active_positions low <= 0: {positions_config['low']}")
        
        if issues:
            print("НАЙДЕННЫЕ ПРОБЛЕМЫ В SEARCH_SPACE:")
            for issue in issues:
                print(f"  ✗ {issue}")
            pytest.fail(f"Найдены проблемы в search_space: {issues}")
        else:
            print("✓ Search space выглядит корректно")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
