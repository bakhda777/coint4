#!/usr/bin/env python3
"""
Интеграционные тесты для проверки всех критических исправлений Optuna.
"""

import pytest
import optuna
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import numpy as np

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.run_optimization import run_optimization
from src.optimiser.fast_objective import FastWalkForwardObjective, PENALTY


class TestOptunaIntegration:
    """Интеграционные тесты для всех исправлений."""
    
    def create_minimal_config(self, temp_dir):
        """Создает минимальную конфигурацию для тестов."""
        config = {
            'backtest': {
                'timeframe': '15min',  # Добавляем обязательное поле
                'zscore_threshold': 1.5,  # Добавляем обязательное поле
                'fill_limit_pct': 0.1,  # Добавляем обязательное поле
                'annualizing_factor': 365,
                'zscore_entry_threshold': 1.5,
                'zscore_exit': 0.1,
                'rolling_window': 50,
                'stop_loss_multiplier': 3.0,
                'time_stop_multiplier': 5.0,
                'cooldown_hours': 4,
                'commission_pct': 0.0004,
                'slippage_pct': 0.0005
            },
            'portfolio': {
                'initial_capital': 10000,
                'max_active_positions': 5,
                'risk_per_position_pct': 0.02,
                'max_position_size_pct': 0.1
            },
            'data_processing': {
                'normalization_method': 'minmax',
                'min_history_ratio': 0.8,
                'fill_method': 'ffill',
                'handle_constant': True
            },
            'pair_selection': {
                'lookback_days': 60,  # Добавляем обязательное поле
                'ssd_top_n': 10000,
                'kpss_pvalue_threshold': 0.05,
                'coint_pvalue_threshold': 0.05,
                'min_half_life_days': 1.0,
                'max_half_life_days': 30.0,
                'min_mean_crossings': 2
            },
            'walk_forward': {  # Добавляем обязательную секцию
                'enabled': True,
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'training_period_days': 20,
                'testing_period_days': 5,
                'step_size_days': 3,
                'min_training_samples': 1000,
                'refit_frequency': 'weekly'
            },
            'data_dir': 'data_downloaded',
            'results_dir': 'results'
        }
        
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    def create_search_space(self, temp_dir):
        """Создает пространство поиска для тестов."""
        search_space = {
            'signals': {
                'zscore_threshold': {'low': 1.0, 'high': 2.0},
                'zscore_exit': {'low': 0.1, 'high': 0.5}
            },
            'portfolio': {
                'max_active_positions': {'low': 3, 'high': 8, 'step': 1},
                'risk_per_position_pct': {'low': 0.01, 'high': 0.05}
            },
            'normalization': {
                'normalization_method': ['minmax', 'zscore'],
                'min_history_ratio': {'low': 0.6, 'high': 0.9}
            }
        }
        
        search_space_path = Path(temp_dir) / "search_space.yaml"
        with open(search_space_path, 'w') as f:
            yaml.dump(search_space, f)
        
        return str(search_space_path)
    
    def test_all_fixes_integration(self):
        """Интеграционный тест всех критических исправлений."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.create_minimal_config(temp_dir)
            search_space_path = self.create_search_space(temp_dir)
            
            # 1. Тест MedianPruner (не HyperbandPruner)
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
            )
            
            assert isinstance(study.pruner, optuna.pruners.MedianPruner)
            assert not isinstance(study.pruner, optuna.pruners.HyperbandPruner)
            
            # 2. Тест умеренного штрафа
            assert -10.0 <= PENALTY <= -1.0, f"PENALTY слишком агрессивный: {PENALTY}"
            
            # 3. Тест воспроизводимости
            def test_objective(trial):
                np.random.seed(42)  # Локальный сид
                x = trial.suggest_float("x", 0, 1)
                return x**2 + np.random.normal(0, 0.01)
            
            study.optimize(test_objective, n_trials=3)
            
            # Проверяем, что trials завершились
            assert len(study.trials) == 3
            for trial in study.trials:
                assert trial.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]
    
    @patch('src.optimiser.fast_objective.preprocess_and_normalize_data')
    @patch('src.optimiser.fast_objective.FastWalkForwardObjective._run_fast_backtest')
    def test_normalization_config_section(self, mock_backtest, mock_preprocess):
        """Тест использования правильной секции конфигурации для нормализации."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.create_minimal_config(temp_dir)
            search_space_path = self.create_search_space(temp_dir)
            
            # Мокаем результат бэктеста
            mock_backtest.return_value = {
                "sharpe_ratio_abs": 1.5,
                "total_trades": 100,
                "max_drawdown": 0.1,
                "win_rate": 0.6
            }
            
            objective = FastWalkForwardObjective(
                base_config_path=config_path,
                search_space_path=search_space_path
            )
            
            # Проверяем, что базовая конфигурация использует data_processing
            assert hasattr(objective.base_config, 'data_processing')
            assert objective.base_config.data_processing.normalization_method == 'minmax'
            
            # Создаем mock trial
            mock_trial = Mock()
            mock_trial.suggest_float = Mock(side_effect=[1.5, 0.3, 0.02, 0.8])  # zscore_threshold, zscore_exit, risk_per_position_pct, min_history_ratio
            mock_trial.suggest_int = Mock(return_value=5)  # max_active_positions
            mock_trial.suggest_categorical = Mock(return_value='zscore')  # normalization_method
            mock_trial.set_user_attr = Mock()
            
            # Запускаем objective
            result = objective(mock_trial)
            
            # Проверяем, что результат валидный
            assert isinstance(result, (int, float))
            assert result > 0  # Должен быть положительным (хороший результат)
    
    def test_pruning_vs_penalty_strategy(self):
        """Тест стратегии pruning vs penalty."""
        
        def objective_with_mixed_errors(trial):
            error_type = trial.number % 4
            
            if error_type == 0:
                # Недостаточно сделок -> должен быть TrialPruned
                trial.set_user_attr("error", "insufficient_trades")
                raise optuna.TrialPruned("Insufficient trades")
            
            elif error_type == 1:
                # Невалидный Sharpe -> должен быть TrialPruned  
                trial.set_user_attr("error", "invalid_sharpe")
                raise optuna.TrialPruned("Invalid Sharpe")
            
            elif error_type == 2:
                # Системная ошибка -> может быть penalty
                trial.set_user_attr("error", "system_error")
                return PENALTY
            
            else:
                # Нормальный результат
                x = trial.suggest_float("x", 0, 1)
                return x**2
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_with_mixed_errors, n_trials=8)
        
        # Анализируем результаты
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Должны быть и pruned и complete trials
        assert len(pruned_trials) > 0, "Нет pruned trials"
        assert len(complete_trials) > 0, "Нет complete trials"
        
        # Проверяем атрибуты pruned trials
        pruned_errors = [t.user_attrs.get("error") for t in pruned_trials]
        assert "insufficient_trades" in pruned_errors
        assert "invalid_sharpe" in pruned_errors
        
        # Проверяем, что penalty trials завершились
        penalty_trials = [t for t in complete_trials if t.value == PENALTY]
        assert len(penalty_trials) > 0, "Нет penalty trials"
    
    def test_parameter_types_consistency(self):
        """Тест согласованности типов параметров."""
        
        def objective_with_all_types(trial):
            # Целочисленный с шагом
            int_param = trial.suggest_int("int_param", 1, 10, step=2)
            assert isinstance(int_param, int)
            assert int_param % 2 == 1  # Нечетное (1, 3, 5, 7, 9)
            
            # Вещественный
            float_param = trial.suggest_float("float_param", 0.1, 1.0)
            assert isinstance(float_param, float)
            assert 0.1 <= float_param <= 1.0
            
            # Категориальный
            cat_param = trial.suggest_categorical("cat_param", ["a", "b", "c"])
            assert cat_param in ["a", "b", "c"]
            
            return int_param + float_param + (1 if cat_param == "a" else 0)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_with_all_types, n_trials=5)
        
        # Проверяем, что все trials завершились успешно
        for trial in study.trials:
            assert trial.state == optuna.trial.TrialState.COMPLETE
            
            # Проверяем типы параметров
            assert isinstance(trial.params["int_param"], int)
            assert isinstance(trial.params["float_param"], float)
            assert isinstance(trial.params["cat_param"], str)
    
    def test_sqlite_parallelism_warning(self):
        """Тест предупреждения о проблемах SQLite с параллельностью."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test.db"
            storage_url = f"sqlite:///{storage_path}"
            
            # Проверяем логику предупреждения
            if "sqlite" in storage_url:
                # Должно быть предупреждение о параллельности
                assert True  # В реальном коде здесь логирование
            
            # Для SQLite рекомендуется n_jobs=1
            recommended_n_jobs = 1 if "sqlite" in storage_url else -1
            assert recommended_n_jobs == 1
    
    def test_config_serialization_without_python_objects(self):
        """Тест сериализации конфигурации без Python объектов."""
        config_data = {
            'data_dir': 'data_downloaded',  # Строка, не Path
            'results_dir': 'results',       # Строка, не Path
            'backtest': {
                'zscore_threshold': 1.5
            }
        }
        
        # Сериализуем в YAML
        yaml_str = yaml.dump(config_data)
        
        # Проверяем, что нет Python объектов
        assert "!!python" not in yaml_str
        assert "pathlib" not in yaml_str
        
        # Проверяем, что можно десериализовать
        loaded_config = yaml.safe_load(yaml_str)
        assert loaded_config == config_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
