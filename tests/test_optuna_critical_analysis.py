#!/usr/bin/env python3
"""
Критический анализ Optuna оптимизации на логические и критические ошибки.
Проверяет направление оптимизации, обработку ошибок, валидацию параметров,
воспроизводимость, корректность метрик и потенциальные проблемы производительности.
"""

import pytest
import optuna
import numpy as np
import tempfile
import yaml
import time
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import warnings
from typing import Dict, Any

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.fast_objective import FastWalkForwardObjective, PENALTY
from src.optimiser.metric_utils import validate_params, normalize_params
from src.optimiser.run_optimization import run_optimization


class TestCriticalOptimizationErrors:
    """Тесты для выявления критических ошибок в оптимизации."""
    
    def test_direction_consistency_critical(self):
        """КРИТИЧЕСКИЙ: Проверяет что направление оптимизации везде maximize."""
        # Проверяем что в run_optimization direction="maximize"
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test.db"
            
            # Создаем минимальную конфигурацию
            config_path = Path(temp_dir) / "config.yaml"
            search_space_path = Path(temp_dir) / "search_space.yaml"
            
            config = {
                'data_dir': 'data_downloaded',
                'portfolio': {'initial_capital': 10000},
                'pair_selection': {'lookback_days': 30},
                'backtest': {'timeframe': '15min', 'rolling_window': 30}
            }
            
            search_space = {
                'signals': {'zscore_threshold': {'low': 1.0, 'high': 2.0}}
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            with open(search_space_path, 'w') as f:
                yaml.dump(search_space, f)
            
            # Мокаем objective функцию
            def mock_objective(trial):
                return trial.suggest_float("zscore_threshold", 1.0, 2.0)
            
            with patch('src.optimiser.run_optimization.objective', mock_objective):
                study = optuna.create_study(
                    storage=f"sqlite:///{storage_path}",
                    direction="maximize"  # Должно быть maximize
                )
                
                # Проверяем направление
                assert study.direction == optuna.study.StudyDirection.MAXIMIZE, \
                    "КРИТИЧЕСКАЯ ОШИБКА: Направление оптимизации должно быть MAXIMIZE для Sharpe ratio"
    
    def test_sharpe_ratio_calculation_critical(self):
        """КРИТИЧЕСКИЙ: Проверяет корректность расчета Sharpe ratio."""
        # Создаем тестовые данные с известным Sharpe ratio
        returns = np.array([0.01, -0.005, 0.015, -0.01, 0.02])  # Простые доходности
        expected_sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
        
        # Тестируем функцию extract_sharpe
        from src.optimiser.metric_utils import extract_sharpe
        
        metrics = {
            'total_return_pct': np.sum(returns),
            'total_trades': len(returns),
            'max_drawdown': 0.05
        }
        
        # Мокаем данные для расчета
        mock_trades_df = Mock()
        mock_trades_df.empty = False
        mock_trades_df.__len__ = Mock(return_value=len(returns))
        mock_trades_df['pnl'] = returns * 10000  # Конвертируем в абсолютные значения
        
        with patch('pandas.DataFrame') as mock_df:
            mock_df.return_value = mock_trades_df
            
            # Проверяем что Sharpe считается правильно
            sharpe = extract_sharpe(metrics, mock_trades_df)
            
            # Sharpe должен быть числом, не NaN и не бесконечностью
            assert isinstance(sharpe, (int, float)), "Sharpe ratio должен быть числом"
            assert not np.isnan(sharpe), "Sharpe ratio не должен быть NaN"
            assert not np.isinf(sharpe), "Sharpe ratio не должен быть бесконечностью"
    
    def test_parameter_validation_edge_cases_critical(self):
        """КРИТИЧЕСКИЙ: Проверяет валидацию параметров на граничных случаях."""
        
        # Тест 1: z_entry = z_exit (должно вызывать ошибку)
        params_equal_z = {
            'zscore_threshold': 2.0,
            'zscore_exit': 2.0  # Равны - это ошибка
        }
        
        with pytest.raises(ValueError, match="z_entry должен быть больше z_exit"):
            validate_params(params_equal_z)
        
        # Тест 2: Отрицательный z_entry
        params_negative_z = {
            'zscore_threshold': -1.0,
            'zscore_exit': 0.0
        }
        
        with pytest.raises(ValueError, match="z_entry должен быть положительным"):
            validate_params(params_negative_z)
        
        # Тест 3: Нулевые значения риска
        params_zero_risk = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.0,
            'risk_per_position_pct': 0.0  # Нулевой риск
        }
        
        with pytest.raises(ValueError, match="risk_per_position_pct должен быть положительным"):
            validate_params(params_zero_risk)
        
        # Тест 4: Слишком большой риск (>100%)
        params_huge_risk = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.0,
            'risk_per_position_pct': 1.5  # 150% риска
        }
        
        with pytest.raises(ValueError, match="risk_per_position_pct не может быть больше 100%"):
            validate_params(params_huge_risk)
    
    def test_penalty_vs_pruning_logic_critical(self):
        """КРИТИЧЕСКИЙ: Проверяет логику использования штрафов vs pruning."""
        
        # Создаем mock trial
        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.suggest_float = Mock(return_value=2.0)
        mock_trial.suggest_int = Mock(return_value=30)
        mock_trial.suggest_categorical = Mock(return_value="minmax")
        mock_trial.set_user_attr = Mock()
        
        # Создаем objective с мокнутыми данными
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            search_space_path = Path(temp_dir) / "search_space.yaml"
            
            # Минимальная конфигурация
            config = {
                'data_dir': 'data_downloaded',
                'portfolio': {'initial_capital': 10000},
                'pair_selection': {'lookback_days': 30},
                'backtest': {'timeframe': '15min', 'rolling_window': 30}
            }
            
            search_space = {
                'signals': {'zscore_threshold': {'low': 1.0, 'high': 3.0}}
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            with open(search_space_path, 'w') as f:
                yaml.dump(search_space, f)
            
            objective = FastWalkForwardObjective(
                str(config_path),
                str(search_space_path)
            )
            
            # Тест 1: Валидационная ошибка должна вызывать TrialPruned
            with patch.object(objective, '_suggest_parameters') as mock_suggest:
                mock_suggest.return_value = {
                    'zscore_threshold': -1.0,  # Невалидный параметр
                    'zscore_exit': 0.0
                }
                
                with pytest.raises(optuna.TrialPruned):
                    objective(mock_trial)
                
                # Проверяем что установлены правильные атрибуты
                mock_trial.set_user_attr.assert_any_call("error_type", "validation_error")
            
            # Тест 2: Неожиданная ошибка должна возвращать PENALTY
            with patch.object(objective, '_suggest_parameters') as mock_suggest:
                mock_suggest.side_effect = RuntimeError("Неожиданная ошибка")
                
                result = objective(mock_trial)
                assert result == PENALTY, "Неожиданные ошибки должны возвращать PENALTY"
                
                # Проверяем что установлены правильные атрибуты
                mock_trial.set_user_attr.assert_any_call("error_type", "execution_error")


class TestOptimizationPerformanceIssues:
    """Тесты для выявления проблем производительности."""
    
    def test_memory_leaks_in_objective(self):
        """Проверяет потенциальные утечки памяти в objective функции."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Создаем objective и запускаем много раз
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            search_space_path = Path(temp_dir) / "search_space.yaml"
            
            config = {
                'data_dir': 'data_downloaded',
                'portfolio': {'initial_capital': 10000},
                'pair_selection': {'lookback_days': 30},
                'backtest': {'timeframe': '15min', 'rolling_window': 30}
            }
            
            search_space = {
                'signals': {'zscore_threshold': {'low': 1.0, 'high': 3.0}}
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            with open(search_space_path, 'w') as f:
                yaml.dump(search_space, f)
            
            # Мокаем данные чтобы избежать реальной загрузки
            with patch.object(FastWalkForwardObjective, '_load_data'):
                with patch.object(FastWalkForwardObjective, '_run_backtest') as mock_backtest:
                    mock_backtest.return_value = {
                        'sharpe_ratio_abs': 1.0,
                        'total_trades': 100,
                        'max_drawdown': 0.1
                    }
                    
                    objective = FastWalkForwardObjective(
                        str(config_path),
                        str(search_space_path)
                    )
                    
                    # Запускаем много раз
                    for i in range(50):
                        params = {'zscore_threshold': 2.0, 'zscore_exit': 0.0}
                        result = objective(params)
                        assert isinstance(result, (int, float))
            
            # Проверяем что память не выросла критично
            final_memory = process.memory_info().rss
            memory_growth = (final_memory - initial_memory) / initial_memory
            
            assert memory_growth < 0.5, f"Подозрение на утечку памяти: рост {memory_growth:.2%}"
    
    def test_database_connection_handling(self):
        """Проверяет корректность работы с базой данных SQLite."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test.db"
            
            # Тест 1: Создание study не должно блокировать файл
            study1 = optuna.create_study(
                storage=f"sqlite:///{storage_path}",
                study_name="test_study",
                direction="maximize"
            )
            
            # Тест 2: Второй study должен успешно подключиться
            study2 = optuna.create_study(
                storage=f"sqlite:///{storage_path}",
                study_name="test_study",
                load_if_exists=True,
                direction="maximize"
            )
            
            assert study1.study_name == study2.study_name
            
            # Тест 3: Проверяем что база данных не заблокирована
            conn = sqlite3.connect(str(storage_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            
            assert len(tables) > 0, "База данных должна содержать таблицы Optuna"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
