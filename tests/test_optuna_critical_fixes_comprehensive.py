#!/usr/bin/env python3
"""
Комплексный тест всех критических исправлений Optuna оптимизации.
Проверяет направление оптимизации, обработку ошибок, повторяемость, валидацию параметров.
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
from src.optimiser.metric_utils import extract_sharpe, validate_params, _validate_cost_parameters, _validate_cross_parameter_constraints
from src.optimiser.run_optimization import run_optimization


class TestDirectionAndMetrics:
    """Тесты направления оптимизации и метрик."""
    
    def test_direction_is_maximize(self):
        """Проверяет что направление оптимизации везде 'maximize'."""
        # Проверяем в run_optimization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as base_config:
            yaml.dump({
                'data_dir': 'test_data',
                'walk_forward': {'start_date': '2024-01-01', 'training_period_days': 10, 'testing_period_days': 5},
                'portfolio': {'initial_capital': 10000},
                'backtest': {'annualizing_factor': 365}
            }, base_config)
            base_config.flush()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as search_space:
                yaml.dump({
                    'signals': {'zscore_threshold': {'low': 1.0, 'high': 2.0}}
                }, search_space)
                search_space.flush()
                
                # Мокаем создание study
                with patch('optuna.create_study') as mock_create_study:
                    mock_study = Mock()
                    mock_create_study.return_value = mock_study
                    
                    # Мокаем objective
                    with patch('src.optimiser.run_optimization.FastWalkForwardObjective'):
                        try:
                            run_optimization(
                                n_trials=1,
                                base_config_path=base_config.name,
                                search_space_path=search_space.name
                            )
                        except:
                            pass  # Ожидаем ошибки из-за моков
                    
                    # Проверяем что direction='maximize'
                    mock_create_study.assert_called_once()
                    call_kwargs = mock_create_study.call_args[1]
                    assert call_kwargs['direction'] == 'maximize', "Направление оптимизации должно быть 'maximize'"
                
                Path(search_space.name).unlink()
            Path(base_config.name).unlink()
    
    def test_extract_sharpe_priority(self):
        """Проверяет приоритет sharpe_ratio_abs над sharpe_ratio."""
        # Случай 1: есть оба ключа
        result_both = {
            'sharpe_ratio_abs': 1.5,
            'sharpe_ratio': 1.2
        }
        assert extract_sharpe(result_both) == 1.5, "Должен возвращать sharpe_ratio_abs при наличии обоих"
        
        # Случай 2: только sharpe_ratio
        result_only_sharpe = {
            'sharpe_ratio': 1.2
        }
        assert extract_sharpe(result_only_sharpe) == 1.2, "Должен возвращать sharpe_ratio если нет sharpe_ratio_abs"
        
        # Случай 3: невалидные значения
        result_invalid = {
            'sharpe_ratio_abs': np.nan,
            'sharpe_ratio': 1.2
        }
        assert extract_sharpe(result_invalid) == 1.2, "Должен пропускать NaN значения"
        
        # Случай 4: нет ключей
        result_empty = {}
        assert extract_sharpe(result_empty) is None, "Должен возвращать None если нет ключей"


class TestErrorHandling:
    """Тесты правильной обработки ошибок через TrialPruned vs PENALTY."""

    def test_invalid_sharpe_logic(self):
        """Проверяет логику обработки невалидного Sharpe."""
        from src.optimiser.metric_utils import extract_sharpe

        # Тест extract_sharpe с невалидными значениями
        assert extract_sharpe({'sharpe_ratio_abs': np.nan}) is None
        assert extract_sharpe({'sharpe_ratio_abs': np.inf}) is None
        assert extract_sharpe({'sharpe_ratio': np.nan}) is None
        assert extract_sharpe({}) is None

        # Валидные значения должны проходить
        assert extract_sharpe({'sharpe_ratio_abs': 1.5}) == 1.5
        assert extract_sharpe({'sharpe_ratio': 1.2}) == 1.2

    def test_validation_error_logic(self):
        """Проверяет логику валидации параметров."""
        from src.optimiser.metric_utils import _validate_cross_parameter_constraints

        # Невалидные параметры должны вызывать ValueError
        with pytest.raises(ValueError, match="abs\\(zscore_exit\\).*должен быть.*abs\\(zscore_threshold\\)"):
            _validate_cross_parameter_constraints({
                'zscore_threshold': 1.5,
                'zscore_exit': 2.0  # Больше threshold
            })

        with pytest.raises(ValueError, match="min_half_life_days.*должен быть.*max_half_life_days"):
            _validate_cross_parameter_constraints({
                'min_half_life_days': 10.0,
                'max_half_life_days': 5.0  # Меньше min
            })

        # Валидные параметры должны проходить
        try:
            _validate_cross_parameter_constraints({
                'zscore_threshold': 2.0,
                'zscore_exit': 0.5,
                'min_half_life_days': 1.0,
                'max_half_life_days': 10.0,
                'ssd_top_n': 50000,
                'rolling_window': 30
            })
        except ValueError:
            pytest.fail("Валидные параметры не должны вызывать ошибку")

    def test_cost_parameters_validation_logic(self):
        """Проверяет логику валидации параметров издержек."""
        from src.optimiser.metric_utils import _validate_cost_parameters

        # Тест двойного учета комиссий
        params = {
            'commission_pct': 0.0004,
            'fee_maker': 0.0002,
            'fee_taker': 0.0004
        }
        _validate_cost_parameters(params)

        # Детальные параметры должны быть удалены
        assert 'fee_maker' not in params
        assert 'fee_taker' not in params
        assert 'commission_pct' in params

        # Тест невалидных значений
        with pytest.raises(ValueError):
            _validate_cost_parameters({'commission_pct': 0.02})  # 2% - слишком много

    def test_penalty_constant_value(self):
        """Проверяет что PENALTY имеет правильное значение."""
        # PENALTY должен быть умеренным для TPE
        assert -10.0 <= PENALTY <= -1.0, f"PENALTY вне рекомендуемого диапазона: {PENALTY}"
        assert PENALTY == -5.0, f"PENALTY должен быть -5.0, получен: {PENALTY}"


class TestParameterValidation:
    """Тесты валидации параметров."""
    
    def test_cross_parameter_constraints(self):
        """Проверяет валидацию пересекающихся параметров."""
        
        # Тест 1: min_half_life > max_half_life
        with pytest.raises(ValueError, match="min_half_life_days.*должен быть.*max_half_life_days"):
            _validate_cross_parameter_constraints({
                'min_half_life_days': 10.0,
                'max_half_life_days': 5.0
            })
        
        # Тест 2: abs(zscore_exit) >= abs(zscore_threshold)
        with pytest.raises(ValueError, match="abs\\(zscore_exit\\).*должен быть.*abs\\(zscore_threshold\\)"):
            _validate_cross_parameter_constraints({
                'zscore_threshold': 1.5,
                'zscore_exit': 1.6
            })
        
        # Тест 3: валидные параметры
        try:
            _validate_cross_parameter_constraints({
                'min_half_life_days': 1.0,
                'max_half_life_days': 10.0,
                'zscore_threshold': 2.0,
                'zscore_exit': 0.5,
                'ssd_top_n': 50000,
                'rolling_window': 30
            })
        except ValueError:
            pytest.fail("Валидные параметры не должны вызывать ошибку")
    
    def test_cost_parameters_validation(self):
        """Проверяет валидацию параметров издержек."""
        
        # Тест 1: двойной учет комиссий
        params_commission = {
            'commission_pct': 0.0004,
            'fee_maker': 0.0002,
            'fee_taker': 0.0004
        }
        _validate_cost_parameters(params_commission)
        
        # Детальные параметры должны быть удалены
        assert 'fee_maker' not in params_commission
        assert 'fee_taker' not in params_commission
        assert 'commission_pct' in params_commission
        
        # Тест 2: двойной учет проскальзывания
        params_slippage = {
            'slippage_pct': 0.0005,
            'slippage_bps': 2.0,
            'half_spread_bps': 1.5
        }
        _validate_cost_parameters(params_slippage)
        
        # Детальные параметры должны быть удалены
        assert 'slippage_bps' not in params_slippage
        assert 'half_spread_bps' not in params_slippage
        assert 'slippage_pct' in params_slippage
        
        # Тест 3: невалидные значения
        with pytest.raises(ValueError, match="commission_pct должен быть в диапазоне"):
            _validate_cost_parameters({'commission_pct': 0.02})  # 2% - слишком много


class TestReproducibility:
    """Тесты воспроизводимости."""
    
    def test_same_seed_same_results(self):
        """Проверяет что одинаковый seed дает одинаковые результаты."""
        
        def simple_objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2
        
        seed = 42
        
        # Первый запуск
        study1 = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
        )
        study1.optimize(simple_objective, n_trials=5)
        
        # Второй запуск с тем же seed
        study2 = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
        )
        study2.optimize(simple_objective, n_trials=5)
        
        # Результаты должны быть идентичными
        assert len(study1.trials) == len(study2.trials)
        for t1, t2 in zip(study1.trials, study2.trials):
            assert t1.params == t2.params, f"Параметры не совпадают: {t1.params} vs {t2.params}"
            assert abs(t1.value - t2.value) < 1e-10, f"Значения не совпадают: {t1.value} vs {t2.value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
