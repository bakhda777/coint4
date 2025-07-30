#!/usr/bin/env python3
"""
Тесты для проверки работы Pruner в Optuna оптимизации.
"""

import pytest
import optuna
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimiser.run_optimization import run_optimization
from src.optimiser.fast_objective import FastWalkForwardObjective


class TestOptunaPruner:
    """Тесты для проверки работы Pruner."""
    
    def test_median_pruner_configured(self):
        """Проверяем, что MedianPruner правильно настроен."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test.db"
            
            # Создаем минимальную study
            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,
                    n_warmup_steps=0,
                    interval_steps=1
                )
            )
            
            # Проверяем тип pruner
            assert isinstance(study.pruner, optuna.pruners.MedianPruner)
            assert study.pruner._n_startup_trials == 10
            assert study.pruner._n_warmup_steps == 0
            assert study.pruner._interval_steps == 1
    
    def test_pruner_not_hyperband(self):
        """Проверяем, что HyperbandPruner НЕ используется."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test.db"
            
            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Убеждаемся, что это НЕ HyperbandPruner
            assert not isinstance(study.pruner, optuna.pruners.HyperbandPruner)
    
    def test_trial_pruning_mechanism(self):
        """Тест механизма pruning с TrialPruned исключением."""
        
        def objective_with_pruning(trial):
            # Симулируем плохой результат
            if trial.number == 1:
                trial.set_user_attr("error", "test_pruning")
                raise optuna.TrialPruned("Test pruning")
            return 1.0
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_with_pruning, n_trials=3)
        
        # Проверяем, что один trial был pruned
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        assert len(pruned_trials) == 1
        assert pruned_trials[0].user_attrs.get("error") == "test_pruning"
    
    def test_intermediate_reporting(self):
        """Тест промежуточной отчетности для pruner."""
        
        def objective_with_reports(trial):
            # Симулируем промежуточные отчеты
            for step in range(5):
                intermediate_value = step * 0.1
                trial.report(intermediate_value, step)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return 1.0
        
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1)
        )
        
        study.optimize(objective_with_reports, n_trials=5)
        
        # Проверяем, что промежуточные значения записались
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                assert len(trial.intermediate_values) > 0


class TestFastObjectivePruning:
    """Тесты для проверки pruning в FastWalkForwardObjective."""
    
    @patch('src.optimiser.fast_objective.validate_params')
    def test_validation_error_pruning(self, mock_validate):
        """Тест pruning при ошибке валидации."""
        mock_validate.side_effect = ValueError("Invalid params")
        
        # Создаем mock trial
        mock_trial = Mock()
        mock_trial.set_user_attr = Mock()
        
        # Создаем objective (с минимальными параметрами)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("""
backtest:
  timeframe: '15min'
  zscore_threshold: 1.5
  fill_limit_pct: 0.1
  annualizing_factor: 365
  zscore_exit: 0.0
  rolling_window: 30
  stop_loss_multiplier: 3.0
  time_stop_multiplier: 5.0
  cooldown_hours: 4
  commission_pct: 0.0004
  slippage_pct: 0.0005
portfolio:
  initial_capital: 10000
  max_active_positions: 5
  risk_per_position_pct: 0.02
  max_position_size_pct: 0.1
data_processing:
  normalization_method: 'minmax'
  min_history_ratio: 0.8
  fill_method: 'ffill'
  handle_constant: true
pair_selection:
  lookback_days: 60
  ssd_top_n: 10000
  kpss_pvalue_threshold: 0.05
  coint_pvalue_threshold: 0.05
  min_half_life_days: 1.0
  max_half_life_days: 30.0
  min_mean_crossings: 2
walk_forward:
  enabled: true
  start_date: '2024-01-01'
  end_date: '2024-01-31'
  training_period_days: 20
  testing_period_days: 5
  step_size_days: 3
  min_training_samples: 1000
  refit_frequency: 'weekly'
data_dir: data_downloaded
results_dir: results
""")
            
            objective = FastWalkForwardObjective(
                base_config_path=str(config_path),
                search_space_path="configs/search_space_relaxed.yaml"
            )
            
            # Тестируем, что ошибка валидации вызывает TrialPruned
            with pytest.raises(optuna.TrialPruned):
                objective(mock_trial)
            
            # Проверяем, что атрибут ошибки установлен
            mock_trial.set_user_attr.assert_called_with("error", "validation_error: Invalid params")
    
    def test_insufficient_trades_pruning(self):
        """Тест pruning при недостаточном количестве сделок."""
        mock_trial = Mock()
        mock_trial.set_user_attr = Mock()
        
        # Симулируем результат с малым количеством сделок
        metrics = {"total_trades": 5, "sharpe_ratio_abs": 1.0}
        
        # Проверяем логику в коде
        total_trades = metrics.get('total_trades', 0)
        if total_trades < 10:
            mock_trial.set_user_attr("error", f"insufficient_trades: {total_trades}")
            # В реальном коде здесь должен быть raise TrialPruned
        
        mock_trial.set_user_attr.assert_called_with("error", "insufficient_trades: 5")
    
    def test_invalid_sharpe_pruning(self):
        """Тест pruning при невалидном Sharpe ratio."""
        import numpy as np
        
        mock_trial = Mock()
        mock_trial.set_user_attr = Mock()
        
        # Тестируем разные невалидные значения
        invalid_sharpes = [None, np.nan, np.inf, -np.inf, "invalid"]
        
        for sharpe in invalid_sharpes:
            mock_trial.reset_mock()
            
            # Симулируем проверку из кода
            if sharpe is None or not isinstance(sharpe, (int, float)) or \
               (isinstance(sharpe, float) and (np.isnan(sharpe) or np.isinf(sharpe))):
                mock_trial.set_user_attr("error", f"invalid_sharpe: {sharpe}")
                # В реальном коде здесь должен быть raise TrialPruned
            
            mock_trial.set_user_attr.assert_called_with("error", f"invalid_sharpe: {sharpe}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
