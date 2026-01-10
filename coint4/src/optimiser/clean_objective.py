"""
Чистая целевая функция для Optuna без лишних штрафов.
Фокус на максимизации Sharpe ratio.
"""

import numpy as np
import optuna
from typing import Dict, Any, Optional
import logging

from .fast_objective import FastWalkForwardObjective
from .metric_utils import extract_sharpe
from .constants import PENALTY_SOFT

logger = logging.getLogger(__name__)

class CleanObjective(FastWalkForwardObjective):
    """
    Упрощенная целевая функция с фокусом на чистом Sharpe ratio.
    Убраны все лишние штрафы и модификации.
    """
    
    def __call__(self, trial_or_params):
        """
        Основная функция оптимизации - возвращает чистый Sharpe ratio.
        
        Args:
            trial_or_params: optuna.Trial или словарь параметров
            
        Returns:
            float: Sharpe ratio (или штраф при ошибке)
        """
        # Определяем тип входных данных
        if hasattr(trial_or_params, 'suggest_float'):  # Это optuna.Trial
            trial = trial_or_params
            params = self._suggest_parameters(trial)
            trial_number = trial.number
        else:  # Это словарь параметров
            params = trial_or_params
            trial_number = params.get("trial_number", -1)
            trial = None
        
        # Быстрая валидация параметров
        is_valid, reason = self.validate_params_simple(params)
        if not is_valid:
            logger.debug(f"Trial #{trial_number} отклонен: {reason}")
            if trial:
                trial.set_user_attr("rejection_reason", reason)
            return PENALTY_SOFT
        
        try:
            # Запускаем бэктест
            metrics = self._run_fast_backtest(params)
            
            # Извлекаем Sharpe ratio
            sharpe = extract_sharpe(metrics)
            
            # Валидация результата
            if sharpe is None or not isinstance(sharpe, (int, float)) or np.isnan(sharpe) or np.isinf(sharpe):
                logger.warning(f"Trial #{trial_number}: Невалидный Sharpe: {sharpe}")
                return PENALTY_SOFT
            
            # Минимальные требования
            total_trades = metrics.get('total_trades', 0)
            if total_trades < 10:  # Минимум 10 сделок для значимости
                logger.debug(f"Trial #{trial_number}: Недостаточно сделок ({total_trades})")
                return PENALTY_SOFT
            
            # Логируем результат
            logger.info(f"Trial #{trial_number}: Sharpe={sharpe:.3f}, Trades={total_trades}")
            
            # Сохраняем метрики в trial
            if trial:
                trial.set_user_attr("sharpe", float(sharpe))
                trial.set_user_attr("total_trades", int(total_trades))
                trial.set_user_attr("max_drawdown", float(metrics.get("max_drawdown", 0)))
                trial.set_user_attr("win_rate", float(metrics.get("win_rate", 0)))
                trial.set_user_attr("total_pnl", float(metrics.get("total_pnl", 0)))
            
            # Возвращаем чистый Sharpe ratio
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Trial #{trial_number}: Ошибка выполнения: {e}")
            if trial:
                trial.set_user_attr("error", str(e))
            return PENALTY_SOFT
    
    def validate_params_simple(self, params: Dict[str, Any]) -> tuple:
        """
        Упрощенная валидация параметров.
        
        Args:
            params: Словарь параметров
            
        Returns:
            tuple: (is_valid, reason)
        """
        # Проверка hysteresis
        zscore_threshold = params.get('zscore_threshold', 1.0)
        zscore_exit = params.get('zscore_exit', 0.0)
        
        if zscore_exit >= zscore_threshold:
            return False, f"zscore_exit ({zscore_exit:.2f}) >= zscore_threshold ({zscore_threshold:.2f})"
        
        hysteresis = zscore_threshold - zscore_exit
        if hysteresis < 0.2:
            return False, f"Гистерезис слишком мал: {hysteresis:.2f}"
        
        # Проверка stop loss
        stop_loss_mult = params.get('stop_loss_multiplier', 3.0)
        time_stop_mult = params.get('time_stop_multiplier', 5.0)
        
        if time_stop_mult <= stop_loss_mult:
            return False, f"time_stop ({time_stop_mult:.1f}) <= stop_loss ({stop_loss_mult:.1f})"
        
        return True, "OK"