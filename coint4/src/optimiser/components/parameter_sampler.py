"""
Сэмплер параметров для оптимизации.
Отвечает за генерацию и валидацию параметров.
"""

import optuna
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from ..metric_utils import validate_params

logger = logging.getLogger(__name__)


class ParameterSampler:
    """
    Сэмплер параметров для Optuna оптимизации.
    Инкапсулирует логику генерации и валидации параметров.
    """
    
    def __init__(self, search_space: Dict[str, Any]):
        """
        Args:
            search_space: Словарь с определением пространства поиска
        """
        self.search_space = search_space
        self._parameter_importance = {}
        self._adaptive_bounds = {}
        
    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Генерирует параметры для Optuna trial.
        
        Args:
            trial: Optuna trial объект
            
        Returns:
            Словарь сгенерированных параметров
        """
        params = {}
        
        # Группа 1: Фильтры отбора пар
        if 'filters' in self.search_space:
            params.update(self._suggest_filter_params(trial))
        
        # Группа 2: Торговые параметры
        if 'trading' in self.search_space:
            params.update(self._suggest_trading_params(trial))
        
        # Группа 3: Риск-менеджмент
        if 'risk' in self.search_space or 'risk_management' in self.search_space:
            params.update(self._suggest_risk_params(trial))
        
        # Группа 4: Портфельные параметры
        if 'portfolio' in self.search_space:
            params.update(self._suggest_portfolio_params(trial))
        
        # Добавляем метаданные
        params['trial_number'] = trial.number
        
        # Валидируем параметры
        try:
            validated_params = validate_params(params)
            return validated_params
        except ValueError as e:
            logger.warning(f"Невалидные параметры в trial {trial.number}: {e}")
            # Возвращаем дефолтные параметры при ошибке валидации
            return self._get_default_params()
    
    def _suggest_filter_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Генерирует параметры фильтров."""
        params = {}
        filters = self.search_space.get('filters', {})
        
        if 'ssd_top_n' in filters:
            cfg = filters['ssd_top_n']
            if cfg.get('step'):
                params['ssd_top_n'] = trial.suggest_int(
                    "ssd_top_n",
                    cfg['low'],
                    cfg['high'],
                    step=cfg['step']
                )
            else:
                params['ssd_top_n'] = trial.suggest_int(
                    "ssd_top_n", 
                    cfg['low'],
                    cfg['high'],
                    log=cfg.get('log', False)
                )
        
        if 'min_half_life_days' in filters:
            cfg = filters['min_half_life_days']
            params['min_half_life_days'] = trial.suggest_float(
                "min_half_life_days",
                cfg['low'],
                cfg['high']
            )
        
        if 'max_half_life_days' in filters:
            cfg = filters['max_half_life_days']
            params['max_half_life_days'] = trial.suggest_float(
                "max_half_life_days",
                cfg['low'],
                cfg['high']
            )
        
        return params
    
    def _suggest_trading_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Генерирует торговые параметры."""
        params = {}
        trading = self.search_space.get('trading', {})
        
        if 'zscore_threshold' in trading:
            cfg = trading['zscore_threshold']
            params['zscore_threshold'] = trial.suggest_float(
                "zscore_threshold",
                cfg['low'],
                cfg['high'],
                step=cfg.get('step')
            )
        
        if 'zscore_exit' in trading:
            cfg = trading['zscore_exit']
            params['zscore_exit'] = trial.suggest_float(
                "zscore_exit",
                cfg['low'],
                cfg['high'],
                step=cfg.get('step')
            )
        
        if 'rolling_window' in trading:
            cfg = trading['rolling_window']
            if isinstance(cfg, dict):
                if 'choices' in cfg:
                    params['rolling_window'] = trial.suggest_categorical(
                        "rolling_window",
                        cfg['choices']
                    )
                else:
                    params['rolling_window'] = trial.suggest_int(
                        "rolling_window",
                        cfg['low'],
                        cfg['high']
                    )
        
        return params
    
    def _suggest_risk_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Генерирует параметры риск-менеджмента."""
        params = {}
        risk = self.search_space.get('risk', self.search_space.get('risk_management', {}))
        
        if 'stop_loss_multiplier' in risk:
            cfg = risk['stop_loss_multiplier']
            params['stop_loss_multiplier'] = trial.suggest_float(
                "stop_loss_multiplier",
                cfg['low'],
                cfg['high'],
                step=cfg.get('step')
            )
        
        if 'time_stop_multiplier' in risk:
            cfg = risk['time_stop_multiplier']
            params['time_stop_multiplier'] = trial.suggest_float(
                "time_stop_multiplier",
                cfg['low'],
                cfg['high'],
                step=cfg.get('step')
            )
        
        if 'max_drawdown_pct' in risk:
            cfg = risk['max_drawdown_pct']
            params['max_drawdown_pct'] = trial.suggest_float(
                "max_drawdown_pct",
                cfg['low'],
                cfg['high']
            )
        
        return params
    
    def _suggest_portfolio_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Генерирует портфельные параметры."""
        params = {}
        portfolio = self.search_space.get('portfolio', {})
        
        if 'max_active_positions' in portfolio:
            cfg = portfolio['max_active_positions']
            params['max_active_positions'] = trial.suggest_int(
                "max_active_positions",
                cfg['low'],
                cfg['high']
            )
        
        if 'risk_per_position_pct' in portfolio:
            cfg = portfolio['risk_per_position_pct']
            params['risk_per_position_pct'] = trial.suggest_float(
                "risk_per_position_pct",
                cfg['low'],
                cfg['high'],
                step=cfg.get('step')
            )
        
        if 'max_position_size_pct' in portfolio:
            cfg = portfolio['max_position_size_pct']
            params['max_position_size_pct'] = trial.suggest_float(
                "max_position_size_pct",
                cfg['low'],
                cfg['high'],
                step=cfg.get('step')
            )
        
        return params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Возвращает дефолтные параметры."""
        return {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.5,
            'stop_loss_multiplier': 3.0,
            'time_stop_multiplier': 5.0,
            'max_active_positions': 10,
            'risk_per_position_pct': 0.02,
            'max_position_size_pct': 0.1,
            'rolling_window': 60,
            'min_half_life_days': 1.0,
            'max_half_life_days': 7.0,
            'ssd_top_n': 5000
        }
    
    def update_adaptive_bounds(
        self, 
        best_trials: List[optuna.trial.FrozenTrial],
        shrink_factor: float = 0.8
    ) -> None:
        """
        Обновляет границы параметров на основе лучших результатов.
        
        Args:
            best_trials: Список лучших trials
            shrink_factor: Коэффициент сужения границ (0.8 = 80% от исходного диапазона)
        """
        if not best_trials:
            return
        
        # Анализируем распределение лучших параметров
        for param_name in self.search_space.get('trading', {}):
            values = [t.params.get(param_name) for t in best_trials 
                     if param_name in t.params]
            
            if values:
                # Вычисляем новые границы
                new_min = np.percentile(values, 10)
                new_max = np.percentile(values, 90)
                
                # Сохраняем адаптивные границы
                self._adaptive_bounds[param_name] = {
                    'low': new_min,
                    'high': new_max
                }
                
                logger.info(f"📊 Обновлены границы {param_name}: [{new_min:.3f}, {new_max:.3f}]")
    
    def set_parameter_importance(self, importance: Dict[str, float]) -> None:
        """
        Устанавливает важность параметров для фокусированной оптимизации.
        
        Args:
            importance: Словарь с важностью каждого параметра
        """
        self._parameter_importance = importance
        
        # Выводим топ-5 важных параметров
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("🎯 Важность параметров:")
        for param, imp in sorted_params[:5]:
            logger.info(f"   {param}: {imp:.3f}")
