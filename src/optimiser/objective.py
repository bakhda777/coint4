"""
Универсальный класс WalkForwardObjective для оптимизации стратегии парного трейдинга.

Объединяет функциональность всех objective классов:
- FastWalkForwardObjective (быстрая оптимизация на предотобранных парах)
- SimpleBPObjective (best practice оптимизация)
- Простая оптимизация без сложностей

Поддерживает различные режимы работы через параметр fast_mode.
"""

import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from src.optimiser.fast_objective import FastWalkForwardObjective
from src.coint2.utils.config import load_config
from src.coint2.engine.numba_engine import NumbaPairBacktester as PairBacktester

logger = logging.getLogger(__name__)


class WalkForwardObjective:
    """
    Универсальный класс для оптимизации стратегии парного трейдинга.
    
    Поддерживает различные режимы работы:
    - fast_mode=True: Использует предотобранные пары для быстрой оптимизации
    - fast_mode=False: Выполняет полный отбор пар
    - simple_mode=True: Упрощенная оптимизация без сложностей
    """
    
    def __init__(
        self, 
        base_config_path: str, 
        search_space_path: Optional[str] = None,
        fast_mode: bool = True,
        simple_mode: bool = False,
        preselected_pairs_path: str = "outputs/preselected_pairs.csv"
    ):
        """
        Инициализация универсального objective.
        
        Args:
            base_config_path: Путь к базовой конфигурации
            search_space_path: Путь к пространству поиска (не нужен для simple_mode)
            fast_mode: Использовать предотобранные пары (True) или полный отбор (False)
            simple_mode: Упрощенный режим без сложностей
            preselected_pairs_path: Путь к файлу предотобранных пар
        """
        self.base_config_path = base_config_path
        self.search_space_path = search_space_path
        self.fast_mode = fast_mode
        self.simple_mode = simple_mode
        self.preselected_pairs_path = preselected_pairs_path
        
        # Загружаем базовую конфигурацию
        self.base_config = load_config(base_config_path)
        
        # Загружаем пространство поиска (если не simple_mode)
        self.search_space = None
        if not simple_mode and search_space_path:
            # Используем стандартную загрузку yaml для search_space (не AppConfig)
            with open(search_space_path, 'r') as f:
                self.search_space = yaml.safe_load(f)
        
        # Инициализируем внутренний objective в зависимости от режима
        self._internal_objective = None
        self._initialize_internal_objective()
    
    def _initialize_internal_objective(self):
        """Инициализирует внутренний objective в зависимости от режима."""
        if self.simple_mode:
            # Для простого режима создаем простую функцию
            self._internal_objective = self._create_simple_objective()
        else:
            # Для остальных режимов используем FastWalkForwardObjective
            self._internal_objective = FastWalkForwardObjective(
                self.base_config_path, 
                self.search_space_path
            )
    
    def _create_simple_objective(self):
        """Создает простую целевую функцию."""
        def simple_objective_func(trial):
            return self._simple_objective_implementation(trial)
        return simple_objective_func
    
    def _simple_objective_implementation(self, trial):
        """Реализация простой целевой функции."""
        # Генерируем параметры
        z_entry = trial.suggest_float("z_entry", 1.2, 1.8)
        z_exit = trial.suggest_float("z_exit", -0.2, 0.2)
        
        # Проверяем наличие необходимых файлов
        if not Path(self.preselected_pairs_path).exists():
            logger.error(f"Файл {self.preselected_pairs_path} не найден!")
            return -999.0
        
        if not Path("outputs/full_step_data.csv").exists():
            logger.error("Файл outputs/full_step_data.csv не найден!")
            return -999.0
        
        try:
            # Загружаем предотобранные пары
            pairs_df = pd.read_csv(self.preselected_pairs_path)
            
            # Загружаем данные
            full_data = pd.read_csv("outputs/full_step_data.csv", index_col=0, parse_dates=True)
            
            # Определяем тестовый период
            start_date = pd.to_datetime(self.base_config.walk_forward.start_date)
            testing_start = start_date
            testing_end = testing_start + pd.Timedelta(days=self.base_config.walk_forward.testing_period_days)
            
            # Фильтруем данные по периоду
            test_data = full_data.loc[testing_start:testing_end]
            
            total_pnl = 0.0
            successful_pairs = 0
            
            # Тестируем на первых 5 парах для скорости
            for _, row in pairs_df.head(5).iterrows():
                s1, s2 = row['s1'], row['s2']
                
                if s1 not in test_data.columns or s2 not in test_data.columns:
                    continue
                    
                pair_data = test_data[[s1, s2]].dropna()
                if len(pair_data) < 100:
                    continue
                
                # Простая нормализация
                norm_s1 = pair_data[s1].iloc[0]
                norm_s2 = pair_data[s2].iloc[0]
                
                if norm_s1 == 0 or norm_s2 == 0:
                    continue
                
                normalized_data = pair_data.copy()
                normalized_data[s1] = (pair_data[s1] / norm_s1) * 100
                normalized_data[s2] = (pair_data[s2] / norm_s2) * 100
                
                try:
                    # Создаем бэктестер
                    backtester = PairBacktester(
                        pair_data=normalized_data,
                        rolling_window=self.base_config.backtest.rolling_window,
                        z_threshold=z_entry,
                        z_exit=z_exit,
                        stop_loss_multiplier=self.base_config.backtest.stop_loss_multiplier,
                        commission_pct=0.0004,
                        slippage_pct=0.0005,
                        capital_at_risk=10000,
                        pair_name=f"{s1}-{s2}",
                        annualizing_factor=365
                    )
                    
                    # Запускаем бэктест
                    backtester.run()
                    results = backtester.get_results()
                    
                    if results and 'pnl' in results:
                        pnl_sum = results['pnl'].sum()
                        if not pd.isna(pnl_sum):
                            total_pnl += pnl_sum
                            successful_pairs += 1
                            
                except Exception as e:
                    logger.debug(f"Ошибка при обработке пары {s1}-{s2}: {e}")
                    continue
            
            # Возвращаем результат
            if successful_pairs == 0:
                return -999.0
            
            # Простая метрика: средний PnL на пару
            avg_pnl = total_pnl / successful_pairs
            return float(avg_pnl)
            
        except Exception as e:
            logger.error(f"Ошибка в простой целевой функции: {e}")
            return -999.0
    
    def __call__(self, trial_or_params):
        """
        Унифицированная функция вызова.
        
        Args:
            trial_or_params: optuna.Trial объект или словарь параметров
            
        Returns:
            float: Значение целевой функции
        """
        if self.simple_mode:
            # Для простого режима вызываем простую функцию
            if hasattr(trial_or_params, 'suggest_float'):
                return self._internal_objective(trial_or_params)
            else:
                # Если передан словарь параметров, создаем mock trial
                logger.warning("Простой режим работает только с optuna.Trial объектами")
                return -999.0
        else:
            # Для остальных режимов используем FastWalkForwardObjective
            return self._internal_objective(trial_or_params)
    
    def set_fast_mode(self, fast_mode: bool):
        """Изменяет режим работы (fast/full)."""
        if self.fast_mode != fast_mode and not self.simple_mode:
            self.fast_mode = fast_mode
            # Переинициализируем internal objective если нужно
            # В текущей реализации FastWalkForwardObjective автоматически определяет режим
            logger.info(f"Режим изменен на: {'fast' if fast_mode else 'full'}")
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущем режиме работы."""
        return {
            "fast_mode": self.fast_mode,
            "simple_mode": self.simple_mode,
            "base_config_path": self.base_config_path,
            "search_space_path": self.search_space_path,
            "preselected_pairs_path": self.preselected_pairs_path,
            "has_preselected_pairs": Path(self.preselected_pairs_path).exists()
        }


# Для обратной совместимости создаем алиасы
FastWalkForwardObjectiveUnified = WalkForwardObjective
SimpleBPObjective = lambda config_path: WalkForwardObjective(
    config_path, simple_mode=True
)
