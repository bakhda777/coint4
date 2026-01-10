"""
Валидатор для проверки корректности расчета Sharpe ratio.
Обеспечивает консистентность расчетов между разными модулями.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass
from .annualization import (
    get_annualization_factor,
    calculate_sharpe_ratio as calc_sharpe_unified,
    validate_sharpe_ratio as validate_sharpe_unified
)

logger = logging.getLogger(__name__)


@dataclass
class SharpeValidationResult:
    """Результат валидации расчета Sharpe."""
    is_valid: bool
    sharpe_ratio: float
    issue: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SharpeValidator:
    """
    Валидатор для проверки корректности расчета Sharpe ratio.
    
    Проверяет:
    - Правильность annualization factor
    - Обработку нулевых PnL
    - Границы разумных значений
    - Консистентность между методами расчета
    """
    
    # Константы берутся из единого модуля annualization
    # Для 15-минутных данных: sqrt(365 * 96) = ~187
    # Для дневных данных: sqrt(365) = ~19.1
    
    # Границы разумных значений
    MIN_REASONABLE_SHARPE = -10.0
    MAX_REASONABLE_SHARPE = 10.0
    MIN_TRADES_FOR_SIGNIFICANCE = 10
    
    def __init__(self, timeframe_minutes: int = 15):
        """
        Args:
            timeframe_minutes: Размер бара в минутах (по умолчанию 15)
        """
        self.timeframe_minutes = timeframe_minutes
        self.bars_per_day = 24 * 60 // timeframe_minutes
        self.annualization_factor = np.sqrt(365 * self.bars_per_day)
        
    def calculate_sharpe_from_pnl(
        self, 
        pnl_series: Union[np.ndarray, pd.Series],
        method: str = "standard"
    ) -> float:
        """
        Рассчитывает Sharpe ratio из серии PnL.
        
        Args:
            pnl_series: Серия PnL (прибыли/убытки)
            method: Метод расчета ("standard", "robust", "block_bootstrap")
            
        Returns:
            Sharpe ratio
        """
        # Фильтруем нулевые значения
        if isinstance(pnl_series, pd.Series):
            non_zero_pnl = pnl_series[pnl_series != 0].values
        else:
            non_zero_pnl = pnl_series[pnl_series != 0]
        
        if len(non_zero_pnl) < 2:
            logger.warning(f"Недостаточно ненулевых PnL: {len(non_zero_pnl)}")
            return 0.0
        
        if method == "standard":
            return self._calculate_standard_sharpe(non_zero_pnl)
        elif method == "robust":
            return self._calculate_robust_sharpe(non_zero_pnl)
        elif method == "block_bootstrap":
            return self._calculate_bootstrap_sharpe(non_zero_pnl)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def _calculate_standard_sharpe(self, pnl: np.ndarray) -> float:
        """Стандартный расчет Sharpe ratio."""
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl, ddof=1)  # Используем ddof=1 для sample std
        
        if std_pnl < 1e-8:
            logger.warning("Стандартное отклонение близко к нулю")
            return 0.0
        
        # Sharpe = mean/std * sqrt(periods_per_year)
        sharpe = (mean_pnl / std_pnl) * self.annualization_factor
        
        return sharpe
    
    def _calculate_robust_sharpe(self, pnl: np.ndarray) -> float:
        """
        Робастный расчет Sharpe с использованием медианы и MAD.
        Менее чувствителен к выбросам.
        """
        median_pnl = np.median(pnl)
        mad = np.median(np.abs(pnl - median_pnl))
        
        if mad < 1e-8:
            logger.warning("MAD близко к нулю")
            return 0.0
        
        # Коэффициент 1.4826 для приведения MAD к std для нормального распределения
        std_equivalent = mad * 1.4826
        
        sharpe = (median_pnl / std_equivalent) * self.annualization_factor
        
        return sharpe
    
    def _calculate_bootstrap_sharpe(self, pnl: np.ndarray, n_bootstrap: int = 100) -> float:
        """
        Расчет Sharpe с block bootstrap для учета автокорреляции.
        """
        block_size = max(5, int(np.sqrt(len(pnl))))
        sharpe_samples = []
        
        for _ in range(n_bootstrap):
            # Block bootstrap
            n_blocks = len(pnl) // block_size + 1
            blocks = []
            
            for _ in range(n_blocks):
                start = np.random.randint(0, max(1, len(pnl) - block_size))
                blocks.append(pnl[start:start + block_size])
            
            bootstrap_pnl = np.concatenate(blocks)[:len(pnl)]
            sharpe = self._calculate_standard_sharpe(bootstrap_pnl)
            sharpe_samples.append(sharpe)
        
        # Возвращаем медиану bootstrap выборки
        return np.median(sharpe_samples)
    
    def validate_sharpe(
        self,
        sharpe: float,
        pnl_series: Optional[Union[np.ndarray, pd.Series]] = None,
        num_trades: Optional[int] = None
    ) -> SharpeValidationResult:
        """
        Валидирует значение Sharpe ratio.
        
        Args:
            sharpe: Значение Sharpe для проверки
            pnl_series: Опциональная серия PnL для перепроверки
            num_trades: Количество сделок
            
        Returns:
            Результат валидации
        """
        # Проверка на NaN/Inf
        if np.isnan(sharpe) or np.isinf(sharpe):
            return SharpeValidationResult(
                is_valid=False,
                sharpe_ratio=0.0,
                issue="Sharpe is NaN or Inf"
            )
        
        # Проверка границ
        if sharpe < self.MIN_REASONABLE_SHARPE or sharpe > self.MAX_REASONABLE_SHARPE:
            return SharpeValidationResult(
                is_valid=False,
                sharpe_ratio=sharpe,
                issue=f"Sharpe {sharpe:.2f} вне разумных границ [{self.MIN_REASONABLE_SHARPE}, {self.MAX_REASONABLE_SHARPE}]"
            )
        
        # Проверка минимального количества сделок
        if num_trades is not None and num_trades < self.MIN_TRADES_FOR_SIGNIFICANCE:
            return SharpeValidationResult(
                is_valid=False,
                sharpe_ratio=sharpe,
                issue=f"Недостаточно сделок ({num_trades}) для статистической значимости"
            )
        
        # Перепроверка расчета если есть PnL
        if pnl_series is not None:
            recalculated = self.calculate_sharpe_from_pnl(pnl_series)
            diff = abs(sharpe - recalculated)
            
            if diff > 0.1:  # Допускаем небольшую разницу из-за округления
                return SharpeValidationResult(
                    is_valid=False,
                    sharpe_ratio=sharpe,
                    issue=f"Несоответствие расчета: {sharpe:.3f} vs {recalculated:.3f}",
                    details={"original": sharpe, "recalculated": recalculated, "diff": diff}
                )
        
        return SharpeValidationResult(
            is_valid=True,
            sharpe_ratio=sharpe
        )
    
    def compare_sharpe_methods(
        self,
        pnl_series: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Сравнивает разные методы расчета Sharpe.
        
        Args:
            pnl_series: Серия PnL
            
        Returns:
            Словарь с результатами разных методов
        """
        methods = ["standard", "robust", "block_bootstrap"]
        results = {}
        
        for method in methods:
            try:
                sharpe = self.calculate_sharpe_from_pnl(pnl_series, method=method)
                results[method] = sharpe
            except Exception as e:
                logger.error(f"Ошибка в методе {method}: {e}")
                results[method] = None
        
        # Добавляем статистику
        valid_results = [v for v in results.values() if v is not None]
        if valid_results:
            results["mean"] = np.mean(valid_results)
            results["std"] = np.std(valid_results)
            results["consensus"] = np.median(valid_results)
        
        return results
    
    def get_confidence_interval(
        self,
        pnl_series: Union[np.ndarray, pd.Series],
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float, float]:
        """
        Рассчитывает доверительный интервал для Sharpe ratio.
        
        Args:
            pnl_series: Серия PnL
            confidence: Уровень доверия (по умолчанию 95%)
            n_bootstrap: Количество bootstrap итераций
            
        Returns:
            Tuple (нижняя граница, точечная оценка, верхняя граница)
        """
        if isinstance(pnl_series, pd.Series):
            pnl = pnl_series[pnl_series != 0].values
        else:
            pnl = pnl_series[pnl_series != 0]
        
        if len(pnl) < 10:
            logger.warning("Слишком мало данных для доверительного интервала")
            point_estimate = self.calculate_sharpe_from_pnl(pnl)
            return (point_estimate, point_estimate, point_estimate)
        
        # Bootstrap
        sharpe_samples = []
        for _ in range(n_bootstrap):
            # Resample с возвратом
            resampled = np.random.choice(pnl, size=len(pnl), replace=True)
            sharpe = self._calculate_standard_sharpe(resampled)
            sharpe_samples.append(sharpe)
        
        # Вычисляем percentiles
        alpha = 1 - confidence
        lower = np.percentile(sharpe_samples, alpha/2 * 100)
        upper = np.percentile(sharpe_samples, (1 - alpha/2) * 100)
        point = np.median(sharpe_samples)
        
        return (lower, point, upper)
    
    def adjust_for_costs(
        self,
        gross_sharpe: float,
        num_trades: int,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        total_capital: float = 100000
    ) -> float:
        """
        Корректирует Sharpe ratio с учетом торговых издержек.
        
        Args:
            gross_sharpe: Sharpe до учета издержек
            num_trades: Количество сделок
            commission_pct: Комиссия в процентах
            slippage_pct: Проскальзывание в процентах
            total_capital: Общий капитал
            
        Returns:
            Скорректированный Sharpe ratio
        """
        if num_trades == 0:
            return gross_sharpe
        
        # Оценка влияния издержек
        total_cost_pct = (commission_pct + slippage_pct) * 2  # Вход и выход
        cost_per_trade = total_capital * total_cost_pct
        total_costs = cost_per_trade * num_trades
        
        # Грубая оценка коррекции
        # Предполагаем, что costs уменьшают mean return
        cost_impact = total_costs / total_capital
        
        # Корректируем Sharpe пропорционально
        # Это упрощенная модель, в реальности нужен полный пересчет
        adjusted_sharpe = gross_sharpe * (1 - cost_impact)
        
        logger.info(f"Sharpe коррекция: {gross_sharpe:.3f} -> {adjusted_sharpe:.3f} (costs: {cost_impact:.1%})")
        
        return adjusted_sharpe


def create_sharpe_validator(config: Optional[Dict[str, Any]] = None) -> SharpeValidator:
    """
    Фабричная функция для создания валидатора.
    
    Args:
        config: Опциональная конфигурация
        
    Returns:
        Настроенный SharpeValidator
    """
    if config:
        # Проверяем тип конфигурации
        if hasattr(config, 'backtest'):
            # Pydantic модель
            timeframe = getattr(config.backtest, 'timeframe', '15m')
        else:
            # Словарь
            timeframe = config.get('backtest', {}).get('timeframe', '15m')
        # Извлекаем минуты из строки типа '15m'
        if isinstance(timeframe, str) and timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
        else:
            minutes = 15
    else:
        minutes = 15
    
    return SharpeValidator(timeframe_minutes=minutes)