"""
Калькулятор метрик для оптимизации.
Отвечает за расчет и валидацию торговых метрик.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradingMetrics:
    """Структура торговых метрик."""
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    total_trades: int
    avg_trade_duration: float
    total_return: float
    volatility: float
    profit_factor: float
    recovery_factor: float
    

class MetricsCalculator:
    """
    Калькулятор торговых метрик.
    Унифицированный расчет метрик для оптимизации.
    """
    
    def __init__(self, annual_trading_days: int = 252):
        """
        Args:
            annual_trading_days: Количество торговых дней в году
        """
        self.annual_trading_days = annual_trading_days
        self._cache = {}
        
    def calculate_metrics(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        initial_capital: float = 100000.0
    ) -> TradingMetrics:
        """
        Рассчитывает все торговые метрики.
        
        Args:
            returns: Временной ряд доходностей
            trades: DataFrame со сделками (опционально)
            initial_capital: Начальный капитал
            
        Returns:
            TradingMetrics с рассчитанными метриками
        """
        # Проверяем валидность данных
        if returns.empty or len(returns) < 2:
            return self._get_default_metrics()
        
        # Очищаем returns от NaN
        returns = returns.dropna()
        if returns.empty:
            return self._get_default_metrics()
        
        # Рассчитываем базовые метрики
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        total_return = self.calculate_total_return(returns)
        volatility = self.calculate_volatility(returns)
        
        # Рассчитываем производные метрики
        calmar_ratio = self.calculate_calmar_ratio(total_return, max_drawdown)
        recovery_factor = self.calculate_recovery_factor(total_return, max_drawdown)
        
        # Метрики по сделкам
        if trades is not None and not trades.empty:
            win_rate = self.calculate_win_rate(trades)
            total_trades = len(trades)
            avg_trade_duration = self.calculate_avg_trade_duration(trades)
            profit_factor = self.calculate_profit_factor(trades)
        else:
            win_rate = 0.0
            total_trades = 0
            avg_trade_duration = 0.0
            profit_factor = 0.0
        
        return TradingMetrics(
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_duration=avg_trade_duration,
            total_return=total_return,
            volatility=volatility,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor
        )
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Рассчитывает Sharpe ratio.
        
        Args:
            returns: Временной ряд доходностей
            
        Returns:
            Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        # Средняя доходность
        mean_return = returns.mean()
        
        # Стандартное отклонение
        std_return = returns.std()
        
        # Sharpe ratio (annualized)
        periods_per_year = self._estimate_periods_per_year(returns)
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        
        # Ограничиваем диапазон
        return np.clip(sharpe, -10.0, 10.0)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Рассчитывает максимальную просадку.
        
        Args:
            returns: Временной ряд доходностей
            
        Returns:
            Максимальная просадка (положительное значение)
        """
        if returns.empty:
            return 0.0
        
        # Кумулятивная доходность
        cum_returns = (1 + returns).cumprod()
        
        # Running maximum
        running_max = cum_returns.expanding().max()
        
        # Drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Максимальная просадка (делаем положительной)
        max_dd = abs(drawdown.min())
        
        return max_dd
    
    def calculate_total_return(self, returns: pd.Series) -> float:
        """
        Рассчитывает общую доходность.
        
        Args:
            returns: Временной ряд доходностей
            
        Returns:
            Общая доходность
        """
        if returns.empty:
            return 0.0
        
        # Кумулятивная доходность
        total_return = (1 + returns).prod() - 1
        
        return total_return
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """
        Рассчитывает волатильность (annualized).
        
        Args:
            returns: Временной ряд доходностей
            
        Returns:
            Годовая волатильность
        """
        if returns.empty:
            return 0.0
        
        periods_per_year = self._estimate_periods_per_year(returns)
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        return volatility
    
    def calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """
        Рассчитывает Calmar ratio.
        
        Args:
            total_return: Общая доходность
            max_drawdown: Максимальная просадка
            
        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0
        
        # Annualized return / Max Drawdown
        calmar = total_return / max_drawdown
        
        # Ограничиваем диапазон
        return np.clip(calmar, -10.0, 10.0)
    
    def calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """
        Рассчитывает Recovery factor.
        
        Args:
            total_return: Общая доходность
            max_drawdown: Максимальная просадка
            
        Returns:
            Recovery factor
        """
        if max_drawdown == 0:
            return 0.0
        
        return total_return / max_drawdown
    
    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """
        Рассчитывает процент выигрышных сделок.
        
        Args:
            trades: DataFrame со сделками (должен содержать колонку 'pnl')
            
        Returns:
            Win rate [0, 1]
        """
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0
        
        winning_trades = (trades['pnl'] > 0).sum()
        total_trades = len(trades)
        
        if total_trades == 0:
            return 0.0
        
        return winning_trades / total_trades
    
    def calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """
        Рассчитывает Profit factor.
        
        Args:
            trades: DataFrame со сделками
            
        Returns:
            Profit factor
        """
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0
        
        profits = trades[trades['pnl'] > 0]['pnl'].sum()
        losses = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        if losses == 0:
            return 10.0 if profits > 0 else 0.0
        
        return min(profits / losses, 10.0)
    
    def calculate_avg_trade_duration(self, trades: pd.DataFrame) -> float:
        """
        Рассчитывает среднюю длительность сделки в днях.
        
        Args:
            trades: DataFrame со сделками (должен содержать 'entry_time' и 'exit_time')
            
        Returns:
            Средняя длительность в днях
        """
        if trades.empty:
            return 0.0
        
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            durations = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 86400
            return durations.mean()
        
        return 0.0
    
    def _estimate_periods_per_year(self, returns: pd.Series) -> float:
        """
        Оценивает количество периодов в году на основе частоты данных.
        
        Args:
            returns: Временной ряд доходностей
            
        Returns:
            Количество периодов в году
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            return self.annual_trading_days
        
        # Оцениваем частоту
        if len(returns) > 1:
            avg_delta = (returns.index[-1] - returns.index[0]) / (len(returns) - 1)
            
            # Определяем частоту
            if avg_delta < pd.Timedelta(hours=1):
                # 15-минутные бары (96 баров в день * 252 дня)
                return 96 * 252
            elif avg_delta < pd.Timedelta(days=1):
                # Часовые бары (24 * 365)
                return 24 * 365
            elif avg_delta < pd.Timedelta(days=7):
                # Дневные бары
                return 252
            else:
                # Недельные или месячные
                return 52
        
        return self.annual_trading_days
    
    def _get_default_metrics(self) -> TradingMetrics:
        """Возвращает дефолтные метрики при ошибке."""
        return TradingMetrics(
            sharpe_ratio=-10.0,
            max_drawdown=1.0,
            calmar_ratio=-10.0,
            win_rate=0.0,
            total_trades=0,
            avg_trade_duration=0.0,
            total_return=-1.0,
            volatility=1.0,
            profit_factor=0.0,
            recovery_factor=0.0
        )
    
    def validate_metrics(self, metrics: TradingMetrics) -> Tuple[bool, str]:
        """
        Валидирует рассчитанные метрики.
        
        Args:
            metrics: Рассчитанные метрики
            
        Returns:
            (is_valid, error_message)
        """
        # Проверяем диапазоны
        if not -10 <= metrics.sharpe_ratio <= 10:
            return False, f"Sharpe ratio вне диапазона: {metrics.sharpe_ratio}"
        
        if not 0 <= metrics.max_drawdown <= 1:
            return False, f"Max drawdown вне диапазона: {metrics.max_drawdown}"
        
        if not 0 <= metrics.win_rate <= 1:
            return False, f"Win rate вне диапазона: {metrics.win_rate}"
        
        if metrics.total_trades < 0:
            return False, f"Отрицательное количество сделок: {metrics.total_trades}"
        
        if metrics.avg_trade_duration < 0:
            return False, f"Отрицательная длительность сделок: {metrics.avg_trade_duration}"
        
        return True, "OK"