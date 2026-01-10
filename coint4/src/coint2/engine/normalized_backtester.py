"""
Нормализованный бэктестер с правильным расчетом PnL.
Исправляет все критические проблемы оригинальной версии.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..core.numba_kernels_v2 import (
    calculate_positions_and_pnl_normalized,
    calculate_metrics_suite,
    calculate_max_drawdown_correct,
    calculate_sharpe_ratio_correct
)


@dataclass
class NormalizedBacktestResult:
    """Результат нормализованного бэктеста."""
    positions: np.ndarray
    pnl_pct: np.ndarray  # PnL в процентах
    cumulative_return: np.ndarray  # Кумулятивная доходность в %
    equity_curve: np.ndarray  # Кривая капитала в $
    metrics: Dict[str, float]
    
    @property
    def sharpe_ratio(self) -> float:
        return self.metrics.get('sharpe_ratio', 0.0)
    
    @property
    def max_drawdown(self) -> float:
        return self.metrics.get('max_drawdown', 0.0)
    
    @property
    def total_return(self) -> float:
        return self.metrics.get('total_return', 0.0)
    
    @property
    def num_trades(self) -> int:
        return int(self.metrics.get('num_trades', 0))


class NormalizedPairBacktester:
    """
    Бэктестер с правильной нормализацией и расчетом PnL.
    
    Ключевые улучшения:
    1. PnL в процентах от капитала, не в абсолютных значениях
    2. Ограничение размера позиции (max 2% на сделку)
    3. Устранение lookahead bias
    4. Правильный расчет max drawdown
    5. Реалистичные ограничения на Sharpe
    """
    
    def __init__(
        self,
        pair_data: pd.DataFrame,
        rolling_window: int = 20,
        z_threshold: float = 0.5,
        z_exit: float = 0.0,
        commission_pct: float = 0.0004,  # 0.04% - реалистичная комиссия Binance
        slippage_pct: float = 0.0005,  # 0.05% - реалистичный slippage
        max_position_pct: float = 0.02,  # Максимум 2% капитала на позицию
        initial_capital: float = 100000.0,
        **kwargs
    ):
        """
        Инициализация бэктестера.
        
        Args:
            pair_data: DataFrame с колонками ['symbol1', 'symbol2']
            rolling_window: Размер окна для расчета статистик
            z_threshold: Порог входа по z-score
            z_exit: Порог выхода по z-score
            commission_pct: Комиссия в процентах
            slippage_pct: Проскальзывание в процентах
            max_position_pct: Максимальный размер позиции в % от капитала
            initial_capital: Начальный капитал
        """
        self.pair_data = pair_data
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold
        self.z_exit = z_exit
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self.initial_capital = initial_capital
        
        # Валидация параметров
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Валидация параметров бэктестера."""
        if self.pair_data.empty:
            raise ValueError("Pair data is empty")
        
        if len(self.pair_data.columns) < 2:
            raise ValueError("Pair data must have at least 2 columns")
        
        if self.rolling_window < 5:
            raise ValueError("Rolling window must be at least 5")
        
        if self.rolling_window > len(self.pair_data) // 2:
            raise ValueError(f"Rolling window ({self.rolling_window}) too large for data size ({len(self.pair_data)})")
        
        if self.z_threshold <= 0:
            raise ValueError("Z-score threshold must be positive")
        
        if self.commission_pct < 0 or self.commission_pct > 0.01:
            raise ValueError("Commission must be between 0 and 1%")
        
        if self.slippage_pct < 0 or self.slippage_pct > 0.01:
            raise ValueError("Slippage must be between 0 and 1%")
        
        if self.max_position_pct <= 0 or self.max_position_pct > 0.1:
            raise ValueError("Max position must be between 0 and 10% of capital")
    
    def run(self) -> NormalizedBacktestResult:
        """
        Запускает бэктест с нормализованным расчетом PnL.
        
        Returns:
            NormalizedBacktestResult с результатами
        """
        # Извлекаем цены
        y_prices = self.pair_data.iloc[:, 0].values.astype(np.float64)
        x_prices = self.pair_data.iloc[:, 1].values.astype(np.float64)
        
        # Проверяем на валидность
        if np.any(np.isnan(y_prices)) or np.any(np.isnan(x_prices)):
            # Заполняем пропуски методом forward fill
            y_prices = pd.Series(y_prices).fillna(method='ffill').values
            x_prices = pd.Series(x_prices).fillna(method='ffill').values
        
        # Проверяем на нулевые или отрицательные цены
        if np.any(y_prices <= 0) or np.any(x_prices <= 0):
            raise ValueError("Prices must be positive")
        
        # Запускаем нормализованный бэктест
        positions, pnl_pct, cumulative_return, equity_curve = calculate_positions_and_pnl_normalized(
            y_prices=y_prices,
            x_prices=x_prices,
            rolling_window=self.rolling_window,
            entry_threshold=self.z_threshold,
            exit_threshold=self.z_exit,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            max_position_pct=self.max_position_pct,
            initial_capital=self.initial_capital
        )
        
        # Рассчитываем метрики
        metrics = self._calculate_metrics(pnl_pct, equity_curve, positions)
        
        return NormalizedBacktestResult(
            positions=positions,
            pnl_pct=pnl_pct,
            cumulative_return=cumulative_return,
            equity_curve=equity_curve,
            metrics=metrics
        )
    
    def _calculate_metrics(self, pnl_pct: np.ndarray, equity_curve: np.ndarray, positions: np.ndarray) -> Dict[str, float]:
        """Рассчитывает метрики производительности."""
        # Базовые метрики
        total_return = (equity_curve[-1] / self.initial_capital - 1) * 100
        
        # Подсчет сделок
        position_changes = np.diff(positions)
        num_trades = np.sum(position_changes != 0) // 2  # Вход + выход = 1 сделка
        
        # PnL статистика
        non_zero_pnl = pnl_pct[pnl_pct != 0]
        
        if len(non_zero_pnl) > 0:
            win_rate = np.sum(non_zero_pnl > 0) / len(non_zero_pnl)
            
            winning_pnl = non_zero_pnl[non_zero_pnl > 0]
            losing_pnl = non_zero_pnl[non_zero_pnl < 0]
            
            avg_win = np.mean(winning_pnl) if len(winning_pnl) > 0 else 0
            avg_loss = np.mean(losing_pnl) if len(losing_pnl) > 0 else 0
            
            gross_wins = np.sum(winning_pnl) if len(winning_pnl) > 0 else 0
            gross_losses = abs(np.sum(losing_pnl)) if len(losing_pnl) > 0 else 0
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Risk метрики
        max_drawdown = calculate_max_drawdown_correct(equity_curve)
        sharpe_ratio = calculate_sharpe_ratio_correct(pnl_pct)
        
        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio (downside deviation)
        if len(non_zero_pnl) > 0:
            negative_returns = non_zero_pnl[non_zero_pnl < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                sortino_ratio = np.mean(non_zero_pnl) / (downside_std + 1e-8) * np.sqrt(252 * 96)
            else:
                sortino_ratio = sharpe_ratio  # Если нет убытков, Sortino = Sharpe
        else:
            sortino_ratio = 0
        
        # Защита от NaN значений
        def safe_value(val, default=0.0):
            return default if np.isnan(val) or np.isinf(val) else val
        
        final_capital = safe_value(equity_curve[-1], self.initial_capital)
        
        return {
            'total_return': safe_value(total_return),
            'num_trades': int(num_trades),
            'win_rate': safe_value(win_rate),
            'avg_win': safe_value(avg_win),
            'avg_loss': safe_value(avg_loss),
            'profit_factor': safe_value(profit_factor),
            'max_drawdown': safe_value(max_drawdown),
            'sharpe_ratio': safe_value(sharpe_ratio),
            'calmar_ratio': safe_value(calmar_ratio),
            'sortino_ratio': safe_value(sortino_ratio),
            'final_capital': final_capital,
            'total_pnl': final_capital - self.initial_capital
        }