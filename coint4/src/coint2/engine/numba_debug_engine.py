"""
Numba engine с поддержкой debug режима для трассировки.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..core.numba_parity_v3 import (
    compute_positions_parity,
    compute_positions_parity_debug
)
from ..utils.traces import save_trace


class NumbaDebugEngine:
    """
    Numba движок с опциональным debug режимом для детальной трассировки.
    """
    
    def __init__(
        self,
        rolling_window: int = 60,
        z_enter: float = 2.0,
        z_exit: float = 0.5,
        max_holding_period: int = 100,
        commission_pct: float = 0.0004,
        slippage_pct: float = 0.0005,
        debug: bool = False,
        save_traces: bool = False,
        trace_dir: Optional[str] = None
    ):
        """
        Инициализация движка.
        
        Parameters
        ----------
        rolling_window : int
            Размер окна для rolling статистик
        z_enter : float
            Z-score порог для входа в позицию
        z_exit : float
            Z-score порог для выхода из позиции
        max_holding_period : int
            Максимальный период удержания позиции
        commission_pct : float
            Комиссия в процентах
        slippage_pct : float
            Проскальзывание в процентах
        debug : bool
            Включить debug режим с расширенным выводом
        save_traces : bool
            Сохранять трассировки в файлы
        trace_dir : Optional[str]
            Директория для сохранения трассировок
        """
        self.rolling_window = rolling_window
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.max_holding_period = max_holding_period
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.debug = debug
        self.save_traces = save_traces
        self.trace_dir = trace_dir or "artifacts/traces"
        
        # Счетчик для уникальных имен трассировок
        self._trace_counter = 0
    
    def backtest(
        self,
        df: pd.DataFrame,
        pair_name: Optional[str] = None,
        save_trace_as: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Запуск бэктеста с опциональной debug трассировкой.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с колонками 'symbol1' и 'symbol2'
        pair_name : Optional[str]
            Название пары для трассировки
        save_trace_as : Optional[str]
            Имя файла для сохранения трассировки
            
        Returns
        -------
        Dict[str, Any]
            Результаты бэктеста с опциональными debug данными
        """
        # Извлекаем массивы
        y = df['symbol1'].to_numpy()
        x = df['symbol2'].to_numpy()
        
        if self.debug:
            # Debug режим - получаем расширенные данные
            (positions, trades, pnl_series, z_scores, spreads, 
             entries_idx, exits_idx, mu, sigma, beta, alpha) = compute_positions_parity_debug(
                y=y,
                x=x,
                rolling_window=self.rolling_window,
                z_enter=self.z_enter,
                z_exit=self.z_exit,
                max_holding_period=self.max_holding_period,
                commission=self.commission_pct,
                slippage=self.slippage_pct
            )
            
            # Сохраняем трассировку если нужно
            if self.save_traces:
                trace_path = self._save_trace(
                    df, z_scores, entries_idx, exits_idx, 
                    positions, pnl_series, mu, sigma, beta, alpha,
                    pair_name, save_trace_as
                )
            else:
                trace_path = None
            
            # Формируем расширенный результат
            results = {
                # Основные метрики
                'positions': positions,
                'trades': trades,
                'pnl_series': pnl_series,
                'z_scores': z_scores,
                'spreads': spreads,
                'num_trades': np.sum(np.abs(np.diff(positions)) > 0),
                'total_pnl': np.sum(pnl_series),
                'sharpe_ratio': self._calculate_sharpe(pnl_series),
                
                # Debug данные
                'debug': {
                    'entries_idx': entries_idx,
                    'exits_idx': exits_idx,
                    'mu': mu,
                    'sigma': sigma,
                    'beta': beta,
                    'alpha': alpha,
                    'num_entries': np.sum(entries_idx),
                    'num_exits': np.sum(exits_idx),
                    'first_valid_z': next((i for i, z in enumerate(z_scores) if not np.isnan(z)), -1),
                    'max_abs_z': np.nanmax(np.abs(z_scores)),
                    'trace_path': trace_path
                }
            }
        else:
            # Обычный режим - минимальный вывод
            positions, trades, pnl_series, z_scores, spreads = compute_positions_parity(
                y=y,
                x=x,
                rolling_window=self.rolling_window,
                z_enter=self.z_enter,
                z_exit=self.z_exit,
                max_holding_period=self.max_holding_period,
                commission=self.commission_pct,
                slippage=self.slippage_pct
            )
            
            results = {
                'positions': positions,
                'trades': trades,
                'pnl_series': pnl_series,
                'z_scores': z_scores,
                'spreads': spreads,
                'num_trades': np.sum(np.abs(np.diff(positions)) > 0),
                'total_pnl': np.sum(pnl_series),
                'sharpe_ratio': self._calculate_sharpe(pnl_series)
            }
        
        return results
    
    def _calculate_sharpe(self, pnl_series: np.ndarray) -> float:
        """Расчет коэффициента Шарпа."""
        if len(pnl_series) < 2:
            return 0.0
        
        returns = pnl_series[pnl_series != 0]
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-8:
            return 0.0
        
        # Annualized Sharpe (assuming 15-min bars)
        periods_per_year = 252 * 24 * 4  # 252 trading days * 24 hours * 4 (15-min bars)
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        
        return sharpe
    
    def _save_trace(
        self,
        df: pd.DataFrame,
        z_scores: np.ndarray,
        entries_idx: np.ndarray,
        exits_idx: np.ndarray,
        positions: np.ndarray,
        pnl_series: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        beta: np.ndarray,
        alpha: np.ndarray,
        pair_name: Optional[str] = None,
        save_as: Optional[str] = None
    ) -> str:
        """Сохранение трассировки."""
        self._trace_counter += 1
        
        # Метаданные
        meta = {
            'rolling_window': self.rolling_window,
            'z_enter': self.z_enter,
            'z_exit': self.z_exit,
            'max_holding_period': self.max_holding_period,
            'commission_pct': self.commission_pct,
            'slippage_pct': self.slippage_pct
        }
        
        # Определяем имя файла
        if save_as:
            out_path = Path(self.trace_dir) / save_as
        else:
            pair = pair_name or f"pair_{self._trace_counter}"
            out_path = None  # Будет автогенерировано
        
        # Сохраняем основную трассировку
        trace_path = save_trace(
            df_index=df.index,
            z_scores=z_scores,
            entries_idx=entries_idx,
            exits_idx=exits_idx,
            positions=positions,
            pnl=pnl_series,
            out_path=out_path,
            meta=meta,
            pair=pair_name or f"pair_{self._trace_counter}",
            engine="numba_debug"
        )
        
        # Опционально: сохраняем дополнительные данные
        if save_as:
            # Сохраняем расширенные данные рядом
            extended_path = Path(trace_path).with_suffix('.extended.csv')
            extended_df = pd.DataFrame({
                'mu': mu,
                'sigma': sigma,
                'beta': beta,
                'alpha': alpha
            }, index=df.index[:len(mu)])
            extended_df.to_csv(extended_path)
        
        return trace_path
    
    def set_debug(self, enabled: bool = True):
        """Включить/выключить debug режим."""
        self.debug = enabled
    
    def set_trace_saving(self, enabled: bool = True):
        """Включить/выключить сохранение трассировок."""
        self.save_traces = enabled