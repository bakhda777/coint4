"""Numba-оптимизированная версия PairBacktester."""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from ..core.numba_backtest import (
    rolling_ols, 
    simulate_trades, 
    calculate_z_scores,
    calculate_positions_and_pnl
)
from .backtest_engine import PairBacktester


@dataclass
class NumbaBacktestResult:
    """Результат Numba-бэктеста."""
    total_pnl: float
    positions: np.ndarray
    pnl_series: np.ndarray
    trades_series: np.ndarray
    z_scores: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    spread: np.ndarray


class NumbaPairBacktester(PairBacktester):
    """Numba-оптимизированная версия PairBacktester.
    
    Наследует от оригинального PairBacktester для совместимости,
    но использует Numba-компилированные функции для критических вычислений.
    """
    
    def __init__(self, *args, **kwargs):
        """Инициализация с теми же параметрами, что и у базового класса."""
        super().__init__(*args, **kwargs)
        self.numba_result: Optional[NumbaBacktestResult] = None
        
    def run_numba(self) -> NumbaBacktestResult:
        """Запуск Numba-оптимизированного бэктеста.
        
        Returns
        -------
        NumbaBacktestResult
            Результат бэктеста с основными метриками
        """
        if self.pair_data.empty or len(self.pair_data.columns) < 2:
            # Возвращаем пустой результат для пустых данных
            n = 0
            return NumbaBacktestResult(
                total_pnl=0.0,
                positions=np.array([]),
                pnl_series=np.array([]),
                trades_series=np.array([]),
                z_scores=np.array([]),
                beta=np.array([]),
                mu=np.array([]),
                sigma=np.array([]),
                spread=np.array([])
            )
            
        # Извлекаем чистые числовые данные
        y = self.pair_data.iloc[:, 0].to_numpy(dtype=np.float32, copy=False)
        x = self.pair_data.iloc[:, 1].to_numpy(dtype=np.float32, copy=False)
        
        # Быстрый rolling OLS
        beta, mu, sigma = rolling_ols(y, x, self.rolling_window)
        
        # Расчет spread
        spread = y - beta * x
        
        # Расчет z-scores
        z_scores = calculate_z_scores(spread, mu, sigma)
        
        # Вычисляем позиции и PnL
        positions, trades_series, pnl_series, total_pnl = calculate_positions_and_pnl(
            y, x, beta, mu, sigma,
            self.z_threshold, self.z_exit,
            self.commission_pct, self.slippage_pct
        )
        
        result = NumbaBacktestResult(
            total_pnl=total_pnl,
            positions=positions,
            pnl_series=pnl_series,
            trades_series=trades_series,
            z_scores=z_scores,
            beta=beta,
            mu=mu,
            sigma=sigma,
            spread=spread
        )
        
        self.numba_result = result
        return result
    
    def run(self) -> None:
        """Переопределенный метод run, использующий Numba-оптимизацию.
        
        Сохраняет совместимость с оригинальным интерфейсом,
        но использует быстрые Numba-функции.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Логируем начало обработки пары
        logger.info(f"🔄 Начинаем Numba-бэктест пары {self.pair_name or 'Unknown'} с {len(self.pair_data)} периодами данных")
        
        # Запускаем Numba-версию
        numba_result = self.run_numba()
        
        if self.pair_data.empty:
            self.results = pd.DataFrame(
                columns=["spread", "z_score", "position", "pnl", "cumulative_pnl"]
            )
            return
            
        # Создаем DataFrame с результатами для совместимости
        df = self.pair_data.copy()
        df = df.rename(columns={
            df.columns[0]: "y",
            df.columns[1]: "x"
        })
        
        # Заполняем результаты из Numba-расчетов
        df["beta"] = numba_result.beta
        df["mean"] = numba_result.mu
        df["std"] = numba_result.sigma
        df["spread"] = numba_result.spread
        df["z_score"] = numba_result.z_scores
        df["position"] = numba_result.positions
        df["pnl"] = numba_result.pnl_series
        df["trades"] = numba_result.trades_series
        df["costs"] = 0.0  # Уже учтены в pnl
        
        # Расчет кумулятивного PnL
        df["cumulative_pnl"] = df["pnl"].cumsum()
        
        # Дополнительные столбцы для совместимости
        df["entry_price_s1"] = np.nan
        df["entry_price_s2"] = np.nan
        df["exit_price_s1"] = np.nan
        df["exit_price_s2"] = np.nan
        df["entry_z"] = np.nan
        df["exit_z"] = np.nan
        df["exit_reason"] = ""
        df["trade_duration"] = 0.0
        df["entry_date"] = pd.NaT if isinstance(df.index, pd.DatetimeIndex) else np.nan
        df["commission_costs"] = 0.0
        df["slippage_costs"] = 0.0
        df["bid_ask_costs"] = 0.0
        df["impact_costs"] = 0.0
        df["market_regime"] = "neutral"
        df["hurst_exponent"] = np.nan
        df["variance_ratio"] = np.nan
        df["rolling_correlation"] = np.nan
        df["half_life_estimate"] = np.nan
        df["adf_pvalue"] = np.nan
        df["structural_break_detected"] = False
        
        self.results = df
        
        # Логируем итоговую статистику
        total_pnl = numba_result.total_pnl
        total_trades = np.sum(numba_result.trades_series > 0)
        logger.info(f"✅ Завершен Numba-бэктест пары {self.pair_name or 'Unknown'}: PnL={total_pnl:.4f}, Сделок={total_trades}")
        
    def get_performance_metrics(self) -> dict:
        """Получение основных метрик производительности.
        
        Returns
        -------
        dict
            Словарь с метриками производительности
        """
        if self.numba_result is None:
            return {}
            
        result = self.numba_result
        
        # Базовые метрики
        total_pnl = result.total_pnl
        total_trades = np.sum(result.trades_series > 0)
        
        # Расчет дополнительных метрик
        pnl_series = result.pnl_series[~np.isnan(result.pnl_series)]
        
        if len(pnl_series) > 0:
            sharpe_ratio = np.mean(pnl_series) / np.std(pnl_series) if np.std(pnl_series) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(np.cumsum(pnl_series))
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_pnl_per_trade': total_pnl / max(total_trades, 1)
        }
        
    def _calculate_max_drawdown(self, cumulative_pnl: np.ndarray) -> float:
        """Расчет максимальной просадки.
        
        Parameters
        ----------
        cumulative_pnl : np.ndarray
            Кумулятивный PnL
            
        Returns
        -------
        float
            Максимальная просадка
        """
        if len(cumulative_pnl) == 0:
            return 0.0
            
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = peak - cumulative_pnl
        return np.max(drawdown)
        
    def compare_with_original(self, tolerance: float = 1e-6) -> dict:
        """Сравнение результатов Numba-версии с оригинальной.
        
        Parameters
        ----------
        tolerance : float
            Допустимая погрешность для сравнения
            
        Returns
        -------
        dict
            Результат сравнения
        """
        # Сохраняем Numba результат
        numba_result = self.numba_result
        numba_pnl = numba_result.total_pnl if numba_result else 0.0
        
        # Временно отключаем оптимизации для честного сравнения
        original_market_regime = self.market_regime_detection
        original_structural_break = self.structural_break_protection
        
        self.market_regime_detection = False
        self.structural_break_protection = False
        
        try:
            # Запускаем оригинальную версию
            super().run()
            original_pnl = self.results['pnl'].sum() if self.results is not None else 0.0
            
            # Сравниваем результаты
            pnl_diff = abs(numba_pnl - original_pnl)
            is_equivalent = pnl_diff <= tolerance
            
            return {
                'numba_pnl': numba_pnl,
                'original_pnl': original_pnl,
                'difference': pnl_diff,
                'is_equivalent': is_equivalent,
                'relative_error': pnl_diff / max(abs(original_pnl), 1e-8)
            }
            
        finally:
            # Восстанавливаем настройки
            self.market_regime_detection = original_market_regime
            self.structural_break_protection = original_structural_break
            # Восстанавливаем Numba результат
            self.numba_result = numba_result