"""Полная Numba-оптимизированная версия PairBacktester с всеми функциями оригинала."""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from ..core.numba_kernels import (
    rolling_ols,
    calculate_positions_and_pnl_full
)
from .base_engine import BasePairBacktester


@dataclass
class FullNumbaBacktestResult:
    """Результат полного Numba бэктеста."""
    spread: np.ndarray
    z_scores: np.ndarray
    positions: np.ndarray
    trades_series: np.ndarray
    pnl_series: np.ndarray
    total_pnl: float
    beta: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray


class FullNumbaPairBacktester(BasePairBacktester):
    
    def _validate_parameters(self) -> None:
        """Override validation to handle edge cases more gracefully."""
        # Skip validation for empty data or very short data
        if self.pair_data.empty or len(self.pair_data) < 3:
            return
        
        # For short data, just warn but don't raise error
        if len(self.pair_data) < self.rolling_window + 2:
            import warnings
            warnings.warn(
                f"Data length ({len(self.pair_data)}) is less than recommended "
                f"minimum ({self.rolling_window + 2}). Results may be limited.",
                UserWarning
            )
            return
            
        # Override the parent's rolling window validation for edge cases
        if self.rolling_window > len(self.pair_data) // 2:
            import warnings
            warnings.warn(
                f"rolling_window ({self.rolling_window}) is large relative to data size ({len(self.pair_data)}). "
                f"Results may be limited.",
                UserWarning
            )
            return
        
        # Call parent validation for normal cases
        super()._validate_parameters()
    """Полная Numba-оптимизированная версия PairBacktester.
    
    Эта версия полностью повторяет функциональность оригинального PairBacktester,
    включая:
    - Определение рыночных режимов (Hurst Exponent, Variance Ratio)
    - Защиту от структурных сдвигов (корреляция, полупериод, ADF тест)
    - Адаптивные пороги на основе волатильности
    - Временные стоп-лоссы
    - Расширенные стоп-лоссы
    - Полный расчет торговых издержек
    
    При этом обеспечивает значительное ускорение за счет Numba JIT компиляции.
    """
    
    def __init__(self, *args, beta=None, **kwargs):
        """Инициализация с теми же параметрами, что и оригинальный PairBacktester.
        
        Args:
            beta: Фиксированный коэффициент beta из коинтеграционного теста.
                  Если None, используется rolling OLS для вычисления beta.
        """
        adaptive_threshold_factor = kwargs.pop("adaptive_threshold_factor", 0.0)
        min_volatility = kwargs.pop("min_volatility", 0.001)
        kwargs.setdefault("market_regime_detection", False)
        kwargs.setdefault("structural_break_protection", False)
        super().__init__(*args, **kwargs)
        self.fixed_beta = beta  # Сохраняем фиксированную beta если передана
        self.adaptive_threshold_factor = adaptive_threshold_factor
        self.min_volatility = min_volatility
        
    def run_numba_full(self) -> FullNumbaBacktestResult:
        """Запуск полного Numba бэктеста с всеми функциями оригинала.
        
        Returns:
            FullNumbaBacktestResult: Полные результаты бэктеста
        """
        if self.pair_data.empty or len(self.pair_data.columns) < 2:
            n = len(self.pair_data) if not self.pair_data.empty else 0
            return FullNumbaBacktestResult(
                spread=np.full(n, np.nan, dtype=np.float32),
                z_scores=np.full(n, np.nan, dtype=np.float32),
                positions=np.zeros(n, dtype=np.float32),
                trades_series=np.zeros(n, dtype=np.float32),
                pnl_series=np.zeros(n, dtype=np.float32),
                total_pnl=0.0,
                beta=np.full(n, np.nan, dtype=np.float32),
                mu=np.full(n, np.nan, dtype=np.float32),
                sigma=np.full(n, np.nan, dtype=np.float32)
            )
        
        # Check for minimum data length requirement
        n = len(self.pair_data)
        min_required_length = self.rolling_window + 10
        if n < min_required_length:
            return FullNumbaBacktestResult(
                spread=np.full(n, np.nan, dtype=np.float32),
                z_scores=np.full(n, np.nan, dtype=np.float32),
                positions=np.zeros(n, dtype=np.float32),
                trades_series=np.zeros(n, dtype=np.float32),
                pnl_series=np.zeros(n, dtype=np.float32),
                total_pnl=0.0,
                beta=np.full(n, np.nan, dtype=np.float32),
                mu=np.full(n, np.nan, dtype=np.float32),
                sigma=np.full(n, np.nan, dtype=np.float32)
            )
        
        # Извлекаем данные как float32 numpy массивы
        y = self.pair_data.iloc[:, 0].values.astype(np.float32)
        x = self.pair_data.iloc[:, 1].values.astype(np.float32)
        
        # Быстрый rolling OLS
        beta, mu, sigma = rolling_ols(y, x, self.rolling_window)
        
        bar_minutes = 15
        if isinstance(self.pair_data.index, pd.DatetimeIndex) and len(self.pair_data.index) > 1:
            deltas = self.pair_data.index.to_series().diff().dropna()
            if not deltas.empty:
                median_seconds = deltas.dt.total_seconds().median()
                if median_seconds and median_seconds > 0:
                    bar_minutes = int(round(median_seconds / 60))

        min_hold_periods = 0
        if getattr(self, "min_position_hold_minutes", 0) > 0 and bar_minutes > 0:
            min_hold_periods = int(np.ceil(self.min_position_hold_minutes / bar_minutes))

        cooldown_periods = int(getattr(self, "cooldown_periods", 0) or 0)
        if getattr(self, "anti_churn_cooldown_minutes", 0) > 0 and bar_minutes > 0:
            anti_churn_periods = int(np.ceil(self.anti_churn_cooldown_minutes / bar_minutes))
            if anti_churn_periods > cooldown_periods:
                cooldown_periods = anti_churn_periods

        max_holding_period = getattr(self, "max_holding_period", 100)
        if getattr(self, "time_stop_multiplier", None) is not None and getattr(self, "half_life", None) is not None:
            try:
                time_stop_days = float(self.half_life) * float(self.time_stop_multiplier)
                if time_stop_days > 0 and bar_minutes > 0:
                    max_holding_period = int(np.ceil(time_stop_days * 1440 / bar_minutes))
                    if max_holding_period < 1:
                        max_holding_period = 1
            except (TypeError, ValueError):
                pass

        min_notional = 0.0
        max_notional = 0.0
        if getattr(self, "portfolio", None) is not None and getattr(self.portfolio, "config", None) is not None:
            min_notional = float(getattr(self.portfolio.config, "min_notional_per_trade", 0.0) or 0.0)
            max_notional = float(getattr(self.portfolio.config, "max_notional_per_trade", 0.0) or 0.0)

        capital_at_risk = float(getattr(self, "capital_at_risk", 0.0) or 0.0)

        # Полный расчет позиций и PnL с всеми функциями
        positions, pnl_series, cumulative_pnl, _costs_series = calculate_positions_and_pnl_full(
            y=y,
            x=x,
            beta=beta,
            mu=mu,
            sigma=sigma,
            rolling_window=self.rolling_window,
            entry_threshold=self.z_threshold,
            exit_threshold=self.z_exit,
            commission=self.commission_pct,
            slippage=self.slippage_pct,
            max_holding_period=max_holding_period,
            enable_regime_detection=getattr(self, 'market_regime_detection', True),
            enable_structural_breaks=getattr(self, 'structural_break_protection', True),
            market_regime_factor_min=float(getattr(self, "market_regime_factor_min", 0.5) or 0.5),
            market_regime_factor_max=float(getattr(self, "market_regime_factor_max", 1.5) or 1.5),
            structural_break_min_correlation=float(getattr(self, "structural_break_min_correlation", 0.3) or 0.3),
            structural_break_entry_multiplier=float(getattr(self, "structural_break_entry_multiplier", 1.5) or 1.5),
            structural_break_exit_multiplier=float(getattr(self, "structural_break_exit_multiplier", 1.2) or 1.2),
            min_volatility=getattr(self, 'min_volatility', 0.001),
            adaptive_threshold_factor=getattr(self, 'adaptive_threshold_factor', 1.0),
            max_var_multiplier=getattr(self, 'max_var_multiplier', 3.0),
            cooldown_periods=cooldown_periods,
            min_hold_periods=min_hold_periods,
            stop_loss_zscore=float(getattr(self, "pair_stop_loss_zscore", 0.0) or 0.0),
            min_spread_move_sigma=float(getattr(self, "min_spread_move_sigma", 0.0) or 0.0),
            capital_at_risk=capital_at_risk,
            min_notional_per_trade=min_notional,
            max_notional_per_trade=max_notional,
            pair_stop_loss_usd=float(getattr(self, "pair_stop_loss_usd", 0.0) or 0.0),
        )
        
        # Calculate spread and z_scores for compatibility
        spread = np.zeros(len(y), dtype=np.float32)
        z_scores = np.zeros(len(y), dtype=np.float32)
        trades_series = np.zeros(len(y), dtype=np.float32)
        
        # КРИТИЧНО: Используем фиксированную beta из коинтеграционного теста если передана
        if self.fixed_beta is not None:
            # Используем фиксированную beta для всего периода
            fixed_beta_value = float(self.fixed_beta)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"🎯 Используем фиксированную beta={fixed_beta_value:.4f}")
            
            for i in range(self.rolling_window, len(y)):
                # Спред с фиксированной beta
                spread[i] = y[i] - fixed_beta_value * x[i]
                
                # Вычисляем rolling статистики для спреда
                if i >= self.rolling_window:
                    spread_window = spread[i-self.rolling_window+1:i+1]
                    spread_mean = np.mean(spread_window)
                    spread_std = np.std(spread_window)
                    if spread_std > 1e-6:
                        z_scores[i] = (spread[i] - spread_mean) / spread_std
                
                # Mark trade points where position changes
                if i > 0 and positions[i] != positions[i-1]:
                    trades_series[i] = 1.0
        else:
            # Используем rolling beta (старое поведение)
            for i in range(self.rolling_window, len(y)):
                if not np.isnan(beta[i]) and not np.isnan(mu[i]) and not np.isnan(sigma[i]) and sigma[i] >= 1e-6:
                    spread[i] = y[i] - beta[i] * x[i]
                    z_scores[i] = (spread[i] - mu[i]) / sigma[i]
                    
                    # Mark trade points where position changes
                    if i > 0 and positions[i] != positions[i-1]:
                        trades_series[i] = 1.0
        
        total_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0.0
        
        return FullNumbaBacktestResult(
            spread=spread,
            z_scores=z_scores,
            positions=positions,
            trades_series=trades_series,
            pnl_series=pnl_series,
            total_pnl=total_pnl,
            beta=beta,
            mu=mu,
            sigma=sigma
        )
    
    def run(self) -> None:
        """Переопределяем метод run для использования полной Numba версии.
        
        Этот метод заменяет оригинальный run() и использует полную Numba реализацию
        со всеми функциями оригинального алгоритма.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Логируем начало обработки пары
        logger.info(f"🔄 Начинаем полный Numba-бэктест пары {getattr(self, 'pair_name', 'Unknown')} с {len(self.pair_data)} периодами данных")
        
        # Запускаем полный Numba бэктест
        numba_result = self.run_numba_full()
        
        # Создаем результирующий DataFrame в том же формате, что и оригинал
        if self.pair_data.empty:
            self.results = pd.DataFrame(
                columns=["spread", "z_score", "position", "pnl", "cumulative_pnl"]
            )
            return
        
        # Обеспечиваем отсутствие look-ahead bias: первые rolling_window значений должны быть NaN
        beta_corrected = numba_result.beta.copy()
        mu_corrected = numba_result.mu.copy()
        sigma_corrected = numba_result.sigma.copy()
        z_scores_corrected = numba_result.z_scores.copy()
        spread_corrected = numba_result.spread.copy()
        
        # Устанавливаем NaN для первых rolling_window периодов
        beta_corrected[:self.rolling_window] = np.nan
        mu_corrected[:self.rolling_window] = np.nan
        sigma_corrected[:self.rolling_window] = np.nan
        z_scores_corrected[:self.rolling_window] = np.nan
        spread_corrected[:self.rolling_window] = np.nan
        
        # Формируем результаты в формате оригинального PairBacktester
        results_data = {
            'spread': spread_corrected,
            'z_score': z_scores_corrected,
            'position': numba_result.positions,
            'pnl': numba_result.pnl_series,
            'cumulative_pnl': np.cumsum(numba_result.pnl_series),
            'beta': beta_corrected,
            'mu': mu_corrected,
            'sigma': sigma_corrected,
            'std': sigma_corrected,  # Алиас для совместимости с тестами
            'trades': numba_result.trades_series,
            'x': self.pair_data.iloc[:, 1].values,  # Второй актив
            'y': self.pair_data.iloc[:, 0].values   # Первый актив
        }
        
        # Создаем DataFrame с тем же индексом, что и исходные данные
        self.results = pd.DataFrame(results_data, index=self.pair_data.index)
        
        # Добавляем дополнительные столбцы для совместимости
        self.results['entry_price_s1'] = np.nan
        self.results['entry_price_s2'] = np.nan
        self.results['entry_z'] = np.nan
        self.results['entry_date'] = pd.Series(dtype='object', index=self.results.index)
        self.results['exit_reason'] = ''
        self.results['exit_price_s1'] = np.nan
        self.results['exit_price_s2'] = np.nan
        self.results['exit_z'] = np.nan
        self.results['trade_duration'] = np.nan
        
        # Заполняем информацию о сделках там, где есть изменения позиции
        trade_indices = np.where(numba_result.trades_series != 0)[0]
        for idx in trade_indices:
            if idx < len(self.pair_data):
                self.results.iloc[idx, self.results.columns.get_loc('entry_price_s1')] = self.pair_data.iloc[idx, 0]
                self.results.iloc[idx, self.results.columns.get_loc('entry_price_s2')] = self.pair_data.iloc[idx, 1]
                self.results.iloc[idx, self.results.columns.get_loc('entry_z')] = numba_result.z_scores[idx]
                if isinstance(self.pair_data.index, pd.DatetimeIndex):
                    self.results.at[self.results.index[idx], 'entry_date'] = self.pair_data.index[idx]
                else:
                    self.results.at[self.results.index[idx], 'entry_date'] = float(idx)
        
        # Логируем итоговую статистику
        total_pnl = numba_result.total_pnl
        total_trades = np.sum(numba_result.trades_series > 0)
        logger.info(f"✅ Завершен полный Numba-бэктест пары {getattr(self, 'pair_name', 'Unknown')}: PnL={total_pnl:.4f}, Сделок={total_trades}")
    
    def get_performance_summary(self) -> dict:
        """Получение сводки производительности.
        
        Returns:
            dict: Словарь с ключевыми метриками производительности
        """
        if self.results is None or self.results.empty:
            return {}
        
        # Ensure total_pnl is always numeric
        if len(self.results) > 0 and 'cumulative_pnl' in self.results.columns:
            total_pnl = self.results['cumulative_pnl'].iloc[-1]
            # Convert to float and handle NaN/inf values
            if pd.isna(total_pnl) or np.isinf(total_pnl):
                total_pnl = 0.0
            else:
                total_pnl = float(total_pnl)
        else:
            total_pnl = 0.0
        
        # Базовые метрики
        metrics = {
            'total_pnl': total_pnl,
            'total_trades': len(self.results[self.results['trades'] != 0]),
            'winning_trades': len(self.results[self.results['pnl'] > 0]),
            'losing_trades': len(self.results[self.results['pnl'] < 0]),
        }
        
        # Дополнительные метрики
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
            
            winning_pnl = self.results[self.results['pnl'] > 0]['pnl']
            losing_pnl = self.results[self.results['pnl'] < 0]['pnl']
            
            if len(winning_pnl) > 0:
                metrics['avg_winning_trade'] = winning_pnl.mean()
                metrics['max_winning_trade'] = winning_pnl.max()
            
            if len(losing_pnl) > 0:
                metrics['avg_losing_trade'] = losing_pnl.mean()
                metrics['max_losing_trade'] = losing_pnl.min()
            
            # Коэффициент Шарпа (приблизительный)
            if len(self.results['pnl']) > 1:
                pnl_std = self.results['pnl'].std()
                if pnl_std > 0:
                    # Правильная аннуализация для 15-минутных баров (96 баров в день)
                    bars_per_day = 96  # 24 часа * 4 бара в час для 15-минутных данных
                    metrics['sharpe_ratio'] = self.results['pnl'].mean() / pnl_std * np.sqrt(252 * bars_per_day)
        
        return metrics
    
    def compare_with_original(self, original_backtester: BasePairBacktester, 
                            tolerance: float = 0.01) -> dict:
        """Сравнение результатов с оригинальным бэктестером.
        
        Args:
            original_backtester: Оригинальный BasePairBacktester для сравнения
            tolerance: Допустимая относительная погрешность
            
        Returns:
            dict: Результаты сравнения
        """
        # Запускаем оригинальный бэктест
        original_backtester.run()
        
        if (original_backtester.results is None or original_backtester.results.empty or
            self.results is None or self.results.empty):
            return {'error': 'One or both backtesters have empty results'}
        
        # Сравниваем ключевые метрики
        original_pnl = original_backtester.results['cumulative_pnl'].iloc[-1]
        numba_pnl = self.results['cumulative_pnl'].iloc[-1]
        
        pnl_diff = abs(numba_pnl - original_pnl)
        pnl_relative_error = pnl_diff / abs(original_pnl) if abs(original_pnl) > 1e-6 else pnl_diff
        
        comparison = {
            'original_pnl': original_pnl,
            'numba_pnl': numba_pnl,
            'pnl_difference': pnl_diff,
            'pnl_relative_error': pnl_relative_error,
            'within_tolerance': pnl_relative_error <= tolerance,
            'tolerance': tolerance
        }
        
        # Сравниваем количество сделок
        original_trades = len(original_backtester.results[original_backtester.results.get('trades', pd.Series()) != 0])
        numba_trades = len(self.results[self.results['trades'] != 0])
        
        comparison.update({
            'original_trades': original_trades,
            'numba_trades': numba_trades,
            'trades_difference': abs(numba_trades - original_trades)
        })
        
        return comparison
