"""
Orchestrator for Walk-Forward Analysis pipeline.
"""
from pathlib import Path
import os
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Any
from joblib import Parallel, delayed

from coint2.core.data_loader import DataHandler
from coint2.core.memory_optimization import get_price_data_view
from coint2.pipeline.filters import enhanced_pair_screening as apply_filters
from coint2.core.math_utils import safe_div, calculate_ssd
import itertools
import numpy as np
from coint2.engine.numba_engine import NumbaPairBacktester
from coint2.core.performance import calculate_metrics, sharpe_ratio_on_returns, safe_sharpe, safe_max_dd
from coint2.utils.logger import get_logger
from coint2.utils import create_performance_report, calculate_extended_metrics
from coint2.utils.timing_utils import logged_time, time_block
from coint2.utils.config import AppConfig


class QuarantineManager:
    """Manages pair quarantine state based on performance metrics."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.quarantined_pairs: Dict[str, pd.Timestamp] = {} # pair -> release_date
        self.logger = get_logger("quarantine_manager")
        
        # Params
        self.enabled = getattr(config.backtest, 'enable_pair_quarantine', False)
        self.pnl_sigma_threshold = getattr(config.backtest, 'quarantine_pnl_threshold_sigma', 2.0)
        self.dd_threshold = getattr(config.backtest, 'quarantine_drawdown_threshold_pct', 0.05)
        self.duration_days = getattr(config.backtest, 'quarantine_period_days', 7)
        
    def is_quarantined(self, pair: str, current_date: pd.Timestamp) -> bool:
        if not self.enabled:
            return False
            
        if pair in self.quarantined_pairs:
            release_date = self.quarantined_pairs[pair]
            if current_date < release_date:
                return True
            else:
                # Release
                del self.quarantined_pairs[pair]
                self.logger.info(f"[QUARANTINE_EXIT] Pair: {pair}, Date: {current_date.date()}")
                return False
        return False
        
    def update_from_results(self, step_results: List[dict], current_date: pd.Timestamp):
        """Update quarantine list based on step results."""
        if not self.enabled:
            return
            
        for res in step_results:
            if not res['success']:
                continue
                
            stats = res['trade_stat']
            pair = stats['pair']
            pnl = stats['total_pnl']
            max_dd = abs(stats.get('max_dd_pct', 0.0)) # Use calculated max_dd_pct
            
            # Calculate sigma of daily PnL if available
            pnl_series = res.get('pnl_series')
            sigma_loss = 0.0
            std_dev = 0.0
            if pnl_series is not None and not pnl_series.empty:
                daily_pnl = pnl_series.resample('D').sum()
                if not daily_pnl.empty:
                    std_dev = daily_pnl.std()
                    if std_dev > 0:
                         sigma_loss = -pnl / std_dev
            
            # Check triggers
            reasons = []
            
            # NEW: Check for Hard Stop or Step Risk Breach
            trades = res.get('trades_log', [])
            r_breaches = 0
            
            # Calculate cumulative PnL R from trades to detect Step Risk Breach
            cum_pnl_r = 0.0
            
            for t in trades:
                if isinstance(t, dict):
                    # Check R-Stop
                    reason = t.get('exit_reason', '')
                    if reason == 'PnLStopHard':
                        r_breaches += 1
                    
                    # Check Cumulative R
                    if 'final_pnl_r' in t:
                         cum_pnl_r += t['final_pnl_r']
            
            # Get Step Limit from config (magnitude)
            step_limit_magnitude = 3.0
            if hasattr(self.config, 'backtest') and hasattr(self.config.backtest, 'pair_step_r_limit'):
                 step_limit = getattr(self.config.backtest, 'pair_step_r_limit', None)
                 if step_limit is not None:
                     step_limit_magnitude = float(step_limit)
            elif hasattr(self.config, 'pair_step_r_limit'):
                 step_limit = getattr(self.config, 'pair_step_r_limit', None)
                 if step_limit is not None:
                     step_limit_magnitude = float(step_limit)

            # Ensure threshold is negative
            step_limit_r = -abs(step_limit_magnitude)

            # Check Step Breach
            if cum_pnl_r <= step_limit_r:
                 reasons.append(f"StepRiskBreach {cum_pnl_r:.2f}R <= {step_limit_r}R")
            
            # Check R-Stop Breaches (N=1 strict)
            if r_breaches >= 1:
                 reasons.append(f"HardStopBreach count={r_breaches}")

            if sigma_loss > self.pnl_sigma_threshold:
                reasons.append(f"SigmaLoss {sigma_loss:.2f} > {self.pnl_sigma_threshold}")
            
            if max_dd > self.dd_threshold * 100: # dd_threshold is 0.03, max_dd_pct is 3.0
                reasons.append(f"Drawdown {max_dd:.2f}% > {self.dd_threshold*100:.2f}%")
            
            if reasons:
                # Продлеваем до конца шага минимум (testing_period_days), чтобы не ловить повторные стопы
                step_days = getattr(self.config.walk_forward, 'testing_period_days', self.duration_days)
                quarantine_days = max(self.duration_days, step_days)
                release_date = current_date + pd.Timedelta(days=quarantine_days)
                self.quarantined_pairs[pair] = release_date
                
                reason_str = "; ".join(reasons)
                self.logger.info(
                    f"[QUARANTINE_ENTER] Pair: {pair}, Reason: {reason_str}, "
                    f"PnL: {pnl:.2f}, Sigma: {sigma_loss:.2f}, DD: {max_dd:.2f}%, "
                    f"Threshold: {self.dd_threshold*100:.2f}%, "
                    f"Until: {release_date.date()}"
                )


def _safe_cfg_snapshot(cfg: AppConfig) -> Dict[str, Any]:
    """
    Компактный безопасный снимок конфигурации для логов.
    Защищает от AttributeError и не выводит большие структуры.
    """
    snapshot: Dict[str, Any] = {}
    try:
        snapshot["backtest"] = {
            "risk_pct": getattr(cfg.backtest, "risk_per_position_pct", None),
            "pnl_stop_r": getattr(cfg.backtest, "pnl_stop_loss_r_multiple", None),
            "pair_step_r_limit": getattr(cfg.backtest, "pair_step_r_limit", None),
            "z_enter": getattr(cfg.backtest, "zscore_entry_threshold", None),
            "z_exit": getattr(cfg.backtest, "z_exit", None),
            "rolling_window": getattr(cfg.backtest, "rolling_window", None),
            "time_stop_multiplier": getattr(cfg.backtest, "time_stop_multiplier", None),
            "commission": getattr(cfg.backtest, "commission_rate_per_leg", None),
            "slippage": getattr(cfg.backtest, "slippage_pct", None),
            "use_kelly": getattr(cfg.backtest, "use_kelly_sizing", None),
            "kelly_f_max": getattr(cfg.backtest, "max_kelly_fraction", None),
            "pnl_stop_usd": getattr(cfg.backtest, "pair_stop_loss_usd", None),
            "enable_pair_quarantine": getattr(cfg.backtest, "enable_pair_quarantine", None),
            "quarantine_sigma": getattr(cfg.backtest, "quarantine_pnl_threshold_sigma", None),
            "quarantine_dd": getattr(cfg.backtest, "quarantine_drawdown_threshold_pct", None),
            "quarantine_days": getattr(cfg.backtest, "quarantine_period_days", None),
        }
    except Exception:
        snapshot["backtest"] = "error"
    try:
        snapshot["portfolio"] = {
            "risk_pct": getattr(cfg.portfolio, "risk_per_position_pct", None),
            "max_active_positions": getattr(cfg.portfolio, "max_active_positions", None),
            "initial_capital": getattr(cfg.portfolio, "initial_capital", None),
        }
    except Exception:
        snapshot["portfolio"] = "error"
    try:
        snapshot["filters"] = {
            "min_profit": getattr(cfg.filter_params, "min_profit_potential_pct", None),
            "min_vol_usd": getattr(cfg.filter_params, "min_daily_volume_usd", None),
            "pvalue": getattr(cfg.filter_params, "pvalue_threshold", None),
            "hurst": getattr(cfg.filter_params, "use_hurst_filter", None),
            "max_hurst": getattr(cfg.filter_params, "max_hurst_exponent", None),
            "min_cross": getattr(cfg.filter_params, "min_mean_crossings", None),
            "min_beta": getattr(cfg.filter_params, "min_beta", None),
            "max_beta": getattr(cfg.filter_params, "max_beta", None),
            "save_reasons": getattr(cfg.filter_params, "save_filter_reasons", None),
        }
    except Exception:
        snapshot["filters"] = "error"
    try:
        pu = getattr(cfg, "pair_universe", None) or getattr(cfg.pair_selection, "pair_universe", None)
        if pu:
            if hasattr(pu, "get"):
                snapshot["pair_universe"] = {
                    "max_pairs_per_symbol": pu.get("max_pairs_per_symbol", None),
                    "max_total_pairs": pu.get("max_total_pairs", None),
                    "blacklist": pu.get("volatile_blacklist", None),
                }
            else:
                snapshot["pair_universe"] = {
                    "max_pairs_per_symbol": getattr(pu, "max_pairs_per_symbol", None),
                    "max_total_pairs": getattr(pu, "max_total_pairs", None),
                    "blacklist": getattr(pu, "volatile_blacklist", None),
                }
        else:
            snapshot["pair_universe"] = None
    except Exception:
        snapshot["pair_universe"] = "error"
    return snapshot


def _run_backtest_for_pair(
    pair_data: pd.DataFrame,
    s1: str,
    s2: str,
    cfg: AppConfig,
    capital_per_pair: float,
    bar_minutes: int,
    metrics: dict,
    period_label: str
) -> dict:
    """
    Helper function to run backtest for a single pair.
    Extracted to avoid duplication between mmap and standard processing.
    """
    from coint2.utils.logger import get_logger
    logger = get_logger("pair_processing")
    
    try:
        logger.info(f"🔧 Starting pair {s1}-{s2}, data shape: {pair_data.shape}")
        
        # Инициализация и запуск бэктеста
        bt = NumbaPairBacktester(
            pair_data=pair_data,
            rolling_window=getattr(cfg.backtest, 'rolling_window', 20),
            z_threshold=getattr(cfg.backtest, 'zscore_entry_threshold', 2.0),
            z_exit=getattr(cfg.backtest, 'z_exit', 0.0),
            capital_at_risk=capital_per_pair,
            stop_loss_multiplier=getattr(cfg.backtest, 'stop_loss_multiplier', 2.0),
            time_stop_multiplier=getattr(cfg.backtest, 'time_stop_multiplier', None),
            max_position_size_pct=getattr(cfg.portfolio, 'max_position_size_pct', 1.0),
            config=cfg.backtest  # CRITICAL FIX: Pass config for costs and advanced params
        )
        
        # Запуск бэктеста
        import time
        start_time = time.time()
        results = bt.run()
        elapsed = time.time() - start_time
        logger.info(f"✅ Pair {s1}-{s2} completed in {elapsed:.2f}s")
        
        # Проверяем что результат не None
        if results is None:
            logger.warning(f"Backtest returned None for pair {s1}-{s2}")
            results = {}
        
        # Извлечение результатов
        pnl_series = results.get('pnl_series', pd.Series(dtype=float))
        positions = results.get('positions', [])
        trades_log = results.get('trades', [])
        costs = results.get('costs', pd.Series(dtype=float))
        
        # Inject pair name into trades
        winning_trades = 0
        for t in trades_log:
            if isinstance(t, dict):
                t['pair'] = f'{s1}-{s2}'
                if t.get('net_pnl', 0) > 0:
                    winning_trades += 1
        
        # Vectorized calculation of pair statistics using numpy
        # PnL here is NET PnL (already processed in engine)
        
        is_valid = True
        equity_nan_count = 0
        returns_nan_count = 0
        
        if not pnl_series.empty:
            import numpy as np
            pnl_values = pnl_series.values.astype(float)
            
            # Check for NaNs in PnL (returns)
            bad_returns_mask = ~np.isfinite(pnl_values)
            returns_nan_count = int(bad_returns_mask.sum())
            if returns_nan_count > 0:
                logger.warning(
                    "NaN/inf in PnL for pair %s-%s. Replacing bad points with 0", 
                    s1, s2
                )
                pnl_values[bad_returns_mask] = 0.0
                is_valid = False
            
            pair_pnl = float(np.sum(pnl_values))
            
            # Calculate Max Drawdown on Pair Equity
            cumulative = np.cumsum(pnl_values)
            equity = capital_per_pair + cumulative
            
            # Check for NaNs in Equity (though cleaning returns should fix this usually)
            bad_equity_mask = ~np.isfinite(equity)
            equity_nan_count = int(bad_equity_mask.sum())
            if equity_nan_count > 0:
                logger.warning(
                    "NaN/inf in equity for pair %s-%s. Replacing bad points with capital", 
                    s1, s2
                )
                equity[bad_equity_mask] = capital_per_pair
                is_valid = False
            
            # Use safe_max_dd
            max_dd_pct, _ = safe_max_dd(equity)
            
            win_days = int(np.sum(pnl_values > 0))
            lose_days = int(np.sum(pnl_values < 0))
            max_daily_gain = float(np.max(pnl_values)) if len(pnl_values) > 0 else 0.0
            max_daily_loss = float(np.min(pnl_values)) if len(pnl_values) > 0 else 0.0
            
            # Update pnl_series with cleaned values
            if not is_valid:
                 pnl_series = pd.Series(pnl_values, index=pnl_series.index)
                 
        else:
            pair_pnl = 0.0
            max_dd_pct = 0.0
            win_days = lose_days = 0
            max_daily_gain = max_daily_loss = 0.0
        
        # CRITICAL FIX: Правильный расчет издержек с обработкой NaN
        if not costs.empty:
            costs_clean = costs.fillna(0)  # Заменяем NaN на 0
            pair_costs = float(np.sum(costs_clean.values))
        else:
            pair_costs = 0.0
            
        pair_gross_pnl = pair_pnl + pair_costs
            
        actual_trade_count = len(trades_log)
        
        # Calculate avg notional
        total_notional = sum(t.get('notional', 0.0) for t in trades_log)
        avg_notional = safe_div(total_notional, actual_trade_count, 0.0)
        
        trade_stat = {
            'pair': f'{s1}-{s2}',
            'period': period_label,
            'total_pnl': pair_pnl, # Net PnL
            'total_gross_pnl': pair_gross_pnl,
            'total_costs': pair_costs,
            'trade_count': actual_trade_count,
            'avg_pnl_per_trade': safe_div(pair_pnl, actual_trade_count, 0.0),
            'avg_notional': avg_notional,
            'win_trades': winning_trades,
            'win_days': win_days,
            'lose_days': lose_days,
            'total_days': len(pnl_series),
            'max_daily_gain': max_daily_gain,
            'max_daily_loss': max_daily_loss,
            'max_dd_pct': max_dd_pct,
            'is_valid': is_valid,
            'equity_nan_count': equity_nan_count,
            'returns_nan_count': returns_nan_count
        }
        
        # Логирование результатов обработки
        logger.debug(f"✅ Пара {s1}-{s2} обработана успешно: PnL={trade_stat['total_pnl']:.2f}, сделок={trade_stat['trade_count']}")
        
        return {
            'pnl_series': pnl_series,
            'trades_log': trades_log,
            'trade_stat': trade_stat,
            'success': True
        }
        
    except Exception as e:
        # Логирование ошибки обработки
        import traceback
        tb = traceback.format_exc()
        logger.warning(f"❌ Ошибка обработки пары {s1}-{s2}: {str(e)}")
        logger.debug(f"Traceback: {tb}")
        
        # Return empty results for failed pairs
        return {
            'pnl_series': pd.Series(dtype=float),
            'trades_log': [],
            'trade_stat': {
                'pair': f'{s1}-{s2}',
                'period': period_label,
                'total_pnl': 0.0,
                'total_costs': 0.0,
                'trade_count': 0,
                'avg_pnl_per_trade': 0.0,
                'win_days': 0,
                'lose_days': 0,
                'total_days': 0,
                'max_daily_gain': 0.0,
                'max_daily_loss': 0.0,
                'error': str(e)
            },
            'success': False,
            'error': str(e)
        }


def process_single_pair_mmap(
    pair_symbols,
    testing_start,
    testing_end,
    cfg,
    capital_per_pair,
    bar_minutes,
    period_label,
    pair_stats=None,
    training_normalization_params=None,
):
    """
    Process a single pair using memory-mapped data.
    """
    from coint2.core.memory_optimization import setup_blas_threading_limits
    setup_blas_threading_limits(num_threads=1, verbose=False)

    s1, s2 = pair_symbols
    logger = get_logger("pair_processing")

    try:
        pair_data_view = get_price_data_view([s1, s2], testing_start, testing_end)
        if pair_data_view is None or pair_data_view.empty:
            return {
                'success': False,
                'error': 'Empty data view',
                'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0},
                'pnl_series': pd.Series(dtype=float),
                'trades_log': []
            }

        if s1 not in pair_data_view.columns or s2 not in pair_data_view.columns:
            return {
                'success': False,
                'error': f'Missing data for symbols {s1} or {s2}',
                'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0},
                'pnl_series': pd.Series(dtype=float),
                'trades_log': []
            }

        pair_data = pair_data_view[[s1, s2]].dropna()
        if pair_data.empty:
            return {
                'success': False,
                'error': 'No valid data after filtering',
                'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0},
                'pnl_series': pd.Series(dtype=float),
                'trades_log': []
            }

        metrics = {}
        if pair_stats and len(pair_stats) >= 4:
            metrics = pair_stats[3] or {}

        return _run_backtest_for_pair(
            pair_data=pair_data,
            s1=s1,
            s2=s2,
            cfg=cfg,
            capital_per_pair=capital_per_pair,
            bar_minutes=bar_minutes,
            metrics=metrics,
            period_label=period_label
        )

    except Exception as e:
        logger.warning(f"❌ Ошибка обработки пары {s1}-{s2}: {str(e)}")
        return {
            'pnl_series': pd.Series(dtype=float),
            'trades_log': [],
            'trade_stat': {
                'pair': f'{s1}-{s2}',
                'period': period_label,
                'total_pnl': 0.0,
                'total_costs': 0.0,
                'trade_count': 0,
                'avg_pnl_per_trade': 0.0,
                'win_days': 0,
                'lose_days': 0,
                'total_days': 0,
                'max_daily_gain': 0.0,
                'max_daily_loss': 0.0,
                'error': str(e)
            },
            'success': False,
            'error': str(e)
        }


def process_single_pair(
    pair_data_tuple,
    step_df,
    testing_start,
    testing_end,
    cfg,
    capital_per_pair,
    bar_minutes,
    period_label,
    training_normalization_params=None,
):
    """
    Process a single pair for parallel execution.
    """
    from coint2.core.memory_optimization import setup_blas_threading_limits
    setup_blas_threading_limits(num_threads=1, verbose=False)

    logger = get_logger(__name__)

    try:
        s1 = pair_data_tuple[0]
        s2 = pair_data_tuple[1]
        metrics = pair_data_tuple[5] if len(pair_data_tuple) > 5 else {}
    except Exception as e:
        return {
            'pnl_series': pd.Series(dtype=float),
            'trades_log': [],
            'trade_stat': {
                'pair': 'Unknown',
                'period': period_label,
                'total_pnl': 0.0,
                'error': f'Invalid pair tuple: {e}'
            },
            'success': False,
            'error': f'Invalid pair tuple: {e}'
        }

    try:
        pair_data = step_df.loc[testing_start:testing_end, [s1, s2]].dropna()
        if pair_data.empty:
            return {
                'success': False,
                'error': 'No valid data after filtering',
                'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0},
                'pnl_series': pd.Series(dtype=float),
                'trades_log': []
            }

        return _run_backtest_for_pair(
            pair_data=pair_data,
            s1=s1,
            s2=s2,
            cfg=cfg,
            capital_per_pair=capital_per_pair,
            bar_minutes=bar_minutes,
            metrics=metrics,
            period_label=period_label
        )

    except Exception as e:
        logger.warning(f"❌ Ошибка обработки пары {s1}-{s2}: {str(e)}")
        return {
            'pnl_series': pd.Series(dtype=float),
            'trades_log': [],
            'trade_stat': {
                'pair': f'{s1}-{s2}',
                'period': period_label,
                'total_pnl': 0.0,
                'total_costs': 0.0,
                'trade_count': 0,
                'avg_pnl_per_trade': 0.0,
                'win_days': 0,
                'lose_days': 0,
                'total_days': 0,
                'max_daily_gain': 0.0,
                'max_daily_loss': 0.0
            },
            'success': False,
            'error': str(e)
        }


def _simulate_realistic_portfolio(all_pnls, cfg):
    """
    Simulate a portfolio with a max_active_positions constraint.
    """
    if not all_pnls:
        return pd.Series(dtype=float)

    pnl_df = pd.concat(
        {f'pair_{i}': pnl.fillna(0) for i, pnl in enumerate(all_pnls)},
        axis=1
    ).fillna(0)

    signals_df = (pnl_df != 0).astype(int)

    max_positions = 1
    if getattr(cfg, 'portfolio', None) is not None:
        max_positions = int(getattr(cfg.portfolio, 'max_active_positions', 1))
    if max_positions <= 0:
        return pd.Series(0.0, index=pnl_df.index)

    portfolio_pnl = pd.Series(0.0, index=pnl_df.index)
    active_positions = {}

    for timestamp in pnl_df.index:
        current_signals = signals_df.loc[timestamp]
        current_pnls = pnl_df.loc[timestamp]

        for pair_name in list(active_positions.keys()):
            if current_signals[pair_name] == 0:
                del active_positions[pair_name]

        new_signals = []
        for pair_name in current_signals.index:
            if current_signals[pair_name] == 1 and pair_name not in active_positions:
                new_signals.append((pair_name, abs(current_pnls[pair_name])))

        new_signals.sort(key=lambda x: x[1], reverse=True)
        available_slots = max_positions - len(active_positions)
        if available_slots > 0:
            for pair_name, _ in new_signals[:available_slots]:
                active_positions[pair_name] = timestamp

        step_pnl = 0.0
        for pair_name in active_positions:
            step_pnl += float(current_pnls[pair_name])

        portfolio_pnl[timestamp] = step_pnl

    return portfolio_pnl


def select_pairs(
    price_df: pd.DataFrame,
    cfg: AppConfig,
    top_n: int = 1000,
    quarantine_manager: Optional[QuarantineManager] = None,
    current_date: Optional[pd.Timestamp] = None,
    volume_df: Optional[pd.DataFrame] = None, # NEW: Optional volume/turnover data
    runtime_blacklist_pairs: Optional[set] = None, # NEW
    runtime_blacklist_symbols: Optional[set] = None # NEW
) -> List[Tuple[str, str]]:
    """
    Select pairs using SSD and filtering.
    """
    logger = get_logger("pair_selection")
    
    # 1. Calculate SSD
    # Normalize (min-max or similar)
    # Simple normalization: (price - min) / (max - min)
    normalized = (price_df - price_df.min()) / (price_df.max() - price_df.min())
    # Handle constants/NaNs
    normalized = normalized.dropna(axis=1, how='all').fillna(0)
    
    if normalized.empty or len(normalized.columns) < 2:
        return []
        
    # Calculate SSD
    # Use ssd_top_n from config if available, else default to top_n * 5
    ssd_top_k = getattr(cfg.pair_selection, 'ssd_top_n', top_n * 5)
    ssd_series = calculate_ssd(normalized, top_k=ssd_top_k)
    
    # Extract pairs
    candidates = []
    
    for (s1, s2) in ssd_series.index:
        candidates.append((s1, s2))
        
    logger.info(f"[ENHANCED SCREENING] Thresholds: "
                f"pvalue={getattr(cfg.filter_params, 'pvalue_threshold', 0.05):.3f}, "
                f"min_beta={getattr(cfg.filter_params, 'min_beta', 0.05):.2f}, "
                f"max_beta={getattr(cfg.filter_params, 'max_beta', 20.0):.2f}, "
                f"max_half_life={getattr(cfg.filter_params, 'max_half_life_days', 60.0):.1f}d, "
                f"hurst={getattr(cfg.filter_params, 'max_hurst_exponent', 0.5):.2f}, "
                f"use_hurst={getattr(cfg.filter_params, 'use_hurst_filter', False)}, "
                f"use_kpss={getattr(cfg.filter_params, 'use_kpss_filter', False)}, "
                f"min_vol=${getattr(cfg.filter_params, 'min_daily_volume_usd', 50000):.0f}")

    # 2. Apply filters (enhanced_pair_screening)
    # UNIFICATION: Use ONLY filter_params as the single source of truth for Enhanced Screening
    pvalue_threshold = getattr(cfg.filter_params, 'pvalue_threshold', 0.05)
    
    # Determine max_half_life_bars
    max_half_life_bars = getattr(cfg.filter_params, 'max_half_life_bars', None)
    if max_half_life_bars is None:
        max_half_life_days = getattr(cfg.filter_params, 'max_half_life_days', 60.0)
        # Assuming 15min bars (96 bars/day)
        max_half_life_bars = int(max_half_life_days * 96)
    
    min_daily_volume_usd = getattr(cfg.filter_params, 'min_daily_volume_usd', 50000.0)
    
    max_hurst_exponent = getattr(cfg.filter_params, 'max_hurst_exponent', 0.5)
        
    # Extract other filter params from unified source
    min_mean_crossings = getattr(cfg.filter_params, 'min_mean_crossings', 6)
    
    min_beta = getattr(cfg.filter_params, 'min_beta', 0.05)
    max_beta = getattr(cfg.filter_params, 'max_beta', 20.0)

    # NEW: Exotic filter flags - also from filter_params for consistency
    use_kpss_filter = getattr(cfg.filter_params, 'use_kpss_filter', False)
    use_hurst_filter = getattr(cfg.filter_params, 'use_hurst_filter', False)
    
    # NEW: Profit Potential Filter
    min_profit_potential_pct = getattr(cfg.filter_params, 'min_profit_potential_pct', 0.004)
    
    # NEW: Same-base stablecoin exclusion filter
    exclude_same_base_stables = getattr(cfg.filter_params, 'exclude_same_base_stables', False)
    stablecoins = getattr(cfg.filter_params, 'stablecoins', ["USDT", "USDC", "DAI", "FDUSD", "TUSD", "BUSD", "PAX", "USD"])

    # NEW: Whitelist/Blacklist Filtering
    allowed_quotes = getattr(cfg.pair_selection, 'allowed_quotes', ['USDT', 'USDC', 'DAI'])
    blocked_assets = getattr(cfg.pair_selection, 'blocked_assets', ['BRZ', 'EUR', 'METH'])
    
    # Merge with volatile_blacklist if present
    # Check both top-level (new) and nested (legacy) pair_universe
    pair_universe = getattr(cfg, 'pair_universe', None) or getattr(cfg.pair_selection, 'pair_universe', None)
    
    if pair_universe:
         if hasattr(pair_universe, 'get'): # Dictionary
              volatile_blacklist = pair_universe.get('volatile_blacklist', [])
              if volatile_blacklist:
                  blocked_assets = list(set(blocked_assets + volatile_blacklist))
         elif hasattr(pair_universe, 'volatile_blacklist'): # Object
              volatile_blacklist = pair_universe.volatile_blacklist
              if volatile_blacklist:
                  blocked_assets = list(set(blocked_assets + volatile_blacklist))

    filtered_candidates = []
    for s1, s2 in candidates:
        # Check runtime blacklist
        if runtime_blacklist_pairs and f"{s1}-{s2}" in runtime_blacklist_pairs:
             continue
        if runtime_blacklist_symbols and (s1 in runtime_blacklist_symbols or s2 in runtime_blacklist_symbols):
             continue

        # Check quarantine
        if quarantine_manager and current_date:
            pair_str = f"{s1}-{s2}"
            if quarantine_manager.is_quarantined(pair_str, current_date):
                continue

        # Check blacklist (moved to apply_filters but pre-filtering candidates here is also fine for speed)
        # Actually apply_filters handles this now with `blacklist` param, so we can skip it here or keep it.
        # Keeping it here is faster as it avoids loading price data for blacklisted pairs.
        if any(b in s1 or b in s2 for b in blocked_assets):
            continue
            
        # Check whitelist (at least one quote asset must be allowed, or both? usually both should be valid)
        # Let's assume we want pairs where quote currency is in allowed list.
        # Heuristic: check if symbol ends with allowed quote.
        
        # Strict mode: quote asset MUST be in allowed_quotes
        s1_ok = any(s1.endswith(q) for q in allowed_quotes)
        s2_ok = any(s2.endswith(q) for q in allowed_quotes)
        
        if s1_ok and s2_ok:
            filtered_candidates.append((s1, s2))
    
    # NEW: Filter out same-base stablecoin pairs if enabled
    if exclude_same_base_stables:
        def parse_symbol(symbol):
            """Parse symbol into base and quote. Simple heuristic."""
            for stable in sorted(stablecoins, key=len, reverse=True):  # Try longer stablecoins first
                if symbol.endswith(stable):
                    base = symbol[:-len(stable)]
                    quote = stable
                    return base, quote
            # If no stablecoin found, assume last 3-4 chars are quote
            if len(symbol) > 6:
                base, quote = symbol[:-4], symbol[-4:]
            elif len(symbol) > 3:
                base, quote = symbol[:-3], symbol[-3:]
            else:
                base, quote = symbol, ""
            return base, quote
        
        pre_filter_count = len(filtered_candidates)
        same_base_excluded = []
        
        for s1, s2 in filtered_candidates:
            base1, quote1 = parse_symbol(s1)
            base2, quote2 = parse_symbol(s2)
            
            # Exclude if same base asset and both quotes are stablecoins
            if (base1 == base2 and quote1 in stablecoins and quote2 in stablecoins):
                same_base_excluded.append((s1, s2))
        
        # Remove excluded pairs
        filtered_candidates = [(s1, s2) for s1, s2 in filtered_candidates if (s1, s2) not in same_base_excluded]
        
        logger.info(f"[SAME_BASE_FILTER] Excluded {len(same_base_excluded)} same-base stablecoin pairs "
                   f"({pre_filter_count} -> {len(filtered_candidates)})")
        if same_base_excluded[:5]:  # Show first 5 examples
            examples = ", ".join([f"{s1}-{s2}" for s1, s2 in same_base_excluded[:5]])
            logger.info(f"[SAME_BASE_FILTER] Examples excluded: {examples}")
            
    candidates = filtered_candidates

    screened = apply_filters(
        candidates,
        price_df,
        pvalue_threshold=pvalue_threshold,
        max_half_life_bars=max_half_life_bars,
        min_daily_volume_usd=min_daily_volume_usd,
        max_hurst_exponent=max_hurst_exponent,
        use_kpss_filter=use_kpss_filter,
        use_hurst_filter=use_hurst_filter,
        min_mean_crossings=min_mean_crossings,
        min_beta=min_beta,
        max_beta=max_beta,
        min_profit_potential_pct=min_profit_potential_pct,
        blacklist=blocked_assets, # Pass aggregated blacklist
        stable_tokens=stablecoins,
        volume_df=volume_df, # NEW: Pass volume data
        config=cfg # Pass config for train filter
    )

    # Top-N по train edge: сортировка по train_cum_pnl_r, затем g2c
    def _edge_key(item):
        _, _, _, _, _, metrics = item
        return (
            float(metrics.get("train_cum_pnl_r", 0.0)),
            float(metrics.get("train_gross_to_cost_ratio", 0.0))
        )
    top_edge_limit = min(top_n, 15)
    screened_sorted = sorted(screened, key=_edge_key, reverse=True)
    if screened:
        examples = ", ".join([
            f"{a}-{b}(CumR={m.get('train_cum_pnl_r',0.0):.2f},G2C={m.get('train_gross_to_cost_ratio',0.0):.2f})"
            for a,b,_,_,_,m in screened_sorted[:5]
        ])
        logger.info(f"[EDGE_TOP] total_after_filters={len(screened_sorted)}, taking_top={top_edge_limit}, examples: {examples}")
    return [(s1, s2) for s1, s2, _, _, _, _ in screened_sorted][:top_edge_limit]


def _filter_max_pairs_per_symbol(pairs: List[Tuple[str, str]], max_pairs: int) -> List[Tuple[str, str]]:
    """
    Post-filter to limit exposure to any single symbol.
    Keeps the first occurrences (assuming pairs are already sorted by quality/score).
    """
    if max_pairs <= 0:
        return pairs
        
    from collections import Counter
    symbol_counts = Counter()
    filtered = []
    
    for s1, s2 in pairs:
        # Check if adding this pair would exceed limits
        if symbol_counts[s1] < max_pairs and symbol_counts[s2] < max_pairs:
            filtered.append((s1, s2))
            symbol_counts[s1] += 1
            symbol_counts[s2] += 1
            
    return filtered


def _filter_global_limit(pairs: List[Tuple[str, str]], max_total_pairs: int) -> List[Tuple[str, str]]:
    """
    Global limit on total number of pairs.
    Assumes pairs are already sorted by score/quality (SSD).
    """
    if max_total_pairs <= 0:
        return pairs
    return pairs[:max_total_pairs]


def run_walk_forward(cfg: AppConfig, price_df: pd.DataFrame = None, volume_df: pd.DataFrame = None) -> Dict[str, Any]:
    logger = get_logger("walk_forward")
    logger.info("Starting Walk-Forward Analysis")
    
    start_time = time.time()
    
    # --- PROACTIVE CONFIG OPTIMIZATION ---
    # Override parameters to improve performance based on log analysis (High Commissions & Tight Stops)
    disable_opt = os.environ.get("COINT_DISABLE_WF_OPTIMIZATIONS", "").strip().lower() in {
        "1", "true", "yes"
    }
    if disable_opt:
        logger.info("Skipping optimized parameter overrides due to COINT_DISABLE_WF_OPTIMIZATIONS.")
    else:
        logger.info("⚡ APPLYING OPTIMIZED PARAMETERS to increase Profit & Sharpe")
        
        # 1. Стоп-лоссы и риск
        if hasattr(cfg, 'backtest'):
            try:
                # Жесткий лимит на шаг: 1R
                cfg.backtest.pair_step_r_limit = 1.0
            except Exception:
                pass
            try:
                # PnL стоп в R 1.0
                cfg.backtest.pnl_stop_loss_r_multiple = 1.0
            except Exception:
                pass
            try:
                # Усиливаем карантин: sigma 1.0, DD 2%
                cfg.backtest.quarantine_pnl_threshold_sigma = 1.0
                cfg.backtest.quarantine_drawdown_threshold_pct = 0.02
            except Exception:
                pass
                 
            # 2. Вход/выход: раньше, но выход с буфером
            try:
                cfg.backtest.zscore_entry_threshold = 2.5
            except Exception:
                pass
            try:
                cfg.backtest.z_exit = 0.25
            except Exception:
                pass

            # 3. Риск на позицию: 0.3% диагностика
            if hasattr(cfg.backtest, 'risk_per_position_pct'):
                cfg.backtest.risk_per_position_pct = 0.003

        if hasattr(cfg, 'portfolio') and hasattr(cfg.portfolio, 'risk_per_position_pct'):
            cfg.portfolio.risk_per_position_pct = 0.003
            
        # 4. Фильтры качества
        if hasattr(cfg, 'filter_params'):
            try:
                cfg.filter_params.min_profit_potential_pct = 0.025
            except Exception:
                pass
            try:
                current_vol = getattr(cfg.filter_params, 'min_daily_volume_usd', 0)
                cfg.filter_params.min_daily_volume_usd = max(1_200_000, current_vol)
            except Exception:
                pass
             
        # 5. Стабильность сигналов: длиннее окно z-score, если доступно
        if hasattr(cfg, 'backtest'):
            try:
                cfg.backtest.rolling_window = max(getattr(cfg.backtest, 'rolling_window', 30), 30)
            except Exception:
                pass
            try:
                # Быстрее фиксируем неработающие сделки во времени
                current_time_stop = getattr(cfg.backtest, 'time_stop_multiplier', 3.0)
                cfg.backtest.time_stop_multiplier = min(current_time_stop, 2.0)
            except Exception:
                pass
    # -------------------------------------
    # Log compact config snapshot after overrides
    try:
        cfg_snapshot = _safe_cfg_snapshot(cfg)
        logger.info("CONFIG_SNAPSHOT %s", cfg_snapshot)
    except Exception as e:
        logger.warning("Config snapshot logging failed: %s", e)
    
    # Define dates for WFA loop
    start_date = pd.Timestamp(cfg.walk_forward.start_date)
    end_date = pd.Timestamp(cfg.walk_forward.end_date)
    train_days = cfg.walk_forward.training_period_days
    test_days = cfg.walk_forward.testing_period_days
    
    # 1. Load Data if not provided
    if price_df is None:
        logger.info(f"Loading data from {cfg.data_dir}")
        data_handler = DataHandler(cfg)
        
        # Calculate period with buffer
        total_days = (end_date - start_date).days + train_days + 10  # +10 дней запас
        load_start = end_date - pd.Timedelta(days=total_days)
        
        # Use load_data_range to potentially get volume/turnover as well
        df_result = data_handler.load_data_range(load_start, end_date)
        
        if isinstance(df_result.columns, pd.MultiIndex):
             level_zero = df_result.columns.get_level_values(0)
             has_close = "close" in level_zero
             has_turnover = "turnover" in level_zero

             if has_close:
                 price_df = df_result["close"]
             else:
                 logger.warning("Could not extract close from MultiIndex. Using all as price.")
                 price_df = df_result

             if has_turnover:
                 volume_df = df_result["turnover"]
             else:
                 volume_df = None
        else:
             price_df = df_result
    
    if price_df is None or price_df.empty:
        logger.error("No data loaded")
        return {'success': False, 'error': 'No data'}
        
    logger.info(f"Loaded data: {price_df.shape}")
    if volume_df is not None:
        logger.info(f"Loaded volume data: {volume_df.shape}")
    
    # 2. Generate Steps
    
    # INFO: Log WF Config
    logger.info(f"WF_CONFIG "
                f"kelly={getattr(cfg.backtest, 'use_kelly_sizing', True)} "
                f"risk_pct={getattr(cfg.backtest, 'risk_per_position_pct', None) or getattr(cfg.portfolio, 'risk_per_position_pct', 0.0):.4f} "
                f"stop_R={getattr(cfg.backtest, 'pnl_stop_loss_r_multiple', 0.0):.2f} "
                f"stop_USD={getattr(cfg.backtest, 'pair_stop_loss_usd', 'None')} "
                f"min_profit={getattr(cfg.filter_params, 'min_profit_potential_pct', 0.0):.4f} "
                f"min_vol={getattr(cfg.filter_params, 'min_daily_volume_usd', 0.0):.0f} "
                f"hurst={getattr(cfg.filter_params, 'use_hurst_filter', False)} "
                f"max_hurst={getattr(cfg.filter_params, 'max_hurst_exponent', 0.0):.2f} "
                f"min_cross={getattr(cfg.filter_params, 'min_mean_crossings', 0)} "
                f"coint_p={getattr(cfg.filter_params, 'pvalue_threshold', 0.0):.3f} "
                f"z_enter={getattr(cfg.backtest, 'zscore_entry_threshold', 0):.2f} "
                f"z_exit={getattr(cfg.backtest, 'z_exit', 0):.2f} "
                f"comm={getattr(cfg.backtest, 'commission_rate_per_leg', 0):.4f} "
                f"slip={getattr(cfg.backtest, 'slippage_pct', 0):.4f}")

    # Explicit Risk Config Log (Detailed)
    risk_val_pair_step = getattr(cfg.backtest, 'pair_step_r_limit', -3.0)
    risk_config_log = {
        "risk_per_position_pct": getattr(cfg.backtest, 'risk_per_position_pct', None) or getattr(cfg.portfolio, 'risk_per_position_pct', 0.0),
        "pnl_stop_loss_r_multiple": getattr(cfg.backtest, 'pnl_stop_loss_r_multiple', 0.0),
        "pair_stop_loss_usd": getattr(cfg.backtest, 'pair_stop_loss_usd', None),
        "max_negative_pair_step_r": abs(float(risk_val_pair_step)) if risk_val_pair_step is not None else 3.0,
        "quarantine": {
             "enabled": getattr(cfg.backtest, 'enable_pair_quarantine', False),
             "sigma_threshold": getattr(cfg.backtest, 'quarantine_pnl_threshold_sigma', 2.0),
             "dd_threshold": getattr(cfg.backtest, 'quarantine_drawdown_threshold_pct', 0.05),
             "duration_days": getattr(cfg.backtest, 'quarantine_period_days', 7)
        }
    }
    logger.info(f"[RISK_CONFIG_FINAL] {risk_config_log}")

    current_test_start = start_date
    
    # Auto-Blacklist Tracking (Runtime)
    pair_cum_pnl_R: Dict[str, float] = {}
    pair_total_r_over_steps: Dict[str, float] = {}
    pair_total_trades: Dict[str, int] = {}
    times_in_worst_pairs: Dict[str, int] = {}
    symbol_cum_pnl_R: Dict[str, float] = {}
    pair_trade_count: Dict[str, int] = {}
    runtime_blacklist_pairs: set = set()
    runtime_blacklist_symbols: set = set()

    all_results = {
        'trades': [],
        'pnl_series': pd.Series(dtype=float),
        'stats': []
    }
    
    # If data starts after start_date, adjust
    if price_df.index[0] > start_date:
        logger.warning(f"Data starts at {price_df.index[0]}, adjusting start date")
        current_test_start = max(start_date, price_df.index[0] + pd.Timedelta(days=train_days))
    
    step_counter = 0
    # NEW: Respect max_steps from config if provided, otherwise unlimited
    max_steps = getattr(cfg.walk_forward, 'max_steps', None)
    # If max_steps is 0 or less, treat as unlimited
    if max_steps is not None and max_steps <= 0:
        max_steps = None
        
    step_size_days = getattr(cfg.walk_forward, 'step_size_days', None)
    
    # If step_size is not provided, use testing_period_days (default behavior)
    actual_step_size = step_size_days if step_size_days is not None else test_days
    
    # Initialize Quarantine Manager
    quarantine_manager = QuarantineManager(cfg)
    
    while current_test_start + pd.Timedelta(days=test_days) <= end_date:
        if max_steps is not None and step_counter >= max_steps:
            logger.info(f"Reached max_steps ({max_steps}), stopping Walk-Forward.")
            break
            
        step_counter += 1
        test_end = current_test_start + pd.Timedelta(days=test_days)
        train_start = current_test_start - pd.Timedelta(days=train_days)
        
        if train_start < price_df.index[0]:
            logger.warning(f"Training start {train_start} before data start {price_df.index[0]}, skipping")
            current_test_start += pd.Timedelta(days=actual_step_size)
            continue
            
        train_end = current_test_start
        
        logger.info(f"Processing step {step_counter}: Train [{train_start} - {train_end}], Test [{train_end} - {test_end}]")
        
        # Slice data
        train_data = price_df[train_start:train_end]
        test_data = price_df[train_end:test_end]
        
        # Slice volume data if available
        train_volume = None
        if volume_df is not None:
            train_volume = volume_df[train_start:train_end]
        
        if train_data.empty or test_data.empty:
            logger.warning("Empty data for step")
            current_test_start += pd.Timedelta(days=actual_step_size)
            continue
            
        # Select pairs on training data
        selected_pairs = select_pairs(
            train_data, 
            cfg,
            quarantine_manager=quarantine_manager,
            current_date=train_end, # Use train_end (start of test) as current date
            volume_df=train_volume, # Pass sliced volume data
            runtime_blacklist_pairs=runtime_blacklist_pairs,
            runtime_blacklist_symbols=runtime_blacklist_symbols
        )
        
        # POST-FILTER: Max pairs per symbol
        # Check both top-level (new) and nested (legacy) pair_universe
        pair_universe = getattr(cfg, 'pair_universe', None) or getattr(cfg.pair_selection, 'pair_universe', None)
        
        max_pairs = 4
        if pair_universe:
             if hasattr(pair_universe, 'get'):
                 max_pairs = pair_universe.get('max_pairs_per_symbol', 4)
             elif hasattr(pair_universe, 'max_pairs_per_symbol'):
                 max_pairs = pair_universe.max_pairs_per_symbol
        
        pre_limit_count = len(selected_pairs)
        selected_pairs = _filter_max_pairs_per_symbol(selected_pairs, max_pairs)
        if len(selected_pairs) < pre_limit_count:
            logger.info(f"[MAX_PAIRS_FILTER] Reduced from {pre_limit_count} to {len(selected_pairs)} pairs (limit {max_pairs} per symbol)")

        # NEW: Global Limit Filter
        # Try to get max_total_pairs from top-level pair_universe (preferred)
        max_total_pairs = 1000
        if getattr(cfg, 'pair_universe', None) and hasattr(cfg.pair_universe, 'max_total_pairs'):
             max_total_pairs = cfg.pair_universe.max_total_pairs
        elif getattr(cfg.pair_selection, 'pair_universe', None) and hasattr(cfg.pair_selection.pair_universe, 'max_total_pairs'):
             max_total_pairs = cfg.pair_selection.pair_universe.max_total_pairs
        elif getattr(cfg.pair_selection, 'pair_universe', None) and hasattr(cfg.pair_selection.pair_universe, 'get'):
             max_total_pairs = cfg.pair_selection.pair_universe.get('max_total_pairs', 1000)
             
        if max_total_pairs is None: max_total_pairs = 1000
        
        pre_global_limit = len(selected_pairs)
        selected_pairs = _filter_global_limit(selected_pairs, max_total_pairs)
        if len(selected_pairs) < pre_global_limit:
            logger.info(f"[GLOBAL_LIMIT_FILTER] Reduced from {pre_global_limit} to {len(selected_pairs)} pairs (max_total_pairs {max_total_pairs})")

        logger.info(f"Selected {len(selected_pairs)} pairs")
        
        if not selected_pairs:
            current_test_start += pd.Timedelta(days=actual_step_size)
            continue
            
        # Run backtest on test data
        logger.info(f"🔄 Starting backtest for {len(selected_pairs)} pairs...")
        
        # FIX: Calculate capital per pair based on max_active_positions
        # If max_active_positions=20, and initial_capital=10000, then capital_per_pair=500
        initial_capital = getattr(cfg.portfolio, 'initial_capital', 10000.0)
        max_positions = getattr(cfg.portfolio, 'max_active_positions', 20)
        capital_per_pair = initial_capital / max(1, max_positions)
        
        # NEW: Sync trade limits from global config to backtest config
        if cfg.trade_limits:
             if hasattr(cfg.backtest, 'max_round_trips_per_pair_step'):
                 cfg.backtest.max_round_trips_per_pair_step = cfg.trade_limits.max_round_trips_per_pair_step
             if hasattr(cfg.backtest, 'max_new_entries_per_pair_day'):
                 cfg.backtest.max_new_entries_per_pair_day = cfg.trade_limits.max_new_entries_per_pair_day

        # NEW: Sync risk limits from global config to backtest config
        if cfg.risk_limits:
             # Inject risk_limits dict into backtest config wrapper if possible,
             # or just set the attribute dynamically (Pydantic models might block this unless extra='allow')
             # Since we can't easily modify the Pydantic model instance structure if it's strict,
             # we can rely on the fact that NumbaPairBacktester accepts a config object.
             # We can monkey-patch it or use a wrapper.
             # But wait, NumbaPairBacktester reads 'pair_step_r_limit' from config.
             # We can just map it here!
             if hasattr(cfg.risk_limits, 'pair_step_r_multiple'):
                  # Map risk_limits.pair_step_r_multiple -> backtest.pair_step_r_limit
                  # Note: pair_step_r_limit is not defined in BacktestConfig model but might be accepted if extra='ignore' or if we set it on instance
                  # BacktestConfig defaults to extra='ignore' usually? No, default is 'ignore'.
                  # Let's try setting it.
                  try:
                      cfg.backtest.pair_step_r_limit = cfg.risk_limits.pair_step_r_multiple
                  except:
                      pass # If fails, engine will use default or legacy param

        step_results = Parallel(n_jobs=-1)(
            delayed(_run_backtest_for_pair)(
                test_data[[s1, s2]], 
                s1, s2, 
                cfg,
                capital_per_pair=capital_per_pair, # Use dynamic capital
                bar_minutes=15,
                metrics={}, 
                period_label=f"{train_end.date()}"
            )
            for s1, s2 in selected_pairs
        )
        
        logger.info(f"✅ Backtest completed for {len(selected_pairs)} pairs")
        
        # Aggregate step results
        successful_pairs = 0
        failed_pairs = 0
        total_step_pnl = 0.0
        total_step_gross = 0.0
        total_step_costs = 0.0
        total_step_trades = 0
        
        # Separate valid and invalid results
        valid_results = [res for res in step_results if res['success'] and res['trade_stat'].get('is_valid', True)]
        invalid_results = [res for res in step_results if not res['success'] or not res['trade_stat'].get('is_valid', True)]
        
        successful_pairs = len(valid_results)
        failed_pairs = len(invalid_results)
        
        if invalid_results:
             logger.warning(
                 "Step %d has %d pairs with invalid metrics (NaN/inf). Examples: %s",
                 step_counter,
                 len(invalid_results),
                 ", ".join(res['trade_stat']['pair'] for res in invalid_results[:5])
             )

        # Process results
        for res in step_results:
            if res['success']:
                all_results['trades'].extend(res['trades_log'])
                
                # Log individual trades
                exit_reasons = {}
                pnl_r_values = []
                dd_r_values = []
                
                for t in res['trades_log']:
                    if isinstance(t, dict) and 'exit_time' in t: # Only log closed trades
                        # Collect metrics
                        reason = t.get('exit_reason', 'Unknown')
                        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                        
                        if 'final_pnl_r' in t:
                             pnl_r_values.append(t['final_pnl_r'])
                        if 'max_drawdown_r' in t:
                             dd_r_values.append(t['max_drawdown_r'])
                             
                        # DEBUG: Detailed trade log
                        logger.debug(f"TRADE pair={t.get('pair')} "
                                    f"t_open={t.get('entry_time')} t_close={t.get('exit_time')} "
                                    f"hold_min={t.get('hold')} "
                                    f"z_in={t.get('entry_z', 0):.2f} z_out={t.get('exit_z', 0):.2f} "
                                    f"gross={t.get('gross_pnl', 0):.2f} costs={t.get('costs', 0):.2f} net={t.get('net_pnl', 0):.2f} "
                                    f"reason={reason} PnL_R={t.get('final_pnl_r', 0.0):.2f}R DD_R={t.get('max_drawdown_r', 0.0):.2f}R")
                
                # Calculate Aggregates
                min_pnl_r = min(pnl_r_values) if pnl_r_values else 0.0
                avg_pnl_r = sum(pnl_r_values) / len(pnl_r_values) if pnl_r_values else 0.0
                max_pnl_r = max(pnl_r_values) if pnl_r_values else 0.0
                
                min_dd_r = min(dd_r_values) if dd_r_values else 0.0
                avg_dd_r = sum(dd_r_values) / len(dd_r_values) if dd_r_values else 0.0
                max_dd_r = max(dd_r_values) if dd_r_values else 0.0
                
                exit_reasons_str = ", ".join([f"{k}={v}" for k, v in exit_reasons.items()])

                # INFO: Pair Summary
                ts = res['trade_stat']
                win_rate = safe_div(ts.get('win_trades', 0), ts['trade_count'], 0.0) * 100.0
                
                # Enhanced logging
                logger.info(f"PAIR_SUMMARY step={step_counter} pair={ts['pair']} "
                            f"trades={ts['trade_count']} "
                            f"avg_notional={ts.get('avg_notional', 0):.0f} "
                            f"gross={ts.get('total_gross_pnl', 0):.2f} costs={ts.get('total_costs', 0):.2f} net={ts['total_pnl']:.2f} "
                            f"winrate={win_rate:.1f}% "
                            f"maxDD={ts.get('max_dd_pct', 0.0):.2f}% "
                            f"valid_metrics={ts.get('is_valid', True)} "
                            f"equity_nan={ts.get('equity_nan_count', 0)} "
                            f"returns_nan={ts.get('returns_nan_count', 0)} "
                            f"PnL_R=[{min_pnl_r:.2f}, {avg_pnl_r:.2f}, {max_pnl_r:.2f}] "
                            f"DD_R=[{min_dd_r:.2f}, {avg_dd_r:.2f}, {max_dd_r:.2f}] "
                            f"CumPnL_R={sum(pnl_r_values):.2f} " # Log Cumulative PnL in R
                            f"Exits=[{exit_reasons_str}]")
                            
                # DEBUG: Check if PnLStop was triggered but final PnL is acceptable
                if 'PnLStop' in exit_reasons:
                     pnl_stop_threshold = getattr(cfg.backtest, 'pnl_stop_loss_r_multiple', 0.8)
                     # Allow some slippage beyond threshold (e.g. 20%)
                     # But if it's -3R or -5R, that's bad.
                     for val in pnl_r_values:
                         if val < -(pnl_stop_threshold * 1.5): # 50% buffer for slippage/costs
                             logger.warning(f"⚠️ [R-RISK BREACH] Pair {ts['pair']} had trade with {val:.2f}R loss (Threshold: -{pnl_stop_threshold:.2f}R)")
                             
                # DEBUG: Check if pair hit max loss for step
                cum_pnl_r = sum(pnl_r_values)
                max_loss_r_step = 3.0 # Default
                if cum_pnl_r <= -max_loss_r_step:
                     logger.warning(f"⚠️ [STEP-RISK BREACH] Pair {ts['pair']} cumulative loss {cum_pnl_r:.2f}R exceeded limit -{max_loss_r_step}R")
                
                # Aggregate PnL series for step calculation (using cleaned values)
                if all_results['pnl_series'].empty:
                    all_results['pnl_series'] = res['pnl_series']
                else:
                    all_results['pnl_series'] = all_results['pnl_series'].add(res['pnl_series'], fill_value=0)
                
                all_results['stats'].append(res['trade_stat'])
                
        # Calculate Step Aggregates safely
        gross_list = np.asarray([res['trade_stat'].get('total_gross_pnl', 0.0) for res in valid_results], dtype=float)
        net_list = np.asarray([res['trade_stat']['total_pnl'] for res in valid_results], dtype=float)
        costs_list = np.asarray([res['trade_stat'].get('total_costs', 0.0) for res in valid_results], dtype=float)
        trades_list = np.asarray([res['trade_stat']['trade_count'] for res in valid_results], dtype=int)
        
        total_step_gross = float(np.nansum(gross_list))
        total_step_pnl = float(np.nansum(net_list))
        total_step_costs = float(np.nansum(costs_list))
        total_step_trades = int(np.sum(trades_list))
        
        logger.info(f"📊 Step results: {successful_pairs} valid pairs, {failed_pairs} invalid/failed")
        logger.info(f"💰 Step PnL: {total_step_pnl:.2f} (Gross: {total_step_gross:.2f}, Costs: {total_step_costs:.2f}), Trades: {total_step_trades}")
        
        # NEW: WF Step Summary
        # Calculate stats for this step using valid results only
        step_pnl_series = pd.Series(dtype=float)
        for res in valid_results:
            if not res['pnl_series'].empty:
                if step_pnl_series.empty:
                    step_pnl_series = res['pnl_series']
                else:
                    step_pnl_series = step_pnl_series.add(res['pnl_series'], fill_value=0)
        
        # Use initial capital from config or default to 10000.0
        initial_capital = getattr(cfg.portfolio, 'initial_capital', 10000.0)
        
        # SHARPE CORRECTION: 
        # Returns are calculated on 15-min bars (rolling PnL).
        # To get annualized Sharpe, we must multiply by sqrt(bars_per_year).
        # bars_per_year = 365 * 24 * 4 = 35040
        bars_per_year = 365 * 24 * 4
        
        # Use safe_sharpe
        step_sharpe, sharpe_valid = safe_sharpe(
            step_pnl_series.fillna(0) / initial_capital, # Returns
            annualizing_factor=bars_per_year
        )
        
        # Calculate Max Drawdown
        step_cumulative = step_pnl_series.cumsum()
        if not step_cumulative.empty:
            # Add initial capital to get portfolio equity curve
            portfolio_equity = initial_capital + step_cumulative
            step_max_dd, dd_valid = safe_max_dd(portfolio_equity)
            
            # DEBUG: Log portfolio equity diagnostics
            logger.debug(f"WF_PORTFOLIO_EQUITY step={step_counter} "
                        f"initial={initial_capital:.0f} "
                        f"first_equity={portfolio_equity.iloc[0]:.2f} "
                        f"min_equity={portfolio_equity.min():.2f} "
                        f"last_equity={portfolio_equity.iloc[-1]:.2f} "
                        f"dd_pct={step_max_dd:.2f}")
        else:
            step_max_dd = 0.0
            
        # INFO: WF Step Summary
        logger.info(f"WF_STEP step={step_counter} "
                    f"train={train_start.date()}..{train_end.date()} "
                    f"test={test_data.index[0].date()}..{test_data.index[-1].date()} "
                    f"pairs={successful_pairs} trades={total_step_trades} "
                    f"gross={total_step_gross:.2f} costs={total_step_costs:.2f} net={total_step_pnl:.2f} "
                    f"sharpe={step_sharpe:.4f} maxDD={step_max_dd:.2f} "
                    f"step_valid_pairs={successful_pairs} step_invalid_pairs={failed_pairs}")
        
        if failed_pairs > 0:
            logger.info(f"invalid_pairs_examples=[{', '.join(res['trade_stat']['pair'] for res in invalid_results[:5])}]")

        # NEW: Summary PnL Statistics
        avg_per_trade = safe_div(total_step_pnl, total_step_trades, 0.0)
        comm_to_gross_ratio = safe_div(total_step_costs, abs(total_step_gross), 0.0)
        
        logger.info(f"[WF step {step_counter} SUMMARY_PNL] trades={total_step_trades} "
                    f"gross={total_step_gross:.2f} commission={total_step_costs:.2f} "
                    f"net={total_step_pnl:.2f} avg_per_trade={avg_per_trade:.2f} "
                    f"comm_to_gross_ratio={comm_to_gross_ratio:.4f}")
                    
        # NEW: Per-symbol PnL aggregation (using valid results)
        symbol_pnl = {}
        for res in valid_results:
            s1 = res['trade_stat']['pair'].split('-')[0]
            s2 = res['trade_stat']['pair'].split('-')[1]
            p_pnl = res['trade_stat']['total_pnl']
            
            # Split PnL equally
            half_pnl = p_pnl / 2.0
            symbol_pnl[s1] = symbol_pnl.get(s1, 0.0) + half_pnl
            symbol_pnl[s2] = symbol_pnl.get(s2, 0.0) + half_pnl
        
        # Sort and Log
        sorted_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1])
        worst_symbols = sorted_symbols[:5]
        best_symbols = sorted_symbols[-5:][::-1]
        
        worst_str = ", ".join([f"{s}({p:.2f})" for s, p in worst_symbols])
        best_str = ", ".join([f"{s}({p:.2f})" for s, p in best_symbols])
        
        logger.info(f"[WF step {step_counter} SUMMARY] WORST_SYMBOLS: {worst_str}")
        logger.info(f"[WF step {step_counter} SUMMARY] BEST_SYMBOLS: {best_str}")

        # NEW: Worst Pairs
        step_stats = []
        for res in valid_results:
            step_stats.append(res['trade_stat'])
        
        # Sort by PnL ascending (worst first)
        worst_pairs = sorted(step_stats, key=lambda x: x['total_pnl'])[:5]
        worst_pairs_str = ", ".join([f"{p['pair']} ({p['total_pnl']:.2f})" for p in worst_pairs])
        logger.info(f"[WF {step_counter} WORST_PAIRS] {worst_pairs_str}")
        for wp in worst_pairs:
            pid = wp['pair']
            times_in_worst_pairs[pid] = times_in_worst_pairs.get(pid, 0) + 1
        
        # Update quarantine
        # Pass current_date = test_end because we evaluate performance over the test period
        quarantine_manager.update_from_results(step_results, test_end)
        # Расширяем символ-бан для пар, ушедших в карантин
        if quarantine_manager.quarantined_pairs:
            for pair_id in list(quarantine_manager.quarantined_pairs.keys()):
                try:
                    s1, s2 = pair_id.split('-')
                    runtime_blacklist_symbols.add(s1)
                    runtime_blacklist_symbols.add(s2)
                except Exception:
                    continue

        # Доп. защита: если на шаге комиссии съели >60% gross, баним пары с отрицат. gross
        comm_gross_ratio = 0.0
        try:
            comm_gross_ratio = safe_div(total_step_costs, max(total_step_gross, 1e-9), 0.0)
        except Exception:
            comm_gross_ratio = 0.0
        if comm_gross_ratio > 0.6:
            for res in step_results:
                if not res['success']:
                    continue
                g = res['trade_stat'].get('total_gross_pnl', 0.0)
                pair_id = res['trade_stat'].get('pair')
                if g < 0 and pair_id:
                    runtime_blacklist_pairs.add(pair_id)
                    try:
                        s1, s2 = pair_id.split('-')
                        runtime_blacklist_symbols.add(s1)
                        runtime_blacklist_symbols.add(s2)
                    except Exception:
                        pass

        # Auto-Blacklist Update
        for res in step_results:
            if not res['success']: continue
            
            pair_id = res['trade_stat']['pair']
            trades = res.get('trades_log', [])
            
            # Calculate step R PnL
            step_r = 0.0
            step_trades = 0
            for t in trades:
                if isinstance(t, dict) and 'final_pnl_r' in t:
                    step_r += t['final_pnl_r']
                    step_trades += 1
            
            # Update cumulative
            pair_cum_pnl_R[pair_id] = pair_cum_pnl_R.get(pair_id, 0.0) + step_r
            pair_total_r_over_steps[pair_id] = pair_total_r_over_steps.get(pair_id, 0.0) + step_r
            pair_total_trades[pair_id] = pair_total_trades.get(pair_id, 0) + step_trades
            pair_trade_count[pair_id] = pair_trade_count.get(pair_id, 0) + step_trades
            
            # Update symbols
            try:
                s1, s2 = pair_id.split('-')
                # Assign full pair PnL to each symbol (heuristic)
                symbol_cum_pnl_R[s1] = symbol_cum_pnl_R.get(s1, 0.0) + step_r
                symbol_cum_pnl_R[s2] = symbol_cum_pnl_R.get(s2, 0.0) + step_r
            except ValueError:
                pass
            
            # Check Blacklist Rule (агрессивнее: два шага по -1R суммарно)
            if pair_cum_pnl_R[pair_id] <= -2.0 and pair_trade_count[pair_id] >= 2:
                if pair_id not in runtime_blacklist_pairs:
                    runtime_blacklist_pairs.add(pair_id)
                    s1, s2 = pair_id.split('-')
                    runtime_blacklist_symbols.add(s1)
                    runtime_blacklist_symbols.add(s2)
                    
                    logger.warning(f"⛔ [RUNTIME BLACKLIST] Blocked {pair_id} and symbols {s1}, {s2}. "
                                   f"CumPnL_R={pair_cum_pnl_R[pair_id]:.2f}, Trades={pair_trade_count[pair_id]}")

            # NEW: Symbol-level block if symbol cumulative R просел
            try:
                if s1 in symbol_cum_pnl_R and symbol_cum_pnl_R[s1] <= -1.5:
                    runtime_blacklist_symbols.add(s1)
                if s2 in symbol_cum_pnl_R and symbol_cum_pnl_R[s2] <= -1.5:
                    runtime_blacklist_symbols.add(s2)
            except Exception:
                pass

        current_test_start += pd.Timedelta(days=actual_step_size)
        
    # Final metrics
    total_pnl = float(all_results['pnl_series'].sum())
    trade_count = len(all_results['trades'])
    
    # Calculate Sharpe Ratio
    # Use initial capital from config or default to 10000.0
    initial_capital = getattr(cfg.portfolio, 'initial_capital', 10000.0)
    
    # SHARPE CORRECTION for total period
    bars_per_year = 365 * 24 * 4
    
    sharpe_ratio = sharpe_ratio_on_returns(
        all_results['pnl_series'].fillna(0), 
        capital=initial_capital, 
        annualizing_factor=bars_per_year
    )
    if np.isnan(sharpe_ratio):
        sharpe_ratio = 0.0
    
    # Calculate total gross and costs
    total_gross = 0.0
    total_costs = 0.0
    for stat in all_results['stats']:
        total_gross += stat.get('total_gross_pnl', stat.get('total_pnl', 0))
        total_costs += stat.get('total_costs', 0)
        
    # Calculate MaxDD
    cum_pnl = all_results['pnl_series'].fillna(0).cumsum()
    if not cum_pnl.empty:
        # Add initial capital to get portfolio equity curve
        portfolio_equity = initial_capital + cum_pnl
        running_max = portfolio_equity.cummax()
        drawdown = (portfolio_equity - running_max) / running_max * 100  # DD as % of peak
        max_dd = drawdown.min()
        
        # DEBUG: Log final portfolio equity diagnostics
        logger.debug(f"WF_FINAL_PORTFOLIO_EQUITY "
                    f"initial={initial_capital:.0f} "
                    f"first_equity={portfolio_equity.iloc[0]:.2f} "
                    f"min_equity={portfolio_equity.min():.2f} "
                    f"last_equity={portfolio_equity.iloc[-1]:.2f} "
                    f"dd_pct={max_dd:.2f}")
    else:
        max_dd = 0.0
    
    # Best/Worst Pairs
    best_pair = "None"
    worst_pair = "None"
    if all_results['stats']:
        sorted_stats = sorted(all_results['stats'], key=lambda x: x['total_pnl'])
        best_pair = f"{sorted_stats[-1]['pair']} ({sorted_stats[-1]['total_pnl']:.2f})"
        worst_pair = f"{sorted_stats[0]['pair']} ({sorted_stats[0]['total_pnl']:.2f})"

    # BLACKLIST candidates across WF
    blacklist_candidates = []
    for pid, total_r in pair_total_r_over_steps.items():
        worst_hits = times_in_worst_pairs.get(pid, 0)
        if worst_hits >= 2 and total_r <= -3.0:
            blacklist_candidates.append((pid, total_r, worst_hits, pair_total_trades.get(pid, 0)))
    if blacklist_candidates:
        examples = ", ".join([f"{p}(R={r:.2f},worst={w},trades={t})" for p, r, w, t in blacklist_candidates[:10]])
        logger.info(f"[BLACKLIST_CANDIDATES] count={len(blacklist_candidates)} examples: {examples}")
    else:
        logger.info("[BLACKLIST_CANDIDATES] none")

    # INFO: Final Walk-Forward Result
    logger.info(f"WF_RESULT gross={total_gross:.2f} costs={total_costs:.2f} net={total_pnl:.2f} "
                f"trades={trade_count} steps={step_counter} sharpe={sharpe_ratio:.4f} maxDD={max_dd:.2f} "
                f"best_pair={best_pair} worst_pair={worst_pair}")

    # Create basic reports for downstream usage/tests
    results_dir = Path(cfg.results_dir)
    report_pnl_series = all_results['pnl_series'].fillna(0)
    if report_pnl_series.empty:
        report_equity_series = pd.Series([initial_capital], dtype=float)
    else:
        report_equity_series = initial_capital + report_pnl_series.cumsum()

    trade_stats = all_results['stats']
    total_pairs_traded = len({stat.get('pair') for stat in trade_stats if stat.get('pair')})
    total_trades_report = sum(stat.get('trade_count', 0) for stat in trade_stats)
    win_trades_report = sum(stat.get('win_trades', 0) for stat in trade_stats)
    avg_trades_per_pair = safe_div(total_trades_report, total_pairs_traded, 0.0)
    win_rate_trades = safe_div(win_trades_report, total_trades_report, 0.0)
    total_costs_report = sum(stat.get('total_costs', 0.0) for stat in trade_stats)

    best_pair_pnl = max((stat.get('total_pnl', 0.0) for stat in trade_stats), default=0.0)
    worst_pair_pnl = min((stat.get('total_pnl', 0.0) for stat in trade_stats), default=0.0)

    metrics = {
        "total_pnl": float(total_pnl),
        "sharpe_ratio_abs": float(sharpe_ratio),
        "max_drawdown_abs": float(max_dd),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_dd),
        "total_trades": int(total_trades_report),
        "total_pairs_traded": int(total_pairs_traded),
        "avg_trades_per_pair": float(avg_trades_per_pair),
        "win_rate_trades": float(win_rate_trades),
        "total_costs": float(total_costs_report),
        "best_pair_pnl": float(best_pair_pnl),
        "worst_pair_pnl": float(worst_pair_pnl),
    }
    metrics.update(calculate_extended_metrics(report_pnl_series, report_equity_series))

    pair_counts = [(1, total_pairs_traded)] if total_pairs_traded else []
    create_performance_report(
        equity_curve=report_equity_series,
        pnl_series=report_pnl_series,
        metrics=metrics,
        pair_counts=pair_counts,
        results_dir=results_dir,
        strategy_name="CointegrationStrategy"
    )

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / "strategy_metrics.csv", index=False)
    report_equity_series.to_csv(results_dir / "equity_curve.csv", header=['Equity'])
    if not report_pnl_series.empty:
        report_pnl_series.to_csv(results_dir / "daily_pnl.csv", header=['PnL'])
    
    # Prepare serializable results
    # 1. Convert pnl_series to dict with string keys and float values
    pnl_series_clean = all_results['pnl_series'].fillna(0)
    pnl_dict = {
        k.isoformat() if hasattr(k, 'isoformat') else str(k): float(v) 
        for k, v in pnl_series_clean.items()
    }
    
    # 2. Convert trades to serializable format
    serializable_trades = []
    for t in all_results['trades']:
        if hasattr(t, 'isoformat'):
             # It's a timestamp, convert to simple object
             serializable_trades.append({
                 'entry_time': t.isoformat(),
                 'type': 'Unknown', # Numba engine returns timestamps only
                 'pnl': 0.0,
                 'pair': 'Unknown'
             })
        elif isinstance(t, (pd.Timestamp, pd.DatetimeIndex)):
             serializable_trades.append({
                 'entry_time': str(t),
                 'type': 'Unknown',
                 'pnl': 0.0,
                 'pair': 'Unknown'
             })
        elif isinstance(t, dict):
             # Already a dict, ensure serializable
             clean_t = {}
             for k, v in t.items():
                 if hasattr(v, 'isoformat'):
                     clean_t[k] = v.isoformat()
                 elif isinstance(v, (pd.Timestamp, pd.DatetimeIndex)):
                     clean_t[k] = str(v)
                 elif isinstance(v, Path):
                     clean_t[k] = str(v)
                 else:
                     clean_t[k] = v
             serializable_trades.append(clean_t)
        else:
            serializable_trades.append(str(t))
            
    # 3. Serialize Config
    config_dict = cfg.model_dump() if hasattr(cfg, 'model_dump') else cfg.dict() if hasattr(cfg, 'dict') else {}
    
    # Helper to recursively convert Paths to strings in config
    def clean_config(obj):
        if isinstance(obj, dict):
            return {k: clean_config(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_config(i) for i in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
            
    config_dict = clean_config(config_dict)

    return {
        'total_pnl': total_pnl,
        'trade_count': trade_count,
        'total_trades': trade_count,  # Alias for frontend
        'sharpe_ratio_abs': float(sharpe_ratio),
        'max_drawdown_abs': float(max_dd),
        'trades': serializable_trades,
        'pnl_series': pnl_dict,
        'trade_stat': all_results['stats'],
        'config': config_dict,
        'success': True
    }
