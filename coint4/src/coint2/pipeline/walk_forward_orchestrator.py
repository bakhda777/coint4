"""Walk-forward analysis orchestrator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, List
import time
import multiprocessing
import os

import pandas as pd

from coint2.core import math_utils, performance
from coint2.utils.config import AppConfig
from coint2.utils.logging_config import get_trading_logger, setup_logging_from_config
from coint2.utils.logging_utils import get_logger
from coint2.utils.timing_utils import ProgressTracker, logged_time, time_block
from coint2.utils.pairs_loader import load_pair_tuples
from coint2.utils.visualization import calculate_extended_metrics, create_performance_report, format_metrics_summary
from coint2.monitoring.metrics import TradingMetrics, DashboardGenerator

# Import directly from the file path rather than the module
from coint2.core.data_loader import DataHandler, load_master_dataset, resolve_data_filters
from coint2.core.normalization_improvements import (
    preprocess_and_normalize_data,
    apply_production_normalization,
    apply_normalization_with_params,
)
from coint2.core.memory_optimization import (
    consolidate_price_data, initialize_global_price_data, get_price_data_view,
    setup_blas_threading_limits, monitor_memory_usage, verify_no_data_copies,
    cleanup_global_data, GLOBAL_PRICE
)
# from optimiser.metric_utils import validate_params  # Moved to function level to avoid circular import
from coint2.engine.numba_engine import NumbaPairBacktester as PairBacktester
from coint2.core.portfolio import Portfolio
from coint2.utils.vectorized_ops import VectorizedStatsCalculator, vectorized_eval_expression
from coint2.pipeline.cost_model import apply_fully_costed_model
from coint2.pipeline.pair_ranking import apply_entry_rank, rank_pairs

def convert_hours_to_periods(hours: float, bar_minutes: int) -> int:
    """
    Convert hours to number of periods based on bar timeframe.

    Args:
        hours: Number of hours to convert
        bar_minutes: Minutes per bar (timeframe)

    Returns:
        Number of periods
    """
    if hours <= 0:
        return 0
    return int(hours * 60 / bar_minutes)


def _compute_pair_trailing_streaks(
    history_window: list[list[tuple[str, str]]],
) -> dict[tuple[str, str], int]:
    """Compute trailing consecutive presence streak for each historical pair."""
    if not history_window:
        return {}
    all_pairs = {pair for step in history_window for pair in step}
    streaks: dict[tuple[str, str], int] = {}
    reversed_window = list(reversed(history_window))
    for pair in all_pairs:
        if pair not in reversed_window[0]:
            streaks[pair] = 0
            continue
        streak = 0
        for step in reversed_window:
            if pair in step:
                streak += 1
            else:
                break
        streaks[pair] = streak
    return streaks


def _parallel_map(
    func,
    items,
    n_jobs: int | None,
    use_processes: bool = False,
    initializer=None,
    initargs=(),
) -> list:
    if not items:
        return []
    if not n_jobs or n_jobs < 1:
        n_jobs = 1
    max_workers = min(n_jobs, len(items))
    if max_workers <= 1:
        return [func(item) for item in items]
    if use_processes:
        return _fork_process_map(
            func,
            items,
            max_workers,
            initializer=initializer,
            initargs=initargs,
        )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))


def _process_chunk(conn, func, chunk, initializer=None, initargs=()):
    try:
        if initializer:
            initializer(*initargs)
        if func and conn:
            results = [func(item) for item in chunk]
            conn.send({"results": results})
            return
        conn.send({"error": "invalid process chunk"})
    except Exception as exc:
        conn.send({"error": str(exc)})
    finally:
        if conn:
            conn.close()


def _fork_process_map(func, items, n_jobs: int, initializer=None, initargs=()) -> list:
    if not items:
        return []
    max_workers = min(n_jobs, len(items))
    if max_workers <= 1:
        return [func(item) for item in items]
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        ctx = multiprocessing.get_context()
    chunks = [items[i::max_workers] for i in range(max_workers)]
    processes = []
    conns = []
    for chunk in chunks:
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_process_chunk,
            args=(child_conn, func, chunk, initializer, initargs),
        )
        proc.start()
        child_conn.close()
        processes.append(proc)
        conns.append(parent_conn)
    results = []
    errors = []
    for conn in conns:
        payload = conn.recv()
        conn.close()
        if isinstance(payload, dict) and payload.get("error"):
            errors.append(payload["error"])
        else:
            results.extend(payload.get("results", []))
    for proc in processes:
        proc.join()
    if errors:
        raise RuntimeError(f"Process worker error: {errors[0]}")
    return results


def _process_pair_mmap_worker(args):
    return process_single_pair_mmap(*args)


def _process_pair_worker(args):
    return process_single_pair(*args)


def _init_worker_global_price(consolidated_path: str) -> None:
    if not consolidated_path:
        return
    from coint2.core.memory_optimization import GLOBAL_PRICE, initialize_global_price_data
    if GLOBAL_PRICE is None:
        initialize_global_price_data(consolidated_path)

def _simulate_realistic_portfolio(
    all_pnls,
    cfg,
    all_positions=None,
    all_scores=None,
    pair_entry_weights: list[float] | None = None,
    return_diagnostics: bool = False,
):
    """
    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Полноценная симуляция портфеля с реалистичным управлением позициями.

    Вместо простого суммирования PnL всех пар, симулируем реальную работу портфеля:
    1. На каждом временном шаге собираем сигналы от всех пар
    2. Применяем лимит max_active_positions
    3. Выбираем лучшие сигналы для торговли
    4. Рассчитываем реалистичный PnL портфеля

    Args:
        all_pnls: Список PnL серий от всех пар
        cfg: Конфигурация с параметрами портфеля
        all_positions: Список серий позиций (опционально)
        all_scores: Список серий сигналов/скоринга (опционально, например z-score)

    Returns:
        pd.Series or tuple[pd.Series, dict[str, pd.Series]]:
        Портфельный PnL. Если return_diagnostics=True, дополнительно возвращаются
        диагностические серии (turnover_units/exposure_units) для fully-costed агрегации.
    """
    logger = get_logger(__name__)

    if not all_pnls:
        empty = pd.Series(dtype=float)
        diagnostics = {"turnover_units": empty, "exposure_units": empty}
        if return_diagnostics:
            return empty, diagnostics
        return empty

    backtest_cfg = getattr(cfg, "backtest", None)
    portfolio_cfg = getattr(cfg, "portfolio", None)

    initial_capital = float(getattr(portfolio_cfg, "initial_capital", 0.0) or 0.0)
    daily_stop_pct = float(getattr(backtest_cfg, "portfolio_daily_stop_pct", 0.0) or 0.0)
    deleverage_start_raw = getattr(backtest_cfg, "portfolio_deleverage_start_pct", None)
    if deleverage_start_raw is None and daily_stop_pct > 0:
        deleverage_start_pct = daily_stop_pct * 0.6
    else:
        deleverage_start_pct = float(deleverage_start_raw or 0.0)
    if daily_stop_pct > 0 and deleverage_start_pct > daily_stop_pct:
        deleverage_start_pct = daily_stop_pct
    deleverage_factor = float(getattr(backtest_cfg, "portfolio_deleverage_factor", 0.5) or 0.5)
    deleverage_factor = min(max(deleverage_factor, 0.0), 1.0)

    # Создаем DataFrame со всеми PnL сериями
    pnl_df = pd.concat({f'pair_{i}': pnl.fillna(0) for i, pnl in enumerate(all_pnls)}, axis=1)

    positions_df = None
    if all_positions:
        positions_df = pd.concat({f'pair_{i}': pos.fillna(0) for i, pos in enumerate(all_positions)}, axis=1)

    scores_df = None
    if all_scores:
        scores_df = pd.concat({f'pair_{i}': score for i, score in enumerate(all_scores)}, axis=1)

    # Создаем DataFrame с сигналами
    if positions_df is not None:
        signals_df = (positions_df != 0).astype(int)
        entry_signals_df = (positions_df.shift(1).fillna(0) == 0) & (positions_df != 0)
    else:
        signals_df = pd.concat({f'pair_{i}': (pnl != 0).astype(int) for i, pnl in enumerate(all_pnls)}, axis=1)
        entry_signals_df = signals_df

    # Инициализируем портфель
    max_positions = int(getattr(portfolio_cfg, "max_active_positions", 0) or 0)
    if max_positions < 1:
        max_positions = 1

    entry_rank_mode = str(getattr(portfolio_cfg, "entry_rank_mode", "abs_signal") or "abs_signal")
    pair_quality_alpha = float(getattr(portfolio_cfg, "entry_pair_quality_alpha", 0.0) or 0.0)
    pair_quality_by_col: dict[str, float] = {}
    if pair_entry_weights:
        for i, w in enumerate(pair_entry_weights):
            pair_quality_by_col[f"pair_{i}"] = float(w)

    portfolio_pnl = pd.Series(0.0, index=pnl_df.index)
    turnover_units = pd.Series(0.0, index=pnl_df.index)
    exposure_units = pd.Series(0.0, index=pnl_df.index)
    active_positions = {}  # {pair_name: entry_timestamp}
    active_position_sizes = {}  # {pair_name: abs(position_size)}
    current_day = None
    daily_realized_pnl = 0.0
    deleverage_steps = 0
    hard_stop_days = 0
    hard_stop_active = False
    _hard_stop_day_counted = False

    logger.info(f"🎯 СИМУЛЯЦИЯ ПОРТФЕЛЯ: {len(all_pnls)} пар, лимит {max_positions} позиций")
    if entry_rank_mode.strip().lower() != "abs_signal" and pair_quality_alpha > 0:
        logger.info(
            "📌 Entry ranking: mode=%s, pair_quality_alpha=%.2f",
            entry_rank_mode,
            pair_quality_alpha,
        )
    if daily_stop_pct > 0:
        logger.info(
            "🛡️ Portfolio safeguards: daily_stop=%.2f%%, deleverage_start=%.2f%%, deleverage_factor=%.2f",
            daily_stop_pct * 100.0,
            deleverage_start_pct * 100.0,
            deleverage_factor,
        )

    # Симулируем торговлю по каждому временному шагу
    for timestamp in pnl_df.index:
        ts = pd.Timestamp(timestamp)
        day = ts.normalize()
        if current_day is None or day != current_day:
            current_day = day
            daily_realized_pnl = 0.0
            hard_stop_active = False
            _hard_stop_day_counted = False

        daily_loss_pct = 0.0
        if initial_capital > 0:
            daily_loss_pct = max(0.0, -daily_realized_pnl / initial_capital)
        hard_stop_active = hard_stop_active or (
            daily_stop_pct > 0 and daily_loss_pct >= daily_stop_pct
        )
        deleverage_active = (
            not hard_stop_active
            and deleverage_start_pct > 0
            and daily_loss_pct >= deleverage_start_pct
            and deleverage_factor < 1.0
        )
        risk_scale = deleverage_factor if deleverage_active else 1.0
        effective_max_positions = max_positions
        if deleverage_active:
            effective_max_positions = max(1, int(max_positions * risk_scale))
            deleverage_steps += 1

        current_signals = signals_df.loc[timestamp]
        current_entries = entry_signals_df.loc[timestamp]
        current_pnls = pnl_df.loc[timestamp]
        previous_sizes = dict(active_position_sizes)
        current_positions = None

        # 1. Отмечаем позиции на закрытие (по позиции, а не по PnL)
        positions_to_close = []
        if positions_df is not None:
            current_positions = positions_df.loc[timestamp]
            for pair_name in list(active_positions.keys()):
                if current_positions[pair_name] == 0:
                    positions_to_close.append(pair_name)
        else:
            for pair_name in list(active_positions.keys()):
                if current_signals[pair_name] == 0:
                    positions_to_close.append(pair_name)

        if hard_stop_active:
            positions_to_close = list(active_positions.keys())
            new_signals = []
        else:
            # 2. Ищем новые сигналы для открытия позиций (только на входе)
            new_signals = []
            for pair_name in current_signals.index:
                if current_entries[pair_name] and pair_name not in active_positions:
                    if scores_df is not None and pair_name in scores_df.columns:
                        base_strength = abs(scores_df.loc[timestamp, pair_name])
                    else:
                        base_strength = abs(current_pnls[pair_name])
                    strength = apply_entry_rank(
                        float(base_strength),
                        pair_quality=pair_quality_by_col.get(pair_name),
                        entry_rank_mode=entry_rank_mode,
                        pair_quality_alpha=pair_quality_alpha,
                    )
                    new_signals.append((pair_name, float(strength)))

            # 3. Сортируем новые сигналы по силе
            new_signals.sort(key=lambda x: x[1], reverse=True)

            # 4. Открываем новые позиции в пределах лимита (учитываем закрывающиеся)
            available_slots = effective_max_positions - (len(active_positions) - len(positions_to_close))
            if available_slots < 0:
                available_slots = 0
            for i, (pair_name, _strength) in enumerate(new_signals):
                if i >= available_slots:
                    break
                active_positions[pair_name] = timestamp

        # 5. Рассчитываем PnL портфеля на этом шаге
        step_pnl = 0.0
        for pair_name in active_positions:
            step_pnl += current_pnls[pair_name]
        step_pnl *= risk_scale

        portfolio_pnl[timestamp] = step_pnl
        daily_realized_pnl += step_pnl

        if (
            not hard_stop_active
            and daily_stop_pct > 0
            and initial_capital > 0
            and daily_realized_pnl <= -(daily_stop_pct * initial_capital)
        ):
            hard_stop_active = True
            if not _hard_stop_day_counted:
                hard_stop_days += 1
                _hard_stop_day_counted = True
            positions_to_close = list(active_positions.keys())

        # 5b. Диагностика turnover/exposure в единицах позиции.
        current_sizes = {}
        for pair_name in active_positions:
            if current_positions is not None and pair_name in current_positions.index:
                size = abs(float(current_positions[pair_name]))
            else:
                size = 1.0
            current_sizes[pair_name] = size * risk_scale

        turnover_step = 0.0
        for pair_name, prev_size in previous_sizes.items():
            new_size = current_sizes.get(pair_name, 0.0)
            turnover_step += abs(new_size - prev_size)
        for pair_name, new_size in current_sizes.items():
            if pair_name not in previous_sizes:
                turnover_step += abs(new_size)

        turnover_units[timestamp] = float(turnover_step)
        exposure_units[timestamp] = float(sum(current_sizes.values()))
        active_position_sizes = current_sizes

        # 6. Закрываем позиции после учета PnL на текущем баре
        for pair_name in positions_to_close:
            if pair_name in active_positions:
                del active_positions[pair_name]

    # Диагностика
    total_signals = signals_df.sum(axis=1)
    avg_active_pairs = len([p for p in active_positions]) if active_positions else 0
    max_signals = total_signals.max()
    avg_signals = total_signals.mean()

    logger.info(f"   📈 Макс. одновременных сигналов: {max_signals}")
    logger.info(f"   📊 Средн. сигналов за период: {avg_signals:.1f}")
    logger.info(f"   🎯 Финальных активных позиций: {avg_active_pairs}")

    utilization = (total_signals.clip(upper=max_positions) / max_positions).mean()
    logger.info(f"   ⚡ Утилизация лимита позиций: {utilization:.1%}")
    if daily_stop_pct > 0:
        logger.info(f"   🛡️ Deleverage баров: {deleverage_steps}, hard-stop дней: {hard_stop_days}")

    diagnostics = {
        "turnover_units": turnover_units,
        "exposure_units": exposure_units,
    }
    if return_diagnostics:
        return portfolio_pnl, diagnostics
    return portfolio_pnl


def _emit_monitoring_metrics(metrics: dict) -> None:
    logger = get_logger("walk_forward.monitoring")
    try:
        trading_logger = get_trading_logger()
        drawdown_pct = abs(float(metrics.get("max_drawdown_on_equity", 0.0) or 0.0))

        monitoring = TradingMetrics()
        monitoring.total_pnl = float(metrics.get("total_pnl", 0.0) or 0.0)
        monitoring.win_rate = float(metrics.get("win_rate_trades", metrics.get("win_rate", 0.0)) or 0.0)
        monitoring.max_drawdown = drawdown_pct
        monitoring.current_drawdown = drawdown_pct
        monitoring.trade_count = int(metrics.get("total_trades", 0) or 0)
        monitoring.active_positions = int(metrics.get("total_pairs_traded", 0) or 0)
        monitoring.sharpe_ratio = float(metrics.get("sharpe_ratio_abs", 0.0) or 0.0)
        monitoring.avg_trade_duration = float(metrics.get("avg_trade_duration", 0.0) or 0.0)
        monitoring.last_update_time = pd.Timestamp.now().isoformat()

        for key, value in {
            "total_pnl": monitoring.total_pnl,
            "sharpe_ratio_abs": monitoring.sharpe_ratio,
            "max_drawdown_pct": monitoring.max_drawdown,
            "total_trades": monitoring.trade_count,
            "total_pairs_traded": monitoring.active_positions,
            "total_costs": float(metrics.get("total_costs", 0.0) or 0.0),
        }.items():
            trading_logger.log_metric(key, float(value))

        DashboardGenerator().generate_dashboard(monitoring)
    except Exception as exc:
        logger.warning(f"Не удалось записать метрики мониторинга: {exc}")

def _run_backtest_for_pair(pair_data, s1, s2, cfg, capital_per_pair, bar_minutes, period_label, metrics):
    """
    Helper function to run backtest for a pair with given data.
    
    Args:
        pair_data: Normalized pair data DataFrame
        s1, s2: Symbol names
        cfg: Configuration object
        capital_per_pair: Capital allocated per pair
        bar_minutes: Minutes per bar
        period_label: Label for the current period
        metrics: Pair metrics dictionary
        
    Returns:
        Dictionary with pair results
    """
    from coint2.utils.logging_utils import get_logger
    logger = get_logger("pair_processing")
    
    try:
        # Create a temporary portfolio for this pair
        from coint2.core.portfolio import Portfolio
        temp_portfolio = Portfolio(
            initial_capital=capital_per_pair,
            max_active_positions=1,  # Single pair
            config=cfg.portfolio,
        )
        
        bt = PairBacktester(
            pair_data=pair_data,
            rolling_window=cfg.backtest.rolling_window,
            portfolio=temp_portfolio,
            pair_name=f"{s1}-{s2}",
            z_threshold=cfg.backtest.zscore_threshold,
            z_exit=getattr(cfg.backtest, 'zscore_exit', 0.0),
            commission_pct=getattr(cfg.backtest, 'commission_pct', 0.0),
            slippage_pct=getattr(cfg.backtest, 'slippage_pct', 0.0),
            annualizing_factor=getattr(cfg.backtest, 'annualizing_factor', 365),
            capital_at_risk=capital_per_pair,
            stop_loss_multiplier=getattr(cfg.backtest, 'stop_loss_multiplier', 2.0),
            take_profit_multiplier=getattr(cfg.backtest, 'take_profit_multiplier', None),
            cooldown_periods=convert_hours_to_periods(getattr(cfg.backtest, 'cooldown_hours', 0), bar_minutes),
            wait_for_candle_close=getattr(cfg.backtest, 'wait_for_candle_close', False),
            max_margin_usage=getattr(cfg.portfolio, 'max_margin_usage', 1.0),
            half_life=metrics.get('half_life'),
            time_stop_multiplier=getattr(cfg.backtest, 'time_stop_multiplier', None),
            min_volatility=getattr(cfg.backtest, 'min_volatility', 0.001),
            # Enhanced risk management parameters
            use_kelly_sizing=getattr(cfg.backtest, 'use_kelly_sizing', True),
            max_kelly_fraction=getattr(cfg.backtest, 'max_kelly_fraction', 0.25),
            volatility_lookback=getattr(cfg.backtest, 'volatility_lookback', 96),
            adaptive_thresholds=getattr(cfg.backtest, 'adaptive_thresholds', True),
            var_confidence=getattr(cfg.backtest, 'var_confidence', 0.05),
            max_var_multiplier=getattr(cfg.backtest, 'max_var_multiplier', 3.0),
            # Market regime detection parameters
            market_regime_detection=getattr(cfg.backtest, 'market_regime_detection', True),
            hurst_window=getattr(cfg.backtest, 'hurst_window', 720),
            hurst_trending_threshold=getattr(cfg.backtest, 'hurst_trending_threshold', 0.5),
            variance_ratio_window=getattr(cfg.backtest, 'variance_ratio_window', 480),
            variance_ratio_trending_min=getattr(cfg.backtest, 'variance_ratio_trending_min', 1.2),
            variance_ratio_mean_reverting_max=getattr(cfg.backtest, 'variance_ratio_mean_reverting_max', 0.8),
            # Structural break protection parameters
            structural_break_protection=getattr(cfg.backtest, 'structural_break_protection', True),
            cointegration_test_frequency=getattr(cfg.backtest, 'cointegration_test_frequency', 2688),
            adf_pvalue_threshold=getattr(cfg.backtest, 'adf_pvalue_threshold', 0.05),
            exclusion_period_days=getattr(cfg.backtest, 'exclusion_period_days', 30),
            max_half_life_days=getattr(cfg.backtest, 'max_half_life_days', 10),
            min_correlation_threshold=getattr(cfg.backtest, 'min_correlation_threshold', 0.6),
            correlation_window=getattr(cfg.backtest, 'correlation_window', 720),
            # Performance optimization parameters
            regime_check_frequency=getattr(cfg.backtest, 'regime_check_frequency', 96),
            use_market_regime_cache=getattr(cfg.backtest, 'use_market_regime_cache', True),
            adf_check_frequency=getattr(cfg.backtest, 'adf_check_frequency', 5376),
            cache_cleanup_frequency=getattr(cfg.backtest, 'cache_cleanup_frequency', 1000),
            lazy_adf_threshold=getattr(cfg.backtest, 'lazy_adf_threshold', 0.1),
            hurst_neutral_band=getattr(cfg.backtest, 'hurst_neutral_band', 0.05),
            vr_neutral_band=getattr(cfg.backtest, 'vr_neutral_band', 0.2),
            # EW correlation parameters
            use_exponential_weighted_correlation=getattr(cfg.backtest, 'use_exponential_weighted_correlation', False),
            ew_correlation_alpha=getattr(cfg.backtest, 'ew_correlation_alpha', 0.1),
            # Cost modeling parameters
            slippage_stress_multiplier=getattr(cfg.backtest, 'slippage_stress_multiplier', 1.0),
            always_model_slippage=getattr(cfg.backtest, 'always_model_slippage', True),
            # Volatility-based position sizing parameters
            volatility_based_sizing=getattr(cfg.portfolio, 'volatility_based_sizing', False),
            volatility_lookback_hours=getattr(cfg.portfolio, 'volatility_lookback_hours', 24),
            min_position_size_pct=getattr(cfg.portfolio, 'min_position_size_pct', 0.005),
            max_position_size_pct=getattr(cfg.portfolio, 'max_position_size_pct', 0.02),
            volatility_adjustment_factor=getattr(cfg.portfolio, 'volatility_adjustment_factor', 2.0),
            config=cfg.backtest,
        )
        
        # Run backtest
        logger.debug(f"🔄 Запускаем бэктест для пары {s1}-{s2}")
        bt.run()
        results = bt.get_results()
        
        # Process results
        pnl_series = results["pnl"]
        trades_log = results.get('trades_log', [])
        logger.debug(f"📈 Бэктест завершен для {s1}-{s2}: {len(pnl_series)} периодов, {len(trades_log)} сделок")
        
        # Calculate trade statistics
        if isinstance(results, dict):
            trades = results.get("trades", pd.Series())
            positions = results.get("position", pd.Series(dtype=float))
            costs = results.get("costs", pd.Series())
            z_scores = results.get("z_score", pd.Series(dtype=float))
        else:
            trades = results.get("trades", pd.Series())
            positions = results.get("position", pd.Series(dtype=float))
            costs = results.get("costs", pd.Series())
            z_scores = results.get("z_score", pd.Series(dtype=float))
        
        # Vectorized calculation of actual trade count
        if not positions.empty:
            import numpy as np
            positions_values = positions.ffill().values
            prev_positions_values = np.concatenate([[0], positions_values[:-1]])
            # Vectorized boolean operations
            is_trade_open_event = (prev_positions_values == 0) & (positions_values != 0)
            actual_trade_count = int(np.sum(is_trade_open_event))
        else:
            actual_trade_count = 0

        # Notional cap diagnostics (entry-level)
        entry_notional_count = 0
        entry_notional_cap_hits = 0
        entry_notional_below_min = 0
        entry_notional_avg = 0.0
        entry_notional_p50 = 0.0
        entry_notional_min = 0.0
        entry_notional_max = 0.0

        min_notional = float(getattr(cfg.portfolio, "min_notional_per_trade", 0.0) or 0.0)
        max_notional = float(getattr(cfg.portfolio, "max_notional_per_trade", 0.0) or 0.0)
        if actual_trade_count > 0 and (min_notional > 0.0 or max_notional > 0.0):
            try:
                import numpy as np

                entry_mask = is_trade_open_event
                if entry_mask.any():
                    if isinstance(results, dict):
                        y_vals = results.get("y")
                        x_vals = results.get("x")
                        beta_vals = results.get("beta")
                    else:
                        y_vals = results.get("y")
                        x_vals = results.get("x")
                        beta_vals = results.get("beta")

                    if y_vals is None or x_vals is None:
                        y_vals = pair_data.iloc[:, 0].values
                        x_vals = pair_data.iloc[:, 1].values
                    else:
                        y_vals = y_vals.values if hasattr(y_vals, "values") else np.asarray(y_vals)
                        x_vals = x_vals.values if hasattr(x_vals, "values") else np.asarray(x_vals)
                        if np.nanmax(np.abs(y_vals)) == 0.0 or np.nanmax(np.abs(x_vals)) == 0.0:
                            y_vals = pair_data.iloc[:, 0].values
                            x_vals = pair_data.iloc[:, 1].values

                    if beta_vals is None:
                        beta_vals = np.ones_like(y_vals, dtype=float)
                    else:
                        beta_vals = beta_vals.values if hasattr(beta_vals, "values") else np.asarray(beta_vals)
                        if beta_vals.shape[0] != y_vals.shape[0]:
                            beta_vals = np.ones_like(y_vals, dtype=float)

                    entry_idx = np.where(entry_mask)[0]
                    entry_positions = positions_values[entry_mask]
                    entry_notional = np.abs(entry_positions) * (
                        np.abs(y_vals[entry_idx]) + np.abs(beta_vals[entry_idx] * x_vals[entry_idx])
                    )
                    entry_notional = np.nan_to_num(entry_notional, nan=0.0, posinf=0.0, neginf=0.0)

                    entry_notional_count = int(entry_notional.size)
                    if entry_notional_count > 0:
                        entry_notional_avg = float(entry_notional.mean())
                        entry_notional_p50 = float(np.median(entry_notional))
                        entry_notional_min = float(entry_notional.min())
                        entry_notional_max = float(entry_notional.max())
                        if max_notional > 0.0:
                            entry_notional_cap_hits = int(
                                np.sum(entry_notional >= max_notional * 0.999)
                            )
                        if min_notional > 0.0:
                            entry_notional_below_min = int(
                                np.sum(entry_notional <= min_notional * 1.001)
                            )
            except Exception as exc:
                logger.debug(f"Notional diagnostics failed for {s1}-{s2}: {exc}")

        # Vectorized calculation of pair statistics using numpy
        if not pnl_series.empty:
            import numpy as np
            pnl_values = pnl_series.values
            pair_pnl = np.sum(pnl_values)
            win_days = int(np.sum(pnl_values > 0))
            lose_days = int(np.sum(pnl_values < 0))
            max_daily_gain = np.max(pnl_values)
            max_daily_loss = np.min(pnl_values)
        else:
            pair_pnl = 0.0
            win_days = lose_days = 0
            max_daily_gain = max_daily_loss = 0.0
        
        # CRITICAL FIX: Правильный расчет издержек с обработкой NaN
        if not costs.empty:
            costs_clean = costs.fillna(0)  # Заменяем NaN на 0
            pair_costs = float(np.sum(costs_clean.values))
        else:
            pair_costs = 0.0
        
        trade_stat = {
            'pair': f'{s1}-{s2}',
            'period': period_label,
            'total_pnl': pair_pnl,
            'total_costs': pair_costs,
            'trade_count': actual_trade_count,
            'avg_pnl_per_trade': pair_pnl / max(actual_trade_count, 1),
            'win_days': win_days,
            'lose_days': lose_days,
            'total_days': len(pnl_series),
            'max_daily_gain': max_daily_gain,
            'max_daily_loss': max_daily_loss,
            'entry_notional_count': entry_notional_count,
            'entry_notional_cap_hits': entry_notional_cap_hits,
            'entry_notional_below_min': entry_notional_below_min,
            'entry_notional_avg': entry_notional_avg,
            'entry_notional_p50': entry_notional_p50,
            'entry_notional_min': entry_notional_min,
            'entry_notional_max': entry_notional_max,
        }
        
        # Логирование результатов обработки
        logger.debug(f"✅ Пара {s1}-{s2} обработана успешно: PnL={trade_stat['total_pnl']:.2f}, сделок={trade_stat['trade_count']}")
        
        return {
            'pnl_series': pnl_series,
            'trades_log': trades_log,
            'trade_stat': trade_stat,
            'positions': positions,
            'z_scores': z_scores,
            'success': True
        }
        
    except Exception as e:
        # Логирование ошибки обработки
        logger.warning(f"❌ Ошибка обработки пары {s1}-{s2}: {str(e)}")
        
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
            'positions': pd.Series(dtype=float),
            'z_scores': pd.Series(dtype=float),
            'success': False,
            'error': str(e)
        }

def process_single_pair_mmap(pair_symbols, testing_start, testing_end, cfg, 
                            capital_per_pair, bar_minutes, period_label, pair_stats=None, training_normalization_params=None):
    """
    Process a single pair using memory-mapped data for optimal performance.
    
    Args:
        pair_symbols: Tuple of (s1, s2) symbol names
        testing_start, testing_end: Testing period boundaries
        cfg: Configuration object
        capital_per_pair: Capital allocated per pair
        bar_minutes: Minutes per bar
        period_label: Label for the current period
        pair_stats: Optional pre-computed pair statistics (beta, mean, std, metrics)
        
    Returns:
        Dictionary with pair results
    """
    # Setup BLAS threading limits to prevent oversubscription
    from coint2.core.memory_optimization import setup_blas_threading_limits
    setup_blas_threading_limits(num_threads=1, verbose=False)
    
    s1, s2 = pair_symbols
    
    # Логирование начала обработки пары
    from coint2.utils.logging_utils import get_logger
    logger = get_logger("pair_processing")
    logger.debug(f"📊 Начинаем обработку пары {s1}-{s2} (memory-mapped)")
    
    # Extract pair statistics if provided
    if pair_stats is not None:
        beta, mean, std, metrics = pair_stats
    else:
        # Fallback values if no statistics provided
        beta, mean, std = 1.0, 0.0, 1.0
        metrics = {}
    
    try:
        # Get memory-mapped data view for this pair (no copy)
        pair_data_view = get_price_data_view([s1, s2], testing_start, testing_end)
        
        # Check if both symbols are available
        if s1 not in pair_data_view.columns or s2 not in pair_data_view.columns:
            return {
                'success': False,
                'error': f'Missing data for symbols {s1} or {s2}',
                'trade_stat': {
                    'pair': f'{s1}-{s2}',
                    'period': period_label,
                    'total_pnl': 0.0,
                    'trade_count': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0
                },
                'pnl_series': pd.Series(dtype=float),
                'trades_log': []
            }
            
        # Work with data view (no copy) and drop NaN
        pair_data = pair_data_view[[s1, s2]].dropna()
        
        if pair_data.empty:
            return {
                'success': False,
                'error': 'No valid data after filtering',
                'trade_stat': {
                    'pair': f'{s1}-{s2}',
                    'period': period_label,
                    'total_pnl': 0.0,
                    'trade_count': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0
                },
                'pnl_series': pd.Series(dtype=float),
                'trades_log': []
            }
        
        # ИСПРАВЛЕНИЕ: Используем статистики нормализации из тренировочного периода для консистентности
        import numpy as np
        
        # Проверяем если training_normalization_params содержит полные статистики (новый формат)
        if training_normalization_params and isinstance(training_normalization_params, dict):
            method = training_normalization_params.get('method')
            if method in ("rolling_zscore", "percent", "log_returns"):
                # Новый формат со статистиками нормализации
                try:
                    pair_data = apply_production_normalization(
                        pair_data,
                        training_normalization_params
                    )
                    logger.debug(f"Применена {method} нормализация для {s1}-{s2}")
                except Exception as e:
                    logger.error(f"Ошибка применения статистик нормализации: {e}")
                    return {
                        'success': False,
                        'error': f'Normalization error: {e}',
                        'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                        'pnl_series': pd.Series(dtype=float),
                        'trades_log': []
                    }
            elif method in ("minmax", "zscore"):
                try:
                    params = training_normalization_params.get('params', {})
                    fill_method = training_normalization_params.get('fill_method', 'ffill')
                    pair_data = apply_normalization_with_params(
                        pair_data,
                        params,
                        norm_method=method,
                        fill_method=fill_method
                    )
                    if s1 not in pair_data.columns or s2 not in pair_data.columns:
                        raise ValueError("Missing symbols after normalization")
                    logger.debug(f"Применена {method} нормализация для {s1}-{s2}")
                except Exception as e:
                    logger.error(f"Ошибка применения параметров нормализации: {e}")
                    return {
                        'success': False,
                        'error': f'Normalization error: {e}',
                        'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                        'pnl_series': pd.Series(dtype=float),
                        'trades_log': []
                    }
            elif s1 in training_normalization_params and s2 in training_normalization_params:
                # Старый формат с первыми значениями (обратная совместимость)
                data_values = pair_data.to_numpy(copy=False)
                norm_s1 = training_normalization_params[s1]
                norm_s2 = training_normalization_params[s2]
                
                if norm_s1 != 0 and norm_s2 != 0:
                    first_row = np.array([norm_s1, norm_s2])
                    logger.debug(f"Используем простую нормализацию для {s1}-{s2}: {first_row}")
                    normalized_values = np.divide(data_values, first_row[np.newaxis, :]) * 100
                    pair_data = pd.DataFrame(normalized_values, index=pair_data.index, columns=pair_data.columns)
                else:
                    logger.warning(f"Нулевой параметр нормализации для {s1} или {s2}. Пропуск.")
                    return {
                        'success': False,
                        'error': f'Zero normalization parameter for {s1} or {s2}',
                        'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                        'pnl_series': pd.Series(dtype=float),
                        'trades_log': []
                    }
            else:
                # Логируем ошибку и пропускаем пару, если нет параметров
                logger.error(f"Нет параметров нормализации для {s1}-{s2}. Пропуск.")
                return {
                    'success': False,
                    'error': f'Missing normalization parameters for {s1} or {s2}',
                    'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                    'pnl_series': pd.Series(dtype=float),
                    'trades_log': []
                }
        else:
            # Логируем ошибку и пропускаем пару, если нет параметров
            logger.error(f"Нет параметров нормализации для {s1}-{s2}. Пропуск.")
            return {
                'success': False,
                'error': f'Missing normalization parameters for {s1} or {s2}',
                'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                'pnl_series': pd.Series(dtype=float),
                'trades_log': []
            }
        
        # Use helper function to run backtest
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
        # Логирование ошибки обработки
        logger.warning(f"❌ Ошибка обработки пары {s1}-{s2}: {str(e)}")
        
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

def process_single_pair(pair_data_tuple, step_df, testing_start, testing_end, cfg, 
                       capital_per_pair, bar_minutes, period_label, training_normalization_params=None):
    """
    Process a single pair for parallel execution with vectorized optimizations.
    
    Args:
        pair_data_tuple: Tuple of (s1, s2, beta, mean, std, metrics)
        step_df: DataFrame with price data
        testing_start, testing_end: Testing period boundaries
        cfg: Configuration object
        capital_per_pair: Capital allocated per pair
        bar_minutes: Minutes per bar
        period_label: Label for the current period
        training_normalization_params: Dict with normalization parameters from training data
        
    Returns:
        Dictionary with pair results
    """
    # Setup BLAS threading limits to prevent oversubscription
    from coint2.core.memory_optimization import setup_blas_threading_limits
    setup_blas_threading_limits(num_threads=1, verbose=False)
    
    from coint2.utils.logger import get_logger
    logger = get_logger(__name__)
    
    s1, s2, beta, mean, std, metrics = pair_data_tuple
    
    # Логирование начала обработки пары
    logger.debug(f"🔄 Начинаем обработку пары {s1}-{s2} (период: {period_label})")
    
    try:
        pair_data = step_df.loc[testing_start:testing_end, [s1, s2]].dropna()
        # ИСПРАВЛЕНИЕ LOOKAHEAD BIAS: Используем нормализационные параметры из тренировочного периода
        if not pair_data.empty:
            import numpy as np
            data_values = pair_data.values

            if training_normalization_params and isinstance(training_normalization_params, dict):
                method = training_normalization_params.get('method')
                if method in ("rolling_zscore", "percent", "log_returns"):
                    try:
                        pair_data = apply_production_normalization(
                            pair_data,
                            training_normalization_params
                        )
                        logger.debug(f"Применена {method} нормализация для {s1}-{s2}")
                    except Exception as e:
                        logger.error(f"Ошибка применения статистик нормализации: {e}")
                        return {
                            'success': False,
                            'error': f'Normalization error: {e}',
                            'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                            'pnl_series': pd.Series(dtype=float),
                            'trades_log': []
                        }
                elif method in ("minmax", "zscore"):
                    try:
                        params = training_normalization_params.get('params', {})
                        fill_method = training_normalization_params.get('fill_method', 'ffill')
                        pair_data = apply_normalization_with_params(
                            pair_data,
                            params,
                            norm_method=method,
                            fill_method=fill_method
                        )
                        if s1 not in pair_data.columns or s2 not in pair_data.columns:
                            raise ValueError("Missing symbols after normalization")
                        logger.debug(f"Применена {method} нормализация для {s1}-{s2}")
                    except Exception as e:
                        logger.error(f"Ошибка применения параметров нормализации: {e}")
                        return {
                            'success': False,
                            'error': f'Normalization error: {e}',
                            'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                            'pnl_series': pd.Series(dtype=float),
                            'trades_log': []
                        }
                elif s1 in training_normalization_params and s2 in training_normalization_params:
                    norm_s1 = training_normalization_params[s1]
                    norm_s2 = training_normalization_params[s2]

                    if norm_s1 != 0 and norm_s2 != 0:
                        # Применяем нормализацию к столбцам DataFrame
                        first_row = np.array([norm_s1, norm_s2])
                        logger.debug(f"Используем нормализацию из тренировочного периода для {s1}-{s2}: {first_row}")

                        # Vectorized division and multiplication
                        normalized_values = np.divide(data_values, first_row[np.newaxis, :]) * 100
                        pair_data = pd.DataFrame(normalized_values, index=pair_data.index, columns=pair_data.columns)
                    else:
                        # Логируем ошибку и пропускаем пару, так как нормализация невозможна
                        logger.warning(f"Нулевой параметр нормализации для {s1} или {s2}. Пропуск.")
                        return {
                            'success': False,
                            'error': f'Zero normalization parameter for {s1} or {s2}',
                            'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                            'pnl_series': pd.Series(dtype=float),
                            'trades_log': []
                        }
                else:
                    # Логируем ошибку и пропускаем пару, если нет параметров
                    logger.error(f"Нет параметров нормализации для {s1}-{s2}. Пропуск.")
                    return {
                        'success': False,
                        'error': f'Missing normalization parameters for {s1} or {s2}',
                        'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                        'pnl_series': pd.Series(dtype=float),
                        'trades_log': []
                    }
            else:
                # Логируем ошибку и пропускаем пару, если нет параметров
                logger.error(f"Нет параметров нормализации для {s1}-{s2}. Пропуск.")
                return {
                    'success': False,
                    'error': f'Missing normalization parameters for {s1} or {s2}',
                    'trade_stat': {'pair': f'{s1}-{s2}', 'period': period_label, 'total_pnl': 0.0, 'trade_count': 0},
                    'pnl_series': pd.Series(dtype=float),
                    'trades_log': []
                }
        
        # Use helper function to run backtest
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
        # Логирование ошибки обработки
        logger.warning(f"❌ Ошибка обработки пары {s1}-{s2}: {str(e)}")
        
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
                'max_daily_loss': 0.0
            },
            'success': False,
            'error': str(e)
        }

def validate_time_windows(
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    testing_start: pd.Timestamp,
    testing_end: pd.Timestamp,
    bar_minutes: int
) -> bool:
    """
    Validate that time windows don't have look-ahead bias.
    
    Args:
        training_start: Start of training period
        training_end: End of training period
        testing_start: Start of testing period
        testing_end: End of testing period
        bar_minutes: Minutes per bar for minimum gap validation
        
    Returns:
        True if windows are valid, False otherwise
    """
    logger = get_logger("walk_forward.validation")
    
    # Check basic chronological order
    if training_start >= training_end:
        logger.error(f"❌ Training period invalid: start {training_start} >= end {training_end}")
        return False
        
    if testing_start >= testing_end:
        logger.error(f"❌ Testing period invalid: start {testing_start} >= end {testing_end}")
        return False
    
    # Critical: Check for look-ahead bias - training must end before testing starts
    if training_end >= testing_start:
        logger.error(f"❌ LOOK-AHEAD BIAS: Training ends {training_end} >= Testing starts {testing_start}")
        logger.error("   This would allow future data to leak into training!")
        return False
    
    # Ensure minimum gap between training and testing (at least one bar)
    min_gap = pd.Timedelta(minutes=bar_minutes)
    actual_gap = testing_start - training_end
    if actual_gap < min_gap:
        logger.warning(f"⚠️  Small gap between training and testing: {actual_gap} < {min_gap}")
        logger.warning("   Consider increasing gap to avoid potential data leakage")
    
    # Log validation success
    logger.debug(f"✅ Time windows validated:")
    logger.debug(f"   Training: {training_start} → {training_end}")
    logger.debug(f"   Testing:  {testing_start} → {testing_end}")
    logger.debug(f"   Gap:      {actual_gap}")
    
    return True

def validate_walk_forward_data(
    df: pd.DataFrame,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    testing_start: pd.Timestamp,
    testing_end: pd.Timestamp,
    min_training_days: float = 30,
    min_testing_days: float = 1,
    bar_minutes: int = 15,
) -> Tuple[bool, str]:
    """
    Validate that sufficient data exists for walk-forward step.
    
    Args:
        df: DataFrame with timestamp index
        training_start: Start of training period
        training_end: End of training period
        testing_start: Start of testing period
        testing_end: End of testing period
        min_training_days: Minimum required training days
        min_testing_days: Minimum required testing days
        bar_minutes: Bar size in minutes (used for duration/coverage calculations)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    logger = get_logger("walk_forward.data_validation")
    
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check data availability for training period
    training_data = df.loc[training_start:training_end]
    if training_data.empty:
        return False, f"No training data available for period {training_start} to {training_end}"
    
    # Check data availability for testing period
    testing_data = df.loc[testing_start:testing_end]
    if testing_data.empty:
        return False, f"No testing data available for period {testing_start} to {testing_end}"
    
    bar_delta = pd.Timedelta(minutes=bar_minutes)

    # Check minimum training period length (inclusive of last bar)
    training_days = (training_end - training_start + bar_delta) / pd.Timedelta(days=1)
    if training_days < min_training_days:
        return False, f"Training period too short: {training_days:.2f} days < {min_training_days} required"
    
    # Check minimum testing period length (inclusive of last bar)
    testing_days = (testing_end - testing_start + bar_delta) / pd.Timedelta(days=1)
    if testing_days < min_testing_days:
        return False, f"Testing period too short: {testing_days:.2f} days < {min_testing_days} required"
    
    # Check for significant data gaps in training period
    training_data_days = len(training_data)
    expected_training_periods = training_days * (24 * 60 / bar_minutes)
    data_coverage_ratio = (
        training_data_days / expected_training_periods if expected_training_periods else 0.0
    )
    
    if data_coverage_ratio < 0.5:  # Less than 50% data coverage
        logger.warning(f"⚠️  Low training data coverage: {data_coverage_ratio:.1%}")
        logger.warning(f"   Expected ~{expected_training_periods:.0f} periods, got {training_data_days}")
    
    # Check for significant data gaps in testing period
    testing_data_days = len(testing_data)
    expected_testing_periods = testing_days * (24 * 60 / bar_minutes)
    testing_coverage_ratio = (
        testing_data_days / expected_testing_periods if expected_testing_periods else 0.0
    )
    
    if testing_coverage_ratio < 0.3:  # Less than 30% data coverage
        logger.warning(f"⚠️  Low testing data coverage: {testing_coverage_ratio:.1%}")
        logger.warning(f"   Expected ~{expected_testing_periods:.0f} periods, got {testing_data_days}")
    
    logger.debug(f"✅ Data validation passed:")
    logger.debug(f"   Training: {training_data_days} periods ({data_coverage_ratio:.1%} coverage)")
    logger.debug(f"   Testing:  {testing_data_days} periods ({testing_coverage_ratio:.1%} coverage)")
    
    return True, "Data validation successful"


def _build_walk_forward_steps(
    *,
    start_date: pd.Timestamp,
    end_ts_inclusive: pd.Timestamp,
    training_period_days: float,
    testing_period_days: float,
    bar_delta: pd.Timedelta,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Build contiguous WFA steps that cover [start_date, end_ts_inclusive] without overlap.

    Semantics:
    - start_date / end_ts_inclusive are *testing* boundaries (inclusive on both ends at bar level).
    - Each step tests up to `testing_period_days` calendar days of bars, but the final step may be shorter
      if end_ts_inclusive isn't aligned to a full window.
    """
    if end_ts_inclusive < start_date:
        return []

    steps: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    current_test_start = start_date
    while current_test_start <= end_ts_inclusive:
        training_start = current_test_start - pd.Timedelta(days=float(training_period_days))
        training_end = current_test_start - bar_delta
        testing_start = current_test_start

        full_testing_end = testing_start + pd.Timedelta(days=float(testing_period_days)) - bar_delta
        testing_end = min(full_testing_end, end_ts_inclusive)
        if testing_end <= testing_start:
            break

        steps.append((training_start, training_end, testing_start, testing_end))
        current_test_start = testing_end + bar_delta
    return steps


@logged_time("run_walk_forward_analysis")
def run_walk_forward(cfg: AppConfig, use_memory_map: bool = True) -> dict[str, float]:
    """Run walk-forward analysis and return aggregated performance metrics.
    
    Args:
        cfg: Application configuration
        use_memory_map: Whether to use memory-mapped data optimization
    """
    start_time = time.time()
    setup_logging_from_config(cfg)
    logger = get_logger("walk_forward")

    # Setup BLAS threading limits before any parallel processing to prevent oversubscription
    from coint2.core.memory_optimization import setup_blas_threading_limits
    blas_info = setup_blas_threading_limits(num_threads=1, verbose=True)
    logger.info(f"🧵 BLAS threading configured: {blas_info['status']}")
    logger.info(f"🧵 Environment variables set: {len(blas_info['env_vars_set'])}")
    
    if use_memory_map:
        logger.info("🧠 Memory-mapped optimization enabled")
    else:
        logger.info("📊 Using traditional data processing")

    # Setup and initial data loading
    with time_block("initializing data handler"):
        handler = DataHandler(cfg)
        handler.clear_cache()
    clean_window, excluded_symbols = resolve_data_filters(cfg)

    raw_start_date = str(cfg.walk_forward.start_date or "").strip()
    raw_end_date = str(cfg.walk_forward.end_date or "").strip()

    start_date = pd.to_datetime(raw_start_date)
    end_date = pd.to_datetime(raw_end_date)

    # Bar size is needed to interpret date-only boundaries as full-day inclusive ranges.
    bar_minutes = getattr(cfg.pair_selection, "bar_minutes", None) or 15
    bar_delta = pd.Timedelta(minutes=bar_minutes)

    # Interpret date-only boundaries as full-day timestamps: [start 00:00, end 23:45] for 15m bars.
    # If the user supplies an explicit timestamp, respect it as-is.
    if len(raw_start_date) <= 10:
        start_date = start_date.normalize()
    end_ts_inclusive = end_date
    if len(raw_end_date) <= 10:
        end_ts_inclusive = end_date.normalize() + pd.Timedelta(days=1) - bar_delta
    # Для новой логики: нужны данные на training_period_days раньше start_date
    full_range_start = start_date - pd.Timedelta(days=cfg.walk_forward.training_period_days)
    
    logger.info("📅 НОВАЯ ЛОГИКА: start_date = начало ТЕСТИРОВАНИЯ")
    logger.info(f"📊 Данные нужны с: {full_range_start.date()} (для тренировки первого шага)")
    
    logger.info(f"🎯 Walk-Forward анализ: {start_date.date()} → {end_date.date()}")
    logger.info(f"Training период: {cfg.walk_forward.training_period_days} дней")
    logger.info(f"Testing период: {cfg.walk_forward.testing_period_days} дней")
    logger.info(f"💰 Начальный капитал: ${cfg.portfolio.initial_capital:,.0f}")
    logger.info(f"⚙️ Максимум позиций: {cfg.portfolio.max_active_positions}")
    logger.info(f"📊 Риск на позицию: {cfg.portfolio.risk_per_position_pct:.1%}")
    
    # Validate configuration parameters
    logger.info("🔍 Валидация параметров конфигурации...")
    try:
        # Extract parameters for validation
        params = {
            'z_entry': getattr(cfg.backtest, 'zscore_threshold', 2.0),
            'z_exit': getattr(cfg.backtest, 'zscore_exit', 0.0),
            'sl_mult': getattr(cfg.backtest, 'stop_loss_multiplier', 2.0),
            'time_stop_mult': getattr(cfg.backtest, 'time_stop_multiplier', None),
            'max_active_positions': cfg.portfolio.max_active_positions,
            'max_position_size_pct': getattr(cfg.portfolio, 'max_position_size_pct', 1.0),
            'risk_per_position_pct': cfg.portfolio.risk_per_position_pct
        }
        
        from optimiser.metric_utils import validate_params
        validated_params = validate_params(params)
        logger.info("✅ Параметры конфигурации валидны")
        
        # Update config with validated parameters if they were corrected
        if validated_params != params:
            logger.info("📝 Некоторые параметры были скорректированы:")
            if hasattr(cfg.backtest, 'zscore_threshold'):
                cfg.backtest.zscore_threshold = validated_params['z_entry']
            if hasattr(cfg.backtest, 'zscore_exit'):
                cfg.backtest.zscore_exit = validated_params['z_exit']
            if hasattr(cfg.backtest, 'stop_loss_multiplier'):
                cfg.backtest.stop_loss_multiplier = validated_params['sl_mult']
            if hasattr(cfg.backtest, 'time_stop_multiplier') and validated_params['time_stop_mult'] is not None:
                cfg.backtest.time_stop_multiplier = validated_params['time_stop_mult']
            cfg.portfolio.max_active_positions = validated_params['max_active_positions']
            if hasattr(cfg.portfolio, 'max_position_size_pct'):
                cfg.portfolio.max_position_size_pct = validated_params['max_position_size_pct']
            cfg.portfolio.risk_per_position_pct = validated_params['risk_per_position_pct']
            
            for key, (old_val, new_val) in zip(params.keys(), zip(params.values(), validated_params.values())):
                if old_val != new_val:
                    logger.info(f"   {key}: {old_val} → {new_val}")
                    
    except ValueError as e:
        logger.error(f"❌ Ошибка валидации параметров: {e}")
        raise

    fixed_pairs = None
    pairs_file = getattr(cfg.walk_forward, "pairs_file", None)
    if pairs_file:
        fixed_pairs = load_pair_tuples(pairs_file)
        if not fixed_pairs:
            raise ValueError(f"Файл pairs_file пуст или не содержит пар: {pairs_file}")
        # Best-effort guardrail: if pairs_file carries a selection period that overlaps
        # the test start, warn about potential lookahead/selection bias.
        try:
            pairs_path = Path(str(pairs_file))
            if not pairs_path.is_absolute() and not pairs_path.exists():
                app_root = Path(__file__).resolve().parents[3]
                candidates = [Path.cwd() / pairs_path, app_root / pairs_path]
                for candidate in candidates:
                    if candidate.exists():
                        pairs_path = candidate
                        break

            if pairs_path.exists() and pairs_path.suffix.lower() in {".yaml", ".yml"}:
                import yaml

                raw = yaml.safe_load(pairs_path.read_text(encoding="utf-8")) or {}
                if isinstance(raw, dict):
                    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
                    period = metadata.get("period") if isinstance(metadata.get("period"), dict) else {}
                    period_end = str(period.get("end") or "").strip()
                    if period_end:
                        period_end_dt = pd.to_datetime(period_end, errors="coerce")
                        if pd.notna(period_end_dt) and period_end_dt >= start_date:
                            logger.warning(
                                "⚠️ pairs_file period_end >= walk_forward.start_date: "
                                f"period_end={period_end_dt.date()} start_date={start_date.date()} "
                                f"(pairs_file={pairs_path}). "
                                "Это может дать lookahead/selection bias в OOS. "
                                "Рекомендуется собирать pairs_file с end_date <= start_date - 1 day "
                                "или включить динамический отбор пар."
                            )
        except Exception:  # noqa: BLE001
            pass
        logger.info(f"🔒 WFA: фиксированный universe из {pairs_file} ({len(fixed_pairs)} пар)")
    else:
        logger.info("🧭 WFA: динамический отбор пар на каждом шаге")

    # Calculate walk-forward steps (cover the configured test window without overshooting end_date).
    walk_forward_steps = _build_walk_forward_steps(
        start_date=start_date,
        end_ts_inclusive=end_ts_inclusive,
        training_period_days=float(cfg.walk_forward.training_period_days),
        testing_period_days=float(cfg.walk_forward.testing_period_days),
        bar_delta=bar_delta,
    )

    logger.info(f"📈 Запланировано {len(walk_forward_steps)} walk-forward шагов")

    max_steps = getattr(cfg.walk_forward, "max_steps", None)
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("walk_forward.max_steps должен быть >= 1 или null")
        if len(walk_forward_steps) > max_steps:
            logger.info(f"⛔ Ограничение WFA шагов: {len(walk_forward_steps)} → {max_steps}")
            walk_forward_steps = walk_forward_steps[:max_steps]
    
    # КРИТИЧНО: Проверка на временные пересечения для предотвращения lookahead bias
    for i, (tr_start, tr_end, te_start, te_end) in enumerate(walk_forward_steps):
        # Проверяем что testing начинается после окончания training
        if te_start < tr_end:
            raise ValueError(
                f"❌ LOOKAHEAD BIAS DETECTED в шаге {i+1}! "
                f"Testing нолжен начинаться после training: "
                f"training_end={tr_end.date()}, testing_start={te_start.date()}"
            )
        
        # Проверяем наличие буфера между training и testing
        gap_days = (te_start - tr_end).days
        if gap_days < 1:
            logger.warning(
                f"⚠️ Маленький буфер между training и testing в шаге {i+1}: {gap_days} дней. "
                f"Рекомендуется минимум 1 день."
            )
        
        # Проверяем на пересечения с предыдущими шагами
        if i > 0:
            prev_te_end = walk_forward_steps[i-1][3]
            if te_start < prev_te_end:
                logger.warning(
                    f"⚠️ Пересечение testing периодов в шагах {i} и {i+1}: "
                    f"предыдущий testing_end={prev_te_end.date()}, текущий testing_start={te_start.date()}"
                )
    
    # ОТЛАДОЧНАЯ ИНФОРМАЦИЯ для диагностики проблем
    if len(walk_forward_steps) == 0:
        logger.warning("⚠️  ДИАГНОСТИКА: Нет walk-forward шагов!")
        logger.warning(f"   start_date (начало теста): {start_date.date()}")
        logger.warning(f"   end_date (конец выборки): {end_date.date()}")
        logger.warning(f"   training_period_days: {cfg.walk_forward.training_period_days}")
        logger.warning(f"   testing_period_days: {cfg.walk_forward.testing_period_days}")
        
        # Рассчитаем, что должно было быть
        would_be_testing_end = start_date + pd.Timedelta(days=float(cfg.walk_forward.testing_period_days)) - bar_delta
        needed_training_start = start_date - pd.Timedelta(days=cfg.walk_forward.training_period_days)
        
        logger.warning(f"   testing_end был бы: {would_be_testing_end.date()}")
        logger.warning(f"   нужны данные с: {needed_training_start.date()}")
        
        if would_be_testing_end > end_ts_inclusive:
            logger.warning(
                f"   ❌ ПРИЧИНА: testing_end ({would_be_testing_end}) > end_ts_inclusive ({end_ts_inclusive})"
            )
            logger.warning(
                f"   ✅ РЕШЕНИЕ: Продлите end_date или сократите testing_period_days "
                f"(для date-only end_date последним баром считается {end_ts_inclusive})"
            )
        
    else:
        logger.info("✅ Шаги успешно запланированы:")
        for i, (tr_start, tr_end, te_start, te_end) in enumerate(walk_forward_steps, 1):
            logger.info(f"   Шаг {i}: тренировка {tr_start.date()}-{tr_end.date()}, тест {te_start.date()}-{te_end.date()}")

    # Memory-mapped data optimization
    memory_map_path: str | None = None
    if use_memory_map and walk_forward_steps:
        logger.info("🗂️ Consolidating price data for memory-mapped access...")
        
        # Calculate full data range needed for all steps
        earliest_start = min(step[0] for step in walk_forward_steps)  # earliest training_start
        latest_end = max(step[3] for step in walk_forward_steps)      # latest testing_end
        
        # Add buffer for safety
        data_start = earliest_start - pd.Timedelta(days=5)
        data_end = latest_end + pd.Timedelta(days=5)
        
        logger.info(f"📊 Consolidating data range: {data_start.date()} → {data_end.date()}")
        
        try:
            # NOTE: WFA queues often run multiple configs in parallel (separate OS processes).
            # If all runs write to the same consolidated parquet path concurrently, it can be
            # corrupted ("Parquet magic bytes not found"). Use a range-keyed cache filename
            # + cross-process lock + atomic replace to make this safe.
            import os
            try:
                import fcntl  # Linux-only (Serverspace) cross-process lock
            except Exception:  # pragma: no cover
                fcntl = None  # type: ignore[assignment]

            def _is_valid_parquet(path: Path) -> bool:
                try:
                    import pyarrow.parquet as pq  # local import to keep module load light
                    _ = pq.ParquetFile(str(path))
                    return True
                except Exception:
                    return False

            cache_dir = Path(cfg.data_dir).parent / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Cache file keyed by data range + timeframe to avoid collisions across runs.
            bar_minutes = int(getattr(cfg.pair_selection, "bar_minutes", 15))
            consolidated_path = (
                cache_dir
                / f"consolidated_prices_{data_start:%Y%m%d}_{data_end:%Y%m%d}_{bar_minutes}m.parquet"
            )
            lock_path = consolidated_path.with_suffix(consolidated_path.suffix + ".lock")

            lock_handle = None
            if fcntl is not None:
                lock_handle = lock_path.open("w")
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

            try:
                # If the cache exists but is corrupted, rebuild it under the lock.
                if consolidated_path.exists() and not _is_valid_parquet(consolidated_path):
                    logger.warning(f"⚠️ Corrupted consolidated parquet detected: {consolidated_path}; rebuilding")
                    try:
                        consolidated_path.unlink()
                    except FileNotFoundError:
                        pass

                if not consolidated_path.exists():
                    tmp_path = consolidated_path.with_suffix(consolidated_path.suffix + f".tmp.{os.getpid()}")
                    consolidated_ok = consolidate_price_data(
                        str(cfg.data_dir),
                        str(tmp_path),
                        data_start,
                        data_end,
                        clean_window=clean_window,
                        exclude_symbols=excluded_symbols,
                    )
                    if not consolidated_ok:
                        raise RuntimeError("consolidate_price_data failed")
                    os.replace(tmp_path, consolidated_path)
            finally:
                if lock_handle is not None and fcntl is not None:
                    try:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        lock_handle.close()

            # Initialize global memory-mapped data
            initialize_global_price_data(str(consolidated_path))
            memory_map_path = str(consolidated_path)

            # Verify memory mapping is working
            if GLOBAL_PRICE is not None and not GLOBAL_PRICE.empty:
                sample_symbols = list(GLOBAL_PRICE.columns[: min(2, len(GLOBAL_PRICE.columns))])
                if sample_symbols:
                    data_view = get_price_data_view(sample_symbols)
                    verify_no_data_copies(data_view, GLOBAL_PRICE)

            logger.info("✅ Memory-mapped data initialized successfully")

            # Start memory monitoring if in debug mode
            if logger.isEnabledFor(10):  # DEBUG level
                monitor_memory_usage()

        except Exception as e:
            logger.warning(f"⚠️ Memory-mapped optimization failed: {e}")
            logger.warning("Falling back to traditional data loading")
            use_memory_map = False

    # Initialize tracking variables
    aggregated_pnl = pd.Series(dtype=float)
    aggregated_turnover_units = pd.Series(dtype=float)
    aggregated_exposure_units = pd.Series(dtype=float)
    daily_pnl = []
    equity_data = []
    pair_count_data = []
    trade_stats = []
    all_trades_log = []
    pair_history: list[list[tuple[str, str]]] = []

    portfolio = Portfolio(
        initial_capital=cfg.portfolio.initial_capital,
        max_active_positions=cfg.portfolio.max_active_positions,
        config=cfg.portfolio,
    )
    equity_data.append((start_date, portfolio.get_current_equity()))

    # Execute walk-forward steps
    step_tracker = ProgressTracker(len(walk_forward_steps), "Walk-forward steps", step=1)
    
    for step_idx, (training_start, training_end, testing_start, testing_end) in enumerate(walk_forward_steps, 1):
        step_tag = f"WF-шаг {step_idx}/{len(walk_forward_steps)}"
        
        # Initialize equity curve with first test window start date (removes artificial 1970-01-01)
        if step_idx == 1:
            portfolio.initialize_equity_curve(testing_start)
            logger.info(f"📈 Инициализация equity кривой с первой тестовой даты: {testing_start}")
        
        # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ: Проверка временных окон на look-ahead bias
        logger.info(f"🔍 {step_tag}: Валидация временных окон...")
        if not validate_time_windows(training_start, training_end, testing_start, testing_end, bar_minutes):
            logger.error(f"❌ {step_tag}: Валидация временных окон не пройдена, пропуск шага")
            continue
        
        with time_block(f"{step_tag}: training {training_start.date()}-{training_end.date()}, testing {testing_start.date()}-{testing_end.date()}"):
            # Load data only for this step
            with time_block("loading step data"):
                step_df_long = load_master_dataset(
                    cfg.data_dir,
                    training_start,
                    testing_end,
                    clean_window=clean_window,
                    exclude_symbols=excluded_symbols,
                )

            if step_df_long.empty:
                logger.warning(f"  Нет данных для шага {step_idx}, пропуск.")
                continue

            step_df = step_df_long.pivot_table(index="timestamp", columns="symbol", values="close", observed=False)
            
            # ДОПОЛНИТЕЛЬНАЯ ВАЛИДАЦИЯ: Проверка наличия данных для временных окон
            expected_testing_days = float((testing_end - testing_start + bar_delta) / pd.Timedelta(days=1))
            min_testing_days = min(float(cfg.walk_forward.testing_period_days), expected_testing_days)
            is_data_valid, data_error = validate_walk_forward_data(
                step_df,
                training_start,
                training_end,
                testing_start,
                testing_end,
                min_training_days=cfg.walk_forward.training_period_days,
                min_testing_days=min_testing_days,
                bar_minutes=bar_minutes,
            )
            if not is_data_valid:
                logger.warning(f"⚠️  {step_tag}: {data_error}, пропуск шага")
                continue
            
            training_slice = step_df.loc[training_start:training_end]
            testing_slice = step_df.loc[testing_start:testing_end]
            
            # Детальное логирование границ данных
            logger.info(f"📊 {step_tag}: Границы данных:")
            logger.info(f"   Training: {training_start} → {training_end} ({training_slice.shape[0]:,} строк × {training_slice.shape[1]} символов)")
            logger.info(f"   Testing:  {testing_start} → {testing_end} ({testing_slice.shape[0]:,} строк × {testing_slice.shape[1]} символов)")
            logger.info(f"   Временной разрыв: {testing_start - training_end}")
            
            # Проверка на перекрытие данных (дополнительная защита)
            if training_slice.index.max() >= testing_slice.index.min():
                logger.error(f"❌ {step_tag}: ОБНАРУЖЕНО ПЕРЕКРЫТИЕ ДАННЫХ!")
                logger.error(f"   Последняя тренировочная метка: {training_slice.index.max()}")
                logger.error(f"   Первая тестовая метка: {testing_slice.index.min()}")
                logger.error("   Это может привести к look-ahead bias! Пропуск шага.")
                continue
                
            training_normalization_params = {}
            if training_slice.empty or len(training_slice.columns) < 2:
                logger.warning(f"  Недостаточно данных для обучения в шаге {step_idx}")
                pairs = []
            else:
                # ИСПРАВЛЕНИЕ LOOKAHEAD BIAS: готовим параметры нормализации на тренировочных данных
                fallback_first_values = {}
                if not training_slice.empty:
                    for col in training_slice.columns:
                        first_valid_idx = training_slice[col].first_valid_index()
                        if first_valid_idx is not None:
                            fallback_first_values[col] = training_slice.loc[first_valid_idx, col]
                
                # Normalize training data
                with time_block("normalizing training data"):
                    logger.debug(f"  Нормализация данных для {len(training_slice.columns)} символов")
                    
                    # Сохраняем список символов до нормализации
                    symbols_before = set(training_slice.columns)
                    
                    # Проверяем символы с постоянной ценой (max=min)
                    constant_price_symbols = [col for col in training_slice.columns
                                             if training_slice[col].max() == training_slice[col].min()]
                    if constant_price_symbols:
                        logger.info(f"  Символы с постоянной ценой: {len(constant_price_symbols)}")
                        logger.debug(f"  Список: {', '.join(sorted(constant_price_symbols))}")
                    
                    # Проверяем символы с NaN значениями
                    nan_symbols = [col for col in training_slice.columns
                                  if training_slice[col].isna().any()]
                    if nan_symbols:
                        logger.info(f"  Символы с NaN значениями: {len(nan_symbols)}")
                        logger.debug(f"  Список: {', '.join(sorted(nan_symbols))}")
                    
                    # Используем улучшенный метод нормализации
                    norm_method = getattr(cfg.data_processing, 'normalization_method', 'minmax')
                    fill_method = getattr(cfg.data_processing, 'fill_method', 'ffill')
                    min_history_ratio = getattr(cfg.data_processing, 'min_history_ratio', 0.8)
                    handle_constant = getattr(cfg.data_processing, 'handle_constant', True)
                    
                    logger.info(f"  Применяем метод нормализации: {norm_method}")
                    rolling_window = getattr(cfg.backtest, 'rolling_window', None)
                    normalized_training, norm_stats = preprocess_and_normalize_data(
                        training_slice,
                        min_history_ratio=min_history_ratio,
                        fill_method=fill_method,
                        norm_method=norm_method,
                        handle_constant=handle_constant,
                        rolling_window=rolling_window,
                        return_stats=True
                    )
                    
                    # Выводим статистику нормализации
                    logger.info("  Статистика нормализации:")
                    logger.info(f"    Исходные символы: {norm_stats['initial_symbols']}")
                    logger.info(f"    Недостаточная история: {norm_stats['low_history_ratio']}")
                    logger.info(f"    Постоянная цена: {norm_stats['constant_price']}")
                    logger.info(f"    NaN после нормализации: {norm_stats['nan_after_norm']}")
                    logger.info(f"    Итоговые символы: {norm_stats['final_symbols']} ({norm_stats['final_symbols']/norm_stats['initial_symbols']*100:.1f}%)")
                    
                    # Анализируем потерянные символы для обратной совместимости
                    symbols_after = set(normalized_training.columns)
                    dropped_symbols = symbols_before - symbols_after
                    logger.info(f"  После нормализации: {len(normalized_training.columns)} символов (потеряно {len(dropped_symbols)})")
                    if dropped_symbols and len(dropped_symbols) <= 20:
                        logger.info(f"  Отброшенные символы: {', '.join(sorted(dropped_symbols))}")
                    elif dropped_symbols:
                        logger.info(f"  Отброшено много символов: {len(dropped_symbols)} из {len(symbols_before)}")
                    
                normalization_stats = norm_stats.get('normalization_stats', {})
                training_normalization_params = {}
                if normalization_stats:
                    method = normalization_stats.get('method', norm_method)
                    if method in ("minmax", "zscore"):
                        params = {}
                        if method == "minmax":
                            mins = normalization_stats.get('min', {})
                            maxs = normalization_stats.get('max', {})
                            for symbol in normalized_training.columns:
                                if symbol in mins and symbol in maxs:
                                    params[symbol] = {'min': mins[symbol], 'max': maxs[symbol]}
                        else:
                            means = normalization_stats.get('mean', {})
                            stds = normalization_stats.get('std', {})
                            for symbol in normalized_training.columns:
                                if symbol in means and symbol in stds:
                                    params[symbol] = {'mean': means[symbol], 'std': stds[symbol]}
                        training_normalization_params = {
                            'method': method,
                            'params': params,
                            'fill_method': fill_method,
                        }
                    else:
                        training_normalization_params = dict(normalization_stats)
                        training_normalization_params['fill_method'] = fill_method
                elif fallback_first_values:
                    training_normalization_params = fallback_first_values

                if len(normalized_training.columns) < 2:
                    logger.warning("  После нормализации осталось менее 2 символов")
                    pairs = []
                else:
                    pairs_for_filter = []
                    if fixed_pairs:
                        available_symbols = set(training_slice.columns)
                        pairs_for_filter = [
                            (s1, s2)
                            for s1, s2 in fixed_pairs
                            if s1 in available_symbols and s2 in available_symbols
                        ]
                        dropped = len(fixed_pairs) - len(pairs_for_filter)
                        logger.info(
                            f"  Фиксированный universe: {len(pairs_for_filter)} пар "
                            f"(отфильтровано {dropped} недоступных)"
                        )
                    else:
                        # SSD computation
                        with time_block("SSD computation"):
                            logger.info("  Расчет SSD для всех пар (без ограничения)")
                            # Сначала считаем SSD для всех пар
                            ssd = math_utils.calculate_ssd(normalized_training, top_k=None)
                            logger.info(f"  SSD результат (все пары): {len(ssd)} пар")

                            # Затем берем только top-N пар для дальнейшей фильтрации
                            ssd_top_n = cfg.pair_selection.ssd_top_n
                            if len(ssd) > ssd_top_n:
                                logger.info(f"  Ограничиваем до top-{ssd_top_n} пар для дальнейшей обработки")
                                ssd = ssd.sort_values().head(ssd_top_n)
                        pairs_for_filter = [(s1, s2) for s1, s2 in ssd.index]

                    if not pairs_for_filter:
                        logger.warning("  Нет пар для фильтрации")
                        pairs = []
                    else:
                        # Filter pairs
                        with time_block("filtering pairs by cointegration and half-life"):
                            from coint2.pipeline.filters import filter_pairs_by_coint_and_half_life
                            filter_n_jobs = getattr(cfg.backtest, 'n_jobs', None)
                            if filter_n_jobs == -1:
                                filter_n_jobs = os.cpu_count() or 1
                            if not filter_n_jobs or filter_n_jobs < 1:
                                filter_n_jobs = 1
                            filter_backend = os.getenv("COINT_FILTER_BACKEND", "threads")
                            requested_liquidity = float(cfg.pair_selection.liquidity_usd_daily or 0.0)
                            requested_bid_ask = float(
                                cfg.pair_selection.max_bid_ask_pct
                                if cfg.pair_selection.max_bid_ask_pct is not None
                                else 1.0
                            )
                            requested_funding = float(
                                cfg.pair_selection.max_avg_funding_pct
                                if cfg.pair_selection.max_avg_funding_pct is not None
                                else 1.0
                            )
                            (
                                liquidity_usd_daily,
                                max_bid_ask_pct,
                                max_avg_funding_pct,
                            ) = cfg.pair_selection.resolved_tradeability_thresholds()
                            if cfg.pair_selection.enable_pair_tradeability_filter and (
                                requested_liquidity != liquidity_usd_daily
                                or requested_bid_ask != max_bid_ask_pct
                                or requested_funding != max_avg_funding_pct
                            ):
                                logger.info(
                                    "  Tradeability guardrail: liquidity %.0f→%.0f, bid_ask %.4f→%.4f, funding %.4f→%.4f",
                                    requested_liquidity,
                                    liquidity_usd_daily,
                                    requested_bid_ask,
                                    max_bid_ask_pct,
                                    requested_funding,
                                    max_avg_funding_pct,
                                )
                            elif not cfg.pair_selection.enable_pair_tradeability_filter:
                                logger.info("  Tradeability filter disabled by config")
                            filtered_pairs = filter_pairs_by_coint_and_half_life(
                                pairs_for_filter,
                                training_slice,
                                pvalue_threshold=cfg.pair_selection.coint_pvalue_threshold,
                                min_beta=cfg.filter_params.min_beta,
                                max_beta=cfg.filter_params.max_beta,
                                max_beta_drift_ratio=cfg.filter_params.max_beta_drift_ratio,
                                min_half_life=cfg.filter_params.min_half_life_days,
                                max_half_life=cfg.filter_params.max_half_life_days,
                                min_mean_crossings=cfg.filter_params.min_mean_crossings,
                                max_hurst_exponent=cfg.filter_params.max_hurst_exponent,
                                ecm_alpha_tstat_threshold=cfg.filter_params.ecm_alpha_tstat_threshold,
                                min_correlation=cfg.pair_selection.min_correlation,
                                liquidity_usd_daily=liquidity_usd_daily,
                                max_bid_ask_pct=max_bid_ask_pct,
                                max_avg_funding_pct=max_avg_funding_pct,
                                require_market_metrics=bool(getattr(cfg.pair_selection, "require_market_metrics", False)),
                                require_same_quote=bool(getattr(cfg.pair_selection, "require_same_quote", False)),
                                min_volume_usd_24h=float(getattr(cfg.pair_selection, "min_volume_usd_24h", 0.0) or 0.0),
                                min_days_live=int(getattr(cfg.pair_selection, "min_days_live", 0) or 0),
                                max_funding_rate_abs=float(
                                    getattr(cfg.pair_selection, "max_funding_rate_abs", 0.0) or 0.0
                                ),
                                max_tick_size_pct=float(getattr(cfg.pair_selection, "max_tick_size_pct", 0.0) or 0.0),
                                save_filter_reasons=cfg.pair_selection.save_filter_reasons,
                                kpss_pvalue_threshold=cfg.pair_selection.kpss_pvalue_threshold,
                                n_jobs=filter_n_jobs,
                                parallel_backend=filter_backend,
                                # Параметры commission_pct и slippage_pct удалены
                            )
                            current_pair_keys = [(s1, s2) for s1, s2, *_ in filtered_pairs]
                            requested_stability_window = int(
                                getattr(cfg.pair_selection, "pair_stability_window_steps", 0) or 0
                            )
                            requested_stability_min = int(
                                getattr(cfg.pair_selection, "pair_stability_min_steps", 0) or 0
                            )
                            (
                                stability_window,
                                stability_min,
                            ) = cfg.pair_selection.resolved_pair_stability()
                            if requested_stability_window and requested_stability_min and (
                                requested_stability_window != stability_window
                                or requested_stability_min != stability_min
                            ):
                                logger.info(
                                    "  Pair stability guardrail: window %d→%d, min_steps %d→%d",
                                    requested_stability_window,
                                    stability_window,
                                    requested_stability_min,
                                    stability_min,
                                )
                            if stability_window and stability_min:
                                history_steps = len(pair_history)
                                if history_steps >= stability_min:
                                    history_window = pair_history[-stability_window:]
                                    streaks = _compute_pair_trailing_streaks(history_window)
                                    stable_pairs = {
                                        pair for pair, streak in streaks.items() if streak >= stability_min
                                    }
                                    before_count = len(filtered_pairs)
                                    rejected_pairs: list[tuple[tuple[str, str], int]] = []
                                    kept_pairs = []
                                    for pair_payload in filtered_pairs:
                                        key = (pair_payload[0], pair_payload[1])
                                        if key in stable_pairs:
                                            kept_pairs.append(pair_payload)
                                        else:
                                            rejected_pairs.append((key, streaks.get(key, 0)))
                                    filtered_pairs = kept_pairs
                                    logger.info(
                                        "  Pair stability filter: %d → %d (window=%d, min_steps=%d, history=%d, rule=trailing_streak)",
                                        before_count,
                                        len(filtered_pairs),
                                        stability_window,
                                        stability_min,
                                        history_steps,
                                    )
                                    if rejected_pairs:
                                        rejected_sample = ", ".join(
                                            f"{s1}/{s2}:streak={streak}"
                                            for (s1, s2), streak in rejected_pairs[:5]
                                        )
                                        logger.info(
                                            "  Pair stability rejects: %d (sample: %s)",
                                            len(rejected_pairs),
                                            rejected_sample,
                                        )
                                else:
                                    logger.info(
                                        "  Pair stability filter: insufficient history (%d < %d), skip",
                                        history_steps,
                                        stability_min,
                                    )
                            pair_history.append(current_pair_keys)
                            logger.info(
                                f"  Фильтрация: {len(pairs_for_filter)} → {len(filtered_pairs)} пар"
                            )
                            pairs = filtered_pairs

        # Select all filtered pairs for trading (no limit here)
        # max_active_positions ограничивает только одновременно открытые позиции.
        pair_quality_weights_by_name: dict[str, float] = {}
        if pairs:
            rank_mode = str(getattr(cfg.pair_selection, "rank_mode", None) or "spread_std")
            quality_sorted_pairs, pair_quality_weights_by_name = rank_pairs(pairs, rank_mode)
            active_pairs = quality_sorted_pairs  # Берем ВСЕ отфильтрованные пары
            max_pairs = int(getattr(cfg.pair_selection, "max_pairs", 0) or 0)
            if max_pairs > 0 and len(active_pairs) > max_pairs:
                active_pairs = active_pairs[:max_pairs]
                logger.info(
                    "  Ограничение max_pairs=%d: %d → %d",
                    max_pairs,
                    len(quality_sorted_pairs),
                    len(active_pairs),
                )
            elif max_pairs > 0:
                logger.info(
                    "  Ограничение max_pairs=%d: %d пар (лимит не достигнут)",
                    max_pairs,
                    len(active_pairs),
                )
            logger.info("  Pair ranking: mode=%s", rank_mode)
            logger.info("  Топ-3 пары по качеству:")
            for i, (s1, s2, beta, mean, std, metrics) in enumerate(active_pairs[:3], 1):
                quality = pair_quality_weights_by_name.get(f"{s1}-{s2}", float("nan"))
                logger.info(
                    "    %d. %s-%s: beta=%.4f, std=%.4f, quality=%.3f",
                    i,
                    s1,
                    s2,
                    beta,
                    std,
                    quality,
                )
        else:
            active_pairs = []
        
        num_active_pairs = len(active_pairs)
        logger.info(f"  Всего пар для торговли: {num_active_pairs} (ограничение {cfg.portfolio.max_active_positions} применяется к одновременно открытым позициям)")

        step_pnl = pd.Series(dtype=float)
        step_turnover_units = pd.Series(dtype=float)
        step_exposure_units = pd.Series(dtype=float)
        total_step_pnl = 0.0

        current_equity = portfolio.get_current_equity()
        
        # Рассчитываем target_concurrency для правильного логирования
        target_concurrency = min(cfg.portfolio.max_active_positions, num_active_pairs) if num_active_pairs > 0 else 0
        
        logger.info(
            f"  💰 Проверка расчета капитала: equity=${current_equity:,.2f}, кандидатных_пар={num_active_pairs}, max_позиций={cfg.portfolio.max_active_positions}"
        )
        logger.info(
            f"  🎯 Target concurrency: {target_concurrency} (ожидаемое одновременное число позиций)"
        )
        
        if num_active_pairs > 0:
            capital_per_pair = portfolio.calculate_position_risk_capital(
                risk_per_position_pct=cfg.portfolio.risk_per_position_pct,
                max_position_size_pct=getattr(cfg.portfolio, 'max_position_size_pct', 1.0),
                num_selected_pairs=num_active_pairs
            )
            
            # Дополнительная диагностика расчетов
            risk_capital = current_equity * cfg.portfolio.risk_per_position_pct
            max_position_capital = current_equity * getattr(cfg.portfolio, 'max_position_size_pct', 1.0)
            base_capital_per_pair = current_equity / target_concurrency if target_concurrency > 0 else 0
            
            logger.info(f"  📊 Детали расчета капитала:")
            logger.info(f"     • Базовый капитал на пару: ${base_capital_per_pair:,.2f} (equity / target_concurrency)")
            logger.info(f"     • Риск-капитал: ${risk_capital:,.2f} ({cfg.portfolio.risk_per_position_pct:.1%} от equity)")
            logger.info(f"     • Макс. размер позиции: ${max_position_capital:,.2f} ({getattr(cfg.portfolio, 'max_position_size_pct', 1.0):.1%} от equity)")
            logger.info(f"     • Итоговый капитал на пару: ${capital_per_pair:,.2f}")
        else:
            capital_per_pair = 0.0

        if capital_per_pair < 0:
            logger.error(
                f"КРИТИЧЕСКАЯ ОШИБКА: Капитал на пару стал отрицательным ({capital_per_pair})."
            )
            capital_per_pair = 0.0

        period_label = f"{training_start.strftime('%m/%d')}-{testing_end.strftime('%m/%d')}"
        pair_count_data.append((period_label, len(active_pairs)))

        # Run backtests for active pairs
        if active_pairs:
            # Create portfolio for position management
            position_portfolio = Portfolio(
                initial_capital=current_equity,
                max_active_positions=cfg.portfolio.max_active_positions,
                config=cfg.portfolio,
            )
            pair_tracker = ProgressTracker(len(active_pairs), f"{step_tag} backtests", step=max(1, len(active_pairs)//5))
            logger.info(f"🚀 {step_tag}: Начинаем бэктесты для {len(active_pairs)} пар...")
            
            # Determine number of parallel jobs
            n_jobs = getattr(cfg.backtest, 'n_jobs', -1)  # -1 means use all available cores
            if n_jobs == -1:
                n_jobs = os.cpu_count()
            
            logger.info(f"🚀 {step_tag}: Запускаем параллельную обработку {len(active_pairs)} пар на {n_jobs} ядрах...")
            
            # Добавляем детальное логирование для больших наборов пар
            if len(active_pairs) > 50:
                logger.info(f"  📊 Большой набор пар ({len(active_pairs)}), ожидаемое время: ~{len(active_pairs)*2//60+1} мин")
                logger.info(f"  🔄 Прогресс будет отображаться каждые {max(1, len(active_pairs)//10)} пар")
            
            # Process pairs in parallel with progress tracking
            start_time = time.time()
            
            if use_memory_map and memory_map_path:
                logger.info(f"🚀 {step_tag}: Режим параллельности: процессы")
                pair_args = [
                    (
                        (s1, s2),
                        testing_start,
                        testing_end,
                        cfg,
                        capital_per_pair,
                        bar_minutes,
                        period_label,
                        (beta, mean, std, metrics),
                        training_normalization_params,
                    )
                    for (s1, s2, beta, mean, std, metrics) in active_pairs
                ]
                pair_results = _parallel_map(
                    _process_pair_mmap_worker,
                    pair_args,
                    n_jobs,
                    use_processes=True,
                    initializer=_init_worker_global_price,
                    initargs=(memory_map_path,),
                )
            else:
                logger.info(f"🚀 {step_tag}: Режим параллельности: потоки")
                pair_args = [
                    (
                        pair_data_tuple,
                        step_df,
                        testing_start,
                        testing_end,
                        cfg,
                        capital_per_pair,
                        bar_minutes,
                        period_label,
                        training_normalization_params,
                    )
                    for pair_data_tuple in active_pairs
                ]
                pair_results = _parallel_map(
                    _process_pair_worker,
                    pair_args,
                    n_jobs,
                    use_processes=False,
                )
            
            processing_time = time.time() - start_time
            logger.info(f"  ⏱️ Параллельная обработка завершена за {processing_time:.1f}с ({len(active_pairs)/processing_time:.1f} пар/с)")
            
            # Vectorized aggregation of results from parallel processing
            import numpy as np
            
            # Separate successful and failed results
            successful_results = [r for r in pair_results if r['success']]
            failed_results = [r for r in pair_results if not r['success']]
            
            successful_count = len(successful_results)
            failed_count = len(failed_results)
            
            if successful_results:
                # Vectorized calculation of statistics
                pnl_values = np.array([r['trade_stat']['total_pnl'] for r in successful_results])
                trade_counts = np.array([r['trade_stat']['trade_count'] for r in successful_results])

                # Vectorized boolean operations
                profitable_count = int(np.sum(pnl_values > 0))
                total_trades_count = int(np.sum(trade_counts))

                all_pnl_series = [result['pnl_series'] for result in successful_results]
                all_positions = [result.get('positions', pd.Series(dtype=float)) for result in successful_results]
                all_scores = [result.get('z_scores', pd.Series(dtype=float)) for result in successful_results]
                pair_entry_weights = [
                    float(pair_quality_weights_by_name.get(result["trade_stat"]["pair"], 0.5))
                    for result in successful_results
                ]
                step_pnl, step_diagnostics = _simulate_realistic_portfolio(
                    all_pnl_series,
                    cfg,
                    all_positions=all_positions,
                    all_scores=all_scores,
                    pair_entry_weights=pair_entry_weights,
                    return_diagnostics=True,
                )
                step_turnover_units = step_diagnostics.get("turnover_units", pd.Series(dtype=float))
                step_exposure_units = step_diagnostics.get("exposure_units", pd.Series(dtype=float))
                total_step_pnl = step_pnl.sum()

                # Aggregate other data
                for result in successful_results:
                    all_trades_log.extend(result['trades_log'])
                    trade_stats.append(result['trade_stat'])
            else:
                profitable_count = 0
                total_trades_count = 0
                total_step_pnl = 0.0
                step_turnover_units = pd.Series(dtype=float)
                step_exposure_units = pd.Series(dtype=float)
            
            # Process failed results
            for result in failed_results:
                logger.warning(f"  ⚠️ Ошибка обработки пары {result['trade_stat']['pair']}: {result.get('error', 'Unknown error')}")
                trade_stats.append(result['trade_stat'])
                
                pair_tracker.update()
                
                # Промежуточная статистика для больших наборов
                if len(active_pairs) > 100 and i % (len(active_pairs)//4) == 0:
                    current_pnl = sum([r['trade_stat']['total_pnl'] for r in pair_results[:i] if r['success']])
                    logger.info(f"  📈 Промежуточно ({i}/{len(active_pairs)}): P&L ${current_pnl:+,.0f}, {profitable_count}/{successful_count} прибыльных")
            
            pair_tracker.finish()
            
            # Детальная статистика по завершении
            logger.info(f"  ✅ Обработка завершена: {successful_count} успешно, {failed_count} с ошибками")
            if successful_count > 0:
                success_rate = (successful_count / len(active_pairs)) * 100
                profit_rate = (profitable_count / successful_count) * 100
                avg_trades_per_pair = total_trades_count / successful_count
                logger.info(f"  📊 Успешность: {success_rate:.1f}%, прибыльность: {profit_rate:.1f}%, сделок/пара: {avg_trades_per_pair:.1f}")
            logger.info(f"  ✅ Завершены бэктесты для {len(active_pairs)} пар")
            
            # Счетчик обработанных пар
            total_processed_pairs = sum(len(pair_results) for pair_results in [pair_results] if pair_results)
            logger.info(f"  📊 Всего обработано пар в этом шаге: {total_processed_pairs}")
            
            # Статистика по результатам бэктестов
            successful_pairs = len([stat for stat in trade_stats if stat['period'] == period_label and stat['total_pnl'] != 0])
            profitable_pairs = len([stat for stat in trade_stats if stat['period'] == period_label and stat['total_pnl'] > 0])
            total_trades = sum([stat['trade_count'] for stat in trade_stats if stat['period'] == period_label])
            
            logger.info(f"  📈 Результаты шага: {profitable_pairs}/{len(active_pairs)} прибыльных пар, {total_trades} сделок")
            logger.info(f"  💰 P&L шага: ${total_step_pnl:+,.2f}")
            
            # Детальная статистика по сделкам
            if all_trades_log:
                step_trades = [t for t in all_trades_log if t.get('period') == period_label]
                if step_trades:
                    winning_trades = [t for t in step_trades if t.get('pnl', 0) > 0]
                    losing_trades = [t for t in step_trades if t.get('pnl', 0) < 0]
                    win_rate = len(winning_trades) / len(step_trades) * 100 if step_trades else 0
                    avg_win = sum([t['pnl'] for t in winning_trades]) / len(winning_trades) if winning_trades else 0
                    avg_loss = sum([t['pnl'] for t in losing_trades]) / len(losing_trades) if losing_trades else 0
                    profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades else float('inf')
                    
                    logger.info(f"  📊 Винрейт: {win_rate:.1f}% ({len(winning_trades)}/{len(step_trades)})")
                    logger.info(f"  💵 Средний выигрыш: ${avg_win:.0f}, средний проигрыш: ${avg_loss:.0f}")
                    logger.info(f"  🎯 Profit Factor: {profit_factor:.2f}")
                    
                    # Топ-3 пары по прибыли в этом шаге
                    step_pair_stats = [ts for ts in trade_stats if ts['period'] == period_label and ts['total_pnl'] > 0]
                    top_pairs = sorted(step_pair_stats, key=lambda x: x['total_pnl'], reverse=True)[:3]
                    if top_pairs:
                        top_pairs_str = ', '.join([f"{p['pair']} ${p['total_pnl']:+,.0f}" for p in top_pairs])
                        logger.info(f"  🏆 Топ пары: {top_pairs_str}")
            
            logger.info(f"  ⇢ Обновление капитала и дневного P&L...")
        else:
            # IMPORTANT: When no pairs pass selection, the portfolio is flat and PnL is 0.
            # We still must record the testing window as zeros; otherwise Sharpe/volatility
            # are biased upward by silently skipping non-trading windows.
            if not testing_slice.empty:
                step_pnl = pd.Series(0.0, index=testing_slice.index)
                step_turnover_units = pd.Series(0.0, index=testing_slice.index)
                step_exposure_units = pd.Series(0.0, index=testing_slice.index)
                total_step_pnl = 0.0
                logger.info(
                    "  Нет пар после отбора: записываем нулевой P&L для окна теста (%d баров).",
                    len(step_pnl),
                )

        # Update equity and daily P&L
        if not step_pnl.empty:
            running_equity = portfolio.get_current_equity()
            for date, pnl in step_pnl.items():
                daily_pnl.append((date, pnl))
                running_equity += pnl
                equity_data.append((date, running_equity))
                portfolio.record_daily_pnl(pd.Timestamp(date), pnl)

        aggregated_pnl = pd.concat([aggregated_pnl, step_pnl])
        aggregated_turnover_units = pd.concat([aggregated_turnover_units, step_turnover_units])
        aggregated_exposure_units = pd.concat([aggregated_exposure_units, step_exposure_units])

        logger.info(
            f"  💼 Шаг P&L: ${total_step_pnl:+,.2f}, Накопленный капитал: ${portfolio.get_current_equity():,.2f}"
        )
        logger.info("-" * 60)
        step_tracker.update()

    step_tracker.finish()
    
    # Final comprehensive results
    final_equity = portfolio.get_current_equity()
    initial_capital = cfg.portfolio.initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100
    total_pnl = final_equity - initial_capital
    
    # Vectorized comprehensive trade statistics
    total_trades = len(all_trades_log)
    if total_trades > 0:
        # Extract PnL values for vectorized operations
        trade_pnls = np.array([t.get('pnl', 0) for t in all_trades_log])
        
        # Vectorized win/loss calculations
        winning_mask = trade_pnls > 0
        losing_mask = trade_pnls < 0
        
        winning_trades_count = int(np.sum(winning_mask))
        losing_trades_count = int(np.sum(losing_mask))
        
        overall_win_rate = winning_trades_count / total_trades * 100
        total_wins_pnl = np.sum(trade_pnls[winning_mask])
        total_losses_pnl = np.sum(trade_pnls[losing_mask])
        overall_profit_factor = abs(total_wins_pnl / total_losses_pnl) if total_losses_pnl != 0 else float('inf')
        avg_trade_pnl = total_pnl / total_trades
        
        # Vectorized pair-level statistics
        unique_pairs = set([ts['pair'] for ts in trade_stats])
        
        # Use numpy for faster filtering
        trade_counts = np.array([ts['trade_count'] for ts in trade_stats])
        pair_pnls = np.array([ts['total_pnl'] for ts in trade_stats])
        
        active_pairs_mask = trade_counts > 0
        profitable_pairs_mask = pair_pnls > 0
        
        active_pairs = [ts for i, ts in enumerate(trade_stats) if active_pairs_mask[i]]
        profitable_pairs = [ts for i, ts in enumerate(trade_stats) if profitable_pairs_mask[i]]
    
    # Подсчет общего количества обработанных пар
    total_processed_pairs_all_steps = len(trade_stats)
    
    logger.info("="*80)
    logger.info(f"🏁 Walk-forward анализ завершен!")
    logger.info(f"⏱️ Обработано {len(walk_forward_steps)} шагов за {time.time() - start_time:.1f} секунд")
    logger.info(f"📊 Всего обработано пар за весь анализ: {total_processed_pairs_all_steps}")
    logger.info(f"💰 Финальный капитал: ${final_equity:,.2f} (было ${initial_capital:,.2f})")
    logger.info(f"📈 Общая доходность: {total_return:+.2f}% (${total_pnl:+,.2f})")
    logger.info(f"📊 Всего сделок: {total_trades}")
    
    if total_trades > 0:
        logger.info(f"🎯 Общий винрейт: {overall_win_rate:.1f}% ({len(winning_trades)}/{total_trades})")
        logger.info(f"💵 Средняя сделка: ${avg_trade_pnl:+.0f}")
        logger.info(f"🔥 Profit Factor: {overall_profit_factor:.2f}")
        logger.info(f"🔄 Активных пар: {len(active_pairs)}/{len(unique_pairs)}, прибыльных: {len(profitable_pairs)}")
        
        # Топ-5 пар по общей прибыли
        top_pairs_overall = sorted(profitable_pairs, key=lambda x: x['total_pnl'], reverse=True)[:5]
        if top_pairs_overall:
            top_pairs_overall_str = ', '.join([f"{p['pair']} ${p['total_pnl']:+,.0f}" for p in top_pairs_overall])
            logger.info(f"🏆 Топ-5 пар: {top_pairs_overall_str}")
    
    logger.info("="*80)
    logger.info(f"✓ Завершены все {len(walk_forward_steps)} WF-шагов")
    logger.info(f"⇢ Обработка финальных результатов и расчет метрик...")

    # Process results and create metrics
    with time_block("processing final results"):
        aggregated_pnl = aggregated_pnl.dropna()
        logger.info(f"  ⇢ Агрегация данных P&L за {len(aggregated_pnl)} торговых дней")
        
        # Create series for analysis
        if daily_pnl:
            dates, pnls = zip(*daily_pnl)
            pnl_series = pd.Series(pnls, index=pd.to_datetime(dates))
            pnl_series = pnl_series.groupby(pnl_series.index.date).sum()
            pnl_series.index = pd.to_datetime(pnl_series.index)
        else:
            pnl_series = pd.Series(dtype=float)
        
        if not portfolio.equity_curve.empty:
            equity_series = portfolio.equity_curve.dropna()
            # Some edge cases (e.g., all-zero PnL windows) can end up with an empty
            # non-NaN equity curve while still having valid daily aggregation.
            # Fallback to a synthetic equity curve from daily pnl so we still
            # persist artifacts and let ranking gates decide later.
            if equity_series.empty and not pnl_series.empty:
                equity_series = (pnl_series.fillna(0).sort_index().cumsum() + cfg.portfolio.initial_capital)
        if ("equity_series" not in locals()) or equity_series.empty:
            if not aggregated_pnl.empty:
                # Last-resort deterministic fallback for robustness.
                if pnl_series.empty:
                    pnl_series = pd.Series([0.0])
                equity_series = (pnl_series.fillna(0).sort_index().cumsum() + cfg.portfolio.initial_capital)
            else:
                equity_series = pd.Series([cfg.portfolio.initial_capital])

        # Calculate metrics
        logger.info(f"  ⇢ Расчет базовых метрик производительности...")
        logger.info(f"    📊 Данные для анализа: {len(aggregated_pnl)} записей P&L, {len(equity_series)} точек equity")
        
        if aggregated_pnl.empty:
            base_metrics = {
                "sharpe_ratio_abs": 0.0,
                "sharpe_ratio_on_returns": 0.0,
                "max_drawdown_abs": 0.0,
                "max_drawdown_on_equity": 0.0,
                "total_pnl": 0.0,
            }
        else:
            cumulative = aggregated_pnl.cumsum()
            # Используем капитал для расчета риска на сделку и доходности
            capital_per_pair = portfolio.calculate_position_risk_capital(
                risk_per_position_pct=cfg.portfolio.risk_per_position_pct,
                max_position_size_pct=getattr(cfg.portfolio, 'max_position_size_pct', 1.0),
                num_selected_pairs=1  # For final metrics calculation, use 1 as default
            )
            equity_returns = equity_series.ffill().pct_change(fill_method=None).dropna()
            periods_per_year = float(cfg.backtest.annualizing_factor)
            if bar_minutes and bar_minutes > 0:
                periods_per_year *= 24 * 60 / bar_minutes

            sharpe_abs = performance.sharpe_ratio(equity_returns, periods_per_year)
            sharpe_on_returns = performance.sharpe_ratio_on_returns(
                aggregated_pnl, capital_per_pair, periods_per_year
            )
            max_dd_abs = performance.max_drawdown(cumulative)
            max_dd_on_equity = performance.max_drawdown_on_equity(equity_series)
            base_metrics = {
                "sharpe_ratio_abs": sharpe_abs,
                "sharpe_ratio_on_returns": sharpe_on_returns,
                "max_drawdown_abs": max_dd_abs,
                "max_drawdown_on_equity": max_dd_on_equity,
                "total_pnl": cumulative.iloc[-1] if not cumulative.empty else 0.0,
            }
        
        logger.info(f"  ⇢ Расчет расширенных метрик (волатильность, Calmar ratio, и др.)...")
        expected_test_start = str(cfg.walk_forward.start_date)
        expected_test_end = str(cfg.walk_forward.end_date)
        # When walk_forward.max_steps truncates execution, the configured end_date can be far
        # beyond the executed test window. Use the effective executed test bounds so coverage_ratio
        # measures "missing days" within what we actually ran.
        if walk_forward_steps:
            try:
                expected_test_start = str(walk_forward_steps[0][2].date())
                expected_test_end = str(walk_forward_steps[-1][3].date())
            except Exception:  # noqa: BLE001
                expected_test_start = str(cfg.walk_forward.start_date)
                expected_test_end = str(cfg.walk_forward.end_date)
        extended_metrics = calculate_extended_metrics(
            pnl_series,
            equity_series,
            expected_test_start=expected_test_start,
            expected_test_end=expected_test_end,
        )
        logger.info(f"  ✅ Расширенные метрики рассчитаны: {len(extended_metrics)} показателей")
        
        # Trade statistics
        logger.info(f"  ⇢ Анализ статистики по {len(trade_stats)} парам...")
        trade_metrics = {}
        if trade_stats:
            trades_df = pd.DataFrame(trade_stats)
            trade_metrics = {
                'total_trades': trades_df['trade_count'].sum(),
                'total_pairs_traded': len(trades_df['pair'].unique()),
                'total_costs': trades_df['total_costs'].sum(),
                'avg_trades_per_pair': trades_df['trade_count'].mean(),
                'win_rate_trades': trades_df['win_days'].sum() / max(trades_df['total_days'].sum(), 1),
                'best_pair_pnl': trades_df['total_pnl'].max(),
                'worst_pair_pnl': trades_df['total_pnl'].min(),
                'avg_pnl_per_pair': trades_df['total_pnl'].mean(),
            }

            if 'entry_notional_count' in trades_df.columns:
                total_entry_count = trades_df['entry_notional_count'].sum()
                total_entry_notional = (trades_df['entry_notional_avg'] * trades_df['entry_notional_count']).sum()
                nonzero_entry = trades_df['entry_notional_count'] > 0
                if nonzero_entry.any():
                    entry_notional_p50 = float(trades_df.loc[nonzero_entry, 'entry_notional_p50'].median())
                    entry_notional_min = float(trades_df.loc[nonzero_entry, 'entry_notional_min'].min())
                    entry_notional_max = float(trades_df.loc[nonzero_entry, 'entry_notional_max'].max())
                else:
                    entry_notional_p50 = 0.0
                    entry_notional_min = 0.0
                    entry_notional_max = 0.0
                trade_metrics.update({
                    'entry_notional_count': float(total_entry_count),
                    'entry_notional_cap_hits': float(trades_df['entry_notional_cap_hits'].sum()),
                    'entry_notional_below_min': float(trades_df['entry_notional_below_min'].sum()),
                    'entry_notional_avg': float(total_entry_notional / max(total_entry_count, 1)),
                    'entry_notional_p50': entry_notional_p50,
                    'entry_notional_min': entry_notional_min,
                    'entry_notional_max': entry_notional_max,
                })
        
        all_metrics = {**base_metrics, **extended_metrics, **trade_metrics}
        logger.info(f"✅ Рассчитаны метрики производительности ({len(all_metrics)} показателей)")
        logger.info(f"  💰 Общий P&L: ${all_metrics.get('total_pnl', 0):+,.2f}")
        logger.info(f"  📊 Sharpe Ratio: {all_metrics.get('sharpe_ratio_abs', 0):.3f}")
        logger.info(f"  📉 Максимальная просадка: {all_metrics.get('max_drawdown_abs', 0):.2%}")
        logger.info(f"  🔄 Всего сделок: {all_metrics.get('total_trades', 0)}")
        logger.info(f"  📈 Торгуемых пар: {all_metrics.get('total_pairs_traded', 0)}")
        logger.info(f"  💸 Общие издержки: ${all_metrics.get('total_costs', 0):,.2f}")

        _emit_monitoring_metrics(all_metrics)
    
    # Create reports
    logger.info(f"⇢ Создание отчетов и сохранение результатов...")
    with time_block("generating reports"):
        results_dir = Path(cfg.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save core artifacts first. Downstream queue/ranking relies on these files.
        logger.info("  ⇢ Сохранение ключевых артефактов (strategy_metrics/equity_curve/stats)...")
        try:
            # Приводим все числовые значения к float для избежания проблем с типами Arrow
            clean_metrics = {}
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    clean_metrics[key] = float(value)
                else:
                    clean_metrics[key] = value

            metrics_df = pd.DataFrame([clean_metrics])
            metrics_path = results_dir / "strategy_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            if not metrics_path.exists() or metrics_path.stat().st_size <= 0:
                raise RuntimeError(f"failed to write non-empty strategy_metrics.csv: {metrics_path}")
            logger.info(f"📋 Метрики сохранены: {metrics_path}")

            if not pnl_series.empty:
                pnl_series.to_csv(results_dir / "daily_pnl.csv", header=["PnL"])
                logger.info(
                    f"📈 Дневные P&L сохранены: {results_dir / 'daily_pnl.csv'} ({len(pnl_series)} записей)"
                )

            # equity_curve.csv is required for canonical recompute + Sharpe audit.
            if equity_series.empty:
                raise RuntimeError("equity_series is empty; refusing to write equity_curve.csv")
            equity_series.to_csv(results_dir / "equity_curve.csv", header=["Equity"])
            logger.info(f"💹 Кривая капитала сохранена: {results_dir / 'equity_curve.csv'} ({len(equity_series)} точек)")

            if trade_stats:
                trades_df = pd.DataFrame(trade_stats)
                trades_df.to_csv(results_dir / "trade_statistics.csv", index=False)
                logger.info(f"📋 Статистика сделок сохранена: {results_dir / 'trade_statistics.csv'} ({len(trades_df)} записей)")

            if all_trades_log:
                trades_log_df = pd.DataFrame(all_trades_log)
                trades_log_df.to_csv(results_dir / "trades_log.csv", index=False)
                logger.info(f"📓 Детальный лог сделок сохранен: {results_dir / 'trades_log.csv'} ({len(trades_log_df)} записей)")
        except Exception as e:
            # Fail-closed: if we cannot write the core artifacts, the run must be treated as failed
            # (otherwise queues can mark "completed" with no metrics and the loop stalls).
            logger.error(f"Ошибка при сохранении ключевых артефактов: {e}")
            raise

        # Optional: performance report and human-readable summary.
        try:
            logger.info("  ⇢ Создание отчета о производительности...")
            create_performance_report(
                equity_curve=equity_series,
                pnl_series=pnl_series,
                metrics=all_metrics,
                pair_counts=pair_count_data,
                results_dir=results_dir,
                strategy_name="CointegrationStrategy",
            )
            logger.info("  ✓ Отчет о производительности создан")
        except Exception as e:
            logger.error(f"Ошибка при создании отчета о производительности: {e}")

        try:
            summary = format_metrics_summary(all_metrics)
            print(summary)
        except Exception as e:
            logger.error(f"Ошибка при форматировании summary: {e}")

        logger.info("🎉 Walk-Forward анализ полностью завершен!")
        logger.info(f"📊 Все отчеты и данные сохранены в: {results_dir}")
        logger.info(f"💼 Итоговый капитал: ${portfolio.get_current_equity():,.2f}")
        logger.info(f"⏱️  Общее время выполнения: {time.time() - start_time:.1f} секунд")

    # Cleanup memory-mapped data
    if use_memory_map:
        try:
            cleanup_global_data()
            logger.info("🧹 Memory-mapped data cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up memory-mapped data: {e}")

    equity_curve = portfolio.equity_curve
    equity_returns = equity_curve.ffill().pct_change(fill_method=None).dropna()
    periods_per_year = float(cfg.backtest.annualizing_factor)
    if bar_minutes and bar_minutes > 0:
        periods_per_year *= 24 * 60 / bar_minutes

    sharpe = performance.sharpe_ratio(equity_returns, periods_per_year)
    print(f"Annualized Sharpe Ratio: {sharpe}")

    return base_metrics


class WalkForwardOrchestrator:
    """Compatibility wrapper for running walk-forward analysis."""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        """Run walk-forward analysis."""
        return run_walk_forward(self.cfg)
