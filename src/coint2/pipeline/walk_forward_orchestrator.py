"""Walk-forward analysis orchestrator."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from coint2.core import math_utils, performance
from coint2.core.data_loader import DataHandler, load_master_dataset
from coint2.engine.backtest_engine import PairBacktester
from coint2.utils.config import AppConfig
from coint2.utils.logging_utils import get_logger
from coint2.utils.timing_utils import logged_time, time_block, ProgressTracker
from coint2.utils.visualization import (
    create_performance_report, 
    format_metrics_summary, 
    calculate_extended_metrics
)


@logged_time("run_walk_forward_analysis")
def run_walk_forward(cfg: AppConfig) -> dict[str, float]:
    """Run walk-forward analysis and return aggregated performance metrics."""
    logger = get_logger("walk_forward")

    # Setup and initial data loading
    with time_block("initializing data handler"):
        handler = DataHandler(cfg)
        handler.clear_cache()

    start_date = pd.to_datetime(cfg.walk_forward.start_date)
    end_date = pd.to_datetime(cfg.walk_forward.end_date)
    # Для новой логики: нужны данные на training_period_days раньше start_date
    full_range_start = start_date - pd.Timedelta(days=cfg.walk_forward.training_period_days)
    
    logger.info(f"📅 НОВАЯ ЛОГИКА: start_date = начало ТЕСТИРОВАНИЯ")
    logger.info(f"📊 Данные нужны с: {full_range_start.date()} (для тренировки первого шага)")
    
    logger.info(f"🎯 Walk-Forward анализ: {start_date.date()} → {end_date.date()}")
    logger.info(f"Training период: {cfg.walk_forward.training_period_days} дней")
    logger.info(f"Testing период: {cfg.walk_forward.testing_period_days} дней")


    # Calculate walk-forward steps
    # ИСПРАВЛЕНИЕ: start_date теперь начало ТЕСТОВОГО периода, а не тренировочного
    current_test_start = start_date
    walk_forward_steps = []
    while current_test_start < end_date:
        # Тренировочный период ПРЕДШЕСТВУЕТ тестовому
        training_start = current_test_start - pd.Timedelta(days=cfg.walk_forward.training_period_days)
        training_end = current_test_start
        testing_start = current_test_start
        testing_end = testing_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days)
        
        # ИСПРАВЛЕНИЕ: Позволяем тестовому окну выходить на несколько дней за end_date
        # если доступны данные. Проверяем наличие данных вместо жесткой проверки дат.
        buffer_days = 5  # Позволяем до 5 дней буфера
        if testing_end > end_date + pd.Timedelta(days=buffer_days):
            logger.info(f"  Тестовое окно {testing_end.date()} слишком далеко за end_date {end_date.date()}")
            break
        
            
        logger.info(f"  Шаг {len(walk_forward_steps)+1}: тренировка {training_start.date()}-{training_end.date()}, тест {testing_start.date()}-{testing_end.date()}")
        walk_forward_steps.append((training_start, training_end, testing_start, testing_end))
        current_test_start = testing_end

    logger.info(f"📈 Запланировано {len(walk_forward_steps)} walk-forward шагов")
    
    # ОТЛАДОЧНАЯ ИНФОРМАЦИЯ для диагностики проблем
    if len(walk_forward_steps) == 0:
        logger.warning("⚠️  ДИАГНОСТИКА: Нет walk-forward шагов!")
        logger.warning(f"   start_date (начало теста): {start_date.date()}")
        logger.warning(f"   end_date (конец выборки): {end_date.date()}")
        logger.warning(f"   training_period_days: {cfg.walk_forward.training_period_days}")
        logger.warning(f"   testing_period_days: {cfg.walk_forward.testing_period_days}")
        
        # Рассчитаем, что должно было быть
        would_be_testing_end = start_date + pd.Timedelta(days=cfg.walk_forward.testing_period_days)
        needed_training_start = start_date - pd.Timedelta(days=cfg.walk_forward.training_period_days)
        
        logger.warning(f"   testing_end был бы: {would_be_testing_end.date()}")
        logger.warning(f"   нужны данные с: {needed_training_start.date()}")
        
        if would_be_testing_end > end_date:
            logger.warning(f"   ❌ ПРИЧИНА: testing_end ({would_be_testing_end.date()}) > end_date ({end_date.date()})")
            logger.warning(f"   ✅ РЕШЕНИЕ: Продлите end_date до {would_be_testing_end.date()} или сократите testing_period_days")
        
    else:
        logger.info(f"✅ Шаги успешно запланированы:")
        for i, (tr_start, tr_end, te_start, te_end) in enumerate(walk_forward_steps, 1):
            logger.info(f"   Шаг {i}: тренировка {tr_start.date()}-{tr_end.date()}, тест {te_start.date()}-{te_end.date()}")

    # Initialize tracking variables
    aggregated_pnl = pd.Series(dtype=float)
    daily_pnl = []
    equity_data = []
    pair_count_data = []
    trade_stats = []
    equity = cfg.portfolio.initial_capital
    equity_curve = [equity]
    equity_data.append((start_date, equity))

    # Execute walk-forward steps
    step_tracker = ProgressTracker(len(walk_forward_steps), "Walk-forward steps", step=1)
    
    for step_idx, (training_start, training_end, testing_start, testing_end) in enumerate(walk_forward_steps, 1):
        step_tag = f"WF-шаг {step_idx}/{len(walk_forward_steps)}"
        
        with time_block(f"{step_tag}: training {training_start.date()}-{training_end.date()}, testing {testing_start.date()}-{testing_end.date()}"):
            # Load data only for this step
            with time_block("loading step data"):
                step_df_long = load_master_dataset(cfg.data_dir, training_start, testing_end)

            if step_df_long.empty:
                logger.warning(f"  Нет данных для шага {step_idx}, пропуск.")
                continue

            step_df = step_df_long.pivot_table(index="timestamp", columns="symbol", values="close")
            training_slice = step_df.loc[training_start:training_end]
            logger.info(f"  Training данные: {training_slice.shape[0]:,} строк × {training_slice.shape[1]} символов")
                
            if training_slice.empty or len(training_slice.columns) < 2:
                logger.warning(f"  Недостаточно данных для обучения в шаге {step_idx}")
                pairs = []
            else:
                # Normalize training data
                with time_block("normalizing training data"):
                    logger.debug(f"  Нормализация данных для {len(training_slice.columns)} символов")
                    normalized_training = (training_slice - training_slice.min()) / (
                        training_slice.max() - training_slice.min()
                    )
                    normalized_training = normalized_training.dropna(axis=1)
                    logger.info(f"  После нормализации: {len(normalized_training.columns)} символов")
                    
                if len(normalized_training.columns) < 2:
                    logger.warning("  После нормализации осталось менее 2 символов")
                    pairs = []
                else:
                    # SSD computation
                    with time_block("SSD computation"):
                        logger.info(f"  Расчет SSD для топ-{cfg.pair_selection.ssd_top_n} пар")
                        ssd = math_utils.calculate_ssd(
                            normalized_training, top_k=cfg.pair_selection.ssd_top_n
                        )
                        logger.info(f"  SSD результат: {len(ssd)} пар")
                        
                    # Filter pairs
                    with time_block("filtering pairs by cointegration and half-life"):
                        from coint2.pipeline.filters import filter_pairs_by_coint_and_half_life
                        ssd_pairs = [(s1, s2) for s1, s2 in ssd.index]
                        filtered_pairs = filter_pairs_by_coint_and_half_life(
                            ssd_pairs,
                            training_slice,
                            pvalue_threshold=cfg.pair_selection.coint_pvalue_threshold,
                            min_half_life=cfg.pair_selection.min_half_life_days,
                            max_half_life=cfg.pair_selection.max_half_life_days
                        )
                        logger.info(f"  Фильтрация: {len(ssd_pairs)} → {len(filtered_pairs)} пар")
                        pairs = filtered_pairs

        # Select active pairs and run backtests
        # Сортируем пары по качеству (по убыванию std, что означает большую волатильность спреда)
        # Alternatively: можно сортировать по другим критериям качества
        if pairs:
            # Сортируем пары по стандартному отклонению спреда (больше = потенциально более прибыльно)
            quality_sorted_pairs = sorted(pairs, key=lambda x: abs(x[4]), reverse=True)  # x[4] = std
            active_pairs = quality_sorted_pairs[:cfg.portfolio.max_active_positions]
            logger.info(f"  Топ-3 пары по волатильности спреда:")
            for i, (s1, s2, beta, mean, std) in enumerate(active_pairs[:3], 1):
                logger.info(f"    {i}. {s1}-{s2}: beta={beta:.4f}, std={std:.4f}")
        else:
            active_pairs = []
        
        logger.info(f"  Активных пар для торговли: {len(active_pairs)}")

        step_pnl = pd.Series(dtype=float)
        total_step_pnl = 0.0

        if active_pairs:
            capital_per_pair = equity * cfg.portfolio.risk_per_position_pct
            logger.info(f"  Капитал на пару: ${capital_per_pair:,.0f}")
        else:
            capital_per_pair = 0.0

        period_label = f"{training_start.strftime('%m/%d')}-{testing_end.strftime('%m/%d')}"
        pair_count_data.append((period_label, len(active_pairs)))

        # Run backtests for active pairs
        if active_pairs:
            pair_tracker = ProgressTracker(len(active_pairs), f"{step_tag} backtests", step=max(1, len(active_pairs)//5))
            
            for pair_idx, (s1, s2, beta, mean, std) in enumerate(active_pairs, 1):
                pair_data = step_df.loc[testing_start:testing_end, [s1, s2]].dropna()
                # Нормализация: оба ряда начинаются с 100
                if not pair_data.empty:
                    pair_data = pair_data / pair_data.iloc[0] * 100
                
                bt = PairBacktester(
                    pair_data,
                    beta=beta,
                    spread_mean=mean,
                    spread_std=std,
                    z_threshold=cfg.backtest.zscore_threshold,
                    z_exit=getattr(cfg.backtest, 'zscore_exit', 0.0),
                    commission_pct=getattr(cfg.backtest, 'commission_pct', 0.0),
                    slippage_pct=getattr(cfg.backtest, 'slippage_pct', 0.0),
                    annualizing_factor=getattr(cfg.backtest, 'annualizing_factor', 365),
                    capital_at_risk=capital_per_pair,
                    stop_loss_multiplier=getattr(cfg.backtest, 'stop_loss_multiplier', 2.0),
                    cooldown_periods=int(getattr(cfg.backtest, 'cooldown_hours', 0) / 24),  # Convert hours to daily periods
                )
                bt.run()
                results = bt.get_results()
                pnl_series = results["pnl"]
                step_pnl = step_pnl.add(pnl_series, fill_value=0)
                total_step_pnl += pnl_series.sum()
                
                # Собираем статистику по сделкам для данной пары
                if isinstance(results, dict):
                    trades = results.get("trades", pd.Series())
                    positions = results.get("position", pd.Series())
                    costs = results.get("costs", pd.Series())
                else:
                    # Если results - DataFrame, используем колонки
                    trades = results.get("trades", results.get("trades", pd.Series()))
                    positions = results.get("position", results.get("position", pd.Series()))
                    costs = results.get("costs", results.get("costs", pd.Series()))
                
                # Считаем количество открытий/закрытий позиций
                if not positions.empty:
                    position_changes = positions.diff().fillna(0).abs()
                    trade_opens = (position_changes > 0).sum()
                else:
                    trade_opens = 0
                
                # Всегда добавляем статистику по парам, даже если сделок не было
                pair_pnl = pnl_series.sum()
                pair_costs = costs.sum() if not costs.empty else 0
                
                trade_stats.append({
                    'pair': f'{s1}-{s2}',
                    'period': period_label,
                    'total_pnl': pair_pnl,
                    'total_costs': pair_costs,
                    'trade_count': trade_opens,
                    'avg_pnl_per_trade': pair_pnl / max(trade_opens, 1),
                    'win_days': (pnl_series > 0).sum(),
                    'lose_days': (pnl_series < 0).sum(),
                    'total_days': len(pnl_series),
                    'max_daily_gain': pnl_series.max(),
                    'max_daily_loss': pnl_series.min()
                })
                
                pair_tracker.update()
            
            pair_tracker.finish()

        # Update equity and daily P&L
        if not step_pnl.empty:
            running_equity = equity
            for date, pnl in step_pnl.items():
                daily_pnl.append((date, pnl))
                running_equity += pnl
                equity_data.append((date, running_equity))

        aggregated_pnl = pd.concat([aggregated_pnl, step_pnl])
        equity += total_step_pnl
        equity_curve.append(equity)
        
        logger.info(f"  Шаг P&L: ${total_step_pnl:+,.2f}, Накопленный: ${equity:,.2f}")
        step_tracker.update()

    step_tracker.finish()

    # Process results and create metrics
    with time_block("processing final results"):
        aggregated_pnl = aggregated_pnl.dropna()
        
        # Create series for analysis
        if daily_pnl:
            dates, pnls = zip(*daily_pnl)
            pnl_series = pd.Series(pnls, index=pd.to_datetime(dates))
            pnl_series = pnl_series.groupby(pnl_series.index.date).sum()
            pnl_series.index = pd.to_datetime(pnl_series.index)
        else:
            pnl_series = pd.Series(dtype=float)
        
        if equity_data:
            eq_dates, eq_values = zip(*equity_data)
            equity_series = pd.Series(eq_values, index=pd.to_datetime(eq_dates))
        else:
            equity_series = pd.Series([cfg.portfolio.initial_capital])

        # Calculate metrics
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
            capital_per_pair = cfg.portfolio.initial_capital / max(cfg.portfolio.max_active_positions, 1)
            sharpe_abs = performance.sharpe_ratio(aggregated_pnl, cfg.backtest.annualizing_factor)
            sharpe_on_returns = performance.sharpe_ratio_on_returns(
                aggregated_pnl, capital_per_pair, cfg.backtest.annualizing_factor
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
        
        extended_metrics = calculate_extended_metrics(pnl_series, equity_series)
        
        # Trade statistics
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
        
        all_metrics = {**base_metrics, **extended_metrics, **trade_metrics}
    
    # Create reports
    with time_block("generating reports"):
        results_dir = Path(cfg.results_dir)
        
        try:
            create_performance_report(
                equity_curve=equity_series,
                pnl_series=pnl_series,
                metrics=all_metrics,
                pair_counts=pair_count_data,
                results_dir=results_dir,
                strategy_name="CointegrationStrategy"
            )
            
            summary = format_metrics_summary(all_metrics)
            print(summary)
            
            # Save data
            metrics_df = pd.DataFrame([all_metrics])
            metrics_df.to_csv(results_dir / "strategy_metrics.csv", index=False)
            logger.info(f"📋 Метрики сохранены: {results_dir / 'strategy_metrics.csv'}")
            
            if not pnl_series.empty:
                pnl_series.to_csv(results_dir / "daily_pnl.csv", header=['PnL'])
                logger.info(f"📈 Дневные P&L сохранены: {results_dir / 'daily_pnl.csv'}")
            
            if not equity_series.empty:
                equity_series.to_csv(results_dir / "equity_curve.csv", header=['Equity'])
                logger.info(f"💰 Кривая капитала сохранена: {results_dir / 'equity_curve.csv'}")
            
            if trade_stats:
                trades_df = pd.DataFrame(trade_stats)
                trades_df.to_csv(results_dir / "trade_statistics.csv", index=False)
                logger.info(f"🔄 Статистика по сделкам сохранена: {results_dir / 'trade_statistics.csv'}")
                
        except Exception as e:
            logger.error(f"Ошибка при создании отчетов: {e}")
    
    return base_metrics
