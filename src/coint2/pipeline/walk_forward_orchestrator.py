"""Walk-forward analysis orchestrator."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from coint2.core import math_utils, performance
# Import directly from the file path rather than the module
from src.coint2.core.data_loader import DataHandler, load_master_dataset
from src.coint2.core.normalization_improvements import preprocess_and_normalize_data
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
    bar_minutes = getattr(cfg.pair_selection, "bar_minutes", None) or 15
    bar_delta = pd.Timedelta(minutes=bar_minutes)
    while current_test_start < end_date:
        # Тренировочный период ПРЕДШЕСТВУЕТ тестовому
        training_start = current_test_start - pd.Timedelta(days=cfg.walk_forward.training_period_days)
        # Завершаем тренировочный период за один бар до начала тестового
        training_end = current_test_start - bar_delta
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
                    normalized_training, norm_stats = preprocess_and_normalize_data(
                        training_slice,
                        min_history_ratio=min_history_ratio,
                        fill_method=fill_method,
                        norm_method=norm_method,
                        handle_constant=handle_constant
                    )
                    
                    # Выводим статистику нормализации
                    logger.info(f"  Статистика нормализации:")
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
                    
                if len(normalized_training.columns) < 2:
                    logger.warning("  После нормализации осталось менее 2 символов")
                    pairs = []
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
                        
                    # Filter pairs
                    with time_block("filtering pairs by cointegration and half-life"):
                        from coint2.pipeline.filters import filter_pairs_by_coint_and_half_life
                        ssd_pairs = [(s1, s2) for s1, s2 in ssd.index]
                        filtered_pairs = filter_pairs_by_coint_and_half_life(
                            ssd_pairs,
                            training_slice,
                            pvalue_threshold=cfg.pair_selection.coint_pvalue_threshold,
                            min_half_life=cfg.pair_selection.min_half_life_days,
                            max_half_life=cfg.pair_selection.max_half_life_days,
                            min_mean_crossings=cfg.pair_selection.min_mean_crossings,
                            min_correlation=cfg.pair_selection.min_correlation,
                            max_correlation=cfg.pair_selection.max_correlation,
                            min_spread_std=cfg.pair_selection.min_spread_std,
                            max_spread_std=cfg.pair_selection.max_spread_std,
                            max_hurst_exponent=getattr(cfg.pair_selection, 'max_hurst_exponent', 0.5),
                            adaptive_quantiles=cfg.pair_selection.adaptive_quantiles,
                            save_filter_reasons=cfg.pair_selection.save_filter_reasons,
                            save_std_histogram=cfg.pair_selection.save_std_histogram,
                            kpss_pvalue_threshold=cfg.pair_selection.kpss_pvalue_threshold,
                            # Параметры commission_pct и slippage_pct удалены
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
            for i, (s1, s2, beta, mean, std, metrics) in enumerate(active_pairs[:3], 1):
                logger.info(f"    {i}. {s1}-{s2}: beta={beta:.4f}, std={std:.4f}")
        else:
            active_pairs = []
        
        num_active_pairs = len(active_pairs)
        logger.info(f"  Активных пар для торговли: {num_active_pairs}")

        step_pnl = pd.Series(dtype=float)
        total_step_pnl = 0.0

        logger.info(
            f"  Проверка расчета капитала: equity={equity}, num_active_pairs={num_active_pairs}"
        )
        if num_active_pairs > 0:
            capital_per_pair = equity * cfg.portfolio.risk_per_position_pct / num_active_pairs
        else:
            capital_per_pair = 0.0

        if capital_per_pair < 0:
            logger.error(
                f"КРИТИЧЕСКАЯ ОШИБКА: Капитал на пару стал отрицательным ({capital_per_pair})."
            )
            capital_per_pair = 0.0

        logger.info(f"  Капитал на пару: ${capital_per_pair:,.2f}")

        period_label = f"{training_start.strftime('%m/%d')}-{testing_end.strftime('%m/%d')}"
        pair_count_data.append((period_label, len(active_pairs)))

        # Run backtests for active pairs
        if active_pairs:
            pair_tracker = ProgressTracker(len(active_pairs), f"{step_tag} backtests", step=max(1, len(active_pairs)//5))
            
            for pair_idx, (s1, s2, beta, mean, std, metrics) in enumerate(active_pairs, 1):
                pair_data = step_df.loc[testing_start:testing_end, [s1, s2]].dropna()
                # Нормализация: оба ряда начинаются с 100
                if not pair_data.empty:
                    pair_data = pair_data / pair_data.iloc[0] * 100

                bt = PairBacktester(
                    pair_data,
                    rolling_window=cfg.backtest.rolling_window,
                    z_threshold=cfg.backtest.zscore_threshold,
                    z_exit=getattr(cfg.backtest, 'zscore_exit', 0.0),
                    commission_pct=getattr(cfg.backtest, 'commission_pct', 0.0),
                    slippage_pct=getattr(cfg.backtest, 'slippage_pct', 0.0),
                    annualizing_factor=getattr(cfg.backtest, 'annualizing_factor', 365),
                    capital_at_risk=capital_per_pair,
                    stop_loss_multiplier=getattr(cfg.backtest, 'stop_loss_multiplier', 2.0),
                    take_profit_multiplier=getattr(cfg.backtest, 'take_profit_multiplier', None),
                    # Convert hours to periods based on 15-minute timeframe
                    cooldown_periods=int(getattr(cfg.backtest, 'cooldown_hours', 0) * 60 / 15),  # 15-минутный таймфрейм
                    wait_for_candle_close=getattr(cfg.backtest, 'wait_for_candle_close', False),
                    max_margin_usage=getattr(cfg.portfolio, 'max_margin_usage', 1.0),
                )
                bt.run()
                results = bt.get_results()
                pnl_series = results["pnl"]
                step_pnl = step_pnl.add(pnl_series, fill_value=0)
                total_step_pnl += pnl_series.sum()
                
                # Собираем статистику по сделкам для данной пары
                if isinstance(results, dict):
                    trades = results.get("trades", pd.Series())
                    positions = results.get("position", pd.Series(dtype=float))
                    costs = results.get("costs", pd.Series())
                else:
                    # Если results - DataFrame, используем колонки
                    trades = results.get("trades", results.get("trades", pd.Series()))
                    positions = results.get("position", results.get("position", pd.Series(dtype=float)))
                    costs = results.get("costs", results.get("costs", pd.Series()))

                # Новая логика подсчёта фактических сделок
                if not positions.empty:
                    positions = positions.fillna(method="ffill")
                    prev_positions = positions.shift(1).fillna(0)
                    is_trade_open_event = (prev_positions == 0) & (positions != 0)
                    actual_trade_count = int(is_trade_open_event.sum())
                else:
                    actual_trade_count = 0
                
                # Всегда добавляем статистику по парам, даже если сделок не было
                pair_pnl = pnl_series.sum()
                pair_costs = costs.sum() if not costs.empty else 0

                trade_stats.append({
                    'pair': f'{s1}-{s2}',
                    'period': period_label,
                    'total_pnl': pair_pnl,
                    'total_costs': pair_costs,
                    'trade_count': actual_trade_count,
                    'avg_pnl_per_trade': pair_pnl / max(actual_trade_count, 1),
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
            # Используем капитал для расчета риска на сделку и доходности
            capital_per_pair = equity * cfg.portfolio.risk_per_position_pct
            if capital_per_pair < 0:
                logger.error(
                    f"КРИТИЧЕСКАЯ ОШИБКА: Капитал на пару стал отрицательным ({capital_per_pair})."
                )
                capital_per_pair = 0.0
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
