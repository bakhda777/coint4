"""Walk-forward analysis orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import time

import pandas as pd
from joblib import Parallel, delayed

from coint2.core import math_utils, performance
from coint2.utils.config import AppConfig
from coint2.utils.logging_utils import get_logger
from coint2.utils.timing_utils import ProgressTracker, logged_time, time_block
from coint2.utils.visualization import calculate_extended_metrics, create_performance_report, format_metrics_summary

# Import directly from the file path rather than the module
from src.coint2.core.data_loader import DataHandler, load_master_dataset
from src.coint2.core.normalization_improvements import preprocess_and_normalize_data
from src.coint2.core.memory_optimization import (
    consolidate_price_data, initialize_global_price_data, get_price_data_view,
    setup_blas_threading_limits, monitor_memory_usage, verify_no_data_copies,
    cleanup_global_data
)
from src.coint2.engine.numba_backtest_engine_full import FullNumbaPairBacktester as PairBacktester
from src.coint2.core.portfolio import Portfolio
from src.coint2.utils.vectorized_ops import VectorizedStatsCalculator, vectorized_eval_expression


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


def process_single_pair_mmap(pair_symbols, testing_start, testing_end, cfg, 
                            capital_per_pair, bar_minutes, period_label, pair_stats=None):
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
        
        # Vectorized normalization using numpy operations (no copy)
        import numpy as np
        data_values = pair_data.to_numpy(copy=False)  # Use memory view
        first_row = data_values[0]
        # Vectorized division and multiplication
        normalized_values = np.divide(data_values, first_row[np.newaxis, :]) * 100
        pair_data = pd.DataFrame(normalized_values, index=pair_data.index, columns=pair_data.columns)
        
        # Create a temporary portfolio for this pair
        from src.coint2.core.portfolio import Portfolio
        temp_portfolio = Portfolio(
            initial_capital=capital_per_pair,
            max_active_positions=1  # Single pair
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
            lazy_adf_threshold=getattr(cfg.backtest, 'lazy_adf_threshold', 0.1),
            # EW correlation parameters
            use_exponential_weighted_correlation=getattr(cfg.backtest, 'use_exponential_weighted_correlation', False),
            ew_correlation_alpha=getattr(cfg.backtest, 'ew_correlation_alpha', 0.1),
            # Volatility-based position sizing parameters
            volatility_based_sizing=getattr(cfg.portfolio, 'volatility_based_sizing', False),
            volatility_lookback_hours=getattr(cfg.portfolio, 'volatility_lookback_hours', 24),
            min_position_size_pct=getattr(cfg.portfolio, 'min_position_size_pct', 0.005),
            max_position_size_pct=getattr(cfg.portfolio, 'max_position_size_pct', 0.02),
            volatility_adjustment_factor=getattr(cfg.portfolio, 'volatility_adjustment_factor', 2.0),
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
        else:
            trades = results.get("trades", pd.Series())
            positions = results.get("position", pd.Series(dtype=float))
            costs = results.get("costs", pd.Series())
        
        # Vectorized calculation of actual trade count
        if not positions.empty:
            positions_values = positions.ffill().values
            prev_positions_values = np.concatenate([[0], positions_values[:-1]])
            # Vectorized boolean operations
            is_trade_open_event = (prev_positions_values == 0) & (positions_values != 0)
            actual_trade_count = int(np.sum(is_trade_open_event))
        else:
            actual_trade_count = 0
        
        # Vectorized calculation of pair statistics using numpy
        if not pnl_series.empty:
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
        
        pair_costs = np.sum(costs.values) if not costs.empty else 0
        
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
            'max_daily_loss': max_daily_loss
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
                       capital_per_pair, bar_minutes, period_label):
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
        # Vectorized normalization using numpy operations
        if not pair_data.empty:
            # Use numpy for faster normalization
            import numpy as np
            data_values = pair_data.values
            first_row = data_values[0]
            # Vectorized division and multiplication
            normalized_values = np.divide(data_values, first_row[np.newaxis, :]) * 100
            pair_data = pd.DataFrame(normalized_values, index=pair_data.index, columns=pair_data.columns)
        
        # Create a temporary portfolio for this pair (will be merged later)
        from src.coint2.core.portfolio import Portfolio
        temp_portfolio = Portfolio(
            initial_capital=capital_per_pair,
            max_active_positions=1  # Single pair
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
            lazy_adf_threshold=getattr(cfg.backtest, 'lazy_adf_threshold', 0.1),
            # EW correlation parameters
            use_exponential_weighted_correlation=getattr(cfg.backtest, 'use_exponential_weighted_correlation', False),
            ew_correlation_alpha=getattr(cfg.backtest, 'ew_correlation_alpha', 0.1),
            # Volatility-based position sizing parameters
            volatility_based_sizing=getattr(cfg.portfolio, 'volatility_based_sizing', False),
            volatility_lookback_hours=getattr(cfg.portfolio, 'volatility_lookback_hours', 24),
            min_position_size_pct=getattr(cfg.portfolio, 'min_position_size_pct', 0.005),
            max_position_size_pct=getattr(cfg.portfolio, 'max_position_size_pct', 0.02),
            volatility_adjustment_factor=getattr(cfg.portfolio, 'volatility_adjustment_factor', 2.0),
        )
        
        # Run backtest with timing for debugging
        backtest_start = time.time()
        bt.run()
        backtest_time = time.time() - backtest_start
        results = bt.get_results()
        
        # Process results
        pnl_series = results["pnl"]
        trades_log = results.get('trades_log', [])
        
        # Calculate trade statistics
        if isinstance(results, dict):
            trades = results.get("trades", pd.Series())
            positions = results.get("position", pd.Series(dtype=float))
            costs = results.get("costs", pd.Series())
        else:
            trades = results.get("trades", pd.Series())
            positions = results.get("position", pd.Series(dtype=float))
            costs = results.get("costs", pd.Series())
        
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
        
        # Vectorized calculation of pair statistics using numpy
        if not pnl_series.empty:
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
        
        pair_costs = np.sum(costs.values) if not costs.empty else 0
        
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
            'max_daily_loss': max_daily_loss
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
    min_training_days: int = 30,
    min_testing_days: int = 1
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
    
    # Check minimum training period length
    training_days = (training_end - training_start).days
    if training_days < min_training_days:
        return False, f"Training period too short: {training_days} days < {min_training_days} required"
    
    # Check minimum testing period length
    testing_days = (testing_end - testing_start).days
    if testing_days < min_testing_days:
        return False, f"Testing period too short: {testing_days} days < {min_testing_days} required"
    
    # Check for significant data gaps in training period
    training_data_days = len(training_data)
    expected_training_periods = training_days * (24 * 60 / 15)  # Assuming 15-min bars
    data_coverage_ratio = training_data_days / expected_training_periods
    
    if data_coverage_ratio < 0.5:  # Less than 50% data coverage
        logger.warning(f"⚠️  Low training data coverage: {data_coverage_ratio:.1%}")
        logger.warning(f"   Expected ~{expected_training_periods:.0f} periods, got {training_data_days}")
    
    # Check for significant data gaps in testing period
    testing_data_days = len(testing_data)
    expected_testing_periods = testing_days * (24 * 60 / 15)  # Assuming 15-min bars
    testing_coverage_ratio = testing_data_days / expected_testing_periods
    
    if testing_coverage_ratio < 0.3:  # Less than 30% data coverage
        logger.warning(f"⚠️  Low testing data coverage: {testing_coverage_ratio:.1%}")
        logger.warning(f"   Expected ~{expected_testing_periods:.0f} periods, got {testing_data_days}")
    
    logger.debug(f"✅ Data validation passed:")
    logger.debug(f"   Training: {training_data_days} periods ({data_coverage_ratio:.1%} coverage)")
    logger.debug(f"   Testing:  {testing_data_days} periods ({testing_coverage_ratio:.1%} coverage)")
    
    return True, "Data validation successful"


@logged_time("run_walk_forward_analysis")
def run_walk_forward(cfg: AppConfig, use_memory_map: bool = True) -> dict[str, float]:
    """Run walk-forward analysis and return aggregated performance metrics.
    
    Args:
        cfg: Application configuration
        use_memory_map: Whether to use memory-mapped data optimization
    """
    start_time = time.time()
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

    start_date = pd.to_datetime(cfg.walk_forward.start_date)
    end_date = pd.to_datetime(cfg.walk_forward.end_date)
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
        logger.info("✅ Шаги успешно запланированы:")
        for i, (tr_start, tr_end, te_start, te_end) in enumerate(walk_forward_steps, 1):
            logger.info(f"   Шаг {i}: тренировка {tr_start.date()}-{tr_end.date()}, тест {te_start.date()}-{te_end.date()}")

    # Memory-mapped data optimization
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
            # Consolidate all price data into a single memory-mapped file
            consolidated_file = consolidate_price_data(
                cfg.data_dir, 
                data_start, 
                data_end,
                output_file=cfg.data_dir / "consolidated_prices.parquet"
            )
            
            # Initialize global memory-mapped data
            initialize_global_price_data(consolidated_file)
            
            # Verify memory mapping is working
            verify_no_data_copies()
            
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
    daily_pnl = []
    equity_data = []
    pair_count_data = []
    trade_stats = []
    all_trades_log = []

    portfolio = Portfolio(
        initial_capital=cfg.portfolio.initial_capital,
        max_active_positions=cfg.portfolio.max_active_positions,
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
                step_df_long = load_master_dataset(cfg.data_dir, training_start, testing_end)

            if step_df_long.empty:
                logger.warning(f"  Нет данных для шага {step_idx}, пропуск.")
                continue

            step_df = step_df_long.pivot_table(index="timestamp", columns="symbol", values="close")
            
            # ДОПОЛНИТЕЛЬНАЯ ВАЛИДАЦИЯ: Проверка наличия данных для временных окон
            is_data_valid, data_error = validate_walk_forward_data(
                step_df, training_start, training_end, testing_start, testing_end, bar_minutes
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
                            min_beta=cfg.filter_params.min_beta,
                            max_beta=cfg.filter_params.max_beta,
                            min_half_life=cfg.filter_params.min_half_life_days,
                            max_half_life=cfg.filter_params.max_half_life_days,
                            min_mean_crossings=cfg.filter_params.min_mean_crossings,
                            max_hurst_exponent=cfg.filter_params.max_hurst_exponent,
                            save_filter_reasons=cfg.pair_selection.save_filter_reasons,
                            kpss_pvalue_threshold=cfg.pair_selection.kpss_pvalue_threshold,
                            # Параметры commission_pct и slippage_pct удалены
                        )
                        logger.info(f"  Фильтрация: {len(ssd_pairs)} → {len(filtered_pairs)} пар")
                        pairs = filtered_pairs

        # Select all filtered pairs for trading (no limit here)
        # Сортируем пары по качеству (по убыванию std, что означает большую волатильность спреда)
        # max_active_positions теперь ограничивает только одновременно открытые позиции
        if pairs:
            # Сортируем пары по стандартному отклонению спреда (больше = потенциально более прибыльно)
            quality_sorted_pairs = sorted(pairs, key=lambda x: abs(x[4]), reverse=True)  # x[4] = std
            active_pairs = quality_sorted_pairs  # Берем ВСЕ отфильтрованные пары
            logger.info("  Топ-3 пары по волатильности спреда:")
            for i, (s1, s2, beta, mean, std, metrics) in enumerate(active_pairs[:3], 1):
                logger.info(f"    {i}. {s1}-{s2}: beta={beta:.4f}, std={std:.4f}")
        else:
            active_pairs = []
        
        num_active_pairs = len(active_pairs)
        logger.info(f"  Всего пар для торговли: {num_active_pairs} (ограничение {cfg.portfolio.max_active_positions} применяется к одновременно открытым позициям)")

        step_pnl = pd.Series(dtype=float)
        total_step_pnl = 0.0

        current_equity = portfolio.get_current_equity()
        logger.info(
            f"  Проверка расчета капитала: equity={current_equity}, num_active_pairs={num_active_pairs}"
        )
        if num_active_pairs > 0:
            capital_per_pair = portfolio.calculate_position_risk_capital(
                cfg.portfolio.risk_per_position_pct
            )
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
            # Create portfolio for position management
            position_portfolio = Portfolio(
                initial_capital=current_equity,
                max_active_positions=cfg.portfolio.max_active_positions
            )
            pair_tracker = ProgressTracker(len(active_pairs), f"{step_tag} backtests", step=max(1, len(active_pairs)//5))
            logger.info(f"🚀 {step_tag}: Начинаем бэктесты для {len(active_pairs)} пар...")
            
            # Determine number of parallel jobs
            n_jobs = getattr(cfg.backtest, 'n_jobs', -1)  # -1 means use all available cores
            if n_jobs == -1:
                import os
                n_jobs = os.cpu_count()
            
            logger.info(f"🚀 {step_tag}: Запускаем параллельную обработку {len(active_pairs)} пар на {n_jobs} ядрах...")
            
            # Добавляем детальное логирование для больших наборов пар
            if len(active_pairs) > 50:
                logger.info(f"  📊 Большой набор пар ({len(active_pairs)}), ожидаемое время: ~{len(active_pairs)*2//60+1} мин")
                logger.info(f"  🔄 Прогресс будет отображаться каждые {max(1, len(active_pairs)//10)} пар")
            
            # Process pairs in parallel with progress tracking
            start_time = time.time()
            
            if use_memory_map:
                # Use memory-mapped version for optimal performance
                pair_results = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(process_single_pair_mmap)(
                        (pair_data_tuple[0], pair_data_tuple[1]),  # (s1, s2)
                        testing_start, testing_end, cfg,
                        capital_per_pair, bar_minutes, period_label,
                        (pair_data_tuple[2], pair_data_tuple[3], pair_data_tuple[4], pair_data_tuple[5])  # (beta, mean, std, metrics)
                    )
                    for pair_data_tuple in active_pairs
                )
            else:
                # Fallback to traditional method
                pair_results = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(process_single_pair)(
                        pair_data_tuple, step_df, testing_start, testing_end, cfg,
                        capital_per_pair, bar_minutes, period_label
                    )
                    for pair_data_tuple in active_pairs
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
                total_step_pnl = np.sum(pnl_values)
                
                # Aggregate PnL series efficiently
                for result in successful_results:
                    all_trades_log.extend(result['trades_log'])
                    pnl_series = result['pnl_series']
                    step_pnl = step_pnl.add(pnl_series, fill_value=0)
                    trade_stats.append(result['trade_stat'])
            else:
                profitable_count = 0
                total_trades_count = 0
                total_step_pnl = 0.0
            
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

        # Update equity and daily P&L
        if not step_pnl.empty:
            running_equity = portfolio.get_current_equity()
            for date, pnl in step_pnl.items():
                daily_pnl.append((date, pnl))
                running_equity += pnl
                equity_data.append((date, running_equity))
                portfolio.record_daily_pnl(pd.Timestamp(date), pnl)

        aggregated_pnl = pd.concat([aggregated_pnl, step_pnl])

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
                cfg.portfolio.risk_per_position_pct
            )
            # Calculate Sharpe ratio using portfolio percentage returns
            daily_returns = equity_series.pct_change().dropna()
            sharpe_abs = performance.sharpe_ratio(
                daily_returns, cfg.backtest.annualizing_factor
            )
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
        
        logger.info(f"  ⇢ Расчет расширенных метрик (волатильность, Calmar ratio, и др.)...")
        extended_metrics = calculate_extended_metrics(pnl_series, equity_series)
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
        
        all_metrics = {**base_metrics, **extended_metrics, **trade_metrics}
        logger.info(f"✅ Рассчитаны метрики производительности ({len(all_metrics)} показателей)")
        logger.info(f"  💰 Общий P&L: ${all_metrics.get('total_pnl', 0):+,.2f}")
        logger.info(f"  📊 Sharpe Ratio: {all_metrics.get('sharpe_ratio_abs', 0):.3f}")
        logger.info(f"  📉 Максимальная просадка: {all_metrics.get('max_drawdown_abs', 0):.2%}")
        logger.info(f"  🔄 Всего сделок: {all_metrics.get('total_trades', 0)}")
        logger.info(f"  📈 Торгуемых пар: {all_metrics.get('total_pairs_traded', 0)}")
        logger.info(f"  💸 Общие издержки: ${all_metrics.get('total_costs', 0):,.2f}")
    
    # Create reports
    logger.info(f"⇢ Создание отчетов и сохранение результатов...")
    with time_block("generating reports"):
        results_dir = Path(cfg.results_dir)
        
        try:
            logger.info(f"  ⇢ Создание отчета о производительности...")
            create_performance_report(
                equity_curve=equity_series,
                pnl_series=pnl_series,
                metrics=all_metrics,
                pair_counts=pair_count_data,
                results_dir=results_dir,
                strategy_name="CointegrationStrategy"
            )
            logger.info(f"  ✓ Отчет о производительности создан")
            
            summary = format_metrics_summary(all_metrics)
            print(summary)
            
            # Save data
            logger.info(f"  ⇢ Сохранение данных в CSV файлы...")
            # Приводим все числовые значения к float для избежания проблем с типами Arrow
            clean_metrics = {}
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    clean_metrics[key] = float(value)
                else:
                    clean_metrics[key] = value
            
            metrics_df = pd.DataFrame([clean_metrics])
            metrics_df.to_csv(results_dir / "strategy_metrics.csv", index=False)
            logger.info(f"📋 Метрики сохранены: {results_dir / 'strategy_metrics.csv'}")
            
            if not pnl_series.empty:
                pnl_series.to_csv(results_dir / "daily_pnl.csv", header=['PnL'])
                logger.info(f"📈 Дневные P&L сохранены: {results_dir / 'daily_pnl.csv'} ({len(pnl_series)} записей)")
            
            if not equity_series.empty:
                equity_series.to_csv(results_dir / "equity_curve.csv", header=['Equity'])
                logger.info(f"💹 Кривая капитала сохранена: {results_dir / 'equity_curve.csv'} ({len(equity_series)} точек)")
            
            if trade_stats:
                trades_df = pd.DataFrame(trade_stats)
                trades_df.to_csv(results_dir / "trade_statistics.csv", index=False)
                logger.info(f"📋 Статистика сделок сохранена: {results_dir / 'trade_statistics.csv'} ({len(trades_df)} записей)")

            if all_trades_log:
                trades_log_df = pd.DataFrame(all_trades_log)
                trades_log_df.to_csv(results_dir / "trades_log.csv", index=False)
                logger.info(f"📓 Детальный лог сделок сохранен: {results_dir / 'trades_log.csv'} ({len(trades_log_df)} записей)")
            
            logger.info(f"🎉 Walk-Forward анализ полностью завершен!")
            logger.info(f"📊 Все отчеты и данные сохранены в: {results_dir}")
            logger.info(f"💼 Итоговый капитал: ${portfolio.get_current_equity():,.2f}")
            logger.info(f"⏱️  Общее время выполнения: {time.time() - start_time:.1f} секунд")
                
        except Exception as e:
            logger.error(f"Ошибка при создании отчетов: {e}")

    # Cleanup memory-mapped data
    if use_memory_map:
        try:
            cleanup_global_data()
            logger.info("🧹 Memory-mapped data cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up memory-mapped data: {e}")

    # Calculate and display Sharpe ratio on portfolio returns for verification
    equity_curve = portfolio.equity_curve
    daily_returns = equity_curve.pct_change().dropna()
    sharpe = performance.sharpe_ratio(daily_returns, cfg.backtest.annualizing_factor)
    print(f"Correct Sharpe Ratio: {sharpe}")

    return base_metrics
