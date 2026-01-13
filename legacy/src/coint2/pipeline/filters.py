import numpy as np
import pandas as pd
from coint2.core.math_utils import calculate_half_life, count_mean_crossings
from coint2.core.fast_coint import fast_coint
from coint2.analysis.pair_filter import calculate_hurst_exponent
from statsmodels.tsa.stattools import kpss  # KPSS тест стационарности
from statsmodels.tools.sm_exceptions import InterpolationWarning
import statsmodels.api as sm
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Dict, Any, Optional
from coint2.utils.logger import get_logger
from coint2.engine.numba_engine import NumbaPairBacktester
from coint2.core.performance import calculate_metrics

# --- Placeholder market data helpers ---
_market_cache: dict[str, tuple[float,float,float]] = {}

def _get_market_metrics(symbol: str) -> tuple[float, float, float]:
    """Return (avg_volume_usd, bid_ask_pct, avg_funding_pct).
    In production replace with real data source.
    """
    if symbol in _market_cache:
        return _market_cache[symbol]
    # WARNING: Using placeholder data - integrate real market data in production
    _market_cache[symbol] = (float('inf'), 0.0, 0.0)
    return _market_cache[symbol]


# Разумные границы для коэффициента beta
MIN_BETA = 0.1
MAX_BETA = 10.0

def _evaluate_pair_train_performance(
    s1: str,
    s2: str,
    y_aligned: pd.Series,
    x_aligned: pd.Series,
    config: Any
) -> Dict[str, float]:
    """
    Run a simulation on train data to evaluate pair quality.
    Returns metrics: train_mean_R, train_winrate, train_sharpe, train_gross_to_cost_ratio,
    train_cum_pnl_r, n_trades.
    """
    try:
        pair_data = pd.concat([y_aligned, x_aligned], axis=1)
        pair_data.columns = [s1, s2]
        
        # Determine capital per pair (just for simulation scaling, R metrics are invariant)
        capital_per_pair = 1000.0
        
        bt = NumbaPairBacktester(
            pair_data=pair_data,
            rolling_window=getattr(config.backtest, 'rolling_window', 20),
            z_threshold=getattr(config.backtest, 'zscore_entry_threshold', 2.0),
            z_exit=getattr(config.backtest, 'z_exit', 0.0),
            capital_at_risk=capital_per_pair,
            stop_loss_multiplier=getattr(config.backtest, 'stop_loss_multiplier', 2.0),
            time_stop_multiplier=getattr(config.backtest, 'time_stop_multiplier', None),
            max_position_size_pct=getattr(config.portfolio, 'max_position_size_pct', 1.0),
            config=config.backtest if hasattr(config, 'backtest') else config # Pass backtest config
        )
        
        results = bt.run()
        
        if results is None or not results.get('trades'):
            return {
                'train_mean_R': 0.0,
                'train_winrate': 0.0,
                'train_sharpe': 0.0,
                'train_gross_to_cost_ratio': 0.0,
                'train_n_trades': 0
            }
            
        trades = results['trades']
        n_trades = 0
        winning_trades = 0
        total_pnl_r = 0.0
        
        # Calculate Gross PnL and Costs for Ratio
        total_gross_pnl = 0.0
        total_costs = 0.0
        
        for t in trades:
            if isinstance(t, dict):
                n_trades += 1
                if t.get('net_pnl', 0) > 0:
                    winning_trades += 1
                if 'final_pnl_r' in t:
                    total_pnl_r += t['final_pnl_r']
                
                # Approximate gross and costs if not explicitly separate in trade log (Numba engine usually puts net_pnl)
                # But we can check 'commission' and 'funding' fields if available, or use trade stats
                # Numba engine results['costs'] is a Series of costs over time
        
        if results.get('costs') is not None and not results['costs'].empty:
             total_costs = results['costs'].sum()
        
        # Calculate total PnL (Net)
        total_pnl = results.get('trade_stat', {}).get('total_pnl', 0.0)
        # Gross = Net + Costs
        total_gross_pnl = total_pnl + total_costs
        
        mean_R = total_pnl_r / n_trades if n_trades > 0 else 0.0
        winrate = winning_trades / n_trades if n_trades > 0 else 0.0
        gross_to_cost = total_gross_pnl / total_costs if total_costs > 0 else (10.0 if total_gross_pnl > 0 else 0.0)
        
        # Calculate Sharpe on daily returns
        pnl_series = results.get('pnl_series', pd.Series(dtype=float))
        sharpe = 0.0
        if not pnl_series.empty:
            daily_pnl = pnl_series.resample('D').sum()
            if daily_pnl.std() > 0:
                # Annualized Sharpe (assuming 365 days)
                sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365)
                
        return {
            'train_mean_R': float(mean_R),
            'train_winrate': float(winrate),
            'train_sharpe': float(sharpe),
            'train_gross_to_cost_ratio': float(gross_to_cost),
            'train_cum_pnl_r': float(total_pnl_r),
            'train_n_trades': int(n_trades)
        }
        
    except Exception as e:
        # logger.warning(f"Train filter error for {s1}-{s2}: {e}")
        return {
            'train_mean_R': 0.0,
            'train_winrate': 0.0,
            'train_sharpe': 0.0,
            'train_gross_to_cost_ratio': 0.0,
            'train_cum_pnl_r': 0.0,
            'train_n_trades': 0
        }

def enhanced_pair_screening(
    pairs: List[Tuple[str, str]],
    price_df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    max_half_life_bars: int = 96,  # 96 * 15min = 24 hours for 15-min data
    min_daily_volume_usd: float = 50_000.0,  # $50k per leg
    min_beta: float = MIN_BETA,
    max_beta: float = MAX_BETA,
    min_mean_crossings: int = 8,
    min_history_ratio: float = 0.8,
    save_filter_reasons: bool = False,
    kpss_pvalue_threshold: float = 0.05,
    max_hurst_exponent: float = 0.5,
    use_kpss_filter: bool = False,  # NEW: Flag
    use_hurst_filter: bool = False, # NEW: Flag
    min_profit_potential_pct: float = 0.0, # NEW: Minimum profit potential vs costs
    blacklist: Optional[List[str]] = None, # NEW: Blacklist of symbols
    *,
    stable_tokens: Optional[List[str]] = None,
    volume_df: Optional[pd.DataFrame] = None, # NEW: Optional volume data
    config: Optional[Any] = None # NEW: Config for train filter
) -> List[Tuple[str, str, float, float, float, Dict[str, Any]]]:
    """Enhanced pair screening with strict criteria for 15-minute data.
     
     Criteria:
     1. p-value коинтеграции < 0.05
     2. half-life < N бар (default 96 bars = 24 hours for 15-min data)
     3. среднедневной объём ≥ 50k $ на каждую leg
     4. Фильтрация по blacklist
     
     Parameters
     ----------
     pairs : List[Tuple[str, str]]
         List of symbol pairs to screen
     price_df : pd.DataFrame
         Price data with columns for each symbol
     pvalue_threshold : float, default 0.05
         Maximum p-value for cointegration test
     max_half_life_bars : int, default 96
         Maximum half-life in bars (96 * 15min = 24 hours)
     min_daily_volume_usd : float, default 50_000.0
         Minimum average daily volume in USD per leg
     volume_df : pd.DataFrame, optional
         Turnover/Volume data (in USD)
     
     Returns
    -------
    List[Tuple[str, str, float, float, float, Dict[str, Any]]]
        Filtered pairs with (s1, s2, beta, mean, std, metrics)
    """
    logger = get_logger("enhanced_pair_screening")
    
    # DEBUG: Log filter parameters
    logger.info(f"[ENHANCED SCREENING PARAMS] pvalue_threshold={pvalue_threshold:.3f}, "
                f"min_beta={min_beta:.3f}, max_beta={max_beta:.3f}, "
                f"max_half_life_bars={max_half_life_bars}, "
                f"min_daily_volume_usd={min_daily_volume_usd:.0f}, "
                f"max_hurst_exponent={max_hurst_exponent:.2f}")
    
    # DEBUG: Log detailed reason count for failures
    # We will log this at the end, but good to be aware
    
    if blacklist:
        logger.info(f"[ENHANCED SCREENING] Blacklist active: {len(blacklist)} symbols")

    if stable_tokens is None:
        stable_tokens = [
            "USDT", "USDC", "BUSD", "TUSD", "DAI", "USD", "USDP", "PAX", "SUSD", "GUSD"
        ]
    
    def _is_stable(sym: str) -> bool:
        return sym in stable_tokens
    
    logger.info(f"[ENHANCED SCREENING] Starting with {len(pairs)} pairs")
    
    # Statistics tracking
    filter_stats = {
        "p_value_failed": 0,
        "half_life_failed": 0,
        "volume_failed": 0,
        "beta_failed": 0,
        "data_insufficient": 0,
        "kpss_failed": 0,
        "hurst_failed": 0,
        "mean_crossings_failed": 0,
        "profit_potential_failed": 0,
        "train_filter_failed": 0,
        "spread_empty": 0,
        "recent_data_empty": 0,
        "total_passed": 0
    }
    train_filter_examples = []
    
    # Respect explicit flag (config.filter_params.save_filter_reasons) if provided
    try:
        if config and hasattr(config, "filter_params") and hasattr(config.filter_params, "save_filter_reasons"):
            save_filter_reasons = bool(getattr(config.filter_params, "save_filter_reasons"))
    except Exception:
        pass
    
    filtered_pairs = []
    filter_reasons = []
    
    for s1, s2 in pairs:
        # 0. Blacklist Check
        if blacklist and (s1 in blacklist or s2 in blacklist):
            # filter_stats["blacklist_failed"] = filter_stats.get("blacklist_failed", 0) + 1
            if save_filter_reasons:
                filter_reasons.append((s1, s2, "blacklist"))
            continue

        try:
            # Check if symbols exist in price data
            if s1 not in price_df.columns or s2 not in price_df.columns:
                filter_stats["data_insufficient"] += 1
                if save_filter_reasons:
                    filter_reasons.append((s1, s2, "missing_price_data"))
                continue
            
            # Get price series
            y = price_df[s1].dropna()
            x = price_df[s2].dropna()
            
            # Align series
            common_idx = y.index.intersection(x.index)
            if len(common_idx) < 100:  # Minimum data requirement
                filter_stats["data_insufficient"] += 1
                if save_filter_reasons:
                    filter_reasons.append((s1, s2, "insufficient_data"))
                continue
            
            y_aligned = y.loc[common_idx]
            x_aligned = x.loc[common_idx]
            
            # 1. Cointegration test (p-value criterion)
            try:
                _, pvalue, _ = fast_coint(y_aligned, x_aligned, trend='n')
                
                # DEBUG: Specific logging for suspicious pairs
                # if (s1 == 'ORDIUSDT' and s2 == 'SOLUSDT') or (s1 == 'ETHDAI' and s2 == 'METHUSDT'):
                #      logger.info(f"[DEBUG {s1}-{s2}] pvalue={pvalue} (thresh={pvalue_threshold})")
                
                if pvalue > pvalue_threshold:
                    # logger.debug(f"SKIP_ENTRY_FILTER pair={s1}-{s2} filter=pvalue value={pvalue:.4f} threshold={pvalue_threshold}")
                    filter_stats["p_value_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"p_value_{pvalue:.4f}"))
                    continue
            except Exception as e:
                filter_stats["data_insufficient"] += 1
                if save_filter_reasons:
                    filter_reasons.append((s1, s2, f"coint_error_{str(e)[:50]}"))
                continue
            
            # Calculate regression parameters
            try:
                # Check for degenerate series before regression
                if x_aligned.var() < 1e-10 or y_aligned.var() < 1e-10:
                    filter_stats["data_insufficient"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, "degenerate_series"))
                    continue
                
                X = sm.add_constant(x_aligned)
                model = sm.OLS(y_aligned, X).fit()
                beta = model.params.iloc[1]
                
                # Sanity check for extreme beta values
                if abs(beta) > 100 or np.isnan(beta) or np.isinf(beta):
                    filter_stats["beta_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"beta_extreme_{beta:.2f}"))
                    continue
                
                # DIAGNOSTIC: Track beta distribution before filtering
                if not hasattr(enhanced_pair_screening, '_beta_diagnostics'):
                    enhanced_pair_screening._beta_diagnostics = []
                enhanced_pair_screening._beta_diagnostics.append({
                    'pair': f"{s1}-{s2}", 
                    'beta': beta,
                    'abs_beta': abs(beta)
                })
                
                # 2. Beta criterion
                if not (min_beta <= abs(beta) <= max_beta):
                    filter_stats["beta_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"beta_{beta:.4f}_range_{min_beta}-{max_beta}"))
                    continue
                
                # Calculate spread and statistics
                spread = y_aligned - beta * x_aligned
                
                # Check for empty spread to avoid numpy warnings
                if len(spread) == 0 or spread.isna().all():
                    filter_stats["spread_empty"] = filter_stats.get("spread_empty", 0) + 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, "spread_empty"))
                    continue
                
                mean_spread = spread.mean()
                std_spread = spread.std()
                
                # 3. Half-life criterion (in bars for 15-min data)
                half_life_bars = calculate_half_life(spread)
                
                # if (s1 == 'ORDIUSDT' and s2 == 'SOLUSDT') or (s1 == 'ETHDAI' and s2 == 'METHUSDT'):
                #      logger.info(f"[DEBUG {s1}-{s2}] half_life_bars={half_life_bars} (max={max_half_life_bars})")
                
                # Check for infinite or NaN half-life (e.g., if mean reversion is very weak)
                if np.isinf(half_life_bars) or np.isnan(half_life_bars):
                    filter_stats["half_life_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"half_life_inf_nan"))
                    continue

                if half_life_bars > max_half_life_bars:
                    # logger.debug(f"SKIP_ENTRY_FILTER pair={s1}-{s2} filter=half_life value={half_life_bars:.1f} threshold={max_half_life_bars}")
                    filter_stats["half_life_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"half_life_{half_life_bars:.1f}_bars"))
                    continue
                
                # 4. Volume criterion
                # For 15-min data: 96 bars per day
                bars_per_day = 96
                recent_data_days = min(30, len(common_idx) // bars_per_day)  # Last 30 days or available
                
                if recent_data_days > 0:
                    recent_periods = recent_data_days * bars_per_day
                    
                    # REAL VOLUME CHECK
                    if volume_df is not None and s1 in volume_df.columns and s2 in volume_df.columns:
                         vol_s1_series = volume_df[s1].loc[common_idx].iloc[-recent_periods:]
                         vol_s2_series = volume_df[s2].loc[common_idx].iloc[-recent_periods:]
                         
                         # Calculate average daily turnover (sum of 15m intervals per day)
                         # Since we have turnover (USD volume) per 15m, sum over all periods then divide by days
                         # or resample. Simpler: mean * bars_per_day
                         
                         avg_daily_vol_s1 = vol_s1_series.mean() * bars_per_day
                         avg_daily_vol_s2 = vol_s2_series.mean() * bars_per_day
                         
                         # Handle NaN
                         if np.isnan(avg_daily_vol_s1): avg_daily_vol_s1 = 0.0
                         if np.isnan(avg_daily_vol_s2): avg_daily_vol_s2 = 0.0
                         
                         estimated_volume_s1 = avg_daily_vol_s1
                         estimated_volume_s2 = avg_daily_vol_s2
                         
                    else:
                        # Fallback to heuristic if no volume data
                        recent_y = y_aligned.iloc[-recent_periods:] if len(y_aligned) >= recent_periods else y_aligned
                        recent_x = x_aligned.iloc[-recent_periods:] if len(x_aligned) >= recent_periods else x_aligned
                        
                        # Check for empty recent data
                        if len(recent_y) == 0 or recent_y.isna().all() or len(recent_x) == 0 or recent_x.isna().all():
                            filter_stats["recent_data_empty"] = filter_stats.get("recent_data_empty", 0) + 1
                            if save_filter_reasons:
                                filter_reasons.append((s1, s2, "recent_data_empty"))
                            continue
                        
                        avg_price_s1 = recent_y.mean()
                        avg_price_s2 = recent_x.mean()
                        
                        volatility_s1 = recent_y.pct_change().std() * np.sqrt(bars_per_day)
                        volatility_s2 = recent_x.pct_change().std() * np.sqrt(bars_per_day)
                        
                        estimated_volume_s1 = avg_price_s1 * 1000 * (1 + volatility_s1 * 10)
                        estimated_volume_s2 = avg_price_s2 * 1000 * (1 + volatility_s2 * 10)
                    
                    # DIAGNOSTIC: Track volume distribution before filtering
                    if not hasattr(enhanced_pair_screening, '_volume_diagnostics'):
                        enhanced_pair_screening._volume_diagnostics = []
                    enhanced_pair_screening._volume_diagnostics.append({
                        'pair': f"{s1}-{s2}", 
                        'vol_s1': estimated_volume_s1, 
                        'vol_s2': estimated_volume_s2,
                        'min_vol': min(estimated_volume_s1, estimated_volume_s2)
                    })
                    
                    if (estimated_volume_s1 < min_daily_volume_usd or 
                        estimated_volume_s2 < min_daily_volume_usd):
                        filter_stats["volume_failed"] += 1
                        if save_filter_reasons:
                            filter_reasons.append((s1, s2, f"volume_s1_{estimated_volume_s1:.0f}_s2_{estimated_volume_s2:.0f}"))
                        continue
                
                # Additional quality checks
                
                # KPSS test for spread stationarity
                if use_kpss_filter:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", InterpolationWarning)
                            kpss_stat, kpss_pvalue, _, _ = kpss(spread, regression='c')
                        
                        if kpss_pvalue < kpss_pvalue_threshold:
                            filter_stats["kpss_failed"] += 1
                            if save_filter_reasons:
                                filter_reasons.append((s1, s2, f"kpss_pvalue_{kpss_pvalue:.4f}"))
                            continue
                    except Exception:
                        # If KPSS fails, skip this pair
                        filter_stats["kpss_failed"] += 1
                        if save_filter_reasons:
                            filter_reasons.append((s1, s2, "kpss_error"))
                        continue
                
                # Hurst exponent check
                if use_hurst_filter:
                    try:
                        hurst = calculate_hurst_exponent(spread)
                        
                        if hurst > max_hurst_exponent:
                            # logger.debug(f"SKIP_ENTRY_FILTER pair={s1}-{s2} filter=hurst value={hurst:.3f} threshold={max_hurst_exponent}")
                            filter_stats["hurst_failed"] += 1
                            if save_filter_reasons:
                                filter_reasons.append((s1, s2, f"hurst_{hurst:.4f}"))
                            continue
                    except Exception:
                        # If Hurst calculation fails, continue (don't filter out)
                        pass
                
                # Mean crossings check
                mean_crossings = count_mean_crossings(spread)
                if mean_crossings < min_mean_crossings:
                    filter_stats["mean_crossings_failed"] += 1
                    # logger.debug(f"SKIP_ENTRY_FILTER pair={s1}-{s2} filter=crossings value={mean_crossings} threshold={min_mean_crossings}")
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"mean_crossings_{mean_crossings}"))
                    continue
                
                # If we reach here, the pair passed all filters
                
                # 5. Minimum Profit Potential Check (Vol > Costs)
                if min_profit_potential_pct > 0:
                    # Calculate relative volatility of spread vs prices
                    # Approximate: std_spread / mean_price_of_pair
                    avg_pair_price = (y_aligned.mean() + x_aligned.mean()) / 2
                    if avg_pair_price > 0:
                        # For log prices, std_spread IS percentage volatility approximately
                        # But if using raw prices, we need to normalize
                        
                        # Assuming input prices are RAW (not log returns) based on context
                        rel_volatility = std_spread / avg_pair_price
                        
                        if rel_volatility < min_profit_potential_pct:
                            filter_stats["profit_potential_failed"] += 1
                            if save_filter_reasons:
                                filter_reasons.append((s1, s2, f"low_volatility_{rel_volatility:.6f}"))
                            continue
                
                # 6. Train Edge Filter
                train_metrics = {}
                if config and hasattr(config, 'filter_params') and hasattr(config.filter_params, 'train_edge_filter'):
                    train_conf = config.filter_params.train_edge_filter
                    # Check if train_conf is a dict or object (OmegaConf/Pydantic)
                    if train_conf:
                         # Safe access to dict or object attributes
                         def get_conf_val(obj, key, default):
                             if isinstance(obj, dict):
                                 return obj.get(key, default)
                             return getattr(obj, key, default)

                         min_mean_R = float(get_conf_val(train_conf, 'min_train_mean_R', 0.0))
                         min_sharpe = float(get_conf_val(train_conf, 'min_train_sharpe', 0.0))
                         min_g2c = float(get_conf_val(train_conf, 'min_gross_to_cost_ratio', 2.5))
                         min_cum_r = float(get_conf_val(train_conf, 'min_train_cum_pnl_r', 2.0))
                         
                         train_res = _evaluate_pair_train_performance(s1, s2, y_aligned, x_aligned, config)
                         train_metrics = train_res
                         
                         if (train_res['train_mean_R'] < min_mean_R or 
                             train_res['train_sharpe'] <= min_sharpe or 
                             train_res['train_gross_to_cost_ratio'] < min_g2c or
                             train_res.get('train_cum_pnl_r', 0.0) < min_cum_r):
                             
                             filter_stats["train_filter_failed"] = filter_stats.get("train_filter_failed", 0) + 1
                             if save_filter_reasons:
                                 reason = f"train_R_{train_res['train_mean_R']:.2f}_Sh_{train_res['train_sharpe']:.2f}_G2C_{train_res['train_gross_to_cost_ratio']:.1f}"
                                 filter_reasons.append((s1, s2, reason))
                             train_filter_examples.append(
                                 (s1, s2, train_res.get('train_gross_to_cost_ratio', 0.0), train_res.get('train_cum_pnl_r', 0.0), train_res.get('train_winrate', 0.0))
                             )
                             continue
                             
                         # Log passing pair with train metrics
                         logger.info(f"[TRAIN_FILTER_PASS] {s1}-{s2} R={train_res['train_mean_R']:.2f} Sh={train_res['train_sharpe']:.2f} G2C={train_res['train_gross_to_cost_ratio']:.1f} CumR={train_res.get('train_cum_pnl_r',0.0):.2f} Win={train_res.get('train_winrate',0.0):.2f}")

                metrics = {
                    "p_value": pvalue,
                    **train_metrics, # Merge train metrics
                    "half_life_bars": half_life_bars,
                    "beta": beta,
                    "mean_crossings": mean_crossings,
                    "hurst_exponent": hurst if 'hurst' in locals() else 0.5,
                    "kpss_pvalue": kpss_pvalue if 'kpss_pvalue' in locals() else 1.0,
                    "estimated_volume_s1": estimated_volume_s1 if 'estimated_volume_s1' in locals() else 0,
                    "estimated_volume_s2": estimated_volume_s2 if 'estimated_volume_s2' in locals() else 0,
                }
                
                filtered_pairs.append((s1, s2, beta, mean_spread, std_spread, metrics))
                filter_stats["total_passed"] += 1
                
            except Exception as e:
                filter_stats["data_insufficient"] += 1
                if save_filter_reasons:
                    filter_reasons.append((s1, s2, f"calculation_error_{str(e)[:50]}"))
                continue
                
        except Exception as e:
            filter_stats["data_insufficient"] += 1
            if save_filter_reasons:
                filter_reasons.append((s1, s2, f"general_error_{str(e)[:50]}"))
            continue
    
    # Log results
    logger.info(f"[ENHANCED SCREENING] Results:")
    logger.info(f"  Total pairs processed: {len(pairs)}")
    
    remaining = len(pairs)
    
    # Stage 1: Data Validation
    data_issues = filter_stats['data_insufficient'] + filter_stats.get('recent_data_empty', 0) + filter_stats.get('spread_empty', 0)
    remaining -= data_issues
    logger.info(f"  Stage 1: Data Validation -> Filtered: {data_issues} (Insufficient/Empty: {filter_stats['data_insufficient']}, Spread Empty: {filter_stats.get('spread_empty', 0)}), Remaining: {remaining}")
    
    # Stage 2: Cointegration (p-value)
    remaining -= filter_stats['p_value_failed']
    logger.info(f"  Stage 2: Cointegration (p-value) -> Filtered: {filter_stats['p_value_failed']}, Remaining: {remaining}")

    # Stage 3: Beta
    remaining -= filter_stats['beta_failed']
    logger.info(f"  Stage 3: Beta -> Filtered: {filter_stats['beta_failed']}, Remaining: {remaining}")
    
    # DIAGNOSTIC: Beta distribution analysis
    if hasattr(enhanced_pair_screening, '_beta_diagnostics') and enhanced_pair_screening._beta_diagnostics:
        betas = [d['beta'] for d in enhanced_pair_screening._beta_diagnostics if not np.isnan(d['beta']) and not np.isinf(d['beta'])]
        abs_betas = [d['abs_beta'] for d in enhanced_pair_screening._beta_diagnostics if not np.isnan(d['abs_beta']) and not np.isinf(d['abs_beta'])]
        n_nan = sum(1 for d in enhanced_pair_screening._beta_diagnostics if np.isnan(d['beta']) or np.isinf(d['beta']))
        
        if betas:
            logger.info(f"[BETA_DIAGNOSTIC] bounds: min_beta={min_beta:.3f}, max_beta={max_beta:.3f}")
            logger.info(f"[BETA_DIAGNOSTIC] raw_beta distribution (n={len(betas)}, nan/inf={n_nan}): "
                       f"min={np.min(betas):.4f}, "
                       f"p25={np.percentile(betas, 25):.4f}, "
                       f"median={np.median(betas):.4f}, "
                       f"p75={np.percentile(betas, 75):.4f}, "
                       f"max={np.max(betas):.4f}")
            logger.info(f"[BETA_DIAGNOSTIC] abs_beta distribution: "
                       f"min={np.min(abs_betas):.4f}, "
                       f"median={np.median(abs_betas):.4f}, "
                       f"max={np.max(abs_betas):.4f}")
            
            # Beta bins
            n_very_small = sum(1 for b in abs_betas if b < 0.1)
            n_small = sum(1 for b in abs_betas if 0.1 <= b < 0.5)
            n_medium = sum(1 for b in abs_betas if 0.5 <= b < 1.5)
            n_large = sum(1 for b in abs_betas if b >= 1.5)
            
            logger.info(f"[BETA_DIAGNOSTIC] bins: |beta|<0.1={n_very_small}, "
                       f"0.1-0.5={n_small}, 0.5-1.5={n_medium}, >1.5={n_large}")
            
            # Show examples of filtered pairs
            beta_filtered_examples = [d for d in enhanced_pair_screening._beta_diagnostics 
                            if not (min_beta <= d['abs_beta'] <= max_beta) and 
                            not np.isnan(d['abs_beta']) and not np.isinf(d['abs_beta'])]
            if beta_filtered_examples:
                examples = ", ".join([f"{p['pair']}({p['beta']:.3f})" for p in beta_filtered_examples[:5]])
                logger.info(f"[BETA_DIAGNOSTIC] FILTERED examples: {examples}")
        
        # Reset diagnostics for next run
        enhanced_pair_screening._beta_diagnostics = []

    # Stage 4: Half-life
    remaining -= filter_stats['half_life_failed']
    logger.info(f"  Stage 4: Half-life -> Filtered: {filter_stats['half_life_failed']}, Remaining: {remaining}")

    # Stage 5: Volume
    remaining -= filter_stats['volume_failed']
    logger.info(f"  Stage 5: Volume -> Filtered: {filter_stats['volume_failed']}, Remaining: {remaining}")
    
    # DIAGNOSTIC: Volume distribution analysis
    if hasattr(enhanced_pair_screening, '_volume_diagnostics') and enhanced_pair_screening._volume_diagnostics:
        vols = [d['min_vol'] for d in enhanced_pair_screening._volume_diagnostics]
        logger.info(f"[VOLUME_DIAGNOSTIC] threshold={min_daily_volume_usd:.0f}")
        logger.info(f"[VOLUME_DIAGNOSTIC] min_vol_distribution: "
                   f"min={np.min(vols):.0f}, "
                   f"p10={np.percentile(vols, 10):.0f}, "
                   f"median={np.median(vols):.0f}, "
                   f"p90={np.percentile(vols, 90):.0f}, "
                   f"max={np.max(vols):.0f}")
        
        # Show examples of passed/failed pairs
        passed_pairs = [d for d in enhanced_pair_screening._volume_diagnostics if d['min_vol'] >= min_daily_volume_usd]
        failed_pairs = [d for d in enhanced_pair_screening._volume_diagnostics if d['min_vol'] < min_daily_volume_usd]
        
        if passed_pairs:
            logger.info(f"[VOLUME_DIAGNOSTIC] PASSED examples (first 5): " +
                       ", ".join([f"{p['pair']}({p['min_vol']:.0f})" for p in passed_pairs[:5]]))
        if failed_pairs:
            logger.info(f"[VOLUME_DIAGNOSTIC] FAILED examples (first 5): " +
                       ", ".join([f"{p['pair']}({p['min_vol']:.0f})" for p in failed_pairs[:5]]))
        
        # Reset diagnostics for next run
        enhanced_pair_screening._volume_diagnostics = []
    
    # Stage 6: KPSS
    remaining -= filter_stats['kpss_failed']
    logger.info(f"  Stage 6: KPSS -> Filtered: {filter_stats['kpss_failed']}, Remaining: {remaining}")
    
    # Stage 7: Hurst
    remaining -= filter_stats['hurst_failed']
    logger.info(f"  Stage 7: Hurst -> Filtered: {filter_stats['hurst_failed']}, Remaining: {remaining}")
    
    # Stage 8: Mean Crossings
    remaining -= filter_stats['mean_crossings_failed']
    logger.info(f"  Stage 8: Mean Crossings -> Filtered: {filter_stats['mean_crossings_failed']}, Remaining: {remaining}")

    # Stage 9: Profit Potential
    remaining -= filter_stats['profit_potential_failed']
    logger.info(f"  Stage 9: Profit Potential -> Filtered: {filter_stats['profit_potential_failed']}, Remaining: {remaining}")
    
    # Stage 10: Train Edge Filter
    remaining_before_train = remaining
    remaining -= filter_stats['train_filter_failed']
    logger.info(f"  Stage 10: Train Edge -> Filtered: {filter_stats['train_filter_failed']} (before={remaining_before_train}, after={remaining})")
    if train_filter_examples:
        examples = ", ".join([f"{a}-{b}(G2C={g:.2f},CumR={c:.2f})" for a,b,g,c,_ in train_filter_examples[:5]])
        logger.info(f"  Stage 10: filtered examples: {examples}")

    logger.info(f"  Total passed: {filter_stats['total_passed']}")
    
    # Save filter reasons if requested
    if save_filter_reasons and filter_reasons:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_file = Path(f"enhanced_filter_reasons_{timestamp}.csv")
        
        with open(filter_file, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["symbol1", "symbol2", "filter_reason"])
            writer.writerows(filter_reasons)
        
        logger.info(f"Filter reasons saved to: {filter_file}")

        # Log top rejection reasons
        from collections import Counter
        reasons = [r[2] for r in filter_reasons]
        top_reasons = Counter(reasons).most_common(10)
        logger.info(f"  Top rejection reasons: {top_reasons}")
    
    return filtered_pairs
    
def filter_pairs_by_coint_and_half_life(
    pairs: List[Tuple[str, str]],
    price_df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    min_beta: float = MIN_BETA,
    max_beta: float = MAX_BETA,
    min_half_life: float = 2,
    max_half_life: float = 100,
    min_mean_crossings: int = 8,
    min_history_ratio: float = 0.8,
    liquidity_usd_daily: float = 1_000_000.0,
    max_bid_ask_pct: float = 0.2,
    max_avg_funding_pct: float = 0.03,
    save_filter_reasons: bool = False,
    kpss_pvalue_threshold: float = 0.05,
    max_hurst_exponent: float = 0.5,
    *,
    stable_tokens: Optional[List[str]] = None,
) -> List[Tuple[str, str, float, float, float, Dict[str, Any]]]:
    """
    Фильтрует пары по ключевым критериям качества:
    1. p-value коинтеграции
    2. коэффициент beta
    3. half-life спреда
    4. Количество пересечений среднего
    
    Возвращает пары с параметрами (s1, s2, beta, mean, std, metrics), прошедшие все фильтры.
    """
    logger = get_logger("pair_filter")
    if stable_tokens is None:
        stable_tokens = [
            "USDT",
            "USDC",
            "BUSD",
            "TUSD",
            "DAI",
            "USD",
            "USDP",
            "PAX",
            "SUSD",
            "GUSD",
        ]
    def _is_stable(sym: str) -> bool:
        return sym in stable_tokens
    logger.info(f"[ФИЛЬТР] На входе после SSD: {len(pairs)} пар")
    logger.info(f"[ФИЛЬТР] Применяем оптимизированный порядок: Коинтеграция → Beta → Half-life → Mean crossings → Hurst → KPSS → Market microstructure")
    
    # Словарь для отслеживания причин фильтрации по категориям
    filter_stats = {
        'total': len(pairs),
        'insufficient_data': 0,
        'zero_variance': 0,
        'pvalue': 0,
        'beta': 0,
        'half_life': 0,
        'crossings': 0,
        'hurst': 0,
        'kpss': 0,
        'history': 0,
        'liquidity': 0,
    }
    
    # Фильтр 1: Коинтеграция (p-value) - основной критерий качества
    coint_passed = []
    filter_reasons: list[tuple[str,str,str]] = []  # (s1,s2,reason)
    for s1, s2 in pairs:
        pair_data = price_df[[s1, s2]].dropna()
        if pair_data.empty or len(pair_data) < 30:  # Минимум 30 наблюдений для статистических тестов
            filter_reasons.append((s1, s2, 'insufficient_data'))
            filter_stats['insufficient_data'] += 1
            continue
        if pair_data[s2].var() == 0:
            filter_reasons.append((s1, s2, 'zero_variance'))
            filter_stats['zero_variance'] += 1
            continue
            
        # Корреляция между активами (для метрик)
        corr = pair_data[s1].corr(pair_data[s2])
            
        # Коинтеграционный тест (Engle-Granger) - ускоренная версия
        try:
            _score, pvalue, _ = fast_coint(pair_data[s1], pair_data[s2], trend='n')
            if pvalue is None or np.isnan(pvalue):
                filter_reasons.append((s1, s2, 'invalid_pvalue'))
                filter_stats['invalid_pvalue'] = filter_stats.get('invalid_pvalue', 0) + 1
                continue
        except Exception as e:
            logger.debug(f"Ошибка коинтеграционного теста для пары {s1}-{s2}: {str(e)}")
            filter_reasons.append((s1, s2, 'coint_test_error'))
            filter_stats['coint_test_error'] = filter_stats.get('coint_test_error', 0) + 1
            continue
            
        if pvalue >= pvalue_threshold:
            filter_reasons.append((s1, s2, 'pvalue'))
            filter_stats['pvalue'] += 1
            continue
            
        coint_passed.append((s1, s2, corr, pvalue))
        

    logger.info(f"[ФИЛЬТР] После фильтра коинтеграции: {len(coint_passed)} пар")

    # Фильтр 2: Beta range проверка (очень быстро)
    beta_passed = []
    for s1, s2, corr, pvalue in coint_passed:
        pair_data = price_df[[s1, s2]].dropna()
        beta = pair_data[s1].cov(pair_data[s2]) / pair_data[s2].var()

        if not (min_beta <= abs(beta) <= max_beta):
            filter_reasons.append((s1, s2, f"beta_out_of_range ({beta:.2f})"))
            filter_stats['beta'] += 1
            continue

        beta_passed.append((s1, s2, corr, pvalue, beta))
    
    logger.info(f"[ФИЛЬТР] После фильтра beta: {len(beta_passed)} пар")

    # Фильтр 3: Half-life расчет (быстро)
    half_life_passed = []
    for s1, s2, corr, pvalue, beta in beta_passed:
        pair_data = price_df[[s1, s2]].dropna()
        spread = pair_data[s1] - beta * pair_data[s2]

        # Расчёт half-life (в барах)
        hl_bars = calculate_half_life(spread)
        # Определяем таймфрейм в минутах по разнице индексов
        try:
            bar_minutes = int((pair_data.index[1] - pair_data.index[0]).total_seconds() / 60)
            if bar_minutes <= 0:
                bar_minutes = 15
        except Exception:
            bar_minutes = 15  # fallback
        hl_days = hl_bars * bar_minutes / 1440
        if not (min_half_life < hl_days < max_half_life):
            filter_reasons.append((s1, s2, 'half_life'))
            filter_stats['half_life'] += 1
            continue
        
        half_life_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread))
    
    logger.info(f"[ФИЛЬТР] После фильтра half-life: {len(half_life_passed)} пар")

    # Фильтр 4: Mean crossings (быстро)
    crossings_passed = []
    for s1, s2, corr, pvalue, beta, hl_days, spread in half_life_passed:
        mean_crossings = count_mean_crossings(spread)
        if mean_crossings < min_mean_crossings:
            filter_reasons.append((s1, s2, 'crossings'))
            filter_stats['crossings'] += 1
            continue
        
        crossings_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра mean crossings: {len(crossings_passed)} пар")

    # Фильтр 5: Hurst exponent (умеренно)
    hurst_passed = []
    for s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings in crossings_passed:
        hurst_exponent = calculate_hurst_exponent(spread)
        if hurst_exponent > max_hurst_exponent:
            filter_reasons.append((s1, s2, f"hurst_too_high ({hurst_exponent:.2f})"))
            filter_stats['hurst'] += 1
            continue
        
        hurst_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра Hurst: {len(hurst_passed)} пар")

    # Фильтр 6: KPSS тест (медленно)
    kpss_passed = []
    for s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings in hurst_passed:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InterpolationWarning)
                p_kpss = kpss(spread, regression='c', nlags='auto')[1]
            if p_kpss < kpss_pvalue_threshold:
                filter_reasons.append((s1, s2, 'kpss'))
                filter_stats['kpss'] += 1
                continue  # отвергаем стационарность тренда
        except Exception:
            pass  # если тест не применим, пропускаем
        
        kpss_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра KPSS: {len(kpss_passed)} пар")

    # Фильтр 7: Market microstructure (liquidity, bid-ask, funding)
    microstructure_passed = []
    for s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings in kpss_passed:
        # --- History coverage check ---
        hist_ratio1 = price_df[s1].notna().mean()
        hist_ratio2 = price_df[s2].notna().mean()
        if min(hist_ratio1, hist_ratio2) < min_history_ratio:
            filter_reasons.append((s1, s2, 'history'))
            filter_stats['history'] += 1
            continue

        # --- Liquidity & market microstructure checks ---
        vol1, ba1, fund1 = _get_market_metrics(s1)
        vol2, ba2, fund2 = _get_market_metrics(s2)
        if min(vol1, vol2) < liquidity_usd_daily or max(ba1, ba2) > max_bid_ask_pct or max(abs(fund1), abs(fund2)) > max_avg_funding_pct:
            filter_reasons.append((s1, s2, 'liquidity'))
            filter_stats['liquidity'] += 1
            continue
        
        # Расчет стандартного отклонения спреда
        if len(spread) == 0 or spread.isna().all():
            filter_reasons.append((s1, s2, 'spread_empty_stats'))
            filter_stats['spread_empty_stats'] = filter_stats.get('spread_empty_stats', 0) + 1
            continue
        
        mean = spread.mean()
        std = spread.std()
        
        microstructure_passed.append((s1, s2, corr, pvalue, beta, hl_days, mean, std, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра market microstructure: {len(microstructure_passed)} пар")

    # Формируем финальный результат
    result = []
    for s1, s2, corr, pvalue, beta, hl_days, mean, std, mean_crossings in microstructure_passed:
        # Сохраняем расширенные метрики для пары
        metrics = {
            'half_life': hl_days,
            'correlation': corr,
            'mean_crossings': mean_crossings,
            'spread_std': std,
            'pvalue': pvalue
        }
        
        result.append((s1, s2, beta, mean, std, metrics))
        
    # Логирование итогов по этапам
    total_pairs = len(pairs)
    
    # Детальная статистика по фильтрам в порядке применения
    logger.info(f"[ФИЛЬТР] Статистика фильтрации (оптимизированный порядок):")
    logger.info(f"  1. SSD → Коинтеграция: {total_pairs} → {len(coint_passed)} пар")
    logger.info(f"  2. Коинтеграция → Beta: {len(coint_passed)} → {len(beta_passed)} пар")
    logger.info(f"  3. Beta → Half-life: {len(beta_passed)} → {len(half_life_passed)} пар")
    logger.info(f"  4. Half-life → Mean crossings: {len(half_life_passed)} → {len(crossings_passed)} пар")
    logger.info(f"  5. Mean crossings → Hurst: {len(crossings_passed)} → {len(hurst_passed)} пар")
    logger.info(f"  6. Hurst → KPSS: {len(hurst_passed)} → {len(kpss_passed)} пар")
    logger.info(f"  7. KPSS → Market microstructure: {len(kpss_passed)} → {len(microstructure_passed)} пар")
    
    # Рассчитываем процент отфильтрованных пар для каждой причины
    filter_percentages = {reason: (count / total_pairs * 100) for reason, count in filter_stats.items() if count > 0}
    
    # Детальная статистика по причинам отсева
    logger.info(f"[ФИЛЬТР] Причины отсева:")
    for reason, percent in sorted(filter_percentages.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  • {reason}: {filter_stats[reason]} пар ({percent:.1f}%)")
    
    logger.info(f"[ФИЛЬТР] Итого: {total_pairs} → {len(result)} пар ({len(result)/total_pairs*100:.1f}% прошли все фильтры)")

    # Сохраняем причины отсева в CSV
    if save_filter_reasons and filter_reasons:
        out_dir = Path('results')
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'filter_reasons_{ts}.csv'
        pd.DataFrame(filter_reasons, columns=['s1','s2','reason']).to_csv(out_path, index=False)
        logger.info(f"Причины отсева сохранены в {out_path}")

    return result
