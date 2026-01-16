import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
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
import multiprocessing
import os
from typing import List, Tuple, Dict, Any, Optional

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

import logging
from typing import List, Tuple, Optional, Dict, Any

# Разумные границы для коэффициента beta
MIN_BETA = 0.1
MAX_BETA = 10.0

_FILTER_PRICE_DF: Optional[pd.DataFrame] = None
_FILTER_WORKER_CFG: Optional["_FilterWorkerCfg"] = None


@dataclass(frozen=True)
class _FilterWorkerCfg:
    pvalue_threshold: float
    min_beta: float
    max_beta: float
    min_half_life: float
    max_half_life: float
    min_mean_crossings: int
    min_history_ratio: float
    min_correlation: float
    liquidity_usd_daily: float
    max_bid_ask_pct: float
    max_avg_funding_pct: float
    kpss_pvalue_threshold: Optional[float]
    max_hurst_exponent: float


def _normalize_n_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs is None:
        return 1
    try:
        n_jobs = int(n_jobs)
    except (TypeError, ValueError):
        return 1
    if n_jobs == -1:
        return os.cpu_count() or 1
    return max(1, n_jobs)


def _set_numba_threads(target_threads: int) -> None:
    try:
        from numba import set_num_threads
        set_num_threads(max(1, int(target_threads)))
    except Exception:
        pass


def _init_filter_worker(price_df: pd.DataFrame, cfg: _FilterWorkerCfg, numba_threads: int) -> None:
    global _FILTER_PRICE_DF, _FILTER_WORKER_CFG
    _FILTER_PRICE_DF = price_df
    _FILTER_WORKER_CFG = cfg
    _set_numba_threads(numba_threads)


def _reject_pair(stage: str, reason_key: str, reason_detail: Optional[str], s1: str, s2: str) -> tuple:
    return ("reject", stage, reason_key, reason_detail or reason_key, (s1, s2))


def _evaluate_pair(pair: Tuple[str, str], price_df: pd.DataFrame, cfg: _FilterWorkerCfg) -> tuple:
    s1, s2 = pair
    try:
        pair_data = price_df[[s1, s2]].dropna()
    except KeyError:
        return _reject_pair("data", "insufficient_data", "missing_price_data", s1, s2)

    if pair_data.empty or len(pair_data) < 960:
        return _reject_pair("data", "insufficient_data", "insufficient_data", s1, s2)
    if pair_data[s2].var() == 0:
        return _reject_pair("data", "zero_variance", "zero_variance", s1, s2)

    corr = pair_data[s1].corr(pair_data[s2])
    if corr < cfg.min_correlation:
        return _reject_pair("correlation", "low_correlation", f"low_correlation_{corr:.2f}", s1, s2)

    beta = pair_data[s1].cov(pair_data[s2]) / pair_data[s2].var()
    if not (cfg.min_beta <= abs(beta) <= cfg.max_beta):
        return _reject_pair("beta", "beta", f"beta_out_of_range ({beta:.2f})", s1, s2)

    spread = pair_data[s1] - beta * pair_data[s2]
    mean_crossings = count_mean_crossings(spread)
    if mean_crossings < cfg.min_mean_crossings:
        return _reject_pair("crossings", "crossings", f"crossings_{mean_crossings}", s1, s2)

    hl_bars = calculate_half_life(spread)
    try:
        bar_minutes = int((pair_data.index[1] - pair_data.index[0]).total_seconds() / 60)
        if bar_minutes <= 0:
            bar_minutes = 15
    except Exception:
        bar_minutes = 15
    hl_days = hl_bars * bar_minutes / 1440
    if not (cfg.min_half_life < hl_days < cfg.max_half_life):
        return _reject_pair("half_life", "half_life", f"half_life_{hl_days:.1f}", s1, s2)

    try:
        _score, pvalue, _ = fast_coint(pair_data[s1], pair_data[s2], trend='n')
        if pvalue is None or np.isnan(pvalue):
            return _reject_pair("coint", "invalid_pvalue", "invalid_pvalue", s1, s2)
    except Exception:
        return _reject_pair("coint", "coint_test_error", "coint_test_error", s1, s2)

    if pvalue >= cfg.pvalue_threshold:
        return _reject_pair("coint", "pvalue", f"pvalue_{pvalue:.3f}", s1, s2)

    hurst_exponent = calculate_hurst_exponent(spread)
    if hurst_exponent > cfg.max_hurst_exponent:
        return _reject_pair("hurst", "hurst", f"hurst_too_high ({hurst_exponent:.2f})", s1, s2)

    if cfg.kpss_pvalue_threshold is not None and cfg.kpss_pvalue_threshold < 0.95:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InterpolationWarning)
                p_kpss = kpss(spread, regression='c', nlags='auto')[1]
            if p_kpss < cfg.kpss_pvalue_threshold:
                return _reject_pair("kpss", "kpss", "kpss", s1, s2)
        except Exception:
            pass

    if cfg.liquidity_usd_daily > 0 and cfg.max_bid_ask_pct < 1.0:
        hist_ratio1 = price_df[s1].notna().mean()
        hist_ratio2 = price_df[s2].notna().mean()
        if min(hist_ratio1, hist_ratio2) < cfg.min_history_ratio:
            return _reject_pair("market", "history", "history", s1, s2)

        vol1, ba1, fund1 = _get_market_metrics(s1)
        vol2, ba2, fund2 = _get_market_metrics(s2)
        if (
            min(vol1, vol2) < cfg.liquidity_usd_daily
            or max(ba1, ba2) > cfg.max_bid_ask_pct
            or max(abs(fund1), abs(fund2)) > cfg.max_avg_funding_pct
        ):
            return _reject_pair("market", "liquidity", "liquidity", s1, s2)

    if len(spread) == 0 or spread.isna().all():
        return _reject_pair("market", "spread_empty_stats", "spread_empty_stats", s1, s2)

    mean = spread.mean()
    std = spread.std()
    metrics = {
        'half_life': hl_days,
        'correlation': corr,
        'mean_crossings': mean_crossings,
        'spread_std': std,
        'pvalue': pvalue,
    }

    return ("pass", None, None, None, (s1, s2, beta, mean, std, metrics))


def _filter_pairs_parallel(
    pairs: List[Tuple[str, str]],
    price_df: pd.DataFrame,
    cfg: _FilterWorkerCfg,
    n_jobs: int,
    parallel_backend: str,
    save_filter_reasons: bool,
) -> List[Tuple[str, str, float, float, float, Dict[str, Any]]]:
    logger = logging.getLogger("pair_filter")
    total_pairs = len(pairs)

    stage_order = (
        "data",
        "correlation",
        "beta",
        "crossings",
        "half_life",
        "coint",
        "hurst",
        "kpss",
        "market",
    )
    stage_idx = {stage: idx for idx, stage in enumerate(stage_order)}

    filter_stats: Dict[str, int] = {
        'total': total_pairs,
        'insufficient_data': 0,
        'zero_variance': 0,
        'low_correlation': 0,
        'pvalue': 0,
        'beta': 0,
        'half_life': 0,
        'crossings': 0,
        'hurst': 0,
        'kpss': 0,
        'history': 0,
        'liquidity': 0,
    }

    backend = (parallel_backend or "threads").strip().lower()
    if backend not in ("threads", "processes", "auto"):
        backend = "threads"
    if backend == "auto":
        backend = "processes"

    numba_threads = max(1, (os.cpu_count() or 1) // max(1, n_jobs))
    results = []

    if backend == "processes":
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = multiprocessing.get_context()
        if ctx.get_start_method() != "fork":
            logger.warning("[ФИЛЬТР] Fork недоступен, переключаюсь на threads для фильтрации")
            backend = "threads"

    if backend == "threads":
        logger.info(f"[ФИЛЬТР] Параллельная фильтрация: threads ({n_jobs} workers)")
        _set_numba_threads(numba_threads)
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(lambda p: _evaluate_pair(p, price_df, cfg), pairs))
    else:
        logger.info(f"[ФИЛЬТР] Параллельная фильтрация: processes ({n_jobs} workers)")
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=ctx,
            initializer=_init_filter_worker,
            initargs=(price_df, cfg, numba_threads),
        ) as executor:
            chunksize = max(1, total_pairs // max(1, n_jobs * 4))
            results = list(executor.map(_filter_pair_worker, pairs, chunksize=chunksize))

    filter_reasons: list[tuple[str, str, str]] = []
    stage_indices: list[int] = []
    passed_pairs: list[Tuple[str, str, float, float, float, Dict[str, Any]]] = []

    for status, stage, reason_key, reason_detail, payload in results:
        if status == "pass":
            stage_indices.append(len(stage_order))
            passed_pairs.append(payload)
            continue

        stage_indices.append(stage_idx.get(stage, 0))
        s1, s2 = payload
        if reason_key:
            filter_stats.setdefault(reason_key, 0)
            filter_stats[reason_key] += 1
        if save_filter_reasons:
            filter_reasons.append((s1, s2, reason_detail or reason_key or "unknown"))

    def _count_after(stage_name: str) -> int:
        idx = stage_idx[stage_name]
        return sum(stage_idx_val > idx for stage_idx_val in stage_indices)

    data_passed = _count_after("data")
    correlation_passed = _count_after("correlation")
    beta_passed = _count_after("beta")
    crossings_passed = _count_after("crossings")
    half_life_passed = _count_after("half_life")
    coint_passed = _count_after("coint")
    hurst_passed = _count_after("hurst")
    kpss_passed = _count_after("kpss")
    microstructure_passed = _count_after("market")

    logger.info(f"[ФИЛЬТР] После проверки данных: {data_passed} пар")
    logger.info(f"[ФИЛЬТР] После фильтра корреляции (>{cfg.min_correlation}): {correlation_passed} пар")
    logger.info(f"[ФИЛЬТР] После фильтра beta: {beta_passed} пар")
    logger.info(f"[ФИЛЬТР] После фильтра mean crossings: {crossings_passed} пар")
    logger.info(f"[ФИЛЬТР] После фильтра half-life: {half_life_passed} пар")
    logger.info(f"[ФИЛЬТР] После фильтра коинтеграции: {coint_passed} пар")
    logger.info(f"[ФИЛЬТР] После фильтра Hurst: {hurst_passed} пар")
    logger.info(f"[ФИЛЬТР] После фильтра KPSS: {kpss_passed} пар")
    logger.info(f"[ФИЛЬТР] После фильтра market microstructure: {microstructure_passed} пар")

    logger.info(f"[ФИЛЬТР] Статистика фильтрации (оптимизированный порядок):")
    logger.info(f"  1. SSD → Коинтеграция: {total_pairs} → {coint_passed} пар")
    logger.info(f"  2. Коинтеграция → Beta: {coint_passed} → {beta_passed} пар")
    logger.info(f"  3. Beta → Half-life: {beta_passed} → {half_life_passed} пар")
    logger.info(f"  4. Half-life → Mean crossings: {half_life_passed} → {crossings_passed} пар")
    logger.info(f"  5. Mean crossings → Hurst: {crossings_passed} → {hurst_passed} пар")
    logger.info(f"  6. Hurst → KPSS: {hurst_passed} → {kpss_passed} пар")
    logger.info(f"  7. KPSS → Market microstructure: {kpss_passed} → {microstructure_passed} пар")

    filter_percentages = {
        reason: (count / total_pairs * 100)
        for reason, count in filter_stats.items()
        if count > 0 and reason != "total"
    }

    logger.info(f"[ФИЛЬТР] Причины отсева:")
    for reason, percent in sorted(filter_percentages.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  • {reason}: {filter_stats[reason]} пар ({percent:.1f}%)")

    logger.info(
        f"[ФИЛЬТР] Итого: {total_pairs} → {len(passed_pairs)} пар "
        f"({len(passed_pairs)/total_pairs*100:.1f}% прошли все фильтры)"
    )

    if save_filter_reasons and filter_reasons:
        out_dir = Path('results')
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'filter_reasons_{ts}.csv'
        pd.DataFrame(filter_reasons, columns=['s1','s2','reason']).to_csv(out_path, index=False)
        logger.info(f"Причины отсева сохранены в {out_path}")

    return passed_pairs


def _filter_pair_worker(pair: Tuple[str, str]) -> tuple:
    if _FILTER_PRICE_DF is None or _FILTER_WORKER_CFG is None:
        raise RuntimeError("Filter worker not initialized")
    return _evaluate_pair(pair, _FILTER_PRICE_DF, _FILTER_WORKER_CFG)

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
    *,
    stable_tokens: Optional[List[str]] = None,
) -> List[Tuple[str, str, float, float, float, Dict[str, Any]]]:
    """Enhanced pair screening with strict criteria for 15-minute data.
     
     Criteria:
     1. p-value коинтеграции < 0.05
     2. half-life < N бар (default 96 bars = 24 hours for 15-min data)
     3. среднедневной объём ≥ 50k $ на каждую leg
     
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
     
     Returns
     -------
     List[Tuple[str, str, float, float, float, Dict[str, Any]]]
         Filtered pairs with (s1, s2, beta, mean, std, metrics)
     """
    logger = logging.getLogger("enhanced_pair_screening")
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
        "total_passed": 0
    }
    
    filtered_pairs = []
    filter_reasons = []
    
    for s1, s2 in pairs:
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
            if len(common_idx) < 960:  # Минимум 10 дней для 15-мин баров (960 баров)
                filter_stats["data_insufficient"] += 1
                if save_filter_reasons:
                    filter_reasons.append((s1, s2, "insufficient_data"))
                continue
            
            y_aligned = y.loc[common_idx]
            x_aligned = x.loc[common_idx]
            
            # 1. Cointegration test (p-value criterion)
            try:
                _, pvalue, _ = fast_coint(y_aligned, x_aligned, trend='n')
                if pvalue > pvalue_threshold:
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
                X = sm.add_constant(x_aligned)
                model = sm.OLS(y_aligned, X).fit()
                alpha = model.params.iloc[0]
                beta = model.params.iloc[1]
                
                # 2. Beta criterion
                if not (min_beta <= abs(beta) <= max_beta):
                    filter_stats["beta_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"beta_{beta:.4f}"))
                    continue
                
                # Calculate spread and statistics
                spread = y_aligned - (alpha + beta * x_aligned)
                
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
                if half_life_bars > max_half_life_bars:
                    filter_stats["half_life_failed"] += 1
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"half_life_{half_life_bars:.1f}_bars"))
                    continue
                
                # 4. Volume criterion (estimate daily volume)
                # For 15-min data: 96 bars per day
                bars_per_day = 96
                recent_data_days = min(30, len(common_idx) // bars_per_day)  # Last 30 days or available
                
                if recent_data_days > 0:
                    recent_periods = recent_data_days * bars_per_day
                    recent_y = y_aligned.iloc[-recent_periods:] if len(y_aligned) >= recent_periods else y_aligned
                    recent_x = x_aligned.iloc[-recent_periods:] if len(x_aligned) >= recent_periods else x_aligned
                    
                    # Estimate daily volume (simplified - using price * typical volume multiplier)
                    # This is a placeholder - in real implementation, you'd use actual volume data
                    
                    # Check for empty recent data to avoid numpy warnings
                    if len(recent_y) == 0 or recent_y.isna().all() or len(recent_x) == 0 or recent_x.isna().all():
                        filter_stats["recent_data_empty"] = filter_stats.get("recent_data_empty", 0) + 1
                        if save_filter_reasons:
                            filter_reasons.append((s1, s2, "recent_data_empty"))
                        continue
                    
                    avg_price_s1 = recent_y.mean()
                    avg_price_s2 = recent_x.mean()
                    
                    # Placeholder volume check - in practice, you'd get this from market data
                    # For now, we'll use a simplified heuristic based on price volatility
                    volatility_s1 = recent_y.ffill().pct_change(fill_method=None).std() * np.sqrt(bars_per_day)
                    volatility_s2 = recent_x.ffill().pct_change(fill_method=None).std() * np.sqrt(bars_per_day)
                    
                    # Estimate volume based on volatility (higher vol usually means higher volume)
                    # This is a rough approximation - replace with actual volume data
                    estimated_volume_s1 = avg_price_s1 * 1000 * (1 + volatility_s1 * 10)
                    estimated_volume_s2 = avg_price_s2 * 1000 * (1 + volatility_s2 * 10)
                    
                    if (estimated_volume_s1 < min_daily_volume_usd or 
                        estimated_volume_s2 < min_daily_volume_usd):
                        filter_stats["volume_failed"] += 1
                        if save_filter_reasons:
                            filter_reasons.append((s1, s2, f"volume_s1_{estimated_volume_s1:.0f}_s2_{estimated_volume_s2:.0f}"))
                        continue
                
                # Additional quality checks
                
                # KPSS test for spread stationarity
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
                try:
                    hurst = calculate_hurst_exponent(spread)
                    if hurst > max_hurst_exponent:
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
                    if save_filter_reasons:
                        filter_reasons.append((s1, s2, f"mean_crossings_{mean_crossings}"))
                    continue
                
                # If we reach here, the pair passed all filters
                metrics = {
                    "p_value": pvalue,
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
    logger.info(f"  Passed all filters: {filter_stats['total_passed']}")
    logger.info(f"  Failed p-value: {filter_stats['p_value_failed']}")
    logger.info(f"  Failed half-life: {filter_stats['half_life_failed']}")
    logger.info(f"  Failed volume: {filter_stats['volume_failed']}")
    logger.info(f"  Failed beta: {filter_stats['beta_failed']}")
    logger.info(f"  Failed KPSS: {filter_stats['kpss_failed']}")
    logger.info(f"  Failed Hurst: {filter_stats['hurst_failed']}")
    logger.info(f"  Data insufficient: {filter_stats['data_insufficient']}")
    
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
    min_correlation: float = 0.5,  # НОВЫЙ параметр для фильтрации по корреляции
    liquidity_usd_daily: float = 1_000_000.0,
    max_bid_ask_pct: float = 0.2,
    max_avg_funding_pct: float = 0.03,
    save_filter_reasons: bool = False,
    kpss_pvalue_threshold: Optional[float] = 0.05,
    max_hurst_exponent: float = 0.5,
    *,
    stable_tokens: Optional[List[str]] = None,
    n_jobs: Optional[int] = None,
    parallel_backend: str = "threads",
) -> List[Tuple[str, str, float, float, float, Dict[str, Any]]]:
    """
    Фильтрует пары по ключевым критериям качества:
    1. p-value коинтеграции
    2. коэффициент beta
    3. half-life спреда
    4. Количество пересечений среднего
    
    Возвращает пары с параметрами (s1, s2, beta, mean, std, metrics), прошедшие все фильтры.
    """
    logger = logging.getLogger("pair_filter")
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

    normalized_jobs = _normalize_n_jobs(n_jobs)
    if normalized_jobs > 1 and pairs:
        worker_cfg = _FilterWorkerCfg(
            pvalue_threshold=pvalue_threshold,
            min_beta=min_beta,
            max_beta=max_beta,
            min_half_life=min_half_life,
            max_half_life=max_half_life,
            min_mean_crossings=min_mean_crossings,
            min_history_ratio=min_history_ratio,
            min_correlation=min_correlation,
            liquidity_usd_daily=liquidity_usd_daily,
            max_bid_ask_pct=max_bid_ask_pct,
            max_avg_funding_pct=max_avg_funding_pct,
            kpss_pvalue_threshold=kpss_pvalue_threshold,
            max_hurst_exponent=max_hurst_exponent,
        )
        return _filter_pairs_parallel(
            pairs=pairs,
            price_df=price_df,
            cfg=worker_cfg,
            n_jobs=normalized_jobs,
            parallel_backend=parallel_backend,
            save_filter_reasons=save_filter_reasons,
        )
    logger.info(f"[ФИЛЬТР] На входе после SSD: {len(pairs)} пар")
    logger.info(f"[ФИЛЬТР] Оптимизированный порядок: Данные → Корреляция → Beta → Crossings → Half-life → Коинтеграция → Hurst → KPSS")
    
    # Словарь для отслеживания причин фильтрации по категориям
    filter_stats = {
        'total': len(pairs),
        'insufficient_data': 0,
        'zero_variance': 0,
        'low_correlation': 0,  # НОВЫЙ счетчик
        'pvalue': 0,
        'beta': 0,
        'half_life': 0,
        'crossings': 0,
        'hurst': 0,
        'kpss': 0,
        'history': 0,
        'liquidity': 0,
    }
    
    # Оптимизированный порядок фильтров (от быстрых к медленным)
    data_passed = []
    filter_reasons: list[tuple[str,str,str]] = []  # (s1,s2,reason)
    
    # Фильтр 1: Проверка данных (БЫСТРО)
    for s1, s2 in pairs:
        pair_data = price_df[[s1, s2]].dropna()
        if pair_data.empty or len(pair_data) < 960:  # Минимум 10 дней для 15-мин баров (960 баров) для статистических тестов
            filter_reasons.append((s1, s2, 'insufficient_data'))
            filter_stats['insufficient_data'] += 1
            continue
        if pair_data[s2].var() == 0:
            filter_reasons.append((s1, s2, 'zero_variance'))
            filter_stats['zero_variance'] += 1
            continue
        data_passed.append((s1, s2, pair_data))
    
    logger.info(f"[ФИЛЬТР] После проверки данных: {len(data_passed)} пар")
    
    # Фильтр 2: Корреляция (БЫСТРО)
    correlation_passed = []
    for s1, s2, pair_data in data_passed:
        corr = pair_data[s1].corr(pair_data[s2])
        
        # НОВЫЙ фильтр по корреляции - важно для крипто пар
        if corr < min_correlation:
            filter_reasons.append((s1, s2, f'low_correlation_{corr:.2f}'))
            filter_stats['low_correlation'] += 1
            continue
        
        correlation_passed.append((s1, s2, pair_data, corr))
    
    logger.info(f"[ФИЛЬТР] После фильтра корреляции (>{min_correlation}): {len(correlation_passed)} пар")
    
    # Фильтр 3: Beta (БЫСТРО)
    beta_passed = []
    for s1, s2, pair_data, corr in correlation_passed:
        beta = pair_data[s1].cov(pair_data[s2]) / pair_data[s2].var()
        
        if not (min_beta <= abs(beta) <= max_beta):
            filter_reasons.append((s1, s2, f"beta_out_of_range ({beta:.2f})"))
            filter_stats['beta'] += 1
            continue
        
        beta_passed.append((s1, s2, pair_data, corr, beta))
    
    logger.info(f"[ФИЛЬТР] После фильтра beta: {len(beta_passed)} пар")
    
    # Фильтр 4: Mean crossings (БЫСТРО)
    crossings_passed = []
    for s1, s2, pair_data, corr, beta in beta_passed:
        spread = pair_data[s1] - beta * pair_data[s2]
        mean_crossings = count_mean_crossings(spread)
        
        if mean_crossings < min_mean_crossings:
            filter_reasons.append((s1, s2, f'crossings_{mean_crossings}'))
            filter_stats['crossings'] += 1
            continue
        
        crossings_passed.append((s1, s2, pair_data, corr, beta, spread, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра mean crossings: {len(crossings_passed)} пар")
    
    # Фильтр 5: Half-life (СРЕДНЕ)
    half_life_passed = []
    for s1, s2, pair_data, corr, beta, spread, mean_crossings in crossings_passed:
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
            filter_reasons.append((s1, s2, f'half_life_{hl_days:.1f}'))
            filter_stats['half_life'] += 1
            continue
        
        half_life_passed.append((s1, s2, pair_data, corr, beta, spread, mean_crossings, hl_days))
    
    logger.info(f"[ФИЛЬТР] После фильтра half-life: {len(half_life_passed)} пар")
    
    # Фильтр 6: Коинтеграция (МЕДЛЕННО)
    coint_passed = []
    for s1, s2, pair_data, corr, beta, spread, mean_crossings, hl_days in half_life_passed:
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
            filter_reasons.append((s1, s2, f'pvalue_{pvalue:.3f}'))
            filter_stats['pvalue'] += 1
            continue
            
        coint_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра коинтеграции: {len(coint_passed)} пар")

    # Фильтр 7: Hurst exponent (умеренно)
    hurst_passed = []
    for s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings in coint_passed:
        hurst_exponent = calculate_hurst_exponent(spread)
        if hurst_exponent > max_hurst_exponent:
            filter_reasons.append((s1, s2, f"hurst_too_high ({hurst_exponent:.2f})"))
            filter_stats['hurst'] += 1
            continue
        
        hurst_passed.append((s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings))
    
    logger.info(f"[ФИЛЬТР] После фильтра Hurst: {len(hurst_passed)} пар")

    # Фильтр 6: KPSS тест (медленно)
    kpss_passed = []
    
    # Если порог не задан или >= 0.95, пропускаем все пары (фильтр отключен)
    if kpss_pvalue_threshold is None:
        kpss_passed = hurst_passed
        logger.info("[ФИЛЬТР] KPSS фильтр отключен (threshold=None)")
    elif kpss_pvalue_threshold >= 0.95:
        kpss_passed = hurst_passed
        logger.info(f"[ФИЛЬТР] KPSS фильтр отключен (threshold={kpss_pvalue_threshold:.2f} >= 0.95)")
    else:
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
    
    # Если ликвидность <= 0 или другие параметры максимальны, пропускаем фильтр
    if liquidity_usd_daily <= 0 or max_bid_ask_pct >= 1.0:
        for s1, s2, corr, pvalue, beta, hl_days, spread, mean_crossings in kpss_passed:
            if len(spread) == 0 or spread.isna().all():
                filter_reasons.append((s1, s2, 'spread_empty_stats'))
                filter_stats['spread_empty_stats'] = filter_stats.get('spread_empty_stats', 0) + 1
                continue
            mean = spread.mean()
            std = spread.std()
            microstructure_passed.append((s1, s2, corr, pvalue, beta, hl_days, mean, std, mean_crossings))
        logger.info(f"[ФИЛЬТР] Market microstructure фильтр отключен")
    else:
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
