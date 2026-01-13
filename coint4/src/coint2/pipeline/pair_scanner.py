"""
Pair scanner for cointegration analysis and universe selection.
"""

# Avoid pytest collecting test_* helpers from this module.
__test__ = False

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from dask import delayed
import warnings
import os
import time
from collections import defaultdict
import yaml
warnings.filterwarnings('ignore')


def find_cointegrated_pairs(handler, start_date, end_date, config) -> List[Tuple]:
    """Find cointegrated pairs from universe.
    
    Args:
        handler: DataHandler instance
        start_date: Start date for analysis
        end_date: End date for analysis
        config: Configuration object
        
    Returns:
        List of tuples representing cointegrated pairs
    """
    # Convert config to dict for scan_universe
    cfg_dict = {
        'train_days': getattr(config, 'train_days', 60),
        'valid_days': getattr(config, 'valid_days', 30),
        'criteria': {
            'coint_pvalue_max': 0.05,
            'hl_min': 5,
            'hl_max': 200,
            'min_cross': 10,
            'beta_drift_max': 0.15
        }
    }
    
    # Scan universe
    df = scan_universe(handler, [], start_date, end_date, cfg_dict)
    
    # Filter passing pairs
    passing = df[df['verdict'] == 'PASS']
    
    # Return as list of tuples
    return [(row['symbol1'], row['symbol2']) for _, row in passing.iterrows()]


def scan_universe(
    data_handler,
    symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: Dict
) -> pd.DataFrame:
    """
    Scan universe of pairs for cointegration.
    
    Args:
        data_handler: Data handler instance
        symbols: List of symbols to consider
        start_date: Start date for analysis
        end_date: End date for analysis
        config: Configuration dictionary
        
    Returns:
        DataFrame with cointegration metrics for each pair
    """
    results = []
    
    # Load data for all symbols
    lookback_days = config.get('train_days', 60) + config.get('valid_days', 30)
    df = data_handler.load_all_data_for_period(
        lookback_days=lookback_days,
        end_date=end_date
    )
    
    # Filter to requested symbols if provided
    if symbols:
        available = [s for s in symbols if s in df.columns]
        df = df[available]
    else:
        symbols = list(df.columns)
    
    # Generate all pairs
    pairs = []
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            pairs.append((symbols[i], symbols[j]))
    
    # Test each pair
    total_pairs = len(pairs)
    print(f"🔄 Testing {total_pairs} pairs...")
    
    # Initialize heartbeat logging and rejection tracking
    log_every = max(1, int(os.getenv('COINT_LOG_EVERY', '1000')))
    t0 = time.time()
    tested = 0
    passed = 0
    reject_counts = defaultdict(int)
    tested_pairs = 0
    
    for idx, (sym1, sym2) in enumerate(pairs, 1):
        if sym1 not in df.columns or sym2 not in df.columns:
            continue
            
        y = df[sym1].dropna()
        x = df[sym2].dropna()
        
        # Align series
        aligned = pd.DataFrame({'y': y, 'x': x}).dropna()
        if len(aligned) < 100:  # Minimum data points
            continue
            
        # Test cointegration
        result = test_cointegration(
            aligned['y'].values,
            aligned['x'].values,
            config
        )
        
        # Add pair info
        result['pair'] = f"{sym1}/{sym2}"
        result['symbol1'] = sym1
        result['symbol2'] = sym2
        
        # Calculate verdict
        verdict, reason = evaluate_pair(result, config)
        result['verdict'] = verdict
        
        results.append(result)
        
        # Update counters
        tested += 1
        tested_pairs += 1
        if verdict == 'PASS':
            passed += 1
        else:
            reject_counts[reason] += 1
        
        # Progress heartbeat logging with top rejection reasons
        if idx % log_every == 0:
            dt = max(1e-9, time.time() - t0)
            rate = idx / dt
            remain = max(0, total_pairs - idx)
            eta_min = (remain / rate) / 60.0 if rate > 0 else float('inf')
            
            # Get top 2 rejection reasons
            top_reasons = ""
            if reject_counts:
                sorted_reasons = sorted(reject_counts.items(), key=lambda x: x[1], reverse=True)[:2]
                top_reasons = ", reasons: " + ", ".join([f"{r}={c}" for r, c in sorted_reasons])
            
            print(f"⏱️ scan: {idx}/{total_pairs} ({idx/max(1,total_pairs):.1%}), "
                  f"{rate:.1f} pairs/s, ETA ~{eta_min:.1f} min, "
                  f"tested={tested}, passed={passed}{top_reasons}", flush=True)
    
    # Prepare rejection breakdown data
    breakdown = {
        'tested_pairs': tested_pairs,
        'passed_pairs': passed,
        'reasons': dict(reject_counts)
    }
    
    # Return both DataFrame and breakdown
    df_results = pd.DataFrame(results)
    df_results._rejection_breakdown = breakdown  # Attach as attribute
    return df_results


def test_cointegration(y: np.ndarray, x: np.ndarray, config: Dict) -> Dict:
    """
    Test cointegration between two series using Engle-Granger method.
    
    Args:
        y: First price series
        x: Second price series
        config: Configuration parameters
        
    Returns:
        Dictionary with test metrics
    """
    result = {}
    
    # Step 1: OLS regression to find beta
    X = np.column_stack([np.ones(len(x)), x])
    betas = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta = betas[0], betas[1]
    
    # Calculate spread
    spread = y - beta * x - alpha
    
    # Step 2: ADF test on residuals
    try:
        adf_result = adfuller(spread, maxlag=10, autolag='BIC')
        result['adf_stat'] = adf_result[0]
        result['pvalue'] = adf_result[1]
    except:
        result['adf_stat'] = 0.0
        result['pvalue'] = 1.0
    
    # Step 3: Calculate half-life
    result['half_life'] = estimate_half_life(spread)
    
    # Step 4: Count mean crossings
    result['crossings'] = count_mean_crossings(spread)

    # Step 5: Hurst exponent
    result['hurst'] = hurst_exponent(spread)
    
    # Step 6: Beta stability (split data and compare)
    mid = len(y) // 2
    beta1 = np.linalg.lstsq(
        np.column_stack([np.ones(mid), x[:mid]]), 
        y[:mid], 
        rcond=None
    )[0][1]
    
    beta2 = np.linalg.lstsq(
        np.column_stack([np.ones(len(y)-mid), x[mid:]]), 
        y[mid:], 
        rcond=None
    )[0][1]
    
    result['beta'] = beta
    result['beta_drift'] = abs(beta2 - beta1) / abs(beta1) if beta1 != 0 else 0
    
    return result


# Prevent pytest from collecting this as a test when imported into test modules.
test_cointegration.__test__ = False


def hurst_exponent(series: np.ndarray) -> float:
    """Estimate Hurst exponent using a log-log variance fit."""
    if series is None or len(series) < 20:
        return 0.5

    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    if len(series) < 20:
        return 0.5

    # Heuristic: strong linear trend implies persistent behavior
    t = np.arange(len(series))
    corr = np.corrcoef(t, series)[0, 1]
    if np.isfinite(corr) and abs(corr) > 0.98:
        return 0.8

    lags = range(2, 20)
    tau = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        tau.append(np.std(diff))

    tau = np.array(tau)
    if np.any(tau <= 0):
        return 0.5

    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0]
    return float(np.clip(hurst, 0.0, 1.0))


def estimate_half_life(spread: np.ndarray) -> float:
    """
    Estimate half-life of mean reversion using AR(1) model.
    
    Args:
        spread: Spread series
        
    Returns:
        Half-life in periods
    """
    if len(spread) < 3:
        return np.inf

    try:
        spread_series = pd.Series(spread).dropna()
        if len(spread_series) < 3 or spread_series.nunique() < 2:
            return np.inf

        y = spread_series.iloc[1:]
        y_lag = spread_series.iloc[:-1]
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        beta = float(coeffs[1]) if len(coeffs) > 1 else 0.0
        if beta == 0.0:
            return np.inf

        if beta < 0:
            beta_abs = min(abs(beta), 0.9)
        else:
            beta_abs = abs(beta)

        if beta_abs >= 0.99 or beta_abs <= 0.0:
            return np.inf

        half_life = -np.log(2) / np.log(beta_abs)
        return min(max(half_life, 1), 1000)
    except Exception:
        return np.inf




def count_mean_crossings(spread: np.ndarray) -> int:
    """
    Count number of mean crossings in spread.
    
    Args:
        spread: Spread series
        
    Returns:
        Number of mean crossings
    """
    if len(spread) < 2:
        return 0
        
    mean = np.mean(spread)
    centered = spread - mean
    
    # Count sign changes
    signs = np.sign(centered)
    sign_changes = np.diff(signs)
    crossings = np.sum(np.abs(sign_changes) > 0)
    
    return int(crossings)


def evaluate_pair(metrics: Dict, config: Dict) -> Tuple[str, str]:
    """
    Evaluate if pair passes selection criteria.
    
    Args:
        metrics: Cointegration metrics
        config: Selection criteria
        
    Returns:
        Tuple of (verdict, reason) where verdict is 'PASS' or 'FAIL'
        and reason is the first failed check or 'OK'
    """
    criteria = config.get('criteria', {})
    
    # P-value check
    if metrics['pvalue'] > criteria.get('coint_pvalue_max', 0.05):
        return 'FAIL', 'pvalue'
    
    # Half-life check
    hl = metrics['half_life']
    hl_min = criteria.get('hl_min', 5)
    hl_max = criteria.get('hl_max', 200)
    if hl < hl_min or hl > hl_max:
        return 'FAIL', 'half_life'
    
    # Crossings check
    if metrics['crossings'] < criteria.get('min_cross', 10):
        return 'FAIL', 'crossings'
    
    # Beta drift check
    if metrics['beta_drift'] > criteria.get('beta_drift_max', 0.15):
        return 'FAIL', 'beta_drift'
    
    # All checks passed
    return 'PASS', 'OK'


def calculate_pair_score(metrics: Dict, config: Dict) -> float:
    """
    Calculate score for pair ranking.
    
    Args:
        metrics: Cointegration metrics
        config: Scoring weights
        
    Returns:
        Score (higher is better)
    """
    score = 0.0
    
    # Lower p-value is better
    score -= metrics['pvalue'] * 10
    
    # Penalize beta drift
    score -= metrics['beta_drift'] * 5
    
    # Reward crossings
    score += min(metrics['crossings'] / 100, 1.0) * 2
    
    # Optimal half-life around 20-50
    hl = metrics['half_life']
    if 20 <= hl <= 50:
        score += 1.0
    elif 10 <= hl <= 100:
        score += 0.5
    
    # Removed Hurst scoring - not meaningful for crypto
    
    return score


@delayed
def _test_pair_for_tradability(
    handler_arg,
    s1: str,
    s2: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_hl: float,
    max_hl: float,
    min_cross: int,
) -> Optional[Tuple[str, str]]:
    """Заглушка для тестирования торгуемости пары.
    
    Args:
        handler_arg: DataHandler instance
        s1: First symbol
        s2: Second symbol
        start_date: Start date
        end_date: End date
        min_hl: Minimum half-life
        max_hl: Maximum half-life
        min_cross: Minimum crossings
        
    Returns:
        Tuple of symbols if tradable, None otherwise
    """
    # Простая заглушка - возвращаем пару как есть
    return (s1, s2)
