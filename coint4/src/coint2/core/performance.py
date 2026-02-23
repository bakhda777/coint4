"""Performance metrics for trading strategies."""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from coint2.core.sharpe import annualized_sharpe_ratio


def _canonical_period_sharpe(returns: np.ndarray) -> float:
    """Compute non-annualized Sharpe using the canonical implementation."""
    return annualized_sharpe_ratio(returns, 1.0)


def compute_coverage_metrics(
    pnl: pd.Series | None,
    *,
    start_date: str,
    end_date: str,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute test-period coverage diagnostics for a daily PnL series.

    Definitions (calendar days; inclusive):
    - expected_test_days  = days in [start_date, end_date]
    - observed_test_days  = unique days present in pnl index (after date normalization)
    - coverage_ratio      = observed / expected
    - zero_pnl_days       = count(|pnl| < eps) over observed days
    - zero_pnl_days_pct   = zero_pnl_days / expected
    - missing_test_days   = expected - observed

    Notes:
    - expected days are derived ONLY from the config dates, so missing PnL rows are detectable.
    - pnl is grouped by normalized day to avoid double-counting when timestamps are intraday.
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    expected = int((end_ts - start_ts).days + 1) if end_ts >= start_ts else 0

    observed = 0
    zero_days = 0
    if pnl is not None and not pnl.empty:
        idx = pd.to_datetime(pnl.index, errors="coerce")
        values = pd.to_numeric(pnl, errors="coerce")
        series = pd.Series(values.to_numpy(copy=False), index=idx).dropna()
        if not series.empty:
            series = series.groupby(series.index.normalize()).sum()
            observed = int(series.shape[0])
            zero_days = int((series.abs() < float(eps)).sum())

    coverage = float("nan") if expected <= 0 else float(observed) / float(expected)
    zero_pct = float("nan") if expected <= 0 else float(zero_days) / float(expected)
    missing = float("nan") if expected <= 0 else float(expected - observed)

    return {
        "expected_test_days": float(expected),
        "observed_test_days": float(observed),
        "coverage_ratio": float(coverage),
        "zero_pnl_days": float(zero_days),
        "zero_pnl_days_pct": float(zero_pct),
        "missing_test_days": float(missing),
    }


def sharpe_ratio_on_returns(
    pnl: pd.Series,
    capital: float,
    annualizing_factor: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Sharpe по доходностям (PnL / capital).
    """
    if capital == 0 or pnl.empty:
        return float('nan')
    returns = pnl / capital
    return annualized_sharpe_ratio(
        returns,
        annualizing_factor,
        risk_free_rate=risk_free_rate,
    )

def max_drawdown_on_equity(equity_curve: pd.Series) -> float:
    """
    Max Drawdown по equity curve (в процентах).
    """
    if equity_curve.empty:
        return float('nan')
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())

def sharpe_ratio(returns: pd.Series, annualizing_factor: float) -> float:
    """Calculates the annualized Sharpe ratio from a series of returns.

    Parameters
    ----------
    returns : pd.Series
        Series of strategy returns expressed in percentages (not PnL).
    annualizing_factor : float
        Factor used to annualize the Sharpe ratio (e.g. number of trading days).

    Returns
    -------
    float
        Annualized Sharpe ratio. Returns ``0.0`` if the standard deviation of
        ``returns`` is zero.
    """
    return annualized_sharpe_ratio(returns, annualizing_factor)


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Calculate maximum drawdown from a cumulative PnL series.

    Parameters
    ----------
    cumulative_pnl : pd.Series
        Cumulative profit and loss series.

    Returns
    -------
    float
        Maximum drawdown as a non-positive number.
    """
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    return float(drawdown.min())


def win_rate(pnl_series: pd.Series) -> float:
    """Calculate win rate from a PnL series.
    
    Parameters
    ----------
    pnl_series : pd.Series
        Series of individual trade PnLs or daily PnLs.
        
    Returns
    -------
    float
        Win rate as a percentage (0.0 to 1.0).
    """
    if pnl_series.empty:
        return 0.0
        
    non_zero_pnl = pnl_series[pnl_series != 0]
    if len(non_zero_pnl) == 0:
        return 0.0
        
    winning_trades = (non_zero_pnl > 0).sum()
    total_trades = len(non_zero_pnl)
    
    return winning_trades / total_trades


def expectancy(pnl_series: pd.Series) -> float:
    """Calculate expectancy from a PnL series.
    
    Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)
    
    Parameters
    ----------
    pnl_series : pd.Series
        Series of individual trade PnLs or daily PnLs.
        
    Returns
    -------
    float
        Expectancy value.
    """
    if pnl_series.empty:
        return 0.0
        
    non_zero_pnl = pnl_series[pnl_series != 0]
    if len(non_zero_pnl) == 0:
        return 0.0
        
    winning_trades = non_zero_pnl[non_zero_pnl > 0]
    losing_trades = non_zero_pnl[non_zero_pnl < 0]
    
    if len(winning_trades) == 0 and len(losing_trades) == 0:
        return 0.0
        
    win_rate_val = len(winning_trades) / len(non_zero_pnl)
    loss_rate = len(losing_trades) / len(non_zero_pnl)
    
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
    avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0.0
    
    return (win_rate_val * avg_win) - (loss_rate * avg_loss)


def kelly_criterion(pnl_series: pd.Series) -> float:
    """Calculate Kelly Criterion from a PnL series.
    
    Kelly % = (bp - q) / b
    where:
    - b = odds received on the wager (avg_win / avg_loss)
    - p = probability of winning (win rate)
    - q = probability of losing (1 - p)
    
    Parameters
    ----------
    pnl_series : pd.Series
        Series of individual trade PnLs or daily PnLs.
        
    Returns
    -------
    float
        Kelly percentage (can be negative, indicating no bet should be made).
    """
    if pnl_series.empty:
        return 0.0
        
    non_zero_pnl = pnl_series[pnl_series != 0]
    if len(non_zero_pnl) == 0:
        return 0.0
        
    winning_trades = non_zero_pnl[non_zero_pnl > 0]
    losing_trades = non_zero_pnl[non_zero_pnl < 0]
    
    if len(winning_trades) == 0 or len(losing_trades) == 0:
        return 0.0
        
    win_rate_val = len(winning_trades) / len(non_zero_pnl)
    avg_win = winning_trades.mean()
    avg_loss = abs(losing_trades.mean())
    
    if avg_loss == 0:
        return 0.0
        
    # b = odds = avg_win / avg_loss
    b = avg_win / avg_loss
    p = win_rate_val
    q = 1 - p
    
    kelly_pct = (b * p - q) / b
    
    return kelly_pct


def deflated_sharpe_ratio(
    returns: np.ndarray, 
    trials: int, 
    skew: float = 0.0, 
    kurt: float = 3.0
) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR) to account for multiple testing.
    
    DSR adjusts the Sharpe ratio for the number of strategy variations tested,
    reducing false discovery rate in backtesting.
    
    Args:
        returns: Array of returns
        trials: Number of strategy trials/backtests performed
        skew: Skewness of returns (default 0.0)
        kurt: Excess kurtosis of returns (default 3.0 for normal)
        
    Returns:
        Deflated Sharpe Ratio
        
    Reference:
        Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
    """
    from scipy import stats
    
    if len(returns) < 2:
        return 0.0
        
    # Use canonical Sharpe definition (sample std, ddof=1).
    sr = _canonical_period_sharpe(returns)
    
    # Number of observations
    T = len(returns)
    
    # Expected max Sharpe under null hypothesis
    e_max_sr = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/trials) + \
               np.euler_gamma * stats.norm.ppf(1 - 1/(trials * np.e))
    
    # Variance of Sharpe ratio estimate
    var_sr = (1 + 0.5 * sr**2 - skew * sr + (kurt - 3) * sr**2 / 4) / T
    
    # Deflated Sharpe Ratio
    dsr = (sr - e_max_sr) / np.sqrt(var_sr)
    
    return dsr


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sr: float,
    skew: float = 0.0,
    kurt: float = 3.0
) -> float:
    """
    Calculate Probabilistic Sharpe Ratio (PSR).
    
    PSR gives the probability that the true Sharpe ratio exceeds a benchmark,
    accounting for estimation uncertainty and higher moments.
    
    Args:
        returns: Array of returns
        benchmark_sr: Benchmark Sharpe ratio to compare against
        skew: Skewness of returns (default 0.0) 
        kurt: Excess kurtosis of returns (default 3.0 for normal)
        
    Returns:
        Probability that true SR > benchmark_sr (0 to 1)
        
    Reference:
        Bailey & López de Prado (2012) "The Sharpe Ratio Efficient Frontier"
    """
    from scipy import stats
    
    if len(returns) < 2:
        return 0.0
        
    # Use canonical Sharpe definition (sample std, ddof=1).
    sr = _canonical_period_sharpe(returns)
    
    # Number of observations
    T = len(returns)
    
    # Standard error of Sharpe ratio
    se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt - 3) * sr**2 / 4) / T)
    
    # Z-score
    z_score = (sr - benchmark_sr) / se_sr
    
    # Probability that true SR > benchmark
    psr = stats.norm.cdf(z_score)
    
    return psr
