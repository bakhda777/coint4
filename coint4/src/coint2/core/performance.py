"""Performance metrics for trading strategies."""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


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
    excess_returns = returns - risk_free_rate
    if excess_returns.std(ddof=0) == 0:
        return float('nan')
    daily_sharpe = excess_returns.mean() / excess_returns.std(ddof=0)
    return daily_sharpe * np.sqrt(annualizing_factor)

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

    if returns.std() == 0:
        return 0.0

    # Assume risk free rate is zero
    return np.sqrt(annualizing_factor) * returns.mean() / returns.std()


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
        
    # Calculate standard Sharpe ratio
    sr = np.mean(returns) / (np.std(returns) + 1e-10)
    
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
        
    # Calculate observed Sharpe ratio
    sr = np.mean(returns) / (np.std(returns) + 1e-10)
    
    # Number of observations
    T = len(returns)
    
    # Standard error of Sharpe ratio
    se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt - 3) * sr**2 / 4) / T)
    
    # Z-score
    z_score = (sr - benchmark_sr) / se_sr
    
    # Probability that true SR > benchmark
    psr = stats.norm.cdf(z_score)
    
    return psr
