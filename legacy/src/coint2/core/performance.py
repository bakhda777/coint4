"""Performance metrics for trading strategies."""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def safe_sharpe(returns: np.ndarray | pd.Series, risk_free: float = 0.0, annualizing_factor: float = 365) -> tuple[float, bool]:
    """
    Safely calculate Sharpe ratio.
    Returns (sharpe_ratio, is_valid).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    
    if r.size < 2:
        return 0.0, False
        
    excess = r - risk_free
    vol = np.std(excess, ddof=1)
    
    if vol == 0 or not np.isfinite(vol):
        return 0.0, False
        
    s = np.mean(excess) / vol * np.sqrt(annualizing_factor)
    
    if not np.isfinite(s):
        return 0.0, False
        
    return float(s), True

def safe_max_dd(equity_curve: np.ndarray | pd.Series) -> tuple[float, bool]:
    """
    Safely calculate Max Drawdown (in %).
    Returns (max_dd_pct, is_valid).
    max_dd_pct is positive (e.g. 25.0 for 25% drawdown).
    """
    eq = np.asarray(equity_curve, dtype=float)
    eq = eq[np.isfinite(eq)]
    
    if eq.size < 1:
        return 0.0, False
        
    running_max = np.maximum.accumulate(eq)
    # Avoid division by zero
    valid_max = np.where(running_max > 0, running_max, 1.0)
    
    # Drawdown is positive (running_max - equity)
    dd = (running_max - eq) / valid_max
    
    if dd.size == 0:
        return 0.0, False
        
    mdd = np.max(dd) * 100.0
    
    if not np.isfinite(mdd):
        return 0.0, False
        
    return float(mdd), True

def sharpe_ratio_on_returns(pnl: pd.Series, capital: float, annualizing_factor: int, risk_free_rate: float = 0.0) -> float:
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


def calculate_metrics(
    pnl_series: pd.Series,
    trades: list = None,
    capital: float = 10000.0,
    annualizing_factor: int = 365
) -> dict:
    """
    Calculate a comprehensive set of performance metrics.
    
    Parameters
    ----------
    pnl_series : pd.Series
        Series of daily or periodic PnL values.
    trades : list, optional
        List of trade objects/dictionaries.
    capital : float
        Initial capital used for percentage calculations.
    annualizing_factor : int
        Factor to annualize Sharpe ratio (e.g., 365 for daily crypto).
        
    Returns
    -------
    dict
        Dictionary containing calculated metrics.
    """
    if pnl_series.empty:
        return {
            'total_pnl': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'trade_count': 0
        }
        
    total_pnl = pnl_series.sum()
    total_return = total_pnl / capital if capital > 0 else 0.0
    
    # Equity curve (starting from capital)
    equity_curve = capital + pnl_series.cumsum()
    
    # Calculate basic metrics using helper functions
    sr = sharpe_ratio_on_returns(pnl_series, capital, annualizing_factor)
    mdd = max_drawdown_on_equity(equity_curve)
    wr = win_rate(pnl_series)
    exp = expectancy(pnl_series)
    kelly = kelly_criterion(pnl_series)
    
    # Calculate Profit Factor
    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    return {
        'total_pnl': float(total_pnl),
        'total_return': float(total_return),
        'sharpe_ratio': float(sr) if not np.isnan(sr) else 0.0,
        'max_drawdown': float(mdd) if not np.isnan(mdd) else 0.0,
        'win_rate': float(wr),
        'profit_factor': float(profit_factor),
        'expectancy': float(exp),
        'kelly_criterion': float(kelly),
        'trade_count': len(trades) if trades is not None else len(pnl_series[pnl_series != 0])
    }
