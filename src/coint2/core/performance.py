"""Performance metrics for trading strategies."""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


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
    return drawdown.min()

def sharpe_ratio(pnl: pd.Series, annualizing_factor: int, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio of a PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Profit and loss series.
    risk_free_rate : float, optional
        Daily risk free rate, by default ``0.0``.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    excess_returns = pnl - risk_free_rate
    if excess_returns.std(ddof=0) == 0:
        return np.nan
    daily_sharpe = excess_returns.mean() / excess_returns.std(ddof=0)
    return daily_sharpe * np.sqrt(annualizing_factor)


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
    return drawdown.min()