"""Canonical Sharpe utilities.

This module centralizes Sharpe computations so that WFA, rollups, and audit
tools use exactly the same definition.

Canonical definition (risk-free assumed 0):
  Sharpe = sqrt(periods_per_year) * mean(returns) / std(returns)

Important details:
  - std is the *sample* standard deviation (ddof=1), matching
    pandas.Series.std() default and the WFA pipeline.
  - For equity_curve.csv we infer bar frequency via the median timestamp delta.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable, Optional


def annualized_sharpe_ratio(
    returns: Iterable[float],
    periods_per_year: float,
    *,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio from a stream of returns.

    - Uses sample variance (ddof=1).
    - Excess returns are computed as (return - risk_free_rate), where
      risk_free_rate is expressed in *per-period* units.
    - Skips non-finite values.
    - Returns 0.0 when insufficient data or zero stdev.
    """
    try:
        risk_free_rate_value = float(risk_free_rate)
    except (TypeError, ValueError):
        risk_free_rate_value = 0.0
    if not math.isfinite(risk_free_rate_value):
        risk_free_rate_value = 0.0

    count = 0
    mean = 0.0
    m2 = 0.0

    for value in returns:
        try:
            ret = float(value) - risk_free_rate_value
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ret):
            continue

        count += 1
        diff = ret - mean
        mean += diff / count
        m2 += diff * (ret - mean)

    if count < 2:
        return 0.0

    variance = m2 / (count - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0.0:
        return 0.0

    return math.sqrt(periods_per_year) * mean / std


def annualized_sharpe_ratio_from_equity(
    equities: Iterable[float],
    periods_per_year: float,
    *,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio from an equity series (pct_change returns)."""
    prev_equity: Optional[float] = None
    returns: list[float] = []
    for value in equities:
        try:
            equity = float(value)
        except (TypeError, ValueError):
            continue
        if prev_equity is not None and prev_equity != 0:
            returns.append((equity - prev_equity) / prev_equity)
        prev_equity = equity

    return annualized_sharpe_ratio(
        returns, periods_per_year, risk_free_rate=risk_free_rate
    )


@dataclass(frozen=True)
class EquitySharpeStats:
    n_returns: int
    mean_return: float
    std_return: float
    period_seconds_median: float
    periods_per_year_full: float
    sharpe_full: float
    sharpe_daily_only: float


def compute_equity_sharpe_from_equity_curve_csv(
    equity_curve_path: Path,
    *,
    days_per_year: float = 365.0,
) -> Optional[EquitySharpeStats]:
    """Compute Sharpe ratio from an equity_curve.csv (pandas Series .to_csv format).

    The calculation matches the rollup index builder:
      - returns: (equity_t - equity_{t-1}) / equity_{t-1}
      - mean/std computed with Welford (sample variance, ddof=1)
      - period inferred by median timestamp delta (seconds)
      - periods_per_year = days_per_year * (86400 / period_seconds)
      - sharpe = sqrt(periods_per_year) * mean / std

    Returns None when the file is missing/unparseable/too short (<2 returns).
    """
    if not equity_curve_path.exists():
        return None

    deltas: list[float] = []
    count = 0
    mean = 0.0
    m2 = 0.0

    prev_ts: Optional[datetime] = None
    prev_equity: Optional[float] = None

    with equity_curve_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                ts = datetime.fromisoformat(row[0].strip())
                equity = float(row[1])
            except (ValueError, TypeError):
                continue

            if prev_ts is not None and prev_equity is not None and prev_equity != 0:
                ret = (equity - prev_equity) / prev_equity
                count += 1

                delta_sec = (ts - prev_ts).total_seconds()
                if delta_sec > 0:
                    deltas.append(delta_sec)

                diff = ret - mean
                mean += diff / count
                m2 += diff * (ret - mean)

            prev_ts = ts
            prev_equity = equity

    if count < 2:
        return None

    variance = m2 / (count - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0.0:
        return EquitySharpeStats(
            n_returns=count,
            mean_return=mean,
            std_return=0.0,
            period_seconds_median=float("nan"),
            periods_per_year_full=float("nan"),
            sharpe_full=0.0,
            sharpe_daily_only=0.0,
        )

    if not deltas:
        return None

    period_seconds = float(median(deltas))
    if period_seconds <= 0:
        return None

    periods_per_year = days_per_year * (86400.0 / period_seconds)
    sharpe_full = math.sqrt(periods_per_year) * mean / std
    sharpe_daily_only = math.sqrt(days_per_year) * mean / std

    return EquitySharpeStats(
        n_returns=count,
        mean_return=mean,
        std_return=std,
        period_seconds_median=period_seconds,
        periods_per_year_full=periods_per_year,
        sharpe_full=sharpe_full,
        sharpe_daily_only=sharpe_daily_only,
    )


def compute_sharpe_ratio_abs_from_equity_curve_csv(
    equity_curve_path: Path,
    *,
    days_per_year: float = 365.0,
) -> Optional[float]:
    stats = compute_equity_sharpe_from_equity_curve_csv(
        equity_curve_path, days_per_year=days_per_year
    )
    if stats is None:
        return None
    return stats.sharpe_full
