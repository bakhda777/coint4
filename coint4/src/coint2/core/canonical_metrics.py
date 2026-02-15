"""Canonical metrics computed from equity_curve.csv.

These utilities are intentionally lightweight (no pandas dependency) and are
used by audit/recompute scripts that must not overwrite legacy artifacts like
strategy_metrics.csv.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Optional

from coint2.core.sharpe import annualized_sharpe_ratio_from_equity


@dataclass(frozen=True)
class CanonicalEquityMetrics:
    canonical_sharpe: float
    canonical_pnl_abs: float
    canonical_max_drawdown_abs: float
    equity_first: float
    equity_last: float
    n_points: int
    n_returns: int
    period_seconds_median: Optional[float]


def compute_canonical_metrics_from_equity_curve_csv(
    equity_curve_path: Path,
    *,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
) -> Optional[CanonicalEquityMetrics]:
    """Compute canonical Sharpe/PnL/MaxDD from a typical equity_curve.csv.

    Expected CSV format matches pandas.Series.to_csv output:
      - header: ",Equity"
      - rows: "<iso timestamp>,<equity>"

    Definitions:
      - canonical_pnl_abs = equity_last - equity_first
      - canonical_max_drawdown_abs = min_t(equity_t - peak_equity_to_t) (<= 0)
      - canonical_sharpe is computed via coint2.core.sharpe.annualized_sharpe_ratio_from_equity
        using the provided periods_per_year (fixed annualization).

    Returns None when the file is missing/unparseable (no valid equity points).
    """
    if periods_per_year <= 0 or not math.isfinite(periods_per_year):
        raise ValueError(f"periods_per_year must be positive and finite, got: {periods_per_year!r}")

    if not equity_curve_path.exists():
        return None

    equities: list[float] = []
    deltas_sec: list[float] = []
    prev_ts: Optional[datetime] = None

    equity_first: Optional[float] = None
    equity_last: Optional[float] = None
    peak_equity: Optional[float] = None
    max_drawdown_abs = 0.0

    with equity_curve_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                ts = datetime.fromisoformat(row[0].strip())
                equity = float(row[1])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(equity):
                continue

            equities.append(equity)
            equity_last = equity
            if equity_first is None:
                equity_first = equity

            if peak_equity is None or equity > peak_equity:
                peak_equity = equity
            else:
                drawdown = equity - peak_equity
                if drawdown < max_drawdown_abs:
                    max_drawdown_abs = drawdown

            if prev_ts is not None:
                delta = (ts - prev_ts).total_seconds()
                if delta > 0:
                    deltas_sec.append(delta)
            prev_ts = ts

    if not equities or equity_first is None or equity_last is None:
        return None

    canonical_pnl_abs = equity_last - equity_first

    period_seconds_median = float(median(deltas_sec)) if deltas_sec else None

    canonical_sharpe = annualized_sharpe_ratio_from_equity(
        equities,
        periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    n_points = len(equities)
    n_returns = sum(1 for a, b in zip(equities, equities[1:], strict=False) if a != 0.0)

    return CanonicalEquityMetrics(
        canonical_sharpe=canonical_sharpe,
        canonical_pnl_abs=canonical_pnl_abs,
        canonical_max_drawdown_abs=max_drawdown_abs,
        equity_first=equity_first,
        equity_last=equity_last,
        n_points=n_points,
        n_returns=n_returns,
        period_seconds_median=period_seconds_median,
    )

