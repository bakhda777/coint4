#!/usr/bin/env python3
"""
Audit Sharpe ratio consistency across WFA run artifacts.

Goal:
  - Recompute sharpe_ratio_abs from each run's equity_curve.csv using the same
    formula and annualization approach used by the rollup index builder
    (coint4/src/coint2/ops/run_index.py:_compute_sharpe_from_equity_curve).
  - Compare the recomputed Sharpe to the value stored in strategy_metrics.csv.

Why equity_curve (not daily_pnl):
  - sharpe_ratio_abs in strategy_metrics.csv is computed on equity returns
    (bar-level pct_change of equity). daily_pnl.csv is a different series
    (daily PnL in USD) and will generally not match sharpe_ratio_abs.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "coint4"


@dataclass(frozen=True)
class EquitySharpeStats:
    n_returns: int
    mean_return: float
    std_return: float
    period_seconds_median: float
    periods_per_year_full: float
    sharpe_full: float
    sharpe_daily_only: float


@dataclass(frozen=True)
class DailyPnlSharpeStats:
    cols: list[str]
    rows_read: int
    initial_equity: Optional[float]
    sharpe: Optional[float]


@dataclass(frozen=True)
class RunAuditRow:
    run_dir: str
    metrics_path: str
    equity_curve_path: str
    daily_pnl_path: str
    daily_pnl_cols: str
    daily_pnl_rows_read: Optional[int]
    daily_pnl_initial_equity: Optional[float]
    stored_sharpe_ratio_abs: Optional[float]
    computed_sharpe_full: Optional[float]
    computed_sharpe_daily_only: Optional[float]
    computed_sharpe_from_daily_pnl: Optional[float]
    abs_diff_full: Optional[float]
    abs_diff_daily_only: Optional[float]
    abs_diff_daily_pnl: Optional[float]
    period_seconds_median: Optional[float]
    n_returns: Optional[int]
    likely_underannualized: Optional[bool]


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_first_metrics_row(metrics_path: Path) -> dict[str, object]:
    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return row
    return {}


def _compute_equity_sharpe_from_csv(
    equity_path: Path,
    *,
    days_per_year: float = 365.0,
) -> Optional[EquitySharpeStats]:
    """
    Compute Sharpe from equity_curve.csv exactly like run_index.py does:
      - returns: (equity_t - equity_{t-1}) / equity_{t-1}
      - mean/std computed with Welford (sample variance, ddof=1)
      - period inferred by median timestamp delta (seconds)
      - periods_per_year = days_per_year * (86400 / period_seconds)
      - sharpe = sqrt(periods_per_year) * mean / std

    Also returns "daily-only" Sharpe (sqrt(days_per_year) * mean / std) which is
    useful to detect legacy under-annualization (missing bar-frequency scaling).
    """
    if not equity_path.exists():
        return None

    deltas: list[float] = []
    count = 0
    mean = 0.0
    m2 = 0.0

    prev_ts: Optional[datetime] = None
    prev_equity: Optional[float] = None

    with equity_path.open(newline="") as handle:
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
        # Match performance.sharpe_ratio behavior for zero stdev.
        # (run_index returns 0.0 in this case)
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


def _read_first_equity_value(equity_path: Path) -> Optional[float]:
    if not equity_path.exists():
        return None
    with equity_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                return float(row[1])
            except (ValueError, TypeError):
                continue
    return None


def _compute_sharpe_from_daily_pnl_csv(
    daily_pnl_path: Path,
    *,
    initial_equity: Optional[float],
    days_per_year: float = 365.0,
) -> DailyPnlSharpeStats:
    """
    Compute Sharpe from daily_pnl.csv by reconstructing daily equity and taking
    pct returns (sample variance, ddof=1), annualized by sqrt(days_per_year).

    This is NOT the same metric as sharpe_ratio_abs (which is bar-level equity
    returns). It's included as a diagnostic to show why daily_pnl is not a
    drop-in replacement for sharpe_ratio_abs.
    """
    if not daily_pnl_path.exists():
        return DailyPnlSharpeStats(
            cols=[],
            rows_read=0,
            initial_equity=initial_equity,
            sharpe=None,
        )

    count = 0
    mean = 0.0
    m2 = 0.0
    rows_read = 0
    cols: list[str] = []

    with daily_pnl_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)  # header
        if header:
            cols = [(c.strip() or "<index>") for c in header]
        for row in reader:
            if len(row) < 2:
                continue
            try:
                pnl = float(row[1])
            except (ValueError, TypeError):
                continue
            rows_read += 1

            if initial_equity is None:
                continue

            if count == 0:
                equity = initial_equity
                prev_equity = equity

            if prev_equity == 0:
                return DailyPnlSharpeStats(
                    cols=cols,
                    rows_read=rows_read,
                    initial_equity=initial_equity,
                    sharpe=None,
                )

            equity += pnl
            ret = (equity - prev_equity) / prev_equity
            prev_equity = equity

            count += 1
            diff = ret - mean
            mean += diff / count
            m2 += diff * (ret - mean)

    if initial_equity is None:
        return DailyPnlSharpeStats(
            cols=cols,
            rows_read=rows_read,
            initial_equity=initial_equity,
            sharpe=None,
        )

    if count < 2:
        return DailyPnlSharpeStats(
            cols=cols,
            rows_read=rows_read,
            initial_equity=initial_equity,
            sharpe=None,
        )
    variance = m2 / (count - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0.0:
        return DailyPnlSharpeStats(
            cols=cols,
            rows_read=rows_read,
            initial_equity=initial_equity,
            sharpe=0.0,
        )
    return DailyPnlSharpeStats(
        cols=cols,
        rows_read=rows_read,
        initial_equity=initial_equity,
        sharpe=(math.sqrt(days_per_year) * mean / std),
    )


def _iter_metrics_paths(glob_pattern: str) -> Iterable[Path]:
    # If the user passes a repo-relative pattern like "coint4/artifacts/...",
    # treat it relative to PROJECT_ROOT. Otherwise, expand as-is.
    pattern_path = Path(glob_pattern)
    if not pattern_path.is_absolute():
        candidates = list(PROJECT_ROOT.glob(glob_pattern))
        if candidates:
            yield from candidates
            return
    # Fallback: treat pattern relative to cwd (or as absolute).
    yield from Path().glob(glob_pattern)


def _audit_one_run(metrics_path: Path) -> RunAuditRow:
    run_dir = metrics_path.parent
    equity_path = run_dir / "equity_curve.csv"
    daily_pnl_path = run_dir / "daily_pnl.csv"

    metrics_row = _read_first_metrics_row(metrics_path)
    stored = _to_float(metrics_row.get("sharpe_ratio_abs"))

    computed_stats = _compute_equity_sharpe_from_csv(equity_path)
    computed_full = computed_stats.sharpe_full if computed_stats is not None else None
    computed_daily = computed_stats.sharpe_daily_only if computed_stats is not None else None

    daily_pnl_cols = ""
    daily_pnl_rows_read: Optional[int] = None
    daily_pnl_initial_equity: Optional[float] = None
    computed_from_daily_pnl = None
    if daily_pnl_path.exists():
        initial_equity = _read_first_equity_value(equity_path)
        daily_stats = _compute_sharpe_from_daily_pnl_csv(
            daily_pnl_path,
            initial_equity=initial_equity,
        )
        daily_pnl_cols = "|".join(daily_stats.cols)
        daily_pnl_rows_read = daily_stats.rows_read
        daily_pnl_initial_equity = daily_stats.initial_equity
        computed_from_daily_pnl = daily_stats.sharpe

    abs_diff_full = abs(stored - computed_full) if stored is not None and computed_full is not None else None
    abs_diff_daily = abs(stored - computed_daily) if stored is not None and computed_daily is not None else None
    abs_diff_daily_pnl = (
        abs(stored - computed_from_daily_pnl)
        if stored is not None and computed_from_daily_pnl is not None
        else None
    )

    likely_underannualized = None
    if abs_diff_full is not None and abs_diff_daily is not None:
        # "Underannualized" pattern: stored is closer to daily-only annualization
        # than to full bar-aware annualization.
        likely_underannualized = abs_diff_daily < abs_diff_full

    return RunAuditRow(
        run_dir=run_dir.resolve().as_posix(),
        metrics_path=metrics_path.resolve().as_posix(),
        equity_curve_path=equity_path.resolve().as_posix(),
        daily_pnl_path=daily_pnl_path.resolve().as_posix(),
        daily_pnl_cols=daily_pnl_cols,
        daily_pnl_rows_read=daily_pnl_rows_read,
        daily_pnl_initial_equity=daily_pnl_initial_equity,
        stored_sharpe_ratio_abs=stored,
        computed_sharpe_full=computed_full,
        computed_sharpe_daily_only=computed_daily,
        computed_sharpe_from_daily_pnl=computed_from_daily_pnl,
        abs_diff_full=abs_diff_full,
        abs_diff_daily_only=abs_diff_daily,
        abs_diff_daily_pnl=abs_diff_daily_pnl,
        period_seconds_median=(
            computed_stats.period_seconds_median if computed_stats is not None else None
        ),
        n_returns=(computed_stats.n_returns if computed_stats is not None else None),
        likely_underannualized=likely_underannualized,
    )


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _write_csv(path: Path, rows: list[RunAuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f.name for f in dataclasses.fields(RunAuditRow)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Sharpe ratio across WFA run artifacts.")
    parser.add_argument(
        "--runs-glob",
        default="coint4/artifacts/wfa/runs/**/strategy_metrics.csv",
        help="Glob for strategy_metrics.csv files (repo-relative recommended).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of runs to audit (0 = no limit).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Show top-N discrepancies by abs diff.",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/sharpe_audit_rows.csv",
        help="Write full audit rows CSV (repo-relative path).",
    )
    args = parser.parse_args()

    metrics_paths = sorted(_iter_metrics_paths(args.runs_glob))
    if args.limit and args.limit > 0:
        metrics_paths = metrics_paths[: args.limit]

    rows: list[RunAuditRow] = []
    for metrics_path in metrics_paths:
        rows.append(_audit_one_run(metrics_path))

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = PROJECT_ROOT / out_csv
    _write_csv(out_csv, rows)

    compared = [
        r
        for r in rows
        if r.stored_sharpe_ratio_abs is not None and r.computed_sharpe_full is not None and r.abs_diff_full is not None
    ]
    diffs = sorted([r.abs_diff_full for r in compared if r.abs_diff_full is not None])
    max_abs_diff = max(diffs) if diffs else float("nan")
    p95_abs_diff = _percentile(diffs, 0.95) if diffs else float("nan")

    missing_equity = sum(1 for r in rows if r.computed_sharpe_full is None)
    missing_stored = sum(1 for r in rows if r.stored_sharpe_ratio_abs is None)
    likely_underannualized = sum(1 for r in compared if r.likely_underannualized)

    print(f"runs_scanned: {len(rows)}")
    print(f"runs_compared (stored+equity): {len(compared)}")
    print(f"missing_equity_curve_or_unparseable: {missing_equity}")
    print(f"missing_stored_sharpe: {missing_stored}")
    print(f"max_abs_diff_full: {max_abs_diff}")
    print(f"p95_abs_diff_full: {p95_abs_diff}")
    print(f"likely_underannualized_count: {likely_underannualized}")
    print(f"rows_csv: {out_csv.resolve().as_posix()}")

    if compared and args.top_n > 0:
        top = sorted(compared, key=lambda r: (r.abs_diff_full or 0.0), reverse=True)[: args.top_n]
        print("")
        print(f"top_{args.top_n}_by_abs_diff_full:")
        for r in top:
            stored = r.stored_sharpe_ratio_abs
            comp = r.computed_sharpe_full
            comp_daily = r.computed_sharpe_daily_only
            ratio = (comp / stored) if (stored not in (None, 0.0) and comp is not None) else None
            print(
                "abs_diff_full={abs_diff:.6g} stored={stored:.6g} computed_full={comp:.6g} "
                "computed_daily_only={comp_daily:.6g} ratio_full/stored={ratio} "
                "daily_pnl_sharpe={daily_sharpe:.6g} abs_diff_daily_pnl={abs_diff_daily_pnl:.6g} "
                "daily_pnl_rows_read={daily_rows} daily_pnl_cols={daily_cols} daily_pnl_initial_equity={daily_eq:.6g} "
                "period_sec={period_sec} n={n} underannualized_like_daily={flag} run_dir={run_dir}".format(
                    abs_diff=(r.abs_diff_full or float("nan")),
                    stored=(stored or float("nan")),
                    comp=(comp or float("nan")),
                    comp_daily=(comp_daily or float("nan")),
                    ratio=(f"{ratio:.6g}" if isinstance(ratio, float) else "NA"),
                    daily_sharpe=(
                        r.computed_sharpe_from_daily_pnl
                        if r.computed_sharpe_from_daily_pnl is not None
                        else float("nan")
                    ),
                    abs_diff_daily_pnl=(
                        r.abs_diff_daily_pnl if r.abs_diff_daily_pnl is not None else float("nan")
                    ),
                    daily_rows=(r.daily_pnl_rows_read if r.daily_pnl_rows_read is not None else "NA"),
                    daily_cols=(r.daily_pnl_cols or "NA"),
                    daily_eq=(
                        r.daily_pnl_initial_equity
                        if r.daily_pnl_initial_equity is not None
                        else float("nan")
                    ),
                    period_sec=(f"{r.period_seconds_median:.6g}" if r.period_seconds_median else "NA"),
                    n=(r.n_returns if r.n_returns is not None else "NA"),
                    flag=(r.likely_underannualized if r.likely_underannualized is not None else "NA"),
                    run_dir=r.run_dir,
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
