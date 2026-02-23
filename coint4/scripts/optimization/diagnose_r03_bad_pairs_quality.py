#!/usr/bin/env python3
"""Compute pair-quality metrics for r03 top-1 in key windows (for r04 threshold design).

This script recomputes the same quality metrics used by the selection filter:
  - correlation
  - beta
  - mean crossings
  - half-life (days)
  - cointegration p-value (fast_coint)
  - Hurst exponent
  - optional KPSS p-value

Outputs:
  artifacts/wfa/aggregate/20260222_tailguard_r03/r03_quality_metrics_key_windows.csv
"""

from __future__ import annotations

import argparse
import glob
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.stattools import kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings

# coint2 imports (run with PYTHONPATH=src)
from coint2.core.math_utils import calculate_half_life, count_mean_crossings
from coint2.core.fast_coint import fast_coint
from coint2.analysis.pair_filter import calculate_hurst_exponent


_STEP_LINE_RE = re.compile(
    r"WF-шаг\s+(?P<step>\d+)/(?P<steps>\d+):\s+training\s+"
    r"(?P<tr_start>\d{4}-\d{2}-\d{2})-(?P<tr_end>\d{4}-\d{2}-\d{2}),\s+testing\s+"
    r"(?P<te_start>\d{4}-\d{2}-\d{2})-(?P<te_end>\d{4}-\d{2}-\d{2})"
)


def _to_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _split_pair(pair: str) -> tuple[str, str]:
    if "-" in pair:
        a, b = pair.split("-", 1)
        return a.strip(), b.strip()
    if "/" in pair:
        a, b = pair.split("/", 1)
        return a.strip(), b.strip()
    raise ValueError(f"unexpected pair format: {pair!r}")


@dataclass(frozen=True)
class PeriodDates:
    period: str
    training_start: str
    training_end: str
    testing_start: str
    testing_end: str


def _load_period_dates(run_log_path: Path) -> dict[str, PeriodDates]:
    mapping: dict[str, PeriodDates] = {}
    for line in run_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _STEP_LINE_RE.search(line)
        if not m:
            continue
        tr_start = m.group("tr_start")
        tr_end = m.group("tr_end")
        te_start = m.group("te_start")
        te_end = m.group("te_end")
        period = f"{tr_start[5:].replace('-', '/')}-{te_end[5:].replace('-', '/')}"
        mapping.setdefault(
            period,
            PeriodDates(
                period=period,
                training_start=tr_start,
                training_end=tr_end,
                testing_start=te_start,
                testing_end=te_end,
            ),
        )
    return mapping


def _load_close_wide(*, app_root: Path, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    files = glob.glob(str(app_root / "data_downloaded" / "**" / "*.parquet"), recursive=True)
    if not files:
        raise RuntimeError("no parquet files under data_downloaded/")

    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    # Inclusive end_date: add 1 day and filter ts < next_day
    end_excl = end_dt + pd.Timedelta(days=1)

    scan = (
        pl.scan_parquet(files)
        .select(
            [
                pl.col("symbol"),
                pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("ts"),
                pl.col("close").cast(pl.Float64, strict=False).alias("close"),
            ]
        )
        .filter(pl.col("symbol").is_in(symbols))
        .filter((pl.col("ts") >= pl.lit(start_dt)) & (pl.col("ts") < pl.lit(end_excl)))
    )
    df = scan.collect().to_pandas()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="ts", columns="symbol", values="close").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def _compute_pair_metrics(price_df: pd.DataFrame, a: str, b: str, *, with_kpss: bool) -> dict[str, float]:
    pair_data = price_df[[a, b]].dropna()
    if pair_data.empty or len(pair_data) < 960:
        return {"ok": 0.0}
    if pair_data[b].var() == 0:
        return {"ok": 0.0}

    corr = float(pair_data[a].corr(pair_data[b]))
    beta = float(pair_data[a].cov(pair_data[b]) / pair_data[b].var())

    spread = pair_data[a] - beta * pair_data[b]
    mean_crossings = float(count_mean_crossings(spread))

    hl_bars = float(calculate_half_life(spread))
    try:
        bar_minutes = int((pair_data.index[1] - pair_data.index[0]).total_seconds() / 60)
        if bar_minutes <= 0:
            bar_minutes = 15
    except Exception:
        bar_minutes = 15
    hl_days = float(hl_bars * bar_minutes / 1440.0)

    try:
        _score, pvalue, _ = fast_coint(pair_data[a], pair_data[b], trend="n")
        pvalue = float(pvalue) if pvalue is not None and not np.isnan(pvalue) else float("nan")
    except Exception:
        pvalue = float("nan")

    hurst = float(calculate_hurst_exponent(spread))

    out = {
        "ok": 1.0,
        "corr": corr,
        "beta": beta,
        "abs_beta": abs(beta),
        "mean_crossings": mean_crossings,
        "half_life_days": hl_days,
        "coint_pvalue": pvalue,
        "hurst": hurst,
    }

    if with_kpss:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InterpolationWarning)
                out["kpss_pvalue"] = float(kpss(spread, regression="c", nlags="auto")[1])
        except Exception:
            out["kpss_pvalue"] = float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-run-dir", required=True)
    parser.add_argument("--stress-run-dir", required=True)
    parser.add_argument("--out", default="artifacts/wfa/aggregate/20260222_tailguard_r03/r03_quality_metrics_key_windows.csv")
    parser.add_argument("--worst-periods", type=int, default=5)
    parser.add_argument("--top-bad-per-period", type=int, default=20)
    parser.add_argument("--with-kpss", action="store_true")
    args = parser.parse_args()

    app_root = Path(__file__).resolve().parents[2]
    holdout_run = (Path(args.holdout_run_dir) if Path(args.holdout_run_dir).is_absolute() else (app_root / args.holdout_run_dir)).resolve()
    stress_run = (Path(args.stress_run_dir) if Path(args.stress_run_dir).is_absolute() else (app_root / args.stress_run_dir)).resolve()
    out_path = (Path(args.out) if Path(args.out).is_absolute() else (app_root / args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    holdout_ts = pd.read_csv(holdout_run / "trade_statistics.csv")
    stress_ts = pd.read_csv(stress_run / "trade_statistics.csv")
    merged = holdout_ts.merge(stress_ts, on=["pair", "period"], how="outer", suffixes=("_h", "_s"))
    merged["robust_pnl"] = merged[["total_pnl_h", "total_pnl_s"]].min(axis=1, skipna=True)
    merged["robust_trades"] = merged[["trade_count_h", "trade_count_s"]].min(axis=1, skipna=True)

    period_totals = merged.groupby("period", as_index=False)["robust_pnl"].sum().sort_values("robust_pnl", ascending=True)
    worst_periods = list(period_totals.head(int(args.worst_periods))["period"].astype(str))

    period_dates = _load_period_dates(holdout_run / "run.log")

    # Analyze worst periods and the worst pairs inside them.
    focus_pairs: set[tuple[str, str]] = set()
    for period in worst_periods:
        sub = merged[merged["period"] == period].sort_values("robust_pnl", ascending=True).head(int(args.top_bad_per_period))
        for p in sub["pair"].astype(str).tolist():
            a, b = _split_pair(p)
            focus_pairs.add((period, p))

    # Also include all target pairs (in any period) for easier debugging.
    for p in [
        "CELOUSDT-ENJUSDT",
        "AVAUSDT-FITFIUSDT",
        "CHZUSDC-JSTUSDT",
        "ACSUSDT-HOOKUSDT",
        "CHZUSDC-FILUSDT",
        "BICOUSDT-KASTAUSDT",
        "1INCHUSDT-ACSUSDT",
        "DMAILUSDT-FIDAUSDT",
    ]:
        for period in merged[merged["pair"] == p]["period"].astype(str).unique().tolist():
            focus_pairs.add((period, p))

    rows = []
    # Cache per-period price data.
    price_cache: dict[str, pd.DataFrame] = {}

    for period, pair in sorted(focus_pairs):
        dates = period_dates.get(period)
        if not dates:
            continue
        a, b = _split_pair(pair)
        if period not in price_cache:
            # Load only symbols needed for this period's focus pairs.
            symbols = sorted(
                {
                    s
                    for p2 in [p for pr, p in focus_pairs if pr == period]
                    for s in _split_pair(p2)
                }
            )
            price_cache[period] = _load_close_wide(
                app_root=app_root,
                symbols=symbols,
                start_date=dates.training_start,
                end_date=dates.training_end,
            )
        price_df = price_cache[period]
        metrics = _compute_pair_metrics(price_df, a, b, with_kpss=bool(args.with_kpss)) if not price_df.empty else {"ok": 0.0}

        # Attach robust pnl for this (pair, period).
        sub = merged[(merged["pair"] == pair) & (merged["period"] == period)]
        robust_pnl = float(sub.iloc[0]["robust_pnl"]) if len(sub) else float("nan")
        robust_trades = float(sub.iloc[0]["robust_trades"]) if len(sub) else float("nan")

        rows.append(
            {
                "period": period,
                "training_start": dates.training_start,
                "training_end": dates.training_end,
                "testing_start": dates.testing_start,
                "testing_end": dates.testing_end,
                "pair": pair,
                "robust_pnl": robust_pnl,
                "robust_trades": robust_trades,
                **metrics,
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

