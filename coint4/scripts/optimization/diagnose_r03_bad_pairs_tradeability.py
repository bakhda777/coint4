#!/usr/bin/env python3
"""Diagnose why "bad" pairs passed selection: tradeability metrics vs losses (r03 top-1).

Inputs:
  - holdout/stress top-1 run dirs (trade_statistics.csv)
  - Bybit metrics snapshot CSV (turnover24h/bid-ask/funding/tick/listing age)

Outputs (small, tracked):
  - CSV with per-pair metrics for worst windows + flags (bad/rest)
  - markdown summary with percentile separation + candidate thresholds
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd


TARGET_PAIRS = [
    "CELOUSDT-ENJUSDT",
    "AVAUSDT-FITFIUSDT",
    "CHZUSDC-JSTUSDT",
    "ACSUSDT-HOOKUSDT",
    "CHZUSDC-FILUSDT",
    "BICOUSDT-KASTAUSDT",
    "1INCHUSDT-ACSUSDT",
    "DMAILUSDT-FIDAUSDT",
]

_STEP_LINE_RE = (
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


def _safe_min(a: Optional[float], b: Optional[float], *, missing: float) -> float:
    if a is None or b is None:
        return float(missing)
    return float(min(a, b))


def _safe_max(a: Optional[float], b: Optional[float], *, missing: float) -> float:
    if a is None or b is None:
        return float(missing)
    return float(max(a, b))


@dataclass(frozen=True)
class _Sym:
    symbol: str
    quote: str
    turnover24h_usd: Optional[float]
    bid_ask_pct: Optional[float]
    funding_rate_abs: Optional[float]
    tick_size_pct: Optional[float]
    days_live: Optional[float]


def _load_bybit_metrics(path: Path) -> dict[str, _Sym]:
    out: dict[str, _Sym] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                continue
            out[symbol] = _Sym(
                symbol=symbol,
                quote=(row.get("quote") or "").strip(),
                turnover24h_usd=_to_float(row.get("turnover24h_usd")),
                bid_ask_pct=_to_float(row.get("bid_ask_pct")),
                funding_rate_abs=_to_float(row.get("funding_rate_abs")),
                tick_size_pct=_to_float(row.get("tick_size_pct")),
                days_live=_to_float(row.get("days_live")),
            )
    return out


def _split_pair(pair: str) -> tuple[str, str]:
    if "-" in pair:
        a, b = pair.split("-", 1)
        return a.strip(), b.strip()
    if "/" in pair:
        a, b = pair.split("/", 1)
        return a.strip(), b.strip()
    raise ValueError(f"unexpected pair format: {pair!r}")


def _load_period_dates(run_log_path: Path) -> dict[str, dict[str, str]]:
    import re

    mapping: dict[str, dict[str, str]] = {}
    pattern = re.compile(_STEP_LINE_RE)
    for line in run_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        tr_start = m.group("tr_start")
        tr_end = m.group("tr_end")
        te_start = m.group("te_start")
        te_end = m.group("te_end")
        period = f"{tr_start[5:].replace('-', '/')}-{te_end[5:].replace('-', '/')}"
        # Keep first occurrence; duplicates exist in logs.
        mapping.setdefault(
            period,
            {
                "training_start": tr_start,
                "training_end": tr_end,
                "testing_start": te_start,
                "testing_end": te_end,
            },
        )
    return mapping


def _compute_hist_liquidity(
    *,
    app_root: Path,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily_turnover_df, first_seen_df) for symbols.

    daily_turnover_df columns: symbol, date, turnover_usd
    first_seen_df columns: symbol, first_date
    """
    import glob
    from datetime import date
    import polars as pl

    files = glob.glob(str(app_root / "data_downloaded" / "**" / "*.parquet"), recursive=True)
    if not files:
        raise RuntimeError("no parquet files under data_downloaded/")

    start_d = date.fromisoformat(start_date)
    end_d = date.fromisoformat(end_date)

    # Polars: parse ms epoch -> datetime.
    scan = pl.scan_parquet(files).select(
        [
            pl.col("symbol"),
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("ts"),
            pl.col("turnover").cast(pl.Float64, strict=False).alias("turnover_usd_bar"),
        ]
    )
    scan = scan.filter(pl.col("symbol").is_in(symbols))

    first_seen = (
        scan.select([pl.col("symbol"), pl.col("ts").dt.date().alias("date")])
        .group_by("symbol")
        .agg(pl.col("date").min().alias("first_date"))
        .collect()
    )

    # Daily turnover in [start_date, end_date] (inclusive).
    # Use date filter after deriving date to allow pruning.
    daily = (
        scan.with_columns(pl.col("ts").dt.date().alias("date"))
        .filter((pl.col("date") >= pl.lit(start_d)) & (pl.col("date") <= pl.lit(end_d)))
        .group_by(["symbol", "date"])
        .agg(pl.col("turnover_usd_bar").sum().alias("turnover_usd"))
        .collect()
    )

    daily_df = daily.to_pandas()
    first_df = first_seen.to_pandas()
    return daily_df, first_df


def _robust_merge(
    holdout: pd.DataFrame, stress: pd.DataFrame, *, keys: Iterable[str]
) -> pd.DataFrame:
    keys = list(keys)
    h = holdout.rename(columns={"total_pnl": "holdout_pnl", "trade_count": "holdout_trades"})
    s = stress.rename(columns={"total_pnl": "stress_pnl", "trade_count": "stress_trades"})
    merged = h.merge(s, on=keys, how="outer")
    # Robust = pessimistic (min across holdout/stress when both present).
    merged["robust_pnl"] = merged[["holdout_pnl", "stress_pnl"]].min(axis=1, skipna=True)
    merged["robust_trades"] = merged[["holdout_trades", "stress_trades"]].min(axis=1, skipna=True)
    return merged


def _describe_thresholds(df: pd.DataFrame, bad_mask: pd.Series) -> pd.DataFrame:
    metrics = [
        "min_turnover24h_usd",
        "hist_min_turnover24h_usd",
        "min_days_live",
        "hist_min_days_live",
        "max_bid_ask_pct",
        "max_funding_rate_abs",
        "max_tick_size_pct",
        "quote_mismatch",
        "pair_has_missing",
    ]
    rows = []
    bad = df[bad_mask]
    rest = df[~bad_mask]
    for m in metrics:
        if m not in df.columns:
            continue
        if m == "quote_mismatch":
            rows.append(
                {
                    "metric": m,
                    "bad_rate": float(bad[m].mean()) if len(bad) else 0.0,
                    "rest_rate": float(rest[m].mean()) if len(rest) else 0.0,
                    "bad_median": None,
                    "rest_median": None,
                }
            )
            continue
        if m == "pair_has_missing":
            rows.append(
                {
                    "metric": m,
                    "bad_rate": float(bad[m].mean()) if len(bad) else 0.0,
                    "rest_rate": float(rest[m].mean()) if len(rest) else 0.0,
                    "bad_median": None,
                    "rest_median": None,
                }
            )
            continue
        rows.append(
            {
                "metric": m,
                "bad_rate": None,
                "rest_rate": None,
                "bad_median": float(bad[m].median()) if len(bad) else None,
                "rest_median": float(rest[m].median()) if len(rest) else None,
                "bad_p25": float(bad[m].quantile(0.25)) if len(bad) else None,
                "bad_p75": float(bad[m].quantile(0.75)) if len(bad) else None,
                "rest_p25": float(rest[m].quantile(0.25)) if len(rest) else None,
                "rest_p75": float(rest[m].quantile(0.75)) if len(rest) else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-run-dir", required=True, help="Holdout top-1 run dir")
    parser.add_argument("--stress-run-dir", required=True, help="Stress top-1 run dir")
    parser.add_argument(
        "--bybit-metrics",
        default="configs/market/bybit_linear_metrics_latest.csv",
        help="Bybit metrics snapshot CSV (relative to app root).",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/wfa/aggregate/20260222_tailguard_r03",
        help="Output directory for diagnostics (relative to app root).",
    )
    parser.add_argument("--worst-periods", type=int, default=5, help="How many worst periods to analyze.")
    parser.add_argument("--top-bad-per-period", type=int, default=15, help="How many worst pairs per period.")
    parser.add_argument("--top-bad-total", type=int, default=30, help="How many worst pairs overall.")
    args = parser.parse_args()

    app_root = Path(__file__).resolve().parents[2]
    holdout_run = (Path(args.holdout_run_dir) if Path(args.holdout_run_dir).is_absolute() else (app_root / args.holdout_run_dir)).resolve()
    stress_run = (Path(args.stress_run_dir) if Path(args.stress_run_dir).is_absolute() else (app_root / args.stress_run_dir)).resolve()
    bybit_path = (app_root / args.bybit_metrics).resolve()
    out_dir = (app_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_ts = pd.read_csv(holdout_run / "trade_statistics.csv")
    stress_ts = pd.read_csv(stress_run / "trade_statistics.csv")
    robust = _robust_merge(holdout_ts, stress_ts, keys=["pair", "period"])
    period_dates = _load_period_dates(holdout_run / "run.log")

    # Worst periods by robust total PnL.
    period_totals = (
        robust.groupby("period", as_index=False)["robust_pnl"].sum().sort_values("robust_pnl", ascending=True)
    )
    worst_periods = list(period_totals.head(int(args.worst_periods))["period"])

    # Target pairs: worst period per pair (across all periods).
    target_worst_periods: dict[str, str] = {}
    for pair in TARGET_PAIRS:
        sub = robust[robust["pair"] == pair].sort_values("robust_pnl", ascending=True)
        if len(sub):
            target_worst_periods[pair] = str(sub.iloc[0]["period"])

    # Build "bad pairs" set: (a) user targets, (b) worst by period, (c) worst overall.
    bad_pairs: set[str] = set(TARGET_PAIRS)
    for period in worst_periods:
        sub = robust[robust["period"] == period].sort_values("robust_pnl", ascending=True)
        bad_pairs.update(sub.head(int(args.top_bad_per_period))["pair"].astype(str).tolist())
    total_by_pair = robust.groupby("pair", as_index=False)["robust_pnl"].sum().sort_values("robust_pnl", ascending=True)
    bad_pairs.update(total_by_pair.head(int(args.top_bad_total))["pair"].astype(str).tolist())

    # Active pairs in analysis periods: worst total windows + worst window per target pair.
    analysis_periods = sorted(set(worst_periods).union(set(target_worst_periods.values())))
    active = robust[robust["period"].isin(analysis_periods)].copy()

    bybit = _load_bybit_metrics(bybit_path)

    def sym(symbol: str) -> _Sym:
        # Missing symbols are treated as "untradeable": low volume, huge frictions, zero listing age.
        return bybit.get(
            symbol,
            _Sym(
                symbol=symbol,
                quote="MISSING",
                turnover24h_usd=None,
                bid_ask_pct=None,
                funding_rate_abs=None,
                tick_size_pct=None,
                days_live=None,
            ),
        )

    rows = []
    for _, row in active.iterrows():
        pair = str(row.get("pair") or "").strip()
        if not pair:
            continue
        a, b = _split_pair(pair)
        sa = sym(a)
        sb = sym(b)
        period = str(row.get("period") or "").strip()
        dates = period_dates.get(period) or {}
        testing_start = str(dates.get("testing_start") or "")
        quote_mismatch = int(bool(sa.quote and sb.quote and sa.quote != sb.quote))
        rows.append(
            {
                "pair": pair,
                "period": period,
                "testing_start": testing_start,
                "robust_pnl": float(row.get("robust_pnl") or 0.0),
                "robust_trades": float(row.get("robust_trades") or 0.0),
                "is_bad": int(pair in bad_pairs),
                "a": a,
                "b": b,
                "a_quote": sa.quote,
                "b_quote": sb.quote,
                "quote_mismatch": quote_mismatch,
                "a_turnover24h_usd": sa.turnover24h_usd,
                "b_turnover24h_usd": sb.turnover24h_usd,
                "min_turnover24h_usd": _safe_min(sa.turnover24h_usd, sb.turnover24h_usd, missing=0.0),
                "a_days_live": sa.days_live,
                "b_days_live": sb.days_live,
                "min_days_live": _safe_min(sa.days_live, sb.days_live, missing=0.0),
                "a_bid_ask_pct": sa.bid_ask_pct,
                "b_bid_ask_pct": sb.bid_ask_pct,
                "max_bid_ask_pct": _safe_max(sa.bid_ask_pct, sb.bid_ask_pct, missing=1.0),
                "a_funding_rate_abs": sa.funding_rate_abs,
                "b_funding_rate_abs": sb.funding_rate_abs,
                "max_funding_rate_abs": _safe_max(sa.funding_rate_abs, sb.funding_rate_abs, missing=1.0),
                "a_tick_size_pct": sa.tick_size_pct,
                "b_tick_size_pct": sb.tick_size_pct,
                "max_tick_size_pct": _safe_max(sa.tick_size_pct, sb.tick_size_pct, missing=1.0),
                "a_missing": int(sa.quote == "MISSING" or sa.turnover24h_usd is None),
                "b_missing": int(sb.quote == "MISSING" or sb.turnover24h_usd is None),
                "pair_has_missing": int((sa.turnover24h_usd is None) or (sb.turnover24h_usd is None)),
            }
        )
    metrics_df = pd.DataFrame(rows)

    # Historical liquidity proxies from the downloaded dataset: daily turnover at testing_start + history length.
    hist_periods = sorted({p for p in metrics_df["period"].unique().tolist() if p})
    hist_dates = []
    for p in hist_periods:
        d = period_dates.get(p)
        if not d:
            continue
        if d.get("testing_start"):
            hist_dates.append(str(d["testing_start"]))
    if hist_dates:
        hist_start = min(hist_dates)
        hist_end = max(hist_dates)
        # Need a lookback buffer for median windows.
        from datetime import datetime, timedelta

        start_dt = datetime.fromisoformat(hist_start)
        start_dt = start_dt - timedelta(days=35)
        hist_scan_start = start_dt.date().isoformat()
        hist_scan_end = datetime.fromisoformat(hist_end).date().isoformat()

        symbols_needed = sorted(set(metrics_df["a"].tolist() + metrics_df["b"].tolist()))
        daily_turnover, first_seen = _compute_hist_liquidity(
            app_root=app_root,
            symbols=symbols_needed,
            start_date=hist_scan_start,
            end_date=hist_scan_end,
        )
        daily_turnover["date"] = daily_turnover["date"].astype(str)
        first_seen["first_date"] = first_seen["first_date"].astype(str)
        first_seen_map = dict(zip(first_seen["symbol"], first_seen["first_date"]))

        # Build per-symbol lookup for turnover by date for fast joins.
        daily_map: dict[tuple[str, str], float] = {}
        for sym, date, val in daily_turnover[["symbol", "date", "turnover_usd"]].itertuples(index=False):
            daily_map[(str(sym), str(date))] = float(val) if val == val else 0.0

        def hist_turnover(symbol: str, ref_date: str) -> float:
            return float(daily_map.get((symbol, ref_date), 0.0))

        def hist_days_live(symbol: str, ref_date: str) -> float:
            first = first_seen_map.get(symbol)
            if not first:
                return 0.0
            try:
                d0 = datetime.fromisoformat(first).date()
                d1 = datetime.fromisoformat(ref_date).date()
                return float((d1 - d0).days + 1)
            except Exception:
                return 0.0

        metrics_df["hist_a_turnover24h_usd"] = [
            hist_turnover(a, d) if d else 0.0 for a, d in zip(metrics_df["a"], metrics_df["testing_start"])
        ]
        metrics_df["hist_b_turnover24h_usd"] = [
            hist_turnover(b, d) if d else 0.0 for b, d in zip(metrics_df["b"], metrics_df["testing_start"])
        ]
        metrics_df["hist_min_turnover24h_usd"] = metrics_df[["hist_a_turnover24h_usd", "hist_b_turnover24h_usd"]].min(
            axis=1
        )
        metrics_df["hist_a_days_live"] = [
            hist_days_live(a, d) if d else 0.0 for a, d in zip(metrics_df["a"], metrics_df["testing_start"])
        ]
        metrics_df["hist_b_days_live"] = [
            hist_days_live(b, d) if d else 0.0 for b, d in zip(metrics_df["b"], metrics_df["testing_start"])
        ]
        metrics_df["hist_min_days_live"] = metrics_df[["hist_a_days_live", "hist_b_days_live"]].min(axis=1)
    else:
        # Keep columns present for downstream rendering.
        metrics_df["hist_a_turnover24h_usd"] = 0.0
        metrics_df["hist_b_turnover24h_usd"] = 0.0
        metrics_df["hist_min_turnover24h_usd"] = 0.0
        metrics_df["hist_a_days_live"] = 0.0
        metrics_df["hist_b_days_live"] = 0.0
        metrics_df["hist_min_days_live"] = 0.0

    # Step 1 table: target pairs, worst period (by robust pnl across all periods), and their metrics.
    targets_rows = []
    for pair in TARGET_PAIRS:
        period = target_worst_periods.get(pair)
        if not period:
            continue
        hit = metrics_df[(metrics_df["pair"] == pair) & (metrics_df["period"] == period)]
        if len(hit):
            targets_rows.append(hit.iloc[0])
        else:
            # Fallback: pull directly from robust table (and re-materialize symbol metrics).
            sub = robust[(robust["pair"] == pair) & (robust["period"] == period)]
            if not len(sub):
                continue
            a, b = _split_pair(pair)
            sa = sym(a)
            sb = sym(b)
            quote_mismatch = int(bool(sa.quote and sb.quote and sa.quote != sb.quote))
            targets_rows.append(
                pd.Series(
                    {
                        "pair": pair,
                        "period": period,
                        "robust_pnl": float(sub.iloc[0].get("robust_pnl") or 0.0),
                        "robust_trades": float(sub.iloc[0].get("robust_trades") or 0.0),
                        "is_bad": int(pair in bad_pairs),
                        "a": a,
                        "b": b,
                        "a_quote": sa.quote,
                        "b_quote": sb.quote,
                        "quote_mismatch": quote_mismatch,
                        "a_turnover24h_usd": sa.turnover24h_usd,
                        "b_turnover24h_usd": sb.turnover24h_usd,
                        "min_turnover24h_usd": _safe_min(sa.turnover24h_usd, sb.turnover24h_usd, missing=0.0),
                        "a_days_live": sa.days_live,
                        "b_days_live": sb.days_live,
                        "min_days_live": _safe_min(sa.days_live, sb.days_live, missing=0.0),
                        "a_bid_ask_pct": sa.bid_ask_pct,
                        "b_bid_ask_pct": sb.bid_ask_pct,
                        "max_bid_ask_pct": _safe_max(sa.bid_ask_pct, sb.bid_ask_pct, missing=1.0),
                        "a_funding_rate_abs": sa.funding_rate_abs,
                        "b_funding_rate_abs": sb.funding_rate_abs,
                        "max_funding_rate_abs": _safe_max(sa.funding_rate_abs, sb.funding_rate_abs, missing=1.0),
                        "a_tick_size_pct": sa.tick_size_pct,
                        "b_tick_size_pct": sb.tick_size_pct,
                        "max_tick_size_pct": _safe_max(sa.tick_size_pct, sb.tick_size_pct, missing=1.0),
                        "a_missing": int(sa.quote == "MISSING" or sa.turnover24h_usd is None),
                        "b_missing": int(sb.quote == "MISSING" or sb.turnover24h_usd is None),
                        "pair_has_missing": int((sa.turnover24h_usd is None) or (sb.turnover24h_usd is None)),
                    }
                )
            )
    targets = pd.DataFrame(targets_rows)
    if len(targets):
        targets = targets.sort_values("robust_pnl", ascending=True)
    targets_out = out_dir / "r03_top1_bad_pairs_tradeability.csv"
    targets.to_csv(targets_out, index=False)

    # Step 2: separation report on analysis periods.
    bad_mask = metrics_df["is_bad"].astype(bool)
    sep = _describe_thresholds(metrics_df, bad_mask)
    sep_out = out_dir / "r03_tradeability_separation_worst_periods.csv"
    sep.to_csv(sep_out, index=False)

    # Candidate thresholds (simple heuristics; refined in notebook-like analysis later).
    # We print a compact markdown summary to help pick mild/med/hard.
    md_lines = []
    md_lines.append("# r03 tradeability diagnostics (top-1)\n")
    md_lines.append("## Worst periods (robust total PnL)\n")
    md_lines.append(period_totals.head(int(args.worst_periods)).to_markdown(index=False))
    md_lines.append("\n\n## Target pairs: worst window + Bybit metrics\n")
    if len(targets):
        cols = [
            "pair",
            "period",
            "testing_start",
            "robust_pnl",
            "a_quote",
            "b_quote",
            "min_turnover24h_usd",
            "min_days_live",
            "max_bid_ask_pct",
            "max_funding_rate_abs",
            "max_tick_size_pct",
            "hist_min_turnover24h_usd",
            "hist_min_days_live",
            "a_missing",
            "b_missing",
        ]
        md_lines.append(targets[cols].to_markdown(index=False))
    else:
        md_lines.append("_No target pairs found in worst periods._")
    md_lines.append("\n\n## Separation (bad vs rest within analysis periods)\n")
    md_lines.append(sep.to_markdown(index=False))

    md_path = out_dir / "r03_tradeability_diagnostics.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {targets_out}")
    print(f"Wrote: {sep_out}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
