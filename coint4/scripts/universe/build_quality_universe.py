#!/usr/bin/env python3
"""Build a quality universe and exclusion list from raw data."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

os.environ.setdefault("POLARS_MAX_THREADS", str(os.cpu_count() or 1))

import polars as pl


@dataclass(frozen=True)
class QualityThresholds:
    period_start: str
    period_end: str
    bar_minutes: int
    min_history_days: int
    min_coverage_ratio: float
    min_avg_daily_turnover_usd: float
    max_days_since_last: int
    leveraged_regex: str


def _to_ms(dt: date, end_of_day: bool = False) -> int:
    if end_of_day:
        dt = dt + timedelta(days=1)
        return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp() * 1000) - 1
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp() * 1000)


def _iter_parquet_files(data_root: Path) -> Iterable[Path]:
    yield from data_root.glob("year=*/month=*/data_part_*.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build quality universe from parquet data")
    parser.add_argument("--data-root", default="data_downloaded", help="Data root directory")
    parser.add_argument("--period-start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--period-end", default="2023-09-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bar-minutes", type=int, default=15, help="Bar interval in minutes")
    parser.add_argument("--min-history-days", type=int, default=180, help="Minimum history in days")
    parser.add_argument("--min-coverage-ratio", type=float, default=0.9, help="Minimum coverage ratio")
    parser.add_argument(
        "--min-avg-daily-turnover-usd",
        type=float,
        default=1_000_000.0,
        help="Minimum average daily turnover (USD)",
    )
    parser.add_argument(
        "--max-days-since-last",
        type=int,
        default=14,
        help="Max days since last observation in window",
    )
    parser.add_argument(
        "--leveraged-regex",
        default=r"(?:UP|DOWN|BULL|BEAR|[0-9]+[LS])(?:USDT|USDC|USD|BUSD)$",
        help="Regex for leveraged/meme token suffixes",
    )
    parser.add_argument("--out-dir", default="artifacts/universe/quality_universe_20260115", help="Output directory")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    start_dt = date.fromisoformat(args.period_start)
    end_dt = date.fromisoformat(args.period_end)
    window_days = (end_dt - start_dt).days + 1
    if window_days <= 0:
        raise ValueError("period_end must be >= period_start")

    bars_per_day = int(24 * 60 / args.bar_minutes)
    min_bars = args.min_history_days * bars_per_day
    start_ms = _to_ms(start_dt, end_of_day=False)
    end_ms = _to_ms(end_dt, end_of_day=True)

    thresholds = QualityThresholds(
        period_start=args.period_start,
        period_end=args.period_end,
        bar_minutes=args.bar_minutes,
        min_history_days=args.min_history_days,
        min_coverage_ratio=args.min_coverage_ratio,
        min_avg_daily_turnover_usd=args.min_avg_daily_turnover_usd,
        max_days_since_last=args.max_days_since_last,
        leveraged_regex=args.leveraged_regex,
    )

    parquet_files = list(_iter_parquet_files(data_root))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_root}")

    scan = (
        pl.scan_parquet(parquet_files)
        .select(["timestamp", "symbol", "close", "volume", "turnover"])
        .filter(pl.col("timestamp").is_between(start_ms, end_ms))
        .with_columns(
            [
                pl.col("turnover").cast(pl.Float64, strict=False).alias("turnover_num"),
                (pl.col("close") * pl.col("volume")).alias("turnover_calc"),
            ]
        )
        .with_columns(
            pl.when(pl.col("turnover_num").is_null())
            .then(pl.col("turnover_calc"))
            .otherwise(pl.col("turnover_num"))
            .alias("turnover_usd")
        )
    )

    agg = (
        scan.group_by("symbol")
        .agg(
            [
                pl.len().alias("bars"),
                pl.min("timestamp").alias("min_ts"),
                pl.max("timestamp").alias("max_ts"),
                pl.sum("turnover_usd").alias("turnover_sum"),
            ]
        )
        .with_columns(
            [
                (pl.col("bars") / bars_per_day).alias("days_available"),
                (pl.col("bars") / (window_days * bars_per_day)).alias("coverage_ratio"),
                (pl.col("turnover_sum") / window_days).alias("avg_daily_turnover_usd"),
                ((end_ms - pl.col("max_ts")) / (1000 * 60 * 60 * 24)).alias("days_since_last"),
                pl.col("symbol")
                .str.contains(args.leveraged_regex)
                .fill_null(False)
                .alias("is_leveraged"),
            ]
        )
    )

    df = agg.collect(engine="streaming")
    records = df.to_dicts()

    excluded = []
    included = []
    reason_counts = {
        "short_history": 0,
        "low_coverage": 0,
        "low_turnover": 0,
        "stale": 0,
        "leveraged_token": 0,
    }

    for rec in records:
        reasons = []
        if rec["bars"] < min_bars:
            reasons.append("short_history")
            reason_counts["short_history"] += 1
        if rec["coverage_ratio"] < args.min_coverage_ratio:
            reasons.append("low_coverage")
            reason_counts["low_coverage"] += 1
        if rec["avg_daily_turnover_usd"] < args.min_avg_daily_turnover_usd:
            reasons.append("low_turnover")
            reason_counts["low_turnover"] += 1
        if rec["days_since_last"] > args.max_days_since_last:
            reasons.append("stale")
            reason_counts["stale"] += 1
        if rec["is_leveraged"]:
            reasons.append("leveraged_token")
            reason_counts["leveraged_token"] += 1

        rec["exclude_reasons"] = ",".join(reasons)
        if reasons:
            excluded.append(rec["symbol"])
        else:
            included.append(rec["symbol"])

    excluded = sorted(set(excluded))
    included = sorted(set(included))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "quality_report.csv"
    df = df.with_columns(
        pl.Series("exclude_reasons", [rec["exclude_reasons"] for rec in records])
    )
    df.write_csv(report_path)

    (out_dir / "excluded_symbols.txt").write_text("\n".join(excluded) + "\n")
    (out_dir / "included_symbols.txt").write_text("\n".join(included) + "\n")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": asdict(thresholds),
        "window_days": window_days,
        "bars_per_day": bars_per_day,
        "symbols_total": len(records),
        "symbols_included": len(included),
        "symbols_excluded": len(excluded),
        "exclude_reason_counts": reason_counts,
    }
    (out_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2))

    (out_dir / "exclude_symbols.yaml").write_text(
        "exclude_symbols:\n"
        + "".join([f"  - {sym}\n" for sym in excluded])
        + "metadata:\n"
        + "".join([f"  {k}: {v}\n" for k, v in summary.items() if k != "exclude_reason_counts"])
        + "  exclude_reason_counts:\n"
        + "".join([f"    {k}: {v}\n" for k, v in reason_counts.items()])
    )

    print(f"✅ Quality universe built: {out_dir}")
    print(f"  total symbols: {len(records)}")
    print(f"  included: {len(included)} | excluded: {len(excluded)}")
    for key, val in reason_counts.items():
        print(f"  excluded_{key}: {val}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
