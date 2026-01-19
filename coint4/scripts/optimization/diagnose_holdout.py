#!/usr/bin/env python3
"""Diagnose holdout regression for 2026-01-18 configs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    run_dir: Path


RUNS = {
    "holdout_baseline": RunSpec(
        run_id="holdout_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000",
        run_dir=PROJECT_ROOT
        / "artifacts/wfa/runs/20260118_holdout/holdout_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000",
    ),
    "holdout_corr0p7": RunSpec(
        run_id="holdout_20260118_corr0p7_z0p85_exit0p12_ssd25000",
        run_dir=PROJECT_ROOT
        / "artifacts/wfa/runs/20260118_holdout/holdout_20260118_corr0p7_z0p85_exit0p12_ssd25000",
    ),
    "shortlist_baseline": RunSpec(
        run_id="shortlist_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000",
        run_dir=PROJECT_ROOT
        / "artifacts/wfa/runs/20260118_shortlist/shortlist_20260118_baseline_z0p85_exit0p12_corr0p65_ssd25000",
    ),
}


def _load_daily_pnl(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "daily_pnl.csv")
    df = df.rename(columns={"Unnamed: 0": "date", "PnL": "pnl"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "pnl"]].sort_values("date").reset_index(drop=True)


def _load_trade_stats(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "trade_statistics.csv")
    return df.copy()


def _parse_period(period: str, year: int) -> tuple[datetime, datetime]:
    start_str, end_str = period.split("-")
    start_month, start_day = [int(x) for x in start_str.split("/")]
    end_month, end_day = [int(x) for x in end_str.split("/")]
    start_date = datetime(year, start_month, start_day)
    end_date = datetime(year, end_month, end_day)
    return start_date, end_date


def _extract_filter_files(run_dir: Path) -> list[Path]:
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return []
    text = log_path.read_text()
    matches = re.findall(r"filter_reasons_\d+_\d+\.csv", text)
    # Preserve order while removing duplicates.
    seen = set()
    ordered = []
    for match in matches:
        if match in seen:
            continue
        seen.add(match)
        ordered.append(RESULTS_DIR / match)
    return ordered


def _normalize_reason(reason: str) -> str:
    cleaned = reason.strip()
    if " " in cleaned:
        cleaned = cleaned.split(" ", 1)[0]
    if cleaned.startswith("half_life"):
        return "half_life"
    if cleaned.startswith("hurst_too_high"):
        return "hurst_too_high"
    if cleaned.startswith("low_correlation"):
        return "low_correlation"
    if cleaned.startswith("beta_out_of_range"):
        return "beta_out_of_range"
    if cleaned.startswith("pvalue"):
        return "pvalue"
    if cleaned.startswith("kpss"):
        return "kpss"
    return cleaned


def _compute_sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(returns.mean() / std * (periods_per_year ** 0.5))


def _compute_drawdown_abs(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())


def _step_metrics(
    run_id: str,
    daily_pnl: pd.DataFrame,
    trade_stats: pd.DataFrame,
    test_days: int = 30,
    initial_capital: float = 10000.0,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    year = int(daily_pnl["date"].dt.year.min())
    periods = sorted(
        trade_stats["period"].unique(),
        key=lambda p: _parse_period(p, year)[0],
    )
    rows = []
    for idx, period in enumerate(periods, start=1):
        _, period_end = _parse_period(period, year)
        test_start = period_end - timedelta(days=test_days - 1)
        mask = (daily_pnl["date"] >= test_start) & (daily_pnl["date"] <= period_end)
        pnl_slice = daily_pnl.loc[mask].copy()
        pnl_sum = float(pnl_slice["pnl"].sum())
        equity = initial_capital + pnl_slice["pnl"].cumsum()
        returns = pnl_slice["pnl"] / equity.shift(1)
        returns = returns.fillna(0.0)
        sharpe = _compute_sharpe(returns, periods_per_year=periods_per_year)
        max_dd = _compute_drawdown_abs(equity)
        stats_slice = trade_stats[trade_stats["period"] == period]
        rows.append(
            {
                "run_id": run_id,
                "step_index": idx,
                "period": period,
                "test_start": test_start.date().isoformat(),
                "test_end": period_end.date().isoformat(),
                "days": int(len(pnl_slice)),
                "total_pnl": pnl_sum,
                "sharpe_daily": sharpe,
                "max_drawdown_abs": max_dd,
                "total_trades": float(stats_slice["trade_count"].sum()),
                "total_pairs_traded": int(stats_slice["pair"].nunique()),
                "total_costs": float(stats_slice["total_costs"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _pair_summary(trade_stats: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trade_stats.groupby("pair", as_index=False)
        .agg(
            total_pnl=("total_pnl", "sum"),
            total_costs=("total_costs", "sum"),
            trade_count=("trade_count", "sum"),
        )
        .sort_values("total_pnl", ascending=True)
        .reset_index(drop=True)
    )
    return grouped


def _base_asset(symbol: str) -> str:
    for suffix in ("USDT", "USDC", "USD", "BUSD", "FDUSD", "TUSD", "USDP", "DAI", "BTC", "ETH"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _asset_summary(pair_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pair_stats.iterrows():
        s1, s2 = row["pair"].split("-")
        pnl_half = row["total_pnl"] / 2.0
        costs_half = row["total_costs"] / 2.0
        trades_half = row["trade_count"] / 2.0
        for symbol in (_base_asset(s1), _base_asset(s2)):
            rows.append(
                {
                    "asset": symbol,
                    "total_pnl": pnl_half,
                    "total_costs": costs_half,
                    "trade_count": trades_half,
                }
            )
    df = pd.DataFrame(rows)
    return (
        df.groupby("asset", as_index=False)
        .agg(
            total_pnl=("total_pnl", "sum"),
            total_costs=("total_costs", "sum"),
            trade_count=("trade_count", "sum"),
        )
        .sort_values("total_pnl", ascending=True)
        .reset_index(drop=True)
    )


def _top_concentration(
    run_id: str,
    pair_stats: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    losers = pair_stats.head(top_n).copy()
    losers["side"] = "loser"
    losers["rank"] = range(1, len(losers) + 1)
    winners = pair_stats.sort_values("total_pnl", ascending=False).head(top_n).copy()
    winners["side"] = "winner"
    winners["rank"] = range(1, len(winners) + 1)
    out = pd.concat([losers, winners], ignore_index=True)
    out.insert(0, "run_id", run_id)
    return out[["run_id", "side", "rank", "pair", "total_pnl", "total_costs", "trade_count"]]


def _overlap_summary(
    holdout_pairs: Iterable[str],
    shortlist_pairs: Iterable[str],
    holdout_pair_stats: pd.DataFrame,
) -> dict:
    holdout_set = set(holdout_pairs)
    shortlist_set = set(shortlist_pairs)
    intersection = holdout_set & shortlist_set
    union = holdout_set | shortlist_set
    overlap_pnl = float(
        holdout_pair_stats[holdout_pair_stats["pair"].isin(intersection)]["total_pnl"].sum()
    )
    non_overlap_pnl = float(
        holdout_pair_stats[~holdout_pair_stats["pair"].isin(intersection)]["total_pnl"].sum()
    )
    return {
        "holdout_pairs": len(holdout_set),
        "shortlist_pairs": len(shortlist_set),
        "intersection_pairs": len(intersection),
        "jaccard": (len(intersection) / len(union)) if union else 0.0,
        "overlap_pnl": overlap_pnl,
        "non_overlap_pnl": non_overlap_pnl,
    }


def _filter_summary(run_id: str, run_dir: Path) -> pd.DataFrame:
    files = _extract_filter_files(run_dir)
    rows = []
    for idx, path in enumerate(files, start=1):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["reason_norm"] = df["reason"].map(_normalize_reason)
        counts = df["reason_norm"].value_counts()
        total_rows = int(len(df))
        for reason, count in counts.items():
            rows.append(
                {
                    "run_id": run_id,
                    "step_index": idx,
                    "filter_file": path.name,
                    "reason": reason,
                    "count": int(count),
                    "total_rows": total_rows,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_data = {}
    step_frames = []
    pair_frames = []
    asset_frames = []
    concentration_frames = []

    for key, spec in RUNS.items():
        daily_pnl = _load_daily_pnl(spec.run_dir)
        trade_stats = _load_trade_stats(spec.run_dir)
        run_data[key] = {
            "daily_pnl": daily_pnl,
            "trade_stats": trade_stats,
        }
        if key.startswith("holdout"):
            step_frames.append(_step_metrics(spec.run_id, daily_pnl, trade_stats))
            pair_stats = _pair_summary(trade_stats)
            concentration_frames.append(_top_concentration(spec.run_id, pair_stats))
            pair_stats.insert(0, "run_id", spec.run_id)
            pair_frames.append(pair_stats)
            asset_stats = _asset_summary(pair_stats)
            asset_stats.insert(0, "run_id", spec.run_id)
            asset_frames.append(asset_stats)

    step_metrics = pd.concat(step_frames, ignore_index=True)
    step_metrics.to_csv(RESULTS_DIR / "holdout_20260118_step_metrics.csv", index=False)

    pair_summary = pd.concat(pair_frames, ignore_index=True)
    pair_summary.to_csv(RESULTS_DIR / "holdout_20260118_pair_summary.csv", index=False)

    asset_summary = pd.concat(asset_frames, ignore_index=True)
    asset_summary.to_csv(RESULTS_DIR / "holdout_20260118_asset_summary.csv", index=False)

    concentration = pd.concat(concentration_frames, ignore_index=True)
    concentration.to_csv(RESULTS_DIR / "holdout_20260118_pair_concentration.csv", index=False)

    shortlist_pairs = set(run_data["shortlist_baseline"]["trade_stats"]["pair"].unique())
    overlap = _overlap_summary(
        holdout_pairs=run_data["holdout_baseline"]["trade_stats"]["pair"].unique(),
        shortlist_pairs=shortlist_pairs,
        holdout_pair_stats=pair_summary[pair_summary["run_id"] == RUNS["holdout_baseline"].run_id],
    )
    overlap_path = RESULTS_DIR / "holdout_20260118_overlap_summary.json"
    overlap_path.write_text(json.dumps(overlap, indent=2, sort_keys=True))

    filter_frames = []
    for key in ("holdout_baseline", "holdout_corr0p7", "shortlist_baseline"):
        spec = RUNS[key]
        filter_frames.append(_filter_summary(spec.run_id, spec.run_dir))
    filter_summary = pd.concat(filter_frames, ignore_index=True)
    filter_summary.to_csv(RESULTS_DIR / "holdout_20260118_filter_summary.csv", index=False)

    print("Step metrics:", RESULTS_DIR / "holdout_20260118_step_metrics.csv")
    print("Pair summary:", RESULTS_DIR / "holdout_20260118_pair_summary.csv")
    print("Asset summary:", RESULTS_DIR / "holdout_20260118_asset_summary.csv")
    print("Pair concentration:", RESULTS_DIR / "holdout_20260118_pair_concentration.csv")
    print("Overlap summary:", overlap_path)
    print("Filter summary:", RESULTS_DIR / "holdout_20260118_filter_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
