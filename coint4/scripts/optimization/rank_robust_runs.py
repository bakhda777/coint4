#!/usr/bin/env python3
"""Rank WFA runs by robust Sharpe across holdout/stress pairs.

This script consumes the rollup run index and pairs runs where the run_id starts
with `holdout_` / `stress_` and the remainder (base id) matches, within the same
run_group.

Primary score: robust_sharpe = min(holdout_sharpe, stress_sharpe)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class _Entry:
    run_id: str
    run_group: str
    results_dir: str
    config_path: str
    status: str
    metrics_present: bool
    sharpe: Optional[float]
    pnl: Optional[float]
    max_dd: Optional[float]
    trades: Optional[float]
    pairs: Optional[float]
    costs: Optional[float]


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: str) -> Optional[float]:
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _kind_and_base_id(run_id: str) -> Tuple[Optional[str], str]:
    if run_id.startswith("holdout_"):
        return "holdout", run_id[len("holdout_") :]
    if run_id.startswith("stress_"):
        return "stress", run_id[len("stress_") :]
    return None, run_id


def _matches_all(text: str, needles: Iterable[str]) -> bool:
    hay = text.lower()
    for needle in needles:
        if needle.lower() not in hay:
            return False
    return True


def _fmt(value: Optional[float], *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank paired holdout/stress WFA runs by robust Sharpe")
    parser.add_argument(
        "--run-index",
        default="artifacts/wfa/aggregate/rollup/run_index.csv",
        help="Path to rollup run_index.csv (relative to project root).",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Substring filter (repeatable). All must match combined run metadata.",
    )
    parser.add_argument("--top", type=int, default=20, help="How many rows to print.")
    parser.add_argument(
        "--include-noncompleted",
        action="store_true",
        help="Include pairs where status is not 'completed'.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    run_index_path = project_root / args.run_index
    if not run_index_path.exists():
        raise SystemExit(f"run index not found: {run_index_path}")

    entries: List[_Entry] = []
    with run_index_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                _Entry(
                    run_id=(row.get("run_id") or "").strip(),
                    run_group=(row.get("run_group") or "").strip(),
                    results_dir=(row.get("results_dir") or "").strip(),
                    config_path=(row.get("config_path") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    metrics_present=_to_bool(row.get("metrics_present") or ""),
                    sharpe=_to_float(row.get("sharpe_ratio_abs") or ""),
                    pnl=_to_float(row.get("total_pnl") or ""),
                    max_dd=_to_float(row.get("max_drawdown_abs") or ""),
                    trades=_to_float(row.get("total_trades") or ""),
                    pairs=_to_float(row.get("total_pairs_traded") or ""),
                    costs=_to_float(row.get("total_costs") or ""),
                )
            )

    # Group within run_group to avoid collisions (same run_id can appear in multiple groups).
    grouped: Dict[Tuple[str, str], Dict[str, _Entry]] = {}
    for e in entries:
        kind, base_id = _kind_and_base_id(e.run_id)
        if kind not in {"holdout", "stress"}:
            continue

        meta = " | ".join(
            [
                e.run_group,
                base_id,
                e.run_id,
                e.config_path,
                e.results_dir,
                e.status,
            ]
        )
        if args.contains and not _matches_all(meta, args.contains):
            continue

        key = (e.run_group, base_id)
        slot = grouped.setdefault(key, {})
        slot[kind] = e

    rows = []
    for (run_group, base_id), pair in grouped.items():
        h = pair.get("holdout")
        s = pair.get("stress")
        if h is None or s is None:
            continue
        if not (h.metrics_present and s.metrics_present):
            continue
        if h.sharpe is None or s.sharpe is None:
            continue
        if not args.include_noncompleted:
            if h.status.lower() != "completed" or s.status.lower() != "completed":
                continue

        robust = min(h.sharpe, s.sharpe)
        rows.append((robust, run_group, base_id, h, s))

    rows.sort(key=lambda x: x[0], reverse=True)

    top_n = max(1, args.top)
    rows = rows[:top_n]

    if not rows:
        print("No paired holdout/stress runs matched.")
        return 1

    # Markdown table for easy pasting into docs.
    print(
        "| rank | robust_sharpe | holdout_sh | stress_sh | holdout_pnl | stress_pnl | holdout_dd | stress_dd | trades | pairs | run_group | run_id | config_path |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for idx, (robust, run_group, base_id, h, s) in enumerate(rows, 1):
        config_path = h.config_path if h.config_path == s.config_path else f"H:{h.config_path} / S:{s.config_path}"
        trades = h.trades if h.trades is not None else s.trades
        pairs = h.pairs if h.pairs is not None else s.pairs
        print(
            "| {rank} | {robust} | {h_sh} | {s_sh} | {h_pnl} | {s_pnl} | {h_dd} | {s_dd} | {trades} | {pairs} | {group} | {run_id} | {config_path} |".format(
                rank=idx,
                robust=_fmt(robust, digits=3),
                h_sh=_fmt(h.sharpe, digits=3),
                s_sh=_fmt(s.sharpe, digits=3),
                h_pnl=_fmt(h.pnl, digits=2),
                s_pnl=_fmt(s.pnl, digits=2),
                h_dd=_fmt(h.max_dd, digits=2),
                s_dd=_fmt(s.max_dd, digits=2),
                trades=_fmt(trades, digits=0),
                pairs=_fmt(pairs, digits=0),
                group=run_group,
                run_id=base_id,
                config_path=config_path,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
