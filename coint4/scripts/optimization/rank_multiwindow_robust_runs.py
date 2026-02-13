#!/usr/bin/env python3
"""Rank WFA runs across multiple OOS windows by robust Sharpe, with DD gates.

This script consumes the rollup run index and pairs runs where the run_id starts
with `holdout_` / `stress_` and the remainder (base id) matches, within the same
run_group.

It then aggregates *across OOS windows* for the same variant. OOS windows are
detected by the filename tag pattern: `_oosYYYYMMDD_YYYYMMDD`.

Primary objective (recommended): maximize worst-window robust Sharpe:
  robust_sharpe_window = min(holdout_sharpe, stress_sharpe)
  score = min_window(robust_sharpe_window)

Risk gate (recommended): bound worst-window drawdown on equity:
  dd_pct_window = max(abs(holdout_dd_pct), abs(stress_dd_pct))
  gate: max_window(dd_pct_window) <= --max-dd-pct
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_OOS_RE = re.compile(r"_oos(\d{8})_(\d{8})")


@dataclass(frozen=True)
class _Entry:
    run_id: str
    run_group: str
    config_path: str
    results_dir: str
    status: str
    metrics_present: bool
    sharpe: Optional[float]
    pnl: Optional[float]
    dd_abs: Optional[float]
    dd_pct: Optional[float]
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


def _parse_window(base_id: str) -> str:
    m = _OOS_RE.search(base_id)
    if not m:
        return "-"
    return f"{m.group(1)}-{m.group(2)}"


def _variant_id(base_id: str) -> str:
    return _OOS_RE.sub("", base_id)


def _fmt(value: Optional[float], *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank paired holdout/stress WFA runs across OOS windows (robust Sharpe)")
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
    parser.add_argument("--top", type=int, default=20, help="How many variants to print.")
    parser.add_argument(
        "--include-noncompleted",
        action="store_true",
        help="Include variants where at least one window is not 'completed'.",
    )
    parser.add_argument("--min-windows", type=int, default=3, help="Require at least this many windows per variant.")
    parser.add_argument("--min-trades", type=int, default=200, help="Require min(total_trades) across windows >= this.")
    parser.add_argument("--min-pairs", type=int, default=20, help="Require min(total_pairs_traded) across windows >= this.")
    parser.add_argument(
        "--max-dd-pct",
        type=float,
        default=0.40,
        help="Gate: max window drawdown on equity (abs) must be <= this (default: 0.40 = 40%%).",
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
                    dd_abs=_to_float(row.get("max_drawdown_abs") or ""),
                    dd_pct=_to_float(row.get("max_drawdown_on_equity") or ""),
                    trades=_to_float(row.get("total_trades") or ""),
                    pairs=_to_float(row.get("total_pairs_traded") or ""),
                    costs=_to_float(row.get("total_costs") or ""),
                )
            )

    # Step 1: pair holdout/stress within each run_group by base_id.
    paired: Dict[Tuple[str, str], Dict[str, _Entry]] = {}
    for e in entries:
        kind, base_id = _kind_and_base_id(e.run_id)
        if kind not in {"holdout", "stress"}:
            continue

        meta = " | ".join([e.run_group, base_id, e.run_id, e.config_path, e.results_dir, e.status])
        if args.contains and not _matches_all(meta, args.contains):
            continue

        key = (e.run_group, base_id)
        slot = paired.setdefault(key, {})
        slot[kind] = e

    # Step 2: compute per-window robust metrics and aggregate by variant.
    # variant key: (run_group, variant_id)
    windows_by_variant: Dict[Tuple[str, str], List[Tuple[str, float, float, _Entry, _Entry]]] = {}
    for (run_group, base_id), pair in paired.items():
        h = pair.get("holdout")
        s = pair.get("stress")
        if h is None or s is None:
            continue
        if not (h.metrics_present and s.metrics_present):
            continue
        if h.sharpe is None or s.sharpe is None:
            continue
        if h.dd_pct is None or s.dd_pct is None:
            continue
        if not args.include_noncompleted:
            if h.status.lower() != "completed" or s.status.lower() != "completed":
                continue

        robust_sharpe = min(h.sharpe, s.sharpe)
        dd_pct = max(abs(h.dd_pct), abs(s.dd_pct))
        window = _parse_window(base_id)
        variant = _variant_id(base_id)
        windows_by_variant.setdefault((run_group, variant), []).append((window, robust_sharpe, dd_pct, h, s))

    rows = []
    for (run_group, variant), items in windows_by_variant.items():
        if len(items) < max(1, args.min_windows):
            continue
        # Gates across windows.
        worst_dd = max(item[2] for item in items)
        if worst_dd > max(0.0, args.max_dd_pct):
            continue

        min_trades = min((item[3].trades or 0.0) for item in items)
        min_pairs = min((item[3].pairs or 0.0) for item in items)
        if min_trades < args.min_trades or min_pairs < args.min_pairs:
            continue

        worst_robust = min(item[1] for item in items)
        avg_robust = sum(item[1] for item in items) / len(items)
        avg_dd = sum(item[2] for item in items) / len(items)
        rows.append((worst_robust, avg_robust, worst_dd, avg_dd, run_group, variant, items))

    rows.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    rows = rows[: max(1, args.top)]

    if not rows:
        print("No variants matched (check filters/gates or ensure rollup index is up to date).")
        return 1

    print(
        "| rank | worst_robust_sh | avg_robust_sh | worst_dd_pct | avg_dd_pct | windows | run_group | variant_id | sample_config |"
    )
    print("|---:|---:|---:|---:|---:|---:|---|---|---|")
    for idx, (worst_robust, avg_robust, worst_dd, avg_dd, run_group, variant, items) in enumerate(rows, 1):
        # pick a stable config_path for reference
        sample_cfg = items[0][3].config_path
        print(
            "| {rank} | {worst} | {avg} | {wdd} | {add} | {n} | {group} | {variant} | {cfg} |".format(
                rank=idx,
                worst=_fmt(worst_robust, digits=3),
                avg=_fmt(avg_robust, digits=3),
                wdd=_fmt(worst_dd, digits=3),
                add=_fmt(avg_dd, digits=3),
                n=len(items),
                group=run_group,
                variant=variant,
                cfg=sample_cfg,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
