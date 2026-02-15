#!/usr/bin/env python3
"""Recompute canonical Sharpe from equity_curve.csv files and write a summary.

This script is intentionally read-only w.r.t. run artifacts: it does NOT modify
any existing results directories. It only writes a rollup CSV/JSON report to an
output directory (default: coint4/outputs/, which is gitignored).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from coint2.core.sharpe import compute_equity_sharpe_from_equity_curve_csv


def _discover_equity_curves(*, runs_root: Optional[Path], runs_glob: Optional[str]) -> List[Path]:
    if runs_glob:
        # Interpret as a glob rooted at the current working directory.
        return sorted(Path().glob(runs_glob))
    if runs_root:
        return sorted(runs_root.rglob("equity_curve.csv"))
    raise ValueError("Either --runs-root or --runs-glob must be provided")


def _to_row(equity_curve: Path, *, days_per_year: float) -> dict:
    stats = compute_equity_sharpe_from_equity_curve_csv(
        equity_curve,
        days_per_year=days_per_year,
    )
    base = {
        "run_dir": str(equity_curve.parent),
        "equity_curve": str(equity_curve),
    }
    if stats is None:
        base.update(
            {
                "ok": False,
                "sharpe_full": None,
                "sharpe_daily_only": None,
                "n_returns": None,
                "period_seconds_median": None,
                "periods_per_year_full": None,
            }
        )
        return base

    base.update({"ok": True})
    base.update(
        {
            "sharpe_full": stats.sharpe_full,
            "sharpe_daily_only": stats.sharpe_daily_only,
            "n_returns": stats.n_returns,
            "period_seconds_median": stats.period_seconds_median,
            "periods_per_year_full": stats.periods_per_year_full,
            "mean_return": stats.mean_return,
            "std_return": stats.std_return,
        }
    )
    return base


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=None, help="Root to scan for equity_curve.csv")
    parser.add_argument("--runs-glob", type=str, default=None, help="Glob for equity_curve.csv paths (relative)")
    parser.add_argument("--days-per-year", type=float, default=365.0, help="Annualization base for periods/year")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write rollups (default: coint4/outputs/sharpe_recompute/...)",
    )
    args = parser.parse_args()

    app_root = Path(__file__).resolve().parents[2]  # .../coint4
    output_dir = args.output_dir or (app_root / "outputs" / "sharpe_recompute")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"sharpe_recompute_{stamp}.csv"
    out_json = output_dir / f"sharpe_recompute_{stamp}.json"

    equity_curves = _discover_equity_curves(runs_root=args.runs_root, runs_glob=args.runs_glob)
    rows = [_to_row(p, days_per_year=args.days_per_year) for p in equity_curves]

    _write_csv(out_csv, rows)
    _write_json(out_json, rows)

    ok = sum(1 for r in rows if r.get("ok"))
    total = len(rows)
    print(f"[recompute_sharpe] processed={total} ok={ok} csv={out_csv} json={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

