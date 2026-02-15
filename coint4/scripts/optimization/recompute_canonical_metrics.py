#!/usr/bin/env python3
"""Recompute canonical Sharpe/DD/PnL from equity_curve.csv and write canonical_metrics.json.

This script is intentionally "no-legacy-overwrite":
  - It never modifies strategy_metrics.csv or equity_curve.csv.
  - It writes a NEW file: canonical_metrics.json (per run directory).
  - By default it refuses to overwrite an existing canonical_metrics.json.

Sharpe is computed using coint2.core.sharpe.annualized_sharpe_ratio_from_equity with
FIXED annualization (periods_per_year), not inferred from timestamp deltas.

Typical usage (15m bars, periods_per_year=365*96=35040):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/recompute_canonical_metrics.py \
    --manifest artifacts/wfa/aggregate/clean_cycle_top10/20260215_clean_top10/baseline_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from coint2.core.canonical_metrics import compute_canonical_metrics_from_equity_curve_csv


def _normalize_repo_relative_path(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if value.startswith("coint4/"):
        value = value[len("coint4/") :]
    while value.startswith("./"):
        value = value[2:]
    return value


def _try_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _resolve_under_root(path_str: str, root: Path) -> Optional[Path]:
    raw = str(path_str or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return root / _normalize_repo_relative_path(raw)


def _load_results_dirs_from_manifest(manifest_path: Path) -> List[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"manifest must be a JSON list: {manifest_path}")
    out: List[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        results_dir = str(item.get("results_dir") or "").strip()
        if results_dir:
            out.append(results_dir)
    return out


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.6f}"


def _infer_bar_minutes_from_median_seconds(period_seconds_median: Optional[float]) -> Optional[float]:
    if period_seconds_median is None:
        return None
    if not math.isfinite(period_seconds_median) or period_seconds_median <= 0:
        return None
    return period_seconds_median / 60.0


def _periods_per_year_from_bar_minutes(*, bar_minutes: float, days_per_year: float) -> float:
    if bar_minutes <= 0 or not math.isfinite(bar_minutes):
        raise ValueError(f"bar_minutes must be positive and finite, got: {bar_minutes!r}")
    if days_per_year <= 0 or not math.isfinite(days_per_year):
        raise ValueError(f"days_per_year must be positive and finite, got: {days_per_year!r}")
    return days_per_year * (24.0 * 60.0 / bar_minutes)


def _dedupe_keep_order(values: Iterable[Path]) -> List[Path]:
    seen: set[Path] = set()
    out: List[Path] = []
    for value in values:
        resolved = value.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute canonical Sharpe/DD/PnL and write canonical_metrics.json")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Results directory to process (repeatable; absolute or relative to coint4/).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to baseline_manifest.json with results_dir entries (relative to coint4/ unless absolute).",
    )
    parser.add_argument(
        "--bar-minutes",
        type=float,
        default=None,
        help="Fixed bar timeframe in minutes (default: require inferred=15m).",
    )
    parser.add_argument(
        "--periods-per-year",
        type=float,
        default=None,
        help="Explicit periods_per_year for Sharpe annualization (overrides --bar-minutes).",
    )
    parser.add_argument("--days-per-year", type=float, default=365.0, help="Days/year for --bar-minutes conversion.")
    parser.add_argument("--dry-run", action="store_true", help="Compute, but do not write files.")
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--refuse-overwrite",
        action="store_true",
        help="Refuse overwriting an existing canonical_metrics.json (default).",
    )
    overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing canonical_metrics.json files.",
    )
    args = parser.parse_args()

    app_root = Path(__file__).resolve().parents[2]  # .../coint4

    if args.periods_per_year is not None and args.bar_minutes is not None:
        raise SystemExit("Use only one of --periods-per-year or --bar-minutes.")

    manifest_paths: List[Path] = []
    if args.manifest:
        resolved = _resolve_under_root(args.manifest, app_root)
        if resolved is None:
            raise SystemExit("--manifest path is empty")
        manifest_paths.append(resolved)

    run_dirs: List[Path] = []
    for raw in args.run_dir:
        resolved = _resolve_under_root(raw, app_root)
        if resolved is None:
            continue
        run_dirs.append(resolved)

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            raise SystemExit(f"manifest not found: {manifest_path}")
        for results_dir in _load_results_dirs_from_manifest(manifest_path):
            resolved = _resolve_under_root(results_dir, app_root)
            if resolved is None:
                continue
            run_dirs.append(resolved)

    run_dirs = _dedupe_keep_order(run_dirs)
    if not run_dirs:
        raise SystemExit("No run directories provided (use --run-dir and/or --manifest).")

    # Decide annualization policy (fixed; no timestamp inference for the Sharpe formula).
    periods_per_year_fixed: Optional[float] = None
    bar_minutes_fixed: Optional[float] = None
    if args.periods_per_year is not None:
        periods_per_year_fixed = float(args.periods_per_year)
        bar_minutes_fixed = None
    elif args.bar_minutes is not None:
        bar_minutes_fixed = float(args.bar_minutes)
        periods_per_year_fixed = _periods_per_year_from_bar_minutes(
            bar_minutes=bar_minutes_fixed, days_per_year=float(args.days_per_year)
        )
    else:
        # Safe default: only allow implicit default when the run *looks* like 15m data.
        bar_minutes_fixed = 15.0
        periods_per_year_fixed = _periods_per_year_from_bar_minutes(
            bar_minutes=bar_minutes_fixed, days_per_year=float(args.days_per_year)
        )

    if periods_per_year_fixed is None or periods_per_year_fixed <= 0 or not math.isfinite(periods_per_year_fixed):
        raise SystemExit(f"Invalid periods_per_year: {periods_per_year_fixed!r}")

    target_paths = [run_dir / "canonical_metrics.json" for run_dir in run_dirs]
    if not args.dry_run and not args.overwrite:
        existing = [p for p in target_paths if p.exists()]
        if existing:
            formatted = "\n".join(f"- {p}" for p in existing[:20])
            tail = "" if len(existing) <= 20 else f"\n- ... ({len(existing) - 20} more)"
            raise SystemExit(
                "Refusing to overwrite existing canonical_metrics.json (use --overwrite):\n"
                + formatted
                + tail
            )

    computed_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    ok = 0
    failed: List[str] = []
    for run_dir in run_dirs:
        equity_path = run_dir / "equity_curve.csv"
        if not equity_path.exists():
            failed.append(f"{run_dir}: missing equity_curve.csv")
            continue

        # Compute canonical metrics with FIXED annualization.
        metrics = compute_canonical_metrics_from_equity_curve_csv(
            equity_path,
            periods_per_year=periods_per_year_fixed,
            risk_free_rate=0.0,
        )
        if metrics is None:
            failed.append(f"{run_dir}: failed to parse equity_curve.csv")
            continue

        bar_minutes_inferred = _infer_bar_minutes_from_median_seconds(metrics.period_seconds_median)

        # If the user didn't provide explicit annualization, enforce the implicit 15m assumption.
        if args.periods_per_year is None and args.bar_minutes is None:
            expected_seconds = 15.0 * 60.0
            if metrics.period_seconds_median is None:
                failed.append(
                    f"{run_dir}: cannot infer bar size (no timestamp deltas); pass --bar-minutes or --periods-per-year"
                )
                continue
            if abs(float(metrics.period_seconds_median) - expected_seconds) > 1.0:
                failed.append(
                    f"{run_dir}: inferred bar_minutes={_fmt_float(bar_minutes_inferred)} != 15.0; "
                    "pass --bar-minutes or --periods-per-year explicitly"
                )
                continue

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "metrics": {
                "canonical_sharpe": metrics.canonical_sharpe,
                "canonical_pnl_abs": metrics.canonical_pnl_abs,
                "canonical_max_drawdown_abs": metrics.canonical_max_drawdown_abs,
            },
            "meta": {
                "computed_at_utc": computed_at_utc,
                "risk_free_rate": 0.0,
                "ddof": 1,
                "annualization": {
                    "mode": "periods_per_year" if args.periods_per_year is not None else "bar_minutes",
                    "days_per_year": float(args.days_per_year),
                    "bar_minutes": bar_minutes_fixed,
                    "periods_per_year": periods_per_year_fixed,
                },
                "equity_curve": {
                    "path": _normalize_repo_relative_path(_try_relpath(equity_path, app_root)),
                    "n_points": metrics.n_points,
                    "n_returns": metrics.n_returns,
                    "equity_first": metrics.equity_first,
                    "equity_last": metrics.equity_last,
                    "period_seconds_median": metrics.period_seconds_median,
                    "bar_minutes_inferred": bar_minutes_inferred,
                },
            },
        }

        out_path = run_dir / "canonical_metrics.json"
        if args.dry_run:
            print(
                f"[dry-run] {run_dir}: sharpe={metrics.canonical_sharpe:.6f} pnl={metrics.canonical_pnl_abs:.6f} dd={metrics.canonical_max_drawdown_abs:.6f} -> {out_path}"
            )
            ok += 1
            continue

        try:
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:
            # Sync-back/rsync may leave some artifact dirs root-owned; don't let a single
            # PermissionError abort the entire recompute batch.
            failed.append(f"{run_dir}: failed to write canonical_metrics.json: {exc.__class__.__name__}: {exc}")
            continue
        ok += 1

    total = len(run_dirs)
    print(f"[recompute_canonical_metrics] processed={total} ok={ok} failed={len(failed)} dry_run={bool(args.dry_run)}")
    if failed:
        for issue in failed[:50]:
            print(f" - {issue}")
        if len(failed) > 50:
            print(f" - ... ({len(failed) - 50} more)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
