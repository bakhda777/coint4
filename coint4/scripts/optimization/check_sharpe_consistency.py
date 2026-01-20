#!/usr/bin/env python3
"""Check Sharpe consistency between strategy_metrics.csv and equity_curve.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional

from coint2.ops.run_index import _compute_sharpe_from_equity_curve


def _iter_queue_paths(queue_dir: Path, queue_paths: List[Path]) -> List[Path]:
    if queue_paths:
        return queue_paths
    if not queue_dir.exists():
        return []
    return sorted(queue_dir.rglob("run_queue.csv"))


def _load_run_dirs_from_queue(queue_path: Path, project_root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    with queue_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            results_dir = (row.get("results_dir") or "").strip()
            if not results_dir:
                continue
            run_dirs.append(project_root / results_dir)
    return run_dirs


def _read_metrics_sharpe(metrics_path: Path) -> Optional[float]:
    if not metrics_path.exists():
        return None
    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("sharpe_ratio_abs")
            if value is None or str(value).strip() == "":
                return None
            try:
                return float(value)
            except ValueError:
                return None
    return None


def check_run_dir(run_dir: Path, *, tolerance: float) -> Optional[str]:
    metrics_path = run_dir / "strategy_metrics.csv"
    raw_sharpe = _read_metrics_sharpe(metrics_path)
    if raw_sharpe is None:
        return f"{run_dir}: missing sharpe_ratio_abs in {metrics_path}"

    computed = _compute_sharpe_from_equity_curve(run_dir)
    if computed is None:
        return f"{run_dir}: missing equity_curve.csv for computed sharpe"

    diff = abs(raw_sharpe - computed)
    if diff > tolerance:
        return f"{run_dir}: sharpe mismatch raw={raw_sharpe:.6f} computed={computed:.6f} diff={diff:.6f}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Sharpe consistency")
    parser.add_argument("--run-dir", action="append", default=[], help="Run dir to check (repeatable).")
    parser.add_argument("--queue", action="append", default=[], help="run_queue.csv path (repeatable).")
    parser.add_argument(
        "--queue-dir",
        default="artifacts/wfa/aggregate",
        help="Directory to search for run_queue.csv files.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Allowed Sharpe diff.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    run_dirs = [project_root / path for path in args.run_dir]
    queue_paths = [project_root / path for path in args.queue]

    queue_dir = project_root / args.queue_dir
    for queue_path in _iter_queue_paths(queue_dir, queue_paths):
        run_dirs.extend(_load_run_dirs_from_queue(queue_path, project_root))

    seen = set()
    unique_run_dirs = []
    for run_dir in run_dirs:
        if run_dir in seen:
            continue
        seen.add(run_dir)
        unique_run_dirs.append(run_dir)

    if not unique_run_dirs:
        print("No run directories found.")
        return 1

    mismatches: List[str] = []
    for run_dir in unique_run_dirs:
        issue = check_run_dir(run_dir, tolerance=args.tolerance)
        if issue:
            mismatches.append(issue)

    if mismatches:
        print("Sharpe consistency check failed:")
        for issue in mismatches:
            print(f" - {issue}")
        return 1

    print(f"Sharpe consistency OK ({len(unique_run_dirs)} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
