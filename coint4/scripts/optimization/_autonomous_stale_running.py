#!/usr/bin/env python3
"""Detect stale `running` rows and mark them as `stalled`.

Usage:
  python _autonomous_stale_running.py --queue /path/to/run_queue.csv --stale-sec 900
  python _autonomous_stale_running.py --queue /path/to/run_queue.csv --stale-sec 900 --root /opt/coint4/coint4
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import time
from typing import Optional


def _safe_mtime_candidates(run_dir: Optional[pathlib.Path]) -> list[float]:
    if run_dir is None or not run_dir.exists():
        return []

    mtimes: list[float] = []
    targets = [
        "strategy_metrics.csv",
        "equity_curve.csv",
        "canonical_metrics.json",
        "progress.json",
        "status.json",
    ]
    for name in targets:
        candidate = run_dir / name
        if candidate.exists():
            try:
                mtimes.append(candidate.stat().st_mtime)
            except OSError:
                pass

    for patt in ("*.log", "*.json", "*.csv"):
        for candidate in run_dir.glob(patt):
            if not candidate.is_file():
                continue
            try:
                mtimes.append(candidate.stat().st_mtime)
            except OSError:
                pass
    return mtimes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--stale-sec", type=int, default=900)
    parser.add_argument("--root", default="")
    args = parser.parse_args()

    queue_path = pathlib.Path(args.queue)
    if args.root:
        root = pathlib.Path(args.root)
        if not queue_path.is_absolute():
            queue_path = root / queue_path

    if not queue_path.exists():
        print(0)
        return 0

    rows = list(csv.DictReader(queue_path.open(newline="")))
    now = time.time()
    stale_threshold = max(60, int(args.stale_sec))
    changed = 0

    for row in rows:
        status = (row.get("status") or "").strip().lower()
        if status != "running":
            continue

        results_dir = (row.get("results_dir") or "").strip()
        run_dir = pathlib.Path(results_dir)
        if results_dir and not run_dir.is_absolute() and args.root:
            run_dir = pathlib.Path(args.root) / run_dir
        elif results_dir and not run_dir.is_absolute():
            run_dir = queue_path.parent / run_dir

        mtimes = _safe_mtime_candidates(run_dir)

        if not mtimes:
            row["status"] = "stalled"
            changed += 1
            continue

        age_sec = int(max(0.0, now - max(mtimes)))
        if age_sec >= stale_threshold:
            row["status"] = "stalled"
            changed += 1

    if changed > 0 and rows:
        with queue_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    print(changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
