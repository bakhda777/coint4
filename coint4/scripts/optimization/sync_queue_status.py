#!/usr/bin/env python3
"""Sync run_queue.csv statuses from artifacts on disk.

Use-case: runs executed manually via run_wfa_fullcpu.sh won't update queue status
from "planned"/"running" to "completed", which makes rollups and ranking scripts
misleading. This tool updates queue rows to "completed" when metrics exist.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from coint2.ops.run_queue import load_run_queue, write_run_queue


def _resolve_queue_paths(queue_dir: Path, queue_paths: List[Path]) -> List[Path]:
    if queue_paths:
        return queue_paths
    if not queue_dir.exists():
        return []
    return sorted(queue_dir.rglob("run_queue.csv"))


def _parse_csv_list(value: str) -> List[str]:
    items = [part.strip() for part in value.split(",")]
    return [item for item in items if item]


def _should_update(status: str, allowed_statuses: Iterable[str]) -> bool:
    current = (status or "").strip().lower()
    if not current:
        return True
    return current in {s.lower() for s in allowed_statuses}


def _has_metrics(run_dir: Path, metrics_file: str) -> bool:
    path = run_dir / metrics_file
    try:
        return path.exists() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def _backup(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync run_queue.csv statuses to completed when metrics exist")
    parser.add_argument(
        "--queue-dir",
        default="artifacts/wfa/aggregate",
        help="Directory to search for run_queue.csv files (relative to project root).",
    )
    parser.add_argument(
        "--queue",
        action="append",
        default=[],
        help="Explicit run_queue.csv path (repeatable, relative to project root).",
    )
    parser.add_argument(
        "--metrics-file",
        default="strategy_metrics.csv",
        help="Completion sentinel inside results_dir (default: strategy_metrics.csv).",
    )
    parser.add_argument(
        "--from-statuses",
        default="planned,running,stalled,active",
        help="Only update rows whose current status is in this list (comma-separated). Empty status is always updatable.",
    )
    parser.add_argument(
        "--to-status",
        default="completed",
        help="Status to set when metrics are present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes, do not write files.")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a timestamped .bak copy next to the queue file before editing.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    queue_dir = project_root / args.queue_dir
    queue_paths = [project_root / path for path in args.queue]
    queue_paths = _resolve_queue_paths(queue_dir, queue_paths)
    if not queue_paths:
        print("No run_queue.csv files found.")
        return 1

    allowed = _parse_csv_list(args.from_statuses)
    updated_total = 0
    for queue_path in queue_paths:
        if not queue_path.exists():
            print(f"Missing queue: {queue_path}")
            continue

        entries = load_run_queue(queue_path)
        changed = 0
        completed = 0
        missing = 0
        skipped = 0
        for entry in entries:
            run_dir = project_root / entry.results_dir
            if not _has_metrics(run_dir, args.metrics_file):
                missing += 1
                continue
            completed += 1
            if entry.status.strip().lower() == args.to_status.strip().lower():
                continue
            if not _should_update(entry.status, allowed):
                skipped += 1
                continue
            entry.status = args.to_status
            changed += 1

        if changed:
            updated_total += changed
            if args.dry_run:
                print(f"{queue_path}: would update {changed}/{len(entries)} -> {args.to_status} (metrics_present={completed}, missing={missing}, skipped={skipped})")
            else:
                if args.backup:
                    backup_path = _backup(queue_path)
                    print(f"{queue_path}: backup -> {backup_path}")
                write_run_queue(queue_path, entries)
                print(f"{queue_path}: updated {changed}/{len(entries)} -> {args.to_status} (metrics_present={completed}, missing={missing}, skipped={skipped})")
        else:
            print(f"{queue_path}: no changes (metrics_present={completed}, missing={missing}, skipped={skipped})")

    if updated_total and args.dry_run:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

