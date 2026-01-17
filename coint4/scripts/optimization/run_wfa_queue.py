#!/usr/bin/env python3
"""Run queued WFA configs with resume-friendly logging."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from coint2.ops.run_queue import (
    RunQueueEntry,
    load_run_queue,
    select_by_status,
    write_run_queue,
)


def _rotate_log(log_path: Path) -> None:
    if not log_path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rotated = log_path.with_name(f"{log_path.name}.stalled_{stamp}")
    log_path.rename(rotated)


def _run_entry(
    entry: RunQueueEntry, runner: Path, project_root: Path, rotate_logs: bool
) -> str:
    results_dir = project_root / entry.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run.log"
    if rotate_logs:
        _rotate_log(log_path)
    with log_path.open("w") as handle:
        cmd = [str(runner), entry.config_path, entry.results_dir]
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        handle.write(f"[run_wfa_queue] {timestamp} cmd: {shlex.join(cmd)}\n")
        handle.flush()
        process = subprocess.run(
            cmd,
            cwd=project_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    metrics_path = results_dir / "strategy_metrics.csv"
    if process.returncode == 0 and metrics_path.exists():
        return "completed"
    return "stalled"


def _update_status(
    queue_path: Path,
    entries: List[RunQueueEntry],
    entry: RunQueueEntry,
    status: str,
    lock: threading.Lock,
) -> None:
    with lock:
        entry.status = status
        write_run_queue(queue_path, entries)


def _run_queue(
    queue_path: Path,
    runner: Path,
    statuses: Iterable[str],
    parallel: int,
    rotate_logs: bool,
    project_root: Path,
    dry_run: bool,
) -> None:
    entries = load_run_queue(queue_path)
    selected = select_by_status(entries, statuses)
    if not selected:
        print(f"{queue_path}: nothing to run for {', '.join(statuses)}")
        return

    print(f"{queue_path}: queued {len(selected)} run(s)")
    for entry in selected:
        print(f" - {entry.status}: {entry.config_path} -> {entry.results_dir}")

    if dry_run:
        return

    lock = threading.Lock()
    for entry in selected:
        _update_status(queue_path, entries, entry, "running", lock)

    if parallel <= 1:
        for entry in selected:
            status = _run_entry(entry, runner, project_root, rotate_logs)
            _update_status(queue_path, entries, entry, status, lock)
        return

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(_run_entry, entry, runner, project_root, rotate_logs): entry
            for entry in selected
        }
        for future in as_completed(futures):
            entry = futures[future]
            status = future.result()
            _update_status(queue_path, entries, entry, status, lock)


def _resolve_queue_paths(queue_dir: Path, queue_paths: List[Path]) -> List[Path]:
    if queue_paths:
        return queue_paths
    if not queue_dir.exists():
        return []
    return sorted(queue_dir.rglob("run_queue.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WFA configs from queue")
    parser.add_argument(
        "--queue-dir",
        default="artifacts/wfa/aggregate",
        help="Directory to search for run_queue.csv files.",
    )
    parser.add_argument(
        "--queue",
        action="append",
        default=[],
        help="Explicit run_queue.csv path (repeatable).",
    )
    parser.add_argument(
        "--statuses",
        default="planned,stalled",
        help="Comma-separated statuses to run.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of concurrent runs (default: CPU count).",
    )
    parser.add_argument(
        "--runner",
        default="run_wfa_fullcpu.sh",
        help="Runner script path (relative to project root).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print queue only.")
    parser.add_argument(
        "--no-rotate-logs",
        action="store_true",
        help="Disable run.log rotation before relaunch.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    queue_dir = project_root / args.queue_dir
    runner = project_root / args.runner
    queue_paths = [project_root / path for path in args.queue]
    queue_paths = _resolve_queue_paths(queue_dir, queue_paths)
    if not queue_paths:
        print("No run_queue.csv files found.")
        return 1

    statuses = [status.strip() for status in args.statuses.split(",") if status.strip()]
    for queue_path in queue_paths:
        _run_queue(
            queue_path=queue_path,
            runner=runner,
            statuses=statuses,
            parallel=max(1, args.parallel),
            rotate_logs=not args.no_rotate_logs,
            project_root=project_root,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
