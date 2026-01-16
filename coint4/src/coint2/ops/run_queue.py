"""Run queue helpers for WFA automation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class RunQueueEntry:
    """Single run queue entry."""

    config_path: str
    results_dir: str
    status: str


def load_run_queue(path: Path) -> List[RunQueueEntry]:
    """Load run queue entries from CSV."""
    entries: List[RunQueueEntry] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                RunQueueEntry(
                    config_path=(row.get("config_path") or "").strip(),
                    results_dir=(row.get("results_dir") or "").strip(),
                    status=(row.get("status") or "").strip(),
                )
            )
    return entries


def write_run_queue(path: Path, entries: Sequence[RunQueueEntry]) -> None:
    """Write run queue entries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["config_path", "results_dir", "status"]
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "config_path": entry.config_path,
                    "results_dir": entry.results_dir,
                    "status": entry.status,
                }
            )


def select_by_status(
    entries: Iterable[RunQueueEntry], statuses: Iterable[str]
) -> List[RunQueueEntry]:
    """Select entries matching requested statuses (case-insensitive)."""
    wanted = {status.lower() for status in statuses}
    return [entry for entry in entries if entry.status.lower() in wanted]

