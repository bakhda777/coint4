#!/usr/bin/env python3
"""Build a JSON manifest of WFA runs from run_queue.csv files (lightweight)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _discover_queues(aggregate_root: Path) -> List[Path]:
    return sorted(aggregate_root.rglob("run_queue.csv"))


def _read_queue(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-root",
        type=Path,
        default=None,
        help="Root with aggregate/*/run_queue.csv (default: <app_root>/artifacts/wfa/aggregate)",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Include completed runs (default: only planned, stalled)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <app_root>/outputs/wfa_manifest_<ts>.json)",
    )
    args = parser.parse_args()

    app_root = _app_root()
    aggregate_root = args.aggregate_root or (app_root / "artifacts" / "wfa" / "aggregate")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output or (app_root / "outputs" / f"wfa_manifest_{stamp}.json")

    statuses = {"planned", "stalled"}
    if args.include_completed:
        statuses |= {"completed"}

    entries: List[Dict[str, Any]] = []
    for queue in _discover_queues(aggregate_root):
        for row in _read_queue(queue):
            status = row.get("status", "").strip()
            if status not in statuses:
                continue
            config_path = row.get("config_path", "").strip()
            results_dir = row.get("results_dir", "").strip()
            if not config_path or not results_dir:
                continue
            entries.append(
                {
                    "queue": str(queue.relative_to(app_root)),
                    "config_path": config_path,
                    "results_dir": results_dir,
                    "status": status,
                }
            )

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "aggregateRoot": str(aggregate_root),
        "entries": entries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[build_wfa_manifest] entries={len(entries)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

