#!/usr/bin/env python3
"""Extract a lightweight live system snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def _tail_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return lines[-n:]


def extract_snapshot(logs_dir: Path, metrics_dir: Path, logs: int, trades: int) -> Path:
    """Create a markdown snapshot and return its path."""
    artifacts_dir = Path("artifacts/live")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(logs_dir.glob("*.jsonl"))
    log_lines = []
    for log_file in log_files:
        log_lines.extend(_tail_lines(log_file, logs))

    snapshot_path = artifacts_dir / f"SNAPSHOT_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    content = [
        "# Live System Snapshot",
        "",
        "## System Status Overview",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Logs captured: {min(len(log_lines), logs)}",
        f"- Trades requested: {trades}",
        "",
        "## Performance Metrics",
        "",
        f"- Metrics directory: {metrics_dir}",
        "",
        "## Recent Logs",
        "",
        "```",
        *log_lines[:logs],
        "```",
        "",
    ]
    snapshot_path.write_text("\n".join(content))
    return snapshot_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract live system snapshot")
    parser.add_argument("--logs", type=int, default=50, help="Number of log lines")
    parser.add_argument("--trades", type=int, default=10, help="Number of trades")
    parser.add_argument(
        "--logs-dir", default="artifacts/live/logs", help="Directory with log files"
    )
    parser.add_argument(
        "--metrics-dir",
        default="artifacts/live/metrics",
        help="Directory with metrics files",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    metrics_dir = Path(args.metrics_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = extract_snapshot(logs_dir, metrics_dir, args.logs, args.trades)
    print(f"Snapshot saved: {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
