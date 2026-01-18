#!/usr/bin/env python3
"""Build a rollup index for WFA runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from coint2.ops.run_index import (
    build_run_index,
    write_run_index_csv,
    write_run_index_json,
)


def _render_summary(entries, top_n: int) -> str:
    lines = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines.append("# WFA run index")
    lines.append("")
    lines.append(f"Generated at: {timestamp}")
    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- `sharpe_ratio_abs` is recomputed from `equity_curve.csv` with inferred bar frequency (periods/year = 365 * periods/day)."
    )
    lines.append(
        "- `sharpe_ratio_abs_raw` is the value stored in `strategy_metrics.csv` (legacy runs may be under-annualized)."
    )
    lines.append("")

    metrics_entries = [e for e in entries if e.metrics_present]
    top_sharpe = sorted(
        metrics_entries,
        key=lambda e: (e.sharpe_ratio_abs or float("-inf")),
        reverse=True,
    )[:top_n]
    top_pnl = sorted(
        metrics_entries,
        key=lambda e: (e.total_pnl or float("-inf")),
        reverse=True,
    )[:top_n]

    lines.append("## Top Sharpe")
    if not top_sharpe:
        lines.append("- (no runs with metrics)")
    else:
        for entry in top_sharpe:
            lines.append(
                "- {run_id} | sharpe={sharpe:.4f} pnl={pnl:.2f} dd={dd:.2f} | {path}".format(
                    run_id=entry.run_id,
                    sharpe=entry.sharpe_ratio_abs or 0.0,
                    pnl=entry.total_pnl or 0.0,
                    dd=entry.max_drawdown_abs or 0.0,
                    path=entry.results_dir,
                )
            )

    lines.append("")
    lines.append("## Top PnL")
    if not top_pnl:
        lines.append("- (no runs with metrics)")
    else:
        for entry in top_pnl:
            lines.append(
                "- {run_id} | sharpe={sharpe:.4f} pnl={pnl:.2f} dd={dd:.2f} | {path}".format(
                    run_id=entry.run_id,
                    sharpe=entry.sharpe_ratio_abs or 0.0,
                    pnl=entry.total_pnl or 0.0,
                    dd=entry.max_drawdown_abs or 0.0,
                    path=entry.results_dir,
                )
            )

    lines.append("")
    return "\n".join(lines)


def _resolve_queue_paths(queue_dir: Path, queue_paths: List[Path]) -> List[Path]:
    if queue_paths:
        return queue_paths
    if not queue_dir.exists():
        return []
    return sorted(queue_dir.rglob("run_queue.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build WFA run index rollup")
    parser.add_argument(
        "--runs-dir",
        default="artifacts/wfa/runs",
        help="Root directory with run artifacts.",
    )
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
        "--output-dir",
        default="artifacts/wfa/aggregate/rollup",
        help="Output directory for run_index files.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N runs to list.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    runs_dir = project_root / args.runs_dir
    queue_dir = project_root / args.queue_dir
    output_dir = project_root / args.output_dir
    queue_paths = [project_root / path for path in args.queue]
    queue_paths = _resolve_queue_paths(queue_dir, queue_paths)

    entries = build_run_index(runs_dir, queue_paths, project_root)
    entries = sorted(entries, key=lambda e: (e.run_group, e.run_id))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_index_csv(output_dir / "run_index.csv", entries)
    write_run_index_json(output_dir / "run_index.json", entries)
    summary = _render_summary(entries, args.top_n)
    (output_dir / "run_index.md").write_text(summary)

    print(f"Run index entries: {len(entries)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
