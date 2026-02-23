#!/usr/bin/env python3
"""Build a rollup index for WFA runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from coint2.ops.run_queue import load_run_queue, write_run_queue

from coint2.ops.run_index import (
    build_run_index,
    write_run_index_csv,
    write_run_index_json,
)


def _fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{int(digits)}f}"


def _fmt_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.{int(digits)}f}%"


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
    lines.append(
        "- `psr` / `dsr` are computed from run returns (`equity_curve.csv` fallback: `daily_pnl.csv`); `dsr_trials` is inferred from run_group size."
    )
    lines.append(
        "- `tail_loss_*` fields are derived from `trade_statistics.csv` and show net tail-loss concentration by pair and WF period."
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
    tail_loss_entries = [
        e
        for e in metrics_entries
        if float(e.tail_loss_pair_total_abs or 0.0) > 0.0
        or float(e.tail_loss_period_total_abs or 0.0) > 0.0
    ]
    top_tail_loss = sorted(
        tail_loss_entries,
        key=lambda e: max(
            float(e.tail_loss_worst_pair_share or 0.0),
            float(e.tail_loss_worst_period_share or 0.0),
        ),
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
    lines.append("## Tail-Loss Offenders")
    if not top_tail_loss:
        lines.append("- (no runs with metrics)")
    else:
        for entry in top_tail_loss:
            pair_name = entry.tail_loss_worst_pair or "-"
            period_name = entry.tail_loss_worst_period or "-"
            lines.append(
                "- {run_id} | pair={pair} pnl={pair_pnl} share={pair_share} | period={period} pnl={period_pnl} share={period_share} | {path}".format(
                    run_id=entry.run_id,
                    pair=pair_name,
                    pair_pnl=_fmt_float(entry.tail_loss_worst_pair_pnl, digits=2),
                    pair_share=_fmt_pct(entry.tail_loss_worst_pair_share, digits=1),
                    period=period_name,
                    period_pnl=_fmt_float(entry.tail_loss_worst_period_pnl, digits=2),
                    period_share=_fmt_pct(entry.tail_loss_worst_period_share, digits=1),
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


def _resolve_results_dir(raw: str, *, project_root: Path) -> Path:
    path = Path(str(raw or "").strip())
    if path.is_absolute():
        return path
    return project_root / path


def _sync_queue_status_from_metrics(
    *,
    queue_paths: Iterable[Path],
    project_root: Path,
    metrics_file: str,
    updatable_statuses: Iterable[str],
) -> int:
    updated_rows = 0
    updatable = {str(status).strip().lower() for status in updatable_statuses if str(status).strip()}
    for queue_path in queue_paths:
        if not queue_path.exists():
            continue
        entries = load_run_queue(queue_path)
        changed = 0
        for entry in entries:
            current = str(entry.status or "").strip().lower()
            if current and current not in updatable:
                continue
            run_dir = _resolve_results_dir(entry.results_dir, project_root=project_root)
            metrics_path = run_dir / str(metrics_file)
            try:
                has_metrics = metrics_path.exists() and metrics_path.stat().st_size > 0
            except OSError:
                has_metrics = False
            if not has_metrics:
                continue
            if current == "completed":
                continue
            entry.status = "completed"
            changed += 1

        if changed:
            write_run_queue(queue_path, entries)
            updated_rows += changed
            print(f"{queue_path}: auto-synced {changed} row(s) -> completed")
    return updated_rows


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
    parser.add_argument(
        "--metrics-file",
        default="strategy_metrics.csv",
        help="Completion sentinel filename inside results_dir for auto queue sync.",
    )
    parser.add_argument(
        "--compute-legacy-coverage",
        action="store_true",
        default=False,
        help=(
            "Compute coverage_* fields for legacy runs missing them by reading daily_pnl.csv + config dates. "
            "This can be slow across thousands of runs."
        ),
    )
    parser.add_argument(
        "--auto-sync-status",
        dest="auto_sync_status",
        action="store_true",
        default=True,
        help="Auto-sync queue rows to completed when metrics file exists (default: enabled).",
    )
    parser.add_argument(
        "--no-auto-sync-status",
        dest="auto_sync_status",
        action="store_false",
        help="Disable auto queue status sync before rollup build.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N runs to list.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    runs_dir = project_root / args.runs_dir
    queue_dir = project_root / args.queue_dir
    output_dir = project_root / args.output_dir
    queue_paths = [project_root / path for path in args.queue]
    queue_paths = _resolve_queue_paths(queue_dir, queue_paths)

    if args.auto_sync_status and queue_paths:
        _sync_queue_status_from_metrics(
            queue_paths=queue_paths,
            project_root=project_root,
            metrics_file=str(args.metrics_file),
            updatable_statuses=("planned", "running", "stalled", "active"),
        )

    entries = build_run_index(
        runs_dir,
        queue_paths,
        project_root,
        compute_legacy_coverage=bool(args.compute_legacy_coverage),
    )
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
