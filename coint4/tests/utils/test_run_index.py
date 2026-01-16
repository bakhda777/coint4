"""Tests for WFA run index builder."""

import csv
from pathlib import Path

from coint2.ops.run_index import build_run_index


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sharpe_ratio_abs",
                "total_pnl",
                "max_drawdown_abs",
                "total_trades",
                "total_pairs_traded",
                "best_pair_pnl",
                "worst_pair_pnl",
                "avg_pnl_per_pair",
            ]
        )
        writer.writerow([0.5, 100.0, -20.0, 200, 50, 10.0, -5.0, 2.0])


def _write_queue(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_path", "results_dir", "status"])
        writer.writerows(rows)


def test_build_run_index_merges_queue_and_metrics(tmp_path: Path) -> None:
    runs_dir = tmp_path / "artifacts" / "wfa" / "runs"
    run_a_dir = runs_dir / "group_a" / "run_a"
    run_b_dir = runs_dir / "group_a" / "run_b"

    _write_metrics(run_a_dir / "strategy_metrics.csv")

    queue_path = (
        tmp_path / "artifacts" / "wfa" / "aggregate" / "group_a" / "run_queue.csv"
    )
    _write_queue(
        queue_path,
        [
            ("configs/run_a.yaml", "artifacts/wfa/runs/group_a/run_a", "stalled"),
            ("configs/run_b.yaml", "artifacts/wfa/runs/group_a/run_b", "planned"),
        ],
    )

    entries = build_run_index(runs_dir, [queue_path], tmp_path)

    assert len(entries) == 2

    entry_a = next(entry for entry in entries if entry.run_id == "run_a")
    assert entry_a.metrics_present is True
    assert entry_a.status == "stalled"
    assert entry_a.sharpe_ratio_abs == 0.5
    assert entry_a.total_pnl == 100.0

    entry_b = next(entry for entry in entries if entry.run_id == "run_b")
    assert entry_b.metrics_present is False
    assert entry_b.status == "planned"
    assert entry_b.sharpe_ratio_abs is None

