"""Tests for WFA run index builder."""

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

from coint2.ops.run_index import build_run_index


def _write_metrics(path: Path, *, sharpe: float = 0.5) -> None:
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
        writer.writerow([sharpe, 100.0, -20.0, 200, 50, 10.0, -5.0, 2.0])


def _write_daily_pnl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "pnl"])
        writer.writerows(rows)


def _write_equity_curve(path: Path, equities: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, 0, 0, 0)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "equity"])
        for idx, equity in enumerate(equities):
            ts = start + timedelta(hours=idx)
            writer.writerow([ts.isoformat(), equity])


def _write_queue(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_path", "results_dir", "status"])
        writer.writerows(rows)


def _write_trade_statistics(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pair",
                "period",
                "total_pnl",
                "total_costs",
                "trade_count",
                "avg_pnl_per_trade",
                "win_days",
                "lose_days",
                "total_days",
                "max_daily_gain",
                "max_daily_loss",
            ]
        )
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


def test_build_run_index_computes_psr_dsr_when_equity_exists(tmp_path: Path) -> None:
    runs_dir = tmp_path / "artifacts" / "wfa" / "runs"
    run_a_dir = runs_dir / "group_psr" / "run_a"
    run_b_dir = runs_dir / "group_psr" / "run_b"

    _write_metrics(run_a_dir / "strategy_metrics.csv", sharpe=1.0)
    _write_metrics(run_b_dir / "strategy_metrics.csv", sharpe=0.8)
    _write_equity_curve(run_a_dir / "equity_curve.csv", [100.0, 101.0, 102.5, 103.0, 104.0, 105.0])
    _write_equity_curve(run_b_dir / "equity_curve.csv", [100.0, 100.4, 100.8, 101.4, 102.0, 102.4])

    queue_path = (
        tmp_path / "artifacts" / "wfa" / "aggregate" / "group_psr" / "run_queue.csv"
    )
    _write_queue(
        queue_path,
        [
            ("configs/run_a.yaml", "artifacts/wfa/runs/group_psr/run_a", "completed"),
            ("configs/run_b.yaml", "artifacts/wfa/runs/group_psr/run_b", "completed"),
        ],
    )

    entries = build_run_index(runs_dir, [queue_path], tmp_path)
    assert len(entries) == 2
    for entry in entries:
        assert entry.psr is not None
        assert 0.0 <= float(entry.psr) <= 1.0
        assert entry.dsr is not None
        assert entry.dsr_trials == 2


def test_build_run_index_computes_tail_loss_concentration(tmp_path: Path) -> None:
    runs_dir = tmp_path / "artifacts" / "wfa" / "runs"
    run_dir = runs_dir / "group_tail" / "run_tail"

    _write_metrics(run_dir / "strategy_metrics.csv", sharpe=0.7)
    _write_trade_statistics(
        run_dir / "trade_statistics.csv",
        [
            ("AAAUSDT-BBBUSDT", "03/01-03/31", -8.0, 0.5, 4, -2.0, 2, 2, 4, 1.0, -3.0),
            ("AAAUSDT-BBBUSDT", "04/01-04/30", 2.0, 0.3, 3, 0.66, 2, 1, 3, 1.2, -1.0),
            ("CCCUSDT-DDDUSDT", "03/01-03/31", -3.0, 0.2, 2, -1.5, 1, 1, 2, 0.8, -2.2),
            ("EEEUSDT-FFFUSDT", "04/01-04/30", 1.0, 0.1, 1, 1.0, 1, 0, 1, 1.5, -0.4),
        ],
    )

    queue_path = (
        tmp_path / "artifacts" / "wfa" / "aggregate" / "group_tail" / "run_queue.csv"
    )
    _write_queue(
        queue_path,
        [
            ("configs/run_tail.yaml", "artifacts/wfa/runs/group_tail/run_tail", "completed"),
        ],
    )

    entries = build_run_index(runs_dir, [queue_path], tmp_path)
    assert len(entries) == 1
    entry = entries[0]

    assert entry.tail_loss_pair_total_abs == 9.0
    assert entry.tail_loss_worst_pair == "AAAUSDT-BBBUSDT"
    assert entry.tail_loss_worst_pair_pnl == -6.0
    assert entry.tail_loss_worst_pair_share is not None
    assert math.isclose(float(entry.tail_loss_worst_pair_share), 6.0 / 9.0, rel_tol=1e-9)

    assert entry.tail_loss_period_total_abs == 11.0
    assert entry.tail_loss_worst_period == "03/01-03/31"
    assert entry.tail_loss_worst_period_pnl == -11.0
    assert entry.tail_loss_worst_period_share == 1.0


def test_build_run_index_legacy_coverage_fallback_clips_to_config_dates(tmp_path: Path) -> None:
    runs_dir = tmp_path / "artifacts" / "wfa" / "runs"
    run_dir = runs_dir / "group_cov" / "run_cov"

    _write_metrics(run_dir / "strategy_metrics.csv", sharpe=0.7)  # legacy: no coverage_* fields
    _write_daily_pnl(
        run_dir / "daily_pnl.csv",
        [
            ("2024-01-01", 0.0),  # outside window
            ("2024-01-02", 0.0),  # inside window, zero
            ("2024-01-03", 1.0),  # inside window, non-zero
            ("2024-01-04", 0.0),  # inside window, zero
            ("2024-01-05", 0.0),  # outside window
        ],
    )

    cfg_path = tmp_path / "configs" / "run_cov.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "\n".join(
            [
                "walk_forward:",
                "  start_date: '2024-01-02'",
                "  end_date: '2024-01-04'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "group_cov" / "run_queue.csv"
    _write_queue(queue_path, [("configs/run_cov.yaml", "artifacts/wfa/runs/group_cov/run_cov", "completed")])

    # Default: legacy coverage fallback is disabled for performance.
    entries = build_run_index(runs_dir, [queue_path], tmp_path)
    assert len(entries) == 1
    assert entries[0].coverage_ratio is None

    # Opt-in: compute legacy coverage, but clip observed days to the configured test window.
    entries = build_run_index(runs_dir, [queue_path], tmp_path, compute_legacy_coverage=True)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.expected_test_days == 3.0
    assert entry.observed_test_days == 3.0
    assert entry.coverage_ratio == 1.0
    assert entry.zero_pnl_days == 2.0
    assert math.isclose(float(entry.zero_pnl_days_pct or 0.0), 2.0 / 3.0, rel_tol=1e-9)
    assert entry.missing_test_days == 0.0
