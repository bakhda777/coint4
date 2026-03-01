import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _write_queue(path: Path, *, config_path: Path, results_dir: Path, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_path", "results_dir", "status"])
        writer.writerow([str(config_path), str(results_dir), status])


def test_build_run_index_merges_existing_metrics_when_artifacts_missing(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts" / "optimization" / "build_run_index.py"
    assert script.exists()

    # Simulate a cleaned-up results_dir: queue says completed, but there is no strategy_metrics.csv on disk.
    config_path = tmp_path / "configs" / "sample.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    results_dir = tmp_path / "artifacts" / "wfa" / "runs" / "group_merge" / "run_merge"
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "group_merge" / "run_queue.csv"
    _write_queue(queue_path, config_path=config_path, results_dir=results_dir, status="completed")

    output_dir = tmp_path / "artifacts" / "wfa" / "aggregate" / "rollup"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed existing rollup with metrics for this run_id/run_group.
    seeded = [
        {
            "run_id": "run_merge",
            "run_group": "",
            "results_dir": str(results_dir),
            "metrics_path": str(results_dir / "strategy_metrics.csv"),
            "config_path": str(config_path),
            "universe_path": "",
            "universe_tag": "",
            "universe_pairs_count": None,
            "denylist_count": 0,
            "denylist_hash": "",
            "status": "completed",
            "metrics_present": True,
            "sharpe_ratio_abs": 1.23,
            "sharpe_ratio_abs_raw": 1.23,
            "sharpe_ratio_on_returns": None,
            "psr": None,
            "dsr": None,
            "dsr_trials": None,
            "total_pnl": 42.0,
            "max_drawdown_abs": -3.0,
            "max_drawdown_on_equity": 0.12,
            "total_trades": 200.0,
            "total_pairs_traded": 20.0,
            "total_costs": 0.0,
            "total_days": 100.0,
            "expected_test_days": None,
            "observed_test_days": None,
            "coverage_ratio": None,
            "zero_pnl_days": None,
            "zero_pnl_days_pct": None,
            "missing_test_days": None,
            "volatility": None,
            "win_rate": None,
            "best_pair_pnl": None,
            "worst_pair_pnl": None,
            "avg_pnl_per_pair": None,
            "tail_loss_pair_total_abs": None,
            "tail_loss_worst_pair": "",
            "tail_loss_worst_pair_pnl": None,
            "tail_loss_worst_pair_share": None,
            "tail_loss_period_total_abs": None,
            "tail_loss_worst_period": "",
            "tail_loss_worst_period_pnl": None,
            "tail_loss_worst_period_share": None,
        }
    ]
    (output_dir / "run_index.json").write_text(json.dumps(seeded, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_root / "src")

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runs-dir",
            str(tmp_path / "artifacts" / "wfa" / "runs"),
            "--queue",
            str(queue_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=app_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout

    run_index_csv = output_dir / "run_index.csv"
    assert run_index_csv.exists()
    with run_index_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run_merge"
    assert row["run_group"] == ""
    assert row["status"] == "completed"
    assert row["metrics_present"] == "True"
    assert float(row["sharpe_ratio_abs"]) == 1.23
    assert float(row["total_pnl"]) == 42.0
