import csv
import os
import subprocess
import sys
from pathlib import Path


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sharpe_ratio_abs",
                "total_pnl",
                "max_drawdown_abs",
                "total_trades",
                "total_pairs_traded",
            ]
        )
        writer.writerow([1.25, 123.0, -12.0, 111, 11])


def _write_queue(path: Path, results_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_path", "results_dir", "status"])
        writer.writerow(["configs/manual_run.yaml", str(results_dir), "planned"])


def _read_queue_status(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    return str(rows[0].get("status") or "").strip()


def test_build_run_index_auto_syncs_completed_status(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts" / "optimization" / "build_run_index.py"
    assert script.exists()

    runs_dir = tmp_path / "artifacts" / "wfa" / "runs"
    run_dir = runs_dir / "group_manual" / "run_manual"
    _write_metrics(run_dir / "strategy_metrics.csv")

    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "group_manual" / "run_queue.csv"
    _write_queue(queue_path, run_dir)

    output_dir = tmp_path / "artifacts" / "wfa" / "aggregate" / "rollup"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_root / "src")

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runs-dir",
            str(runs_dir),
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

    assert _read_queue_status(queue_path) == "completed"

    run_index_csv = output_dir / "run_index.csv"
    assert run_index_csv.exists()
    with run_index_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run_manual"
    assert row["status"] == "completed"
