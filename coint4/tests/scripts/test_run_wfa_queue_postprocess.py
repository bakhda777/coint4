import csv
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def _write_queue_csv(path: Path, config_path: Path, results_dir: Path, status: str = "planned") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_path", "results_dir", "status"])
        writer.writerow([str(config_path), str(results_dir), status])


def _write_runner_script(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'config_path="$1"',
                'results_dir="$2"',
                'mkdir -p "$results_dir"',
                "cat > \"$results_dir/strategy_metrics.csv\" <<'EOF'",
                "sharpe_ratio_abs,total_pnl,max_drawdown_abs,total_trades,total_pairs_traded",
                "1.2,42.0,-3.5,200,20",
                "EOF",
                "cat > \"$results_dir/equity_curve.csv\" <<'EOF'",
                ",Equity",
                "2026-01-01 00:00:00,100.0",
                "2026-01-01 00:15:00,101.0",
                "2026-01-01 00:30:00,102.0",
                "EOF",
                'echo "runner config: $config_path" > "$results_dir/runner.log"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _read_queue_status(queue_path: Path) -> str:
    with queue_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    return str(rows[0].get("status") or "").strip()


def test_run_wfa_queue_postprocess_creates_run_artifacts_and_rollup(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/run_wfa_queue.py"
    assert script.exists()

    config_path = tmp_path / "configs" / "sample.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_text = "walk_forward:\n  max_steps: 5\nbacktest:\n  bar_minutes: 15\n"
    config_path.write_text(config_text, encoding="utf-8")

    results_dir = tmp_path / "artifacts" / "wfa" / "runs" / "group_x" / "run_x"
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "group_x" / "run_queue.csv"
    _write_queue_csv(queue_path, config_path, results_dir, status="planned")

    runner_path = tmp_path / "runner.sh"
    _write_runner_script(runner_path)

    rollup_output_dir = tmp_path / "artifacts" / "wfa" / "aggregate" / "rollup"
    rollup_queue_dir = tmp_path / "artifacts" / "wfa" / "aggregate"
    rollup_runs_dir = tmp_path / "artifacts" / "wfa" / "runs"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_root / "src")
    env["ALLOW_HEAVY_RUN"] = "1"
    env["HEAVY_HOSTNAME_ALLOWLIST"] = f"{socket.gethostname()},127.0.0.1,localhost"
    env["HEAVY_MIN_CPU"] = "1"
    env["HEAVY_MIN_RAM_GB"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--queue",
            str(queue_path),
            "--runner",
            str(runner_path),
            "--parallel",
            "1",
            "--postprocess",
            "true",
            "--rollup-output-dir",
            str(rollup_output_dir),
            "--rollup-queue-dir",
            str(rollup_queue_dir),
            "--rollup-runs-dir",
            str(rollup_runs_dir),
        ],
        cwd=app_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout

    assert _read_queue_status(queue_path) == "completed"

    snapshot_path = results_dir / "config_snapshot.yaml"
    commit_path = results_dir / "git_commit.txt"
    canonical_path = results_dir / "canonical_metrics.json"
    assert snapshot_path.exists()
    assert commit_path.exists()
    assert canonical_path.exists()

    assert snapshot_path.read_text(encoding="utf-8") == config_text
    commit_value = commit_path.read_text(encoding="utf-8").strip()
    assert commit_value

    canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    assert isinstance(canonical_payload, dict)
    assert "metrics" in canonical_payload
    assert canonical_payload["metrics"]["canonical_sharpe"] is not None
    assert canonical_payload["metrics"]["canonical_pnl_abs"] is not None
    assert canonical_payload["metrics"]["canonical_max_drawdown_abs"] is not None

    assert (rollup_output_dir / "run_index.csv").exists()
    assert (rollup_output_dir / "run_index.json").exists()
    assert (rollup_output_dir / "run_index.md").exists()


def test_run_wfa_queue_postprocess_can_be_disabled(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/run_wfa_queue.py"
    assert script.exists()

    config_path = tmp_path / "configs" / "sample.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    results_dir = tmp_path / "artifacts" / "wfa" / "runs" / "group_y" / "run_y"
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "group_y" / "run_queue.csv"
    _write_queue_csv(queue_path, config_path, results_dir, status="planned")

    runner_path = tmp_path / "runner.sh"
    _write_runner_script(runner_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_root / "src")
    env["ALLOW_HEAVY_RUN"] = "1"
    env["HEAVY_HOSTNAME_ALLOWLIST"] = f"{socket.gethostname()},127.0.0.1,localhost"
    env["HEAVY_MIN_CPU"] = "1"
    env["HEAVY_MIN_RAM_GB"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--queue",
            str(queue_path),
            "--runner",
            str(runner_path),
            "--parallel",
            "1",
            "--postprocess",
            "false",
        ],
        cwd=app_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout
    assert _read_queue_status(queue_path) == "completed"

    assert not (results_dir / "config_snapshot.yaml").exists()
    assert not (results_dir / "git_commit.txt").exists()
    assert not (results_dir / "canonical_metrics.json").exists()


def test_run_wfa_queue_rebuilds_rollup_on_queue_finish(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/run_wfa_queue.py"
    assert script.exists()

    config_path = tmp_path / "configs" / "sample.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    results_dir = tmp_path / "artifacts" / "wfa" / "runs" / "group_z" / "run_z"
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "group_z" / "run_queue.csv"
    _write_queue_csv(queue_path, config_path, results_dir, status="planned")

    runner_path = tmp_path / "runner.sh"
    _write_runner_script(runner_path)

    rollup_output_dir = tmp_path / "artifacts" / "wfa" / "aggregate" / "rollup"
    rollup_output_dir.mkdir(parents=True, exist_ok=True)
    stale_rollup = rollup_output_dir / "run_index.csv"
    stale_rollup.write_text("stale\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_root / "src")
    env["ALLOW_HEAVY_RUN"] = "1"
    env["HEAVY_HOSTNAME_ALLOWLIST"] = f"{socket.gethostname()},127.0.0.1,localhost"
    env["HEAVY_MIN_CPU"] = "1"
    env["HEAVY_MIN_RAM_GB"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--queue",
            str(queue_path),
            "--runner",
            str(runner_path),
            "--parallel",
            "1",
            "--postprocess",
            "true",
            "--rollup-output-dir",
            str(rollup_output_dir),
            "--rollup-queue-dir",
            str(tmp_path / "artifacts" / "wfa" / "aggregate"),
            "--rollup-runs-dir",
            str(tmp_path / "artifacts" / "wfa" / "runs"),
        ],
        cwd=app_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout

    rebuilt = stale_rollup.read_text(encoding="utf-8")
    assert "stale" not in rebuilt
    assert "run_z" in rebuilt


def test_run_wfa_queue_blocks_when_allow_heavy_run_missing(tmp_path: Path) -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/optimization/run_wfa_queue.py"
    assert script.exists()

    config_path = tmp_path / "configs" / "sample.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    results_dir = tmp_path / "artifacts" / "wfa" / "runs" / "group_guard" / "run_guard"
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "group_guard" / "run_queue.csv"
    _write_queue_csv(queue_path, config_path, results_dir, status="planned")

    runner_path = tmp_path / "runner.sh"
    _write_runner_script(runner_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(app_root / "src")
    env["HEAVY_HOSTNAME_ALLOWLIST"] = f"{socket.gethostname()},127.0.0.1,localhost"
    env["HEAVY_MIN_CPU"] = "1"
    env["HEAVY_MIN_RAM_GB"] = "1"
    env.pop("ALLOW_HEAVY_RUN", None)

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--queue",
            str(queue_path),
            "--runner",
            str(runner_path),
            "--parallel",
            "1",
        ],
        cwd=app_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "BLOCKED" in (proc.stderr + proc.stdout)
    assert _read_queue_status(queue_path) == "planned"
