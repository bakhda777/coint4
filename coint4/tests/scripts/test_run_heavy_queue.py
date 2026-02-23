from __future__ import annotations

import csv
import os
import shutil
import socket
import subprocess
from pathlib import Path
from uuid import uuid4


def test_run_heavy_queue_dry_run_prints_plan() -> None:
    app_root = Path(__file__).resolve().parents[2]
    script = app_root / "scripts/batch/run_heavy_queue.sh"
    assert script.exists()

    token = uuid4().hex[:8]
    run_group = f"test_heavy_queue_{token}"
    cfg_dir = app_root / "configs" / "_test_heavy_queue"
    queue_dir = app_root / "artifacts" / "wfa" / "aggregate" / run_group
    queue_path = queue_dir / "run_queue.csv"
    cfg_path = cfg_dir / f"cfg_{token}.yaml"

    cfg_dir.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("walk_forward:\n  max_steps: 5\n", encoding="utf-8")

    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_path", "results_dir", "status"])
        writer.writerow(
            [
                str(cfg_path.relative_to(app_root)),
                f"artifacts/wfa/runs/{run_group}/holdout_cfg_{token}",
                "planned",
            ]
        )

    env = os.environ.copy()
    env["ALLOW_HEAVY_RUN"] = "1"
    env["HEAVY_HOSTNAME_ALLOWLIST"] = f"{socket.gethostname()},127.0.0.1,localhost"
    env["HEAVY_MIN_CPU"] = "1"
    env["HEAVY_MIN_RAM_GB"] = "0.1"

    try:
        proc = subprocess.run(
            [
                str(script),
                "--queue",
                str(queue_path.relative_to(app_root)),
                "--parallel",
                "2",
                "--dry-run",
            ],
            cwd=app_root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert f"run_group={run_group}" in proc.stdout
        assert "runner=watch" in proc.stdout
        assert "dry-run: no remote execution" in proc.stdout
    finally:
        shutil.rmtree(queue_dir, ignore_errors=True)
        try:
            cfg_path.unlink()
        except FileNotFoundError:
            pass
