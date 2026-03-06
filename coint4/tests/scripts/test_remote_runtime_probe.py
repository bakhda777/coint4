from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "remote_runtime_probe.py"
    spec = importlib.util.spec_from_file_location(f"remote_runtime_probe_{tmp_path.name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Proc:
    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_probe_remote_runtime_reports_child_only_activity(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(tmp_path)
    payload = {
        "reachable": True,
        "load1": 18.0,
        "top_level_queue_jobs": 0,
        "queue_job_pids": [],
        "walk_forward_count": 28,
        "heavy_guardrails_count": 2,
        "run_wfa_fullcpu_count": 0,
        "remote_child_process_count": 30,
        "remote_runner_count": 30,
        "remote_work_active": True,
        "cpu_busy_without_queue_job": True,
    }

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    result = module.probe_remote_runtime(server_user="root", server_ip="85.198.90.128")
    assert result["reachable"] is True
    assert result["top_level_queue_jobs"] == 0
    assert result["remote_child_process_count"] == 30
    assert result["remote_work_active"] is True
    assert result["cpu_busy_without_queue_job"] is True


def test_probe_remote_runtime_treats_watcher_only_queue_as_active_work(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(tmp_path)
    payload = {
        "reachable": True,
        "load1": 0.1,
        "top_level_queue_jobs": 0,
        "queue_job_pids": [],
        "queue_paths": [],
        "watch_queue_count": 1,
        "watch_queue_paths": ["artifacts/wfa/aggregate/group_a/run_queue.csv"],
        "remote_queue_job_count": 1,
        "remote_active_queue_jobs": 1,
        "walk_forward_count": 0,
        "heavy_guardrails_count": 0,
        "run_wfa_fullcpu_count": 0,
        "remote_child_process_count": 0,
        "remote_runner_count": 1,
        "remote_work_active": True,
        "cpu_busy_without_queue_job": False,
    }

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    result = module.probe_remote_runtime(server_user="root", server_ip="85.198.90.128")
    assert result["reachable"] is True
    assert result["top_level_queue_jobs"] == 0
    assert result["watch_queue_count"] == 1
    assert result["watch_queue_paths"] == ["artifacts/wfa/aggregate/group_a/run_queue.csv"]
    assert result["remote_queue_job_count"] == 1
    assert result["remote_active_queue_jobs"] == 1
    assert result["remote_work_active"] is True
    assert result["cpu_busy_without_queue_job"] is False


def test_probe_remote_runtime_exposes_postprocess_and_active_queue_counts(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(tmp_path)
    payload = {
        "reachable": True,
        "load1": 4.2,
        "top_level_queue_jobs": 0,
        "queue_job_pids": [],
        "queue_paths": [],
        "watch_queue_count": 0,
        "watch_queue_paths": [],
        "remote_queue_job_count": 1,
        "remote_active_queue_jobs": 1,
        "walk_forward_count": 0,
        "heavy_guardrails_count": 0,
        "run_wfa_fullcpu_count": 0,
        "postprocess_queue_count": 1,
        "postprocess_active": True,
        "build_run_index_count": 1,
        "build_index_active": True,
        "active_remote_queue_rel": "artifacts/wfa/aggregate/group_a/run_queue.csv",
        "remote_queue_sync_age_sec": 7,
        "active_queues": [
            {
                "queue_rel": "artifacts/wfa/aggregate/group_a/run_queue.csv",
                "counts": {"running": 3, "completed": 2},
                "total": 5,
                "last_progress_epoch": 1700000000,
                "last_progress_age_sec": 7,
            }
        ],
        "remote_child_process_count": 2,
        "remote_runner_count": 2,
        "remote_work_active": True,
        "cpu_busy_without_queue_job": False,
    }

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    result = module.probe_remote_runtime(server_user="root", server_ip="85.198.90.128")
    assert result["postprocess_active"] is True
    assert result["build_index_active"] is True
    assert result["active_remote_queue_rel"] == "artifacts/wfa/aggregate/group_a/run_queue.csv"
    assert result["remote_queue_sync_age_sec"] == 7
    assert result["active_queues"][0]["counts"]["running"] == 3


def test_probe_remote_runtime_handles_ssh_failure(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(tmp_path)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: _Proc(returncode=255, stdout="", stderr="ssh timeout"),
    )

    result = module.probe_remote_runtime(server_user="root", server_ip="85.198.90.128")
    assert result["reachable"] is False
    assert result["probe_status"] == "ssh_error"
    assert "ssh timeout" in str(result.get("probe_error", ""))
