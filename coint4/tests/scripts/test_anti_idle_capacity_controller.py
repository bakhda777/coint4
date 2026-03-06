from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest


def _load_module(script_name: str, tmp_path: Path):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_{tmp_path.name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_queue(path: Path, *, config_path: str, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_path", "status"])
        writer.writerow([config_path, status])


def _write_remote_runtime_state(path: Path, **overrides) -> None:
    payload = {
        "ts": "2026-03-06T09:00:00Z",
        "ts_epoch": int(time.time()),
        "reachable": True,
        "load1": 0.2,
        "top_level_queue_jobs": 0,
        "watch_queue_count": 0,
        "remote_child_process_count": 0,
        "remote_runner_count": 0,
        "remote_work_active": False,
        "cpu_busy_without_queue_job": False,
    }
    payload.update(overrides)
    if "remote_queue_job_count" not in overrides:
        payload["remote_queue_job_count"] = int(payload.get("top_level_queue_jobs", 0) or 0)
    if "remote_active_queue_jobs" not in overrides:
        payload["remote_active_queue_jobs"] = int(payload.get("remote_queue_job_count", 0) or 0)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_anti_idle_policy_and_process_slo_kpi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps({"remote": {"reachable": True, "runner_count": 0, "load1": 0.2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_remote_runtime_state(state_dir / "remote_runtime_state.json")

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["kpi"]["executable_pending_rows"] == 1
    assert process_state["kpi"]["dispatchable_pending_rows"] == 1
    assert process_state["kpi"]["hard_rejected_pending_rows"] == 0
    assert process_state["kpi"]["local_runner_count"] == 0
    assert process_state["kpi"]["remote_runner_count"] == 0
    assert process_state["kpi"]["remote_child_process_count"] == 0
    assert process_state["kpi"]["remote_queue_job_count"] == 0
    assert process_state["kpi"]["remote_work_active"] is False
    assert process_state["kpi"]["cpu_busy_without_queue_job"] is False
    assert process_state["kpi"]["idle_with_executable_pending"] is True
    assert process_state["kpi"]["candidate_pool_status"] == "empty_error"
    assert process_state["progress_source"] == "local_queue"
    assert process_state["active_remote_queue_rel"] == ""
    assert process_state["remote_queue_sync_age_sec"] == -1

    capacity_module = _load_module("vps_capacity_controller_agent.py", tmp_path)
    monkeypatch.setattr(
        capacity_module,
        "probe_remote_runtime_snapshot",
        lambda *_args, **_kwargs: {
            "reachable": True,
            "load1": 3.0,
            "top_level_queue_jobs": 0,
            "remote_child_process_count": 0,
            "remote_runner_count": 0,
            "remote_work_active": False,
            "cpu_busy_without_queue_job": False,
        },
    )
    monkeypatch.setattr(sys, "argv", ["vps_capacity_controller_agent.py", "--root", str(root)])
    assert capacity_module.main() == 0

    capacity_state = json.loads((state_dir / "capacity_controller_state.json").read_text(encoding="utf-8"))
    assert capacity_state["backlog"]["executable_pending"] == 1
    assert capacity_state["backlog"]["dispatchable_pending"] == 1
    assert capacity_state["backlog"]["hard_rejected_pending"] == 0
    assert capacity_state["remote"]["top_level_queue_jobs"] == 0
    assert capacity_state["remote"]["remote_active_queue_jobs"] == 0
    assert capacity_state["remote"]["remote_queue_job_count"] == 0
    assert capacity_state["remote"]["remote_work_active"] is False
    assert "anti_idle_dispatchable_backlog" in capacity_state["reasons"]
    assert int(capacity_state["policy"]["search_parallel_min"]) >= 16
    assert int(capacity_state["policy"]["search_parallel_max"]) >= 48


def test_process_slo_detects_cpu_busy_without_queue_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps({"remote": {"reachable": True, "runner_count": 12, "load1": 8.5}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_remote_runtime_state(
        state_dir / "remote_runtime_state.json",
        load1=8.5,
        top_level_queue_jobs=0,
        remote_child_process_count=12,
        remote_runner_count=12,
        remote_work_active=True,
        cpu_busy_without_queue_job=True,
    )
    (state_dir / "fullspan_decision_state.json").write_text(
        json.dumps({"runtime_metrics": {"remote_active_queue_jobs": 0, "remote_child_process_count": 12}}, ensure_ascii=False),
        encoding="utf-8",
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["queue"]["top_level_queue_jobs"] == 0
    assert process_state["queue"]["remote_child_process_count"] == 12
    assert process_state["queue"]["remote_work_active"] is True
    assert process_state["queue"]["remote_queue_job_count"] == 0
    assert process_state["runtime"]["cpu_busy_without_queue_job"] is True
    assert process_state["queue"]["idle_with_executable_pending"] is False


def test_process_slo_treats_watcher_only_remote_queue_as_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps(
            {
                "remote": {
                    "reachable": True,
                    "runner_count": 1,
                    "load1": 0.2,
                    "remote_queue_job_count": 1,
                    "remote_active_queue_jobs": 1,
                    "top_level_queue_jobs": 0,
                    "watch_queue_count": 1,
                    "remote_child_process_count": 0,
                    "remote_work_active": True,
                    "cpu_busy_without_queue_job": False,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_remote_runtime_state(
        state_dir / "remote_runtime_state.json",
        load1=0.2,
        remote_queue_job_count=1,
        remote_active_queue_jobs=1,
        top_level_queue_jobs=0,
        watch_queue_count=1,
        remote_child_process_count=0,
        remote_runner_count=1,
        remote_work_active=True,
        cpu_busy_without_queue_job=False,
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["queue"]["remote_queue_job_count"] == 1
    assert process_state["queue"]["remote_active_queue_jobs"] == 1
    assert process_state["queue"]["top_level_queue_jobs"] == 0
    assert process_state["queue"]["watch_queue_count"] == 1
    assert process_state["queue"]["remote_work_active"] is True
    assert process_state["queue"]["idle_with_executable_pending"] is False


def test_process_slo_exposes_runtime_orchestration_observability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps({"remote": {"reachable": True, "runner_count": 0, "load1": 0.3}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_remote_runtime_state(state_dir / "remote_runtime_state.json", load1=0.3)
    (state_dir / "fullspan_decision_state.json").write_text(
        json.dumps(
            {
                "runtime_metrics": {
                    "vps_duty_cycle_30m": "0.625",
                    "ready_buffer_policy_mismatch_count": 3,
                    "winner_parent_duplication_rate": "0.333333",
                    "fastlane_replay_pending": 2,
                    "metrics_missing_abort_count_30m": 4,
                    "winner_proximate_dispatch_count_30m": 5,
                    "hot_standby_active": 1,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["queue"]["fastlane_replay_pending"] == 2
    assert process_state["queue"]["hot_standby_active"] is True
    assert process_state["queue"]["top_level_queue_jobs"] == 0
    assert process_state["queue"]["remote_work_active"] is False
    assert process_state["runtime"]["ready_buffer_policy_mismatch_count"] == 3
    assert process_state["runtime"]["fastlane_replay_pending"] == 2
    assert process_state["runtime"]["metrics_missing_abort_count_30m"] == 4
    assert process_state["runtime"]["winner_proximate_dispatch_count_30m"] == 5
    assert process_state["runtime"]["hot_standby_active"] is True
    assert abs(float(process_state["runtime"]["vps_duty_cycle_30m"]) - 0.625) < 1e-9
    assert abs(float(process_state["runtime"]["winner_parent_duplication_rate"]) - 0.333333) < 1e-9


def test_process_slo_prefers_remote_runtime_snapshot_over_stale_runtime_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps({"remote": {"reachable": True, "runner_count": 31, "load1": 18.0}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_remote_runtime_state(
        state_dir / "remote_runtime_state.json",
        load1=18.0,
        top_level_queue_jobs=1,
        remote_child_process_count=30,
        remote_runner_count=30,
        remote_work_active=True,
        cpu_busy_without_queue_job=False,
    )
    (state_dir / "fullspan_decision_state.json").write_text(
        json.dumps({"runtime_metrics": {"remote_active_queue_jobs": 0, "remote_child_process_count": 0}}, ensure_ascii=False),
        encoding="utf-8",
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["queue"]["remote_active_queue_jobs"] == 1
    assert process_state["queue"]["top_level_queue_jobs"] == 1
    assert process_state["queue"]["remote_child_process_count"] == 30
    assert process_state["queue"]["remote_work_active"] is True
    assert process_state["queue"]["idle_with_executable_pending"] is False


def test_process_slo_exposes_remote_progress_source_and_sync_age(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps({"remote": {"reachable": True, "runner_count": 1, "load1": 0.2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_remote_runtime_state(
        state_dir / "remote_runtime_state.json",
        remote_queue_job_count=1,
        remote_active_queue_jobs=1,
        remote_runner_count=1,
        remote_work_active=True,
        postprocess_active=True,
        build_index_active=True,
        active_remote_queue_rel="artifacts/wfa/aggregate/group_a/run_queue.csv",
        remote_queue_sync_age_sec=5,
        active_queues=[
            {
                "queue_rel": "artifacts/wfa/aggregate/group_a/run_queue.csv",
                "counts": {"planned": 1},
                "total": 1,
                "last_progress_epoch": int(time.time()) - 5,
                "last_progress_age_sec": 5,
            }
        ],
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["progress_source"] == "remote_runtime_state"
    assert process_state["active_remote_queue_rel"] == "artifacts/wfa/aggregate/group_a/run_queue.csv"
    assert process_state["remote_queue_sync_age_sec"] == 5
    assert process_state["queue"]["postprocess_active"] is True
    assert process_state["queue"]["build_index_active"] is True
    assert process_state["queue"]["progress_source"] == "remote_runtime_state"


def test_process_slo_exposes_infra_and_auto_seed_gate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps({"remote": {"reachable": True, "runner_count": 0, "load1": 0.2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_remote_runtime_state(state_dir / "remote_runtime_state.json")
    (state_dir / "fullspan_decision_state.json").write_text(
        json.dumps(
            {
                "runtime_metrics": {
                    "infra_gate_status": "hard_block",
                    "infra_gate_reason": "remote_code_drift",
                    "startup_failure_code": "REMOTE_SYNC_FAILED",
                    "auto_seed_blocked": 1,
                    "auto_seed_block_reason": "remote_code_drift",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (state_dir / "yield_governor_state.json").write_text(
        json.dumps(
            {
                "hard_block_active": True,
                "hard_block_reason": "remote_code_drift",
                "hard_block_until_epoch": int(time.time()) + 600,
                "zero_coverage_seed_streak": 2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (state_dir / "ready_queue_buffer.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"queue": "artifacts/wfa/aggregate/group_a/run_queue.csv", "coverage_verified": True},
                    {"queue": "artifacts/wfa/aggregate/group_b/run_queue.csv", "coverage_verified": False},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (state_dir / "queue_seeder.state.json").write_text(
        json.dumps(
            {
                "window_coverage": {
                    "windows": [
                        {"window": "2025-01-01,2025-03-31", "ok": True},
                        {"window": "2025-04-01,2025-06-30", "ok": True},
                        {"window": "2025-07-01,2025-09-30", "ok": False},
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["infra_gate_status"] == "hard_block"
    assert process_state["infra_gate_reason"] == "remote_code_drift"
    assert process_state["auto_seed_blocked"] is True
    assert process_state["auto_seed_block_reason"] == "remote_code_drift"
    assert process_state["covered_window_count"] == 2
    assert process_state["coverage_verified_ready_count"] == 1
    assert process_state["startup_failure_code"] == "REMOTE_SYNC_FAILED"
    assert process_state["queue"]["coverage_verified_ready_count"] == 1
    assert process_state["queue"]["covered_window_count"] == 2
    assert process_state["runtime"]["infra_gate_status"] == "hard_block"
    assert process_state["runtime"]["auto_seed_blocked"] is True


def test_process_slo_falls_back_to_queue_seeder_state_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps({"remote": {"reachable": True, "runner_count": 0, "load1": 0.2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_remote_runtime_state(state_dir / "remote_runtime_state.json")
    (state_dir / "ready_queue_buffer.json").write_text(
        json.dumps(
            {
                "ready_count": 2,
                "entries": [
                    {"queue": "artifacts/wfa/aggregate/group_a/run_queue.csv"},
                    {"queue": "artifacts/wfa/aggregate/group_b/run_queue.csv"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (state_dir / "queue_seeder.state.json").write_text(
        json.dumps(
            {
                "hygiene": {"covered_window_count": 3},
                "ready_queue_buffer": {"coverage_verified_ready_count": 2},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["covered_window_count"] == 3
    assert process_state["coverage_verified_ready_count"] == 2
    assert process_state["queue"]["covered_window_count"] == 3
    assert process_state["queue"]["coverage_verified_ready_count"] == 2
    assert process_state["kpi"]["covered_window_count"] == 3
    assert process_state["runtime"]["coverage_verified_ready_count"] == 2


def test_process_slo_treats_stale_runtime_snapshot_with_high_capacity_load_as_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)
    old_ts = int(time.time()) - 600
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps(
            {
                "ts": "2026-03-06T09:10:00Z",
                "remote": {
                    "reachable": True,
                    "runner_count": 28,
                    "load1": 11.2,
                    "top_level_queue_jobs": 0,
                    "remote_child_process_count": 28,
                    "remote_work_active": True,
                    "cpu_busy_without_queue_job": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_remote_runtime_state(
        state_dir / "remote_runtime_state.json",
        ts_epoch=old_ts,
        load1=0.0,
        top_level_queue_jobs=0,
        remote_child_process_count=0,
        remote_runner_count=0,
        remote_work_active=False,
        cpu_busy_without_queue_job=False,
    )
    (state_dir / "fullspan_decision_state.json").write_text(
        json.dumps({"runtime_metrics": {"remote_active_queue_jobs": 0, "remote_child_process_count": 0}}, ensure_ascii=False),
        encoding="utf-8",
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["queue"]["remote_active_queue_jobs"] == 0
    assert process_state["queue"]["remote_child_process_count"] == 28
    assert process_state["queue"]["remote_work_active"] is True
    assert process_state["runtime"]["cpu_busy_without_queue_job"] is True
    assert process_state["queue"]["remote_snapshot_age_sec"] == -1
    assert process_state["queue"]["idle_with_executable_pending"] is False


def test_capacity_controller_preserves_watcher_only_queue_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_a" / "run_queue.csv"

    config_rel = "configs/sample.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: sample\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="planned")

    state_dir.mkdir(parents=True, exist_ok=True)

    capacity_module = _load_module("vps_capacity_controller_agent.py", tmp_path)
    monkeypatch.setattr(
        capacity_module,
        "probe_remote_runtime_snapshot",
        lambda *_args, **_kwargs: {
            "reachable": True,
            "load1": 0.2,
            "remote_queue_job_count": 1,
            "remote_active_queue_jobs": 1,
            "top_level_queue_jobs": 0,
            "watch_queue_count": 1,
            "remote_child_process_count": 0,
            "remote_runner_count": 1,
            "remote_work_active": True,
            "cpu_busy_without_queue_job": False,
        },
    )
    monkeypatch.setattr(sys, "argv", ["vps_capacity_controller_agent.py", "--root", str(root)])
    assert capacity_module.main() == 0

    capacity_state = json.loads((state_dir / "capacity_controller_state.json").read_text(encoding="utf-8"))
    assert capacity_state["remote"]["remote_queue_job_count"] == 1
    assert capacity_state["remote"]["remote_active_queue_jobs"] == 1
    assert capacity_state["remote"]["top_level_queue_jobs"] == 0
    assert capacity_state["remote"]["watch_queue_count"] == 1
    assert capacity_state["remote"]["remote_work_active"] is True
    assert "anti_idle_dispatchable_backlog" not in capacity_state["reasons"]


def test_process_slo_and_capacity_ignore_fullspan_reject_backlog_for_dispatchable_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "app"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    queue_path = aggregate_root / "group_reject" / "run_queue.csv"

    config_rel = "configs/reject.yaml"
    config_abs = root / config_rel
    config_abs.parent.mkdir(parents=True, exist_ok=True)
    config_abs.write_text("name: reject\n", encoding="utf-8")
    _write_queue(queue_path, config_path=config_rel, status="stalled")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "capacity_controller_state.json").write_text(
        json.dumps({"remote": {"reachable": True, "runner_count": 0, "load1": 0.2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_remote_runtime_state(state_dir / "remote_runtime_state.json")
    queue_rel = "artifacts/wfa/aggregate/group_reject/run_queue.csv"
    (state_dir / "fullspan_decision_state.json").write_text(
        json.dumps(
            {
                "queues": {
                    queue_rel: {
                        "promotion_verdict": "REJECT",
                        "strict_gate_status": "FULLSPAN_PREFILTER_REJECT",
                        "contract_hard_pass": False,
                        "cutover_permission": "FAIL_CLOSED",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (state_dir / "candidate_pool.csv").write_text(
        "queue,planned,running,stalled,failed,completed,total,urgency\n",
        encoding="utf-8",
    )

    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    monkeypatch.setattr(process_module, "detect_local_runner_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["process_slo_guard_agent.py", "--root", str(root)])
    assert process_module.main() == 0

    process_state = json.loads((state_dir / "process_slo_state.json").read_text(encoding="utf-8"))
    assert process_state["queue"]["executable_pending"] == 1
    assert process_state["queue"]["dispatchable_pending"] == 0
    assert process_state["queue"]["hard_rejected_pending"] == 1
    assert process_state["queue"]["candidate_pool_status"] == "empty_expected"
    assert process_state["queue"]["idle_with_executable_pending"] is False
    assert process_state["queue"]["idle_with_dispatchable_pending"] is False

    capacity_module = _load_module("vps_capacity_controller_agent.py", tmp_path)
    monkeypatch.setattr(
        capacity_module,
        "probe_remote_runtime_snapshot",
        lambda *_args, **_kwargs: {
            "reachable": True,
            "load1": 0.2,
            "top_level_queue_jobs": 0,
            "remote_active_queue_jobs": 0,
            "remote_queue_job_count": 0,
            "watch_queue_count": 0,
            "remote_child_process_count": 0,
            "remote_runner_count": 0,
            "remote_work_active": False,
            "cpu_busy_without_queue_job": False,
        },
    )
    monkeypatch.setattr(sys, "argv", ["vps_capacity_controller_agent.py", "--root", str(root)])
    assert capacity_module.main() == 0

    capacity_state = json.loads((state_dir / "capacity_controller_state.json").read_text(encoding="utf-8"))
    assert capacity_state["backlog"]["executable_pending"] == 1
    assert capacity_state["backlog"]["dispatchable_pending"] == 0
    assert capacity_state["backlog"]["hard_rejected_pending"] == 1
    assert "anti_idle_dispatchable_backlog" not in capacity_state["reasons"]
