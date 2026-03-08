from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "autonomous_loop_sentinel.py"
SPEC = importlib.util.spec_from_file_location("autonomous_loop_sentinel", SCRIPT_PATH)
assert SPEC and SPEC.loader
sentinel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sentinel)


def test_normalize_followup_queue_file_drops_invalid_entries_and_dedupes(tmp_path: Path) -> None:
    queue_file = tmp_path / "completion_followup_queue.jsonl"
    queue_file.write_text(
        "\n".join(
            [
                json.dumps({"queue_rel": "0", "trigger_reason": "bad"}),
                json.dumps({"queue_rel": "artifacts/wfa/aggregate/demo/run_queue.csv", "trigger_reason": "ok"}),
                json.dumps({"queue_rel": "artifacts/wfa/aggregate/demo/run_queue.csv", "trigger_reason": "dup"}),
                "{broken-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = sentinel.normalize_followup_queue_file(queue_file)

    assert stats == {"kept": 1, "removed_invalid": 2, "removed_duplicate": 1}
    rows = [json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows == [{"enqueued_epoch": 0, "queue_rel": "artifacts/wfa/aggregate/demo/run_queue.csv", "trigger_reason": "ok"}]


def test_heal_followup_worker_state_clears_invalid_active_queue(tmp_path: Path) -> None:
    app_root = tmp_path
    state_dir = app_root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    state_dir.mkdir(parents=True)
    state_file = state_dir / "completion_followup_worker_state.json"
    pid_file = state_dir / "completion_followup_worker.pid"
    queue_file = state_dir / "completion_followup_queue.jsonl"
    driver_script = app_root / "scripts" / "optimization" / "autonomous_wfa_driver.sh"
    driver_script.parent.mkdir(parents=True)
    driver_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    queue_file.write_text("", encoding="utf-8")
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "active",
                "queue_rel": "0",
                "pid": 0,
                "result": "processing",
                "backlog": 0,
                "ts": "2026-03-08T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    actions = sentinel.heal_followup_worker_state(
        app_root=app_root,
        state_file=state_file,
        pid_file=pid_file,
        queue_file=queue_file,
        driver_script_path=driver_script,
    )

    assert actions == ["worker_state_cleared_invalid_queue"]
    healed = json.loads(state_file.read_text(encoding="utf-8"))
    assert healed["status"] == "idle"
    assert healed["queue_rel"] == ""
    assert healed["result"] == "invalid_queue_rel_cleared"


def test_heal_followup_worker_state_clears_dead_active_pid(tmp_path: Path) -> None:
    app_root = tmp_path
    state_dir = app_root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    queue_dir = app_root / "artifacts" / "wfa" / "aggregate" / "demo"
    queue_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "completion_followup_worker_state.json"
    pid_file = state_dir / "completion_followup_worker.pid"
    queue_file = state_dir / "completion_followup_queue.jsonl"
    driver_script = app_root / "scripts" / "optimization" / "autonomous_wfa_driver.sh"
    driver_script.parent.mkdir(parents=True)
    driver_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (queue_dir / "run_queue.csv").write_text("run_name,config_path,status\n", encoding="utf-8")
    queue_file.write_text("", encoding="utf-8")
    pid_file.write_text("999999\n", encoding="utf-8")
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "active",
                "queue_rel": "artifacts/wfa/aggregate/demo/run_queue.csv",
                "pid": 999999,
                "result": "processing",
                "backlog": 1,
                "ts": "2026-03-08T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    actions = sentinel.heal_followup_worker_state(
        app_root=app_root,
        state_file=state_file,
        pid_file=pid_file,
        queue_file=queue_file,
        driver_script_path=driver_script,
    )

    assert actions == ["worker_state_cleared_dead_pid"]
    healed = json.loads(state_file.read_text(encoding="utf-8"))
    assert healed["status"] == "idle"
    assert healed["queue_rel"] == ""
    assert healed["result"] == "dead_pid_cleared"
    assert not pid_file.exists()


def test_driver_restart_reason_detects_stale_and_dead_pid() -> None:
    assert sentinel.driver_restart_reason({"status": "stale"}) == "runtime_sha_stale"
    assert sentinel.driver_restart_reason(
        {
            "status": "active",
            "runtime_script_sha256": "a",
            "current_script_sha256": "b",
            "pid": 0,
        }
    ) == "runtime_sha_mismatch"
    assert sentinel.driver_restart_reason({"status": "idle", "pid": 0}) == ""
