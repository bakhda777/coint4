#!/usr/bin/env python3
"""Deterministic guard for the autonomous WFA driver loop."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return default


def valid_queue_rel(queue_rel: str) -> bool:
    queue_rel = str(queue_rel or "").strip()
    return bool(queue_rel) and queue_rel != "0" and queue_rel.endswith("/run_queue.csv")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def parse_ts_epoch(value: Any) -> int:
    raw = str(value or "").strip()
    if not raw:
        return 0
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return int(datetime.fromisoformat(raw).timestamp())
    except Exception:
        return 0


def pid_matches_driver(pid: int, driver_script_path: Path) -> bool:
    if not pid_alive(pid):
        return False
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = cmdline_path.read_bytes()
    except Exception:
        return False
    cmdline = " ".join(chunk.decode("utf-8", errors="ignore") for chunk in raw.split(b"\0") if chunk)
    return str(driver_script_path) in cmdline or driver_script_path.name in cmdline


def normalize_followup_queue_file(queue_file: Path) -> dict[str, int]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed_invalid = 0
    removed_duplicate = 0
    if queue_file.exists():
        for raw in queue_file.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                removed_invalid += 1
                continue
            if not isinstance(item, dict):
                removed_invalid += 1
                continue
            queue_rel = str(item.get("queue_rel") or "").strip()
            if not valid_queue_rel(queue_rel):
                removed_invalid += 1
                continue
            if queue_rel in seen:
                removed_duplicate += 1
                continue
            seen.add(queue_rel)
            kept.append(
                {
                    "queue_rel": queue_rel,
                    "trigger_reason": str(item.get("trigger_reason") or "").strip(),
                    "enqueued_epoch": to_int(item.get("enqueued_epoch"), 0),
                }
            )
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    queue_file.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in kept),
        encoding="utf-8",
    )
    return {
        "kept": len(kept),
        "removed_invalid": removed_invalid,
        "removed_duplicate": removed_duplicate,
    }


def clear_followup_worker_state(state_file: Path, backlog: int, result: str) -> None:
    dump_json(
        state_file,
        {
            "version": 1,
            "status": "idle",
            "queue_rel": "",
            "pid": 0,
            "result": result,
            "backlog": max(0, backlog),
            "ts": utc_now_iso(),
        },
    )


def heal_followup_worker_state(
    *,
    app_root: Path,
    state_file: Path,
    pid_file: Path,
    queue_file: Path,
    driver_script_path: Path,
) -> list[str]:
    actions: list[str] = []
    state = load_json(state_file, {})
    if not isinstance(state, dict):
        state = {}
    status = str(state.get("status") or "").strip()
    queue_rel = str(state.get("queue_rel") or "").strip()
    pid = to_int(state.get("pid"), 0)
    state_ts_epoch = parse_ts_epoch(state.get("ts"))
    backlog = normalize_followup_queue_file(queue_file)["kept"]

    if status == "active" and queue_rel and not valid_queue_rel(queue_rel):
        if pid_matches_driver(pid, driver_script_path):
            try:
                os.kill(pid, signal.SIGTERM)
                actions.append("worker_sigterm_invalid_queue")
            except OSError:
                pass
        pid_file.unlink(missing_ok=True)
        clear_followup_worker_state(state_file, backlog, "invalid_queue_rel_cleared")
        actions.append("worker_state_cleared_invalid_queue")
        return actions

    if status == "active" and pid > 0 and not pid_alive(pid):
        pid_file.unlink(missing_ok=True)
        clear_followup_worker_state(state_file, backlog, "dead_pid_cleared")
        actions.append("worker_state_cleared_dead_pid")
        return actions

    if status == "active" and pid <= 0 and state_ts_epoch > 0:
        clear_followup_worker_state(state_file, backlog, "missing_pid_cleared")
        actions.append("worker_state_cleared_missing_pid")
        return actions

    if status == "active" and valid_queue_rel(queue_rel):
        queue_abs = app_root / queue_rel
        if not queue_abs.exists():
            if pid_matches_driver(pid, driver_script_path):
                try:
                    os.kill(pid, signal.SIGTERM)
                    actions.append("worker_sigterm_missing_queue")
                except OSError:
                    pass
            pid_file.unlink(missing_ok=True)
            clear_followup_worker_state(state_file, backlog, "missing_queue_cleared")
            actions.append("worker_state_cleared_missing_queue")
            return actions

    if status != "active" and pid_file.exists():
        pid_file.unlink(missing_ok=True)
        actions.append("worker_pidfile_cleared_idle")
    return actions


def driver_restart_reason(runtime_state: dict[str, Any]) -> str:
    status = str(runtime_state.get("status") or "").strip()
    pid = to_int(runtime_state.get("pid"), 0)
    runtime_sha = str(runtime_state.get("runtime_script_sha256") or "").strip()
    current_sha = str(runtime_state.get("current_script_sha256") or "").strip()
    if status == "stale":
        return "runtime_sha_stale"
    if status == "active" and runtime_sha and current_sha and runtime_sha != current_sha:
        return "runtime_sha_mismatch"
    if status == "active" and pid > 0 and not pid_alive(pid):
        return "driver_pid_dead"
    return ""


def restart_service(service_name: str) -> bool:
    proc = subprocess.run(
        ["systemctl", "--user", "restart", service_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def run(app_root: Path, service_name: str) -> dict[str, Any]:
    state_dir = app_root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    summary: dict[str, Any] = {
        "ts": utc_now_iso(),
        "service_name": service_name,
        "actions": [],
    }

    queue_file = state_dir / "completion_followup_queue.jsonl"
    queue_stats = normalize_followup_queue_file(queue_file)
    summary["queue_stats"] = queue_stats
    if queue_stats["removed_invalid"] > 0 or queue_stats["removed_duplicate"] > 0:
        summary["actions"].append("followup_queue_normalized")

    worker_actions = heal_followup_worker_state(
        app_root=app_root,
        state_file=state_dir / "completion_followup_worker_state.json",
        pid_file=state_dir / "completion_followup_worker.pid",
        queue_file=queue_file,
        driver_script_path=app_root / "scripts" / "optimization" / "autonomous_wfa_driver.sh",
    )
    summary["actions"].extend(worker_actions)

    runtime_state = load_json(state_dir / "driver_runtime_state.json", {})
    if not isinstance(runtime_state, dict):
        runtime_state = {}
    restart_reason = driver_restart_reason(runtime_state)
    summary["driver_restart_reason"] = restart_reason
    if restart_reason:
        if restart_service(service_name):
            summary["actions"].append(f"driver_restarted:{restart_reason}")
        else:
            summary["actions"].append(f"driver_restart_failed:{restart_reason}")

    dump_json(state_dir / "loop_sentinel_state.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic loop sentinel for autonomous WFA driver")
    parser.add_argument(
        "--app-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Absolute path to app root (default: coint4/)",
    )
    parser.add_argument(
        "--service-name",
        default="autonomous-wfa-driver.service",
        help="systemd --user service name to restart when runtime state is stale",
    )
    args = parser.parse_args()

    app_root = Path(args.app_root).resolve()
    summary = run(app_root=app_root, service_name=str(args.service_name))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
