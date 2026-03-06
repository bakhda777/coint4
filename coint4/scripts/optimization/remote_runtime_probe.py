#!/usr/bin/env python3
"""Canonical remote runtime probe for VPS telemetry."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SERVER_IP = "85.198.90.128"
DEFAULT_SERVER_USER = "root"
DEFAULT_MAX_AGE_SEC = 15
DEFAULT_CONNECT_TIMEOUT_SEC = 6


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_epoch() -> int:
    return int(time.time())


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def snapshot_age_sec(snapshot: dict[str, Any], *, now: int | None = None) -> int:
    if now is None:
        now = now_epoch()
    ts_epoch = parse_int(snapshot.get("ts_epoch"), 0)
    if ts_epoch <= 0:
        return 10**9
    return max(0, int(now) - ts_epoch)


def snapshot_is_fresh(snapshot: dict[str, Any], *, max_age_sec: int, now: int | None = None) -> bool:
    if not isinstance(snapshot, dict) or max_age_sec < 0:
        return False
    return snapshot_age_sec(snapshot, now=now) <= max_age_sec


def _resolve_queue_path(cmdline: str) -> str:
    try:
        parts = shlex.split(cmdline)
    except Exception:
        return ""
    for idx, part in enumerate(parts):
        if part == "--queue" and idx + 1 < len(parts):
            return str(parts[idx + 1]).strip()
    return ""


def _probe_remote_processes(server_user: str, server_ip: str, connect_timeout_sec: int) -> dict[str, Any]:
    remote_script = r"""
import json
import os

load1 = 0.0
try:
    load1 = float(open('/proc/loadavg', 'r', encoding='utf-8').read().split()[0])
except Exception:
    load1 = -1.0

top_level_queue_jobs = 0
queue_job_pids = []
queue_paths = []
watch_queue_count = 0
watch_queue_paths = []
run_wfa_fullcpu_count = 0
walk_forward_count = 0
heavy_guardrails_count = 0

for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\x00', b' ').decode('utf-8', 'ignore').strip()
    except Exception:
        continue
    if not cmd:
        continue
    if 'python3 - <<' in cmd or 'pgrep -f' in cmd:
        continue

    if 'run_wfa_queue.py --queue' in cmd and not cmd.startswith('bash -lc '):
        top_level_queue_jobs += 1
        queue_job_pids.append(int(pid))
        tokens = cmd.split()
        try:
            qidx = tokens.index('--queue')
            if qidx + 1 < len(tokens):
                queue_paths.append(tokens[qidx + 1])
        except Exception:
            pass

    if 'watch_wfa_queue.sh --queue' in cmd:
        watch_queue_count += 1
        tokens = cmd.split()
        try:
            qidx = tokens.index('--queue')
            if qidx + 1 < len(tokens):
                watch_queue_paths.append(tokens[qidx + 1])
        except Exception:
            pass
    if 'run_wfa_fullcpu.sh' in cmd:
        run_wfa_fullcpu_count += 1
    if 'coint2 walk-forward' in cmd:
        walk_forward_count += 1
    if 'coint2.ops.heavy_guardrails' in cmd:
        heavy_guardrails_count += 1

remote_child_process_count = walk_forward_count + heavy_guardrails_count + run_wfa_fullcpu_count
owned_queue_paths = sorted(set([path for path in queue_paths if path] + [path for path in watch_queue_paths if path]))
remote_queue_job_count = len(owned_queue_paths)
if remote_queue_job_count <= 0 and (top_level_queue_jobs > 0 or watch_queue_count > 0):
    remote_queue_job_count = top_level_queue_jobs + watch_queue_count
remote_runner_count = max(remote_queue_job_count, top_level_queue_jobs + watch_queue_count, remote_child_process_count)
remote_work_active = bool(remote_queue_job_count > 0 or remote_child_process_count > 0 or load1 >= 1.5)
cpu_busy_without_queue_job = bool(remote_queue_job_count == 0 and (remote_child_process_count > 0 or load1 >= 1.5))

payload = {
    'load1': load1,
    'top_level_queue_jobs': top_level_queue_jobs,
    'queue_job_pids': queue_job_pids,
    'queue_paths': queue_paths,
    'watch_queue_count': watch_queue_count,
    'watch_queue_paths': watch_queue_paths,
    'remote_queue_job_count': remote_queue_job_count,
    'remote_active_queue_jobs': remote_queue_job_count,
    'run_wfa_fullcpu_count': run_wfa_fullcpu_count,
    'walk_forward_count': walk_forward_count,
    'heavy_guardrails_count': heavy_guardrails_count,
    'remote_child_process_count': remote_child_process_count,
    'remote_runner_count': remote_runner_count,
    'remote_work_active': remote_work_active,
    'cpu_busy_without_queue_job': cpu_busy_without_queue_job,
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
"""
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={int(connect_timeout_sec)}",
        f"{server_user}@{server_ip}",
        f"python3 - <<'PY'\n{remote_script}\nPY",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(stderr[-1] if stderr else f"ssh_rc={proc.returncode}")
    stdout = (proc.stdout or "").strip().splitlines()
    if not stdout:
        raise RuntimeError("empty_probe_output")
    payload = json.loads(stdout[-1])
    if not isinstance(payload, dict):
        raise RuntimeError("invalid_probe_payload")
    payload["reachable"] = True
    return payload


def build_remote_runtime_snapshot(
    *,
    server_user: str,
    server_ip: str,
    connect_timeout_sec: int = DEFAULT_CONNECT_TIMEOUT_SEC,
) -> dict[str, Any]:
    ts_epoch = now_epoch()
    ts = utc_now_iso()
    try:
        remote = _probe_remote_processes(server_user, server_ip, connect_timeout_sec)
        remote["reachable"] = True
        remote["probe_status"] = "ok"
        remote["probe_error"] = ""
    except Exception as exc:  # noqa: BLE001
        remote = {
            "reachable": False,
            "load1": -1.0,
            "top_level_queue_jobs": 0,
            "queue_job_pids": [],
            "queue_paths": [],
            "watch_queue_count": 0,
            "watch_queue_paths": [],
            "remote_queue_job_count": 0,
            "remote_active_queue_jobs": 0,
            "run_wfa_fullcpu_count": 0,
            "walk_forward_count": 0,
            "heavy_guardrails_count": 0,
            "remote_child_process_count": 0,
            "remote_runner_count": -1,
            "remote_work_active": False,
            "cpu_busy_without_queue_job": False,
            "probe_status": "ssh_error",
            "probe_error": f"{type(exc).__name__}: {exc}",
        }

    payload = {
        "ts": ts,
        "ts_epoch": ts_epoch,
        "server": {
            "ip": server_ip,
            "user": server_user,
        },
        "reachable": bool(remote.get("reachable")),
        "load1": parse_float(remote.get("load1"), -1.0),
        "top_level_queue_jobs": parse_int(remote.get("top_level_queue_jobs"), 0),
        "remote_active_queue_jobs": parse_int(
            remote.get("remote_active_queue_jobs"),
            parse_int(remote.get("remote_queue_job_count"), parse_int(remote.get("top_level_queue_jobs"), 0)),
        ),
        "remote_queue_job_count": parse_int(
            remote.get("remote_queue_job_count"),
            parse_int(remote.get("remote_active_queue_jobs"), parse_int(remote.get("top_level_queue_jobs"), 0)),
        ),
        "remote_child_process_count": parse_int(remote.get("remote_child_process_count"), 0),
        "remote_runner_count": parse_int(remote.get("remote_runner_count"), -1),
        "watch_queue_count": parse_int(remote.get("watch_queue_count"), 0),
        "watch_queue_paths": list(remote.get("watch_queue_paths") or []),
        "run_wfa_fullcpu_count": parse_int(remote.get("run_wfa_fullcpu_count"), 0),
        "walk_forward_count": parse_int(remote.get("walk_forward_count"), 0),
        "heavy_guardrails_count": parse_int(remote.get("heavy_guardrails_count"), 0),
        "queue_job_pids": list(remote.get("queue_job_pids") or []),
        "queue_paths": list(remote.get("queue_paths") or []),
        "remote_work_active": bool(remote.get("remote_work_active")),
        "cpu_busy_without_queue_job": bool(remote.get("cpu_busy_without_queue_job")),
        "probe_status": str(remote.get("probe_status") or ""),
        "probe_error": str(remote.get("probe_error") or ""),
        "age_sec": 0,
        "fresh": True,
    }
    return payload


def probe_remote_runtime(
    *,
    server_user: str,
    server_ip: str,
    ssh_port: int = 22,
    connect_timeout_sec: int = DEFAULT_CONNECT_TIMEOUT_SEC,
) -> dict[str, Any]:
    del ssh_port  # Reserved for compatibility with shell/python callers.
    return build_remote_runtime_snapshot(
        server_user=server_user,
        server_ip=server_ip,
        connect_timeout_sec=connect_timeout_sec,
    )


def refresh_remote_runtime_state(
    *,
    root: Path,
    server_user: str,
    server_ip: str,
    state_path: Path | None = None,
    max_age_sec: int = DEFAULT_MAX_AGE_SEC,
    force: bool = False,
    dry_run: bool = False,
    connect_timeout_sec: int = DEFAULT_CONNECT_TIMEOUT_SEC,
) -> dict[str, Any]:
    if state_path is None:
        state_path = root / "artifacts" / "wfa" / "aggregate" / ".autonomous" / "remote_runtime_state.json"

    cached = load_json(state_path, {})
    if (
        not force
        and isinstance(cached, dict)
        and snapshot_is_fresh(cached, max_age_sec=max_age_sec)
    ):
        cached["age_sec"] = snapshot_age_sec(cached)
        cached["fresh"] = True
        return cached

    payload = build_remote_runtime_snapshot(
        server_user=server_user,
        server_ip=server_ip,
        connect_timeout_sec=connect_timeout_sec,
    )
    payload["age_sec"] = 0
    payload["fresh"] = True
    if not dry_run:
        dump_json(state_path, payload)
    return payload


def resolve_field(payload: dict[str, Any], field_path: str, default: str = "") -> str:
    cur: Any = payload
    for part in [item for item in str(field_path or "").strip().split(".") if item]:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    if isinstance(cur, bool):
        return "1" if cur else "0"
    if cur is None:
        return default
    if isinstance(cur, (dict, list)):
        return json.dumps(cur, ensure_ascii=False, sort_keys=True)
    return str(cur)


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical remote runtime probe.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--server-ip", default=DEFAULT_SERVER_IP)
    parser.add_argument("--server-user", default=DEFAULT_SERVER_USER)
    parser.add_argument("--state-file", default="", help="Override output state file.")
    parser.add_argument("--max-age-sec", type=int, default=DEFAULT_MAX_AGE_SEC)
    parser.add_argument("--connect-timeout-sec", type=int, default=DEFAULT_CONNECT_TIMEOUT_SEC)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-field", default="", help="Print one field from payload.")
    args = parser.parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_path = Path(args.state_file).resolve() if str(args.state_file or "").strip() else None
    payload = refresh_remote_runtime_state(
        root=root,
        server_user=str(args.server_user or DEFAULT_SERVER_USER),
        server_ip=str(args.server_ip or DEFAULT_SERVER_IP),
        state_path=state_path,
        max_age_sec=max(0, int(args.max_age_sec)),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        connect_timeout_sec=max(1, int(args.connect_timeout_sec)),
    )

    if str(args.print_field or "").strip():
        print(resolve_field(payload, str(args.print_field)))
    else:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
