#!/usr/bin/env python3
"""Adaptive VPS capacity controller for autonomous WFA lanes.

Writes policy into `.autonomous/capacity_controller_state.json` consumed by:
- autonomous_wfa_driver.sh (search parallel bounds)
- confirm_dispatch_agent.sh (confirm lane quotas)
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PENDING_STATUSES = {"planned", "queued", "running", "stalled", "failed", "error", "active"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_under_root(path_text: str, root: Path) -> Path:
    path = Path(str(path_text or "").strip())
    if path.is_absolute():
        return path
    return root / path


def count_executable_pending_rows(*, aggregate_root: Path, app_root: Path) -> int:
    executable_pending = 0

    for queue_path in sorted(aggregate_root.glob("*/run_queue.csv")):
        try:
            rows = list(csv.DictReader(queue_path.open(newline="", encoding="utf-8")))
        except Exception:
            continue

        for row in rows:
            status = str(row.get("status") or "").strip().lower()
            if status not in PENDING_STATUSES:
                continue
            config_path = str(row.get("config_path") or "").strip()
            cfg_exists = False
            if config_path:
                cfg_exists = resolve_under_root(config_path, app_root).exists()
            if cfg_exists or status == "running":
                executable_pending += 1

    return executable_pending


def detect_remote_load(server_user: str, server_ip: str) -> float:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=6",
        f"{server_user}@{server_ip}",
        "cat /proc/loadavg 2>/dev/null | awk '{print $1}'",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return -1.0
    try:
        return float((proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        return -1.0


def detect_remote_runner_count(server_user: str, server_ip: str) -> int:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=6",
        f"{server_user}@{server_ip}",
        "python3 - <<'PY'\n"
        "import os\n"
        "patterns=('watch_wfa_queue.sh','run_wfa_queue.py','run_wfa_fullcpu.sh','walk_forward')\n"
        "count=0\n"
        "for pid in os.listdir('/proc'):\n"
        "    if not pid.isdigit():\n"
        "        continue\n"
        "    try:\n"
        "        cmd=open(f'/proc/{pid}/cmdline','rb').read().replace(b'\\x00',b' ').decode('utf-8','ignore').strip()\n"
        "    except Exception:\n"
        "        continue\n"
        "    if not cmd or 'python3 - <<' in cmd or 'pgrep -f' in cmd:\n"
        "        continue\n"
        "    if any(p in cmd for p in patterns):\n"
        "        count += 1\n"
        "print(count)\n"
        "PY",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return -1
    try:
        return int(float((proc.stdout or "").strip().splitlines()[-1]))
    except Exception:
        return -1


def count_confirm_backlog(state: dict, *, min_groups: int, min_replies: int, sla_confirm_sec: int) -> tuple[int, int]:
    now_epoch = int(time.time())
    pending_confirm = 0
    overdue_confirm = 0

    queues = state.get("queues", {}) if isinstance(state.get("queues"), dict) else {}
    for _queue, entry in queues.items():
        if not isinstance(entry, dict):
            continue
        strict_pass_count = parse_int(entry.get("strict_pass_count"), 0)
        strict_run_groups = parse_int(entry.get("strict_run_group_count"), 0)
        confirm_count = parse_int(entry.get("confirm_count"), 0)
        if strict_pass_count <= 0:
            continue
        if strict_run_groups < min_groups:
            continue
        if confirm_count >= min_replies:
            continue

        pending_confirm += 1
        pending_since = parse_int(entry.get("confirm_pending_since_epoch"), 0)
        if pending_since > 0 and now_epoch - pending_since >= sla_confirm_sec:
            overdue_confirm += 1

    return pending_confirm, overdue_confirm


def read_sla_escalation(path: Path) -> dict[str, Any]:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {}
    until_epoch = parse_int(data.get("until_epoch"), 0)
    now_epoch = int(time.time())
    if not data.get("active"):
        return {}
    if until_epoch > 0 and now_epoch > until_epoch:
        return {}
    policy = data.get("policy", {})
    return policy if isinstance(policy, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive VPS capacity controller.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    lock_path = state_dir / "vps_capacity_controller.lock"
    log_path = state_dir / "vps_capacity_controller.log"
    output_path = state_dir / "capacity_controller_state.json"
    fullspan_state_path = state_dir / "fullspan_decision_state.json"
    sla_state_path = state_dir / "confirm_sla_escalation_state.json"
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"

    server_ip = os.environ.get("SERVER_IP", "85.198.90.128")
    server_user = os.environ.get("SERVER_USER", "root")

    min_groups = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_GROUPS", "2"), 2)
    min_replies = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_REPLIES", "2"), 2)
    sla_confirm_sec = parse_int(os.environ.get("SLA_CONFIRM_PENDING_SEC", "7200"), 7200)
    anti_idle_load_max = parse_float(os.environ.get("ANTI_IDLE_REMOTE_LOAD_MAX", "7.5"), 7.5)
    anti_idle_runner_max = parse_int(os.environ.get("ANTI_IDLE_REMOTE_RUNNER_MAX", "64"), 64)
    anti_idle_search_min = parse_int(os.environ.get("ANTI_IDLE_SEARCH_MIN", "16"), 16)
    anti_idle_search_max = parse_int(os.environ.get("ANTI_IDLE_SEARCH_MAX", "48"), 48)
    search_parallel_hard_min = parse_int(os.environ.get("SEARCH_PARALLEL_HARD_MIN", "2"), 2)
    search_parallel_hard_max = parse_int(os.environ.get("SEARCH_PARALLEL_HARD_MAX", "48"), 48)

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        fullspan_state = load_json(fullspan_state_path, {})
        if not isinstance(fullspan_state, dict):
            fullspan_state = {}

        pending_confirm, overdue_confirm = count_confirm_backlog(
            fullspan_state,
            min_groups=min_groups,
            min_replies=min_replies,
            sla_confirm_sec=sla_confirm_sec,
        )
        executable_pending = count_executable_pending_rows(aggregate_root=aggregate_root, app_root=root)

        load1 = detect_remote_load(server_user, server_ip)
        runner_count = detect_remote_runner_count(server_user, server_ip)
        reachable = bool(load1 >= 0.0 and runner_count >= 0)

        policy = {
            "search_parallel_min": 2,
            "search_parallel_max": 48,
            "confirm_parallel_min": 2,
            "confirm_parallel_max": 4,
            "confirm_dispatches_per_cycle": 2,
            "confirm_lane_max_active": 2,
            "confirm_lane_max_remote_runners": 6,
        }
        reasons: list[str] = []

        if not reachable:
            policy.update({
                "search_parallel_max": 4,
                "confirm_parallel_min": 1,
                "confirm_parallel_max": 2,
                "confirm_dispatches_per_cycle": 1,
                "confirm_lane_max_active": 1,
            })
            reasons.append("remote_unreachable")
        else:
            if load1 >= 12.0:
                policy.update({"search_parallel_min": 4, "search_parallel_max": 10, "confirm_parallel_max": 2})
                reasons.append("vps_overloaded")
            elif load1 >= 9.0:
                policy.update({"search_parallel_min": 8, "search_parallel_max": 24, "confirm_parallel_max": 3})
                reasons.append("vps_high_load")
            elif load1 <= 6.0:
                policy.update({"search_parallel_min": 16, "search_parallel_max": 48, "confirm_parallel_max": 4})
                reasons.append("vps_headroom")
            else:
                policy.update({"search_parallel_min": 12, "search_parallel_max": 36, "confirm_parallel_max": 4})
                reasons.append("vps_balanced")

        anti_idle_candidate = bool(
            reachable
            and overdue_confirm <= 0
            and executable_pending > 0
            and load1 >= 0.0
            and load1 <= anti_idle_load_max
            and runner_count <= anti_idle_runner_max
        )
        if anti_idle_candidate:
            policy["search_parallel_min"] = max(int(policy["search_parallel_min"]), int(anti_idle_search_min))
            policy["search_parallel_max"] = max(
                int(policy["search_parallel_max"]),
                int(anti_idle_search_max),
                int(policy["search_parallel_min"]),
            )
            reasons.append("anti_idle_executable_backlog")

        if overdue_confirm > 0:
            policy["search_parallel_max"] = min(int(policy["search_parallel_max"]), 4)
            policy["confirm_parallel_min"] = max(int(policy["confirm_parallel_min"]), 3)
            policy["confirm_parallel_max"] = max(int(policy["confirm_parallel_max"]), 4)
            policy["confirm_dispatches_per_cycle"] = max(int(policy["confirm_dispatches_per_cycle"]), 3)
            policy["confirm_lane_max_active"] = max(int(policy["confirm_lane_max_active"]), 3)
            policy["confirm_lane_max_remote_runners"] = max(int(policy["confirm_lane_max_remote_runners"]), 8)
            reasons.append("confirm_overdue_boost")

        sla_override = read_sla_escalation(sla_state_path)
        if sla_override:
            reasons.append("sla_escalation_override")
            for key in (
                "search_parallel_min",
                "search_parallel_max",
                "confirm_parallel_min",
                "confirm_parallel_max",
                "confirm_dispatches_per_cycle",
                "confirm_lane_max_active",
                "confirm_lane_max_remote_runners",
            ):
                if key in sla_override:
                    policy[key] = parse_int(sla_override.get(key), parse_int(policy.get(key), 0))

        policy["search_parallel_min"] = max(
            int(search_parallel_hard_min),
            min(int(policy["search_parallel_min"]), int(search_parallel_hard_max)),
        )
        policy["search_parallel_max"] = max(
            int(policy["search_parallel_min"]),
            min(int(policy["search_parallel_max"]), int(search_parallel_hard_max)),
        )
        if int(policy["search_parallel_max"]) < int(policy["search_parallel_min"]):
            policy["search_parallel_max"] = int(policy["search_parallel_min"])
        if int(policy["confirm_parallel_max"]) < int(policy["confirm_parallel_min"]):
            policy["confirm_parallel_max"] = int(policy["confirm_parallel_min"])

        payload = {
            "ts": utc_now_iso(),
            "server": {
                "ip": server_ip,
                "user": server_user,
            },
            "remote": {
                "reachable": reachable,
                "load1": load1,
                "runner_count": runner_count,
            },
            "backlog": {
                "pending_confirm": int(pending_confirm),
                "overdue_confirm": int(overdue_confirm),
                "executable_pending": int(executable_pending),
                "sla_confirm_sec": int(sla_confirm_sec),
            },
            "policy": policy,
            "reasons": reasons,
        }

        if not args.dry_run:
            dump_json(output_path, payload)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | reachable={int(reachable)} load1={load1:.3f} runners={runner_count} "
                f"pending_confirm={pending_confirm} overdue_confirm={overdue_confirm} "
                f"executable_pending={executable_pending} "
                f"search_max={policy['search_parallel_max']} confirm_min={policy['confirm_parallel_min']} "
                f"confirm_max={policy['confirm_parallel_max']} reasons={';'.join(reasons) or 'none'} dry_run={int(bool(args.dry_run))}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
