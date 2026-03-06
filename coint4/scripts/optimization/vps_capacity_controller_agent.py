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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from remote_runtime_probe import dump_json as dump_remote_runtime_json
from remote_runtime_probe import probe_remote_runtime
from _queue_status_contract import (
    load_fullspan_queue_state,
    load_orphan_queue_cooldowns,
    normalize_queue_status,
    queue_dispatch_block_reason,
    queue_rel_path,
    row_counts_dispatchable_pending,
    row_counts_executable,
    row_counts_pending,
)


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


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
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


def count_backlog_rows(*, aggregate_root: Path, app_root: Path) -> dict[str, int]:
    executable_pending = 0
    dispatchable_pending = 0
    hard_rejected_pending = 0
    state_dir = aggregate_root / ".autonomous"
    fullspan_queues = load_fullspan_queue_state(state_dir / "fullspan_decision_state.json")
    orphan_queues = load_orphan_queue_cooldowns(state_dir / "orphan_queues.csv")
    current_epoch = time.time()

    for queue_path in sorted(aggregate_root.glob("*/run_queue.csv")):
        queue_rel = queue_rel_path(queue_path, app_root)
        try:
            rows = list(csv.DictReader(queue_path.open(newline="", encoding="utf-8")))
        except Exception:
            continue
        blocked_reason = queue_dispatch_block_reason(
            queue_rel=queue_rel,
            fullspan_entry=fullspan_queues.get(queue_rel),
            orphan_entry=orphan_queues.get(queue_rel),
            now_epoch=current_epoch,
        )
        queue_dispatchable = not bool(blocked_reason)

        for row in rows:
            status = normalize_queue_status(row.get("status"))
            if not row_counts_pending(status):
                continue
            executable = row_counts_executable(status, row.get("config_path"), app_root)
            dispatchable_row = row_counts_dispatchable_pending(status, row.get("config_path"), app_root)
            if executable:
                executable_pending += 1
                if queue_dispatchable and dispatchable_row:
                    dispatchable_pending += 1
                elif blocked_reason == "FULLSPAN_REJECT" and dispatchable_row:
                    hard_rejected_pending += 1

    return {
        "executable_pending": int(executable_pending),
        "dispatchable_pending": int(dispatchable_pending),
        "hard_rejected_pending": int(hard_rejected_pending),
    }


def probe_remote_runtime_snapshot(root: Path, state_dir: Path, *, server_user: str, server_ip: str) -> dict[str, Any]:
    default = {
        "reachable": False,
        "load1": -1.0,
        "remote_queue_job_count": 0,
        "remote_active_queue_jobs": 0,
        "top_level_queue_jobs": 0,
        "watch_queue_count": 0,
        "remote_child_process_count": 0,
        "remote_runner_count": -1,
        "remote_work_active": False,
        "cpu_busy_without_queue_job": False,
    }
    state_path = state_dir / "remote_runtime_state.json"
    snapshot = probe_remote_runtime(server_user=server_user, server_ip=server_ip)
    if not isinstance(snapshot, dict):
        return dict(default)
    dump_remote_runtime_json(state_path, snapshot)
    return snapshot


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


def _normalize_token_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def read_process_slo_controlled_recovery(path: Path) -> dict[str, Any]:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {
            "active": False,
            "reason": "",
            "attempts_remaining": 0,
            "variants_cap": 0,
            "winner_proximate_positive_contains": [],
            "candidate_pool_status": "",
        }
    search_quality = data.get("search_quality", {})
    if not isinstance(search_quality, dict):
        search_quality = {}
    queue = data.get("queue", {})
    if not isinstance(queue, dict):
        queue = {}
    return {
        "active": parse_bool(
            search_quality.get("controlled_recovery_active"),
            parse_bool(queue.get("controlled_recovery_active"), False),
        ),
        "reason": str(
            search_quality.get("controlled_recovery_reason")
            or queue.get("controlled_recovery_reason")
            or ""
        ).strip(),
        "attempts_remaining": parse_int(
            search_quality.get("controlled_recovery_attempts_remaining"),
            parse_int(queue.get("controlled_recovery_attempts_remaining"), 0),
        ),
        "variants_cap": parse_int(
            search_quality.get("controlled_recovery_variants_cap"),
            parse_int(queue.get("controlled_recovery_variants_cap"), 0),
        ),
        "winner_proximate_positive_contains": _normalize_token_list(
            search_quality.get("winner_proximate_positive_contains"),
        ),
        "candidate_pool_status": str(
            queue.get("candidate_pool_status") or data.get("candidate_pool_status") or ""
        ).strip(),
    }


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
    remote_runtime_state_path = state_dir / "remote_runtime_state.json"
    fullspan_state_path = state_dir / "fullspan_decision_state.json"
    process_slo_state_path = state_dir / "process_slo_state.json"
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
        backlog = count_backlog_rows(aggregate_root=aggregate_root, app_root=root)
        executable_pending = parse_int(backlog.get("executable_pending"), 0)
        dispatchable_pending = parse_int(backlog.get("dispatchable_pending"), 0)
        hard_rejected_pending = parse_int(backlog.get("hard_rejected_pending"), 0)
        controlled_recovery = read_process_slo_controlled_recovery(process_slo_state_path)

        remote_snapshot = probe_remote_runtime_snapshot(root, state_dir, server_user=server_user, server_ip=server_ip)
        reachable = bool(remote_snapshot.get("reachable"))
        load1 = parse_float(remote_snapshot.get("load1"), -1.0)
        runner_count = parse_int(remote_snapshot.get("remote_runner_count"), -1)
        remote_queue_job_count = parse_int(
            remote_snapshot.get("remote_queue_job_count"),
            parse_int(remote_snapshot.get("remote_active_queue_jobs"), parse_int(remote_snapshot.get("top_level_queue_jobs"), 0)),
        )
        remote_active_queue_jobs = parse_int(
            remote_snapshot.get("remote_active_queue_jobs"),
            remote_queue_job_count,
        )
        top_level_queue_jobs = parse_int(remote_snapshot.get("top_level_queue_jobs"), 0)
        watch_queue_count = parse_int(remote_snapshot.get("watch_queue_count"), 0)
        remote_child_process_count = parse_int(remote_snapshot.get("remote_child_process_count"), 0)
        remote_work_active = bool(remote_snapshot.get("remote_work_active"))
        cpu_busy_without_queue_job = bool(remote_snapshot.get("cpu_busy_without_queue_job"))
        remote_snapshot_ts_epoch = parse_int(remote_snapshot.get("ts_epoch"), 0)
        remote_snapshot_age_sec = max(0, int(time.time()) - remote_snapshot_ts_epoch) if remote_snapshot_ts_epoch > 0 else -1
        controlled_recovery_active = bool(controlled_recovery.get("active"))
        controlled_recovery_attempts_remaining = parse_int(
            controlled_recovery.get("attempts_remaining"),
            0,
        )
        controlled_recovery_backlog_active = bool(
            controlled_recovery_active
            and controlled_recovery_attempts_remaining > 0
            and (
                dispatchable_pending > 0
                or remote_queue_job_count > 0
                or remote_active_queue_jobs > 0
                or remote_work_active
                or runner_count > 0
            )
        )

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
            and dispatchable_pending > 0
            and not remote_work_active
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
            reasons.append("anti_idle_dispatchable_backlog")

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

        if controlled_recovery_backlog_active:
            policy.update(
                {
                    "search_parallel_min": 4,
                    "search_parallel_max": 8,
                    "confirm_parallel_min": 1,
                    "confirm_parallel_max": 1,
                    "confirm_dispatches_per_cycle": 1,
                    "confirm_lane_max_active": 1,
                    "confirm_lane_max_remote_runners": 1,
                }
            )
            reasons.append("controlled_recovery_backlog")

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
                "top_level_queue_jobs": int(top_level_queue_jobs),
                "watch_queue_count": int(watch_queue_count),
                "remote_active_queue_jobs": int(remote_active_queue_jobs),
                "remote_queue_job_count": int(remote_queue_job_count),
                "remote_child_process_count": int(remote_child_process_count),
                "remote_work_active": bool(remote_work_active),
                "cpu_busy_without_queue_job": bool(cpu_busy_without_queue_job),
                "remote_snapshot_age_sec": int(remote_snapshot_age_sec),
            },
            "backlog": {
                "pending_confirm": int(pending_confirm),
                "overdue_confirm": int(overdue_confirm),
                "executable_pending": int(executable_pending),
                "dispatchable_pending": int(dispatchable_pending),
                "hard_rejected_pending": int(hard_rejected_pending),
                "sla_confirm_sec": int(sla_confirm_sec),
            },
            "controlled_recovery": {
                "active": controlled_recovery_active,
                "reason": str(controlled_recovery.get("reason") or ""),
                "attempts_remaining": int(controlled_recovery_attempts_remaining),
                "variants_cap": parse_int(controlled_recovery.get("variants_cap"), 0),
                "backlog_active": controlled_recovery_backlog_active,
                "candidate_pool_status": str(controlled_recovery.get("candidate_pool_status") or ""),
                "winner_proximate_positive_contains": list(
                    controlled_recovery.get("winner_proximate_positive_contains") or []
                ),
            },
            "remote_runtime_state_file": str(remote_runtime_state_path),
            "policy": policy,
            "reasons": reasons,
        }

        if not args.dry_run:
            dump_json(output_path, payload)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | reachable={int(reachable)} load1={load1:.3f} runners={runner_count} "
                f"remote_queue_job_count={remote_queue_job_count} remote_active_queue_jobs={remote_active_queue_jobs} "
                f"top_level_queue_jobs={top_level_queue_jobs} watch_queue_count={watch_queue_count} "
                f"remote_child_process_count={remote_child_process_count} "
                f"remote_work_active={int(remote_work_active)} cpu_busy_without_queue_job={int(cpu_busy_without_queue_job)} "
                f"remote_snapshot_age_sec={remote_snapshot_age_sec} "
                f"pending_confirm={pending_confirm} overdue_confirm={overdue_confirm} "
                f"executable_pending={executable_pending} dispatchable_pending={dispatchable_pending} hard_rejected_pending={hard_rejected_pending} "
                f"controlled_recovery_active={int(controlled_recovery_active)} controlled_recovery_backlog_active={int(controlled_recovery_backlog_active)} "
                f"search_max={policy['search_parallel_max']} confirm_min={policy['confirm_parallel_min']} "
                f"confirm_max={policy['confirm_parallel_max']} reasons={';'.join(reasons) or 'none'} dry_run={int(bool(args.dry_run))}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
