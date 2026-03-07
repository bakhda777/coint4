#!/usr/bin/env python3
"""Process SLO/WIP guard for strict fullspan autonomous loop.

Writes machine-readable state consumed by the 10-minute human report:
- artifacts/wfa/aggregate/.autonomous/process_slo_state.json
- artifacts/wfa/aggregate/.autonomous/process_slo_events.jsonl

Fail-closed semantics:
- alerts are observational only; promotion authority stays with gatekeeper/auditor.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _queue_status_contract import (
    FAILED_LIKE_STATUSES,
    PENDING_LIKE_STATUSES,
    load_fullspan_queue_state,
    load_orphan_queue_cooldowns,
    normalize_queue_status,
    queue_dispatch_block_reason,
    queue_rel_path,
    row_counts_dispatchable_pending,
    row_counts_executable,
    row_counts_pending,
)

CONFIRM_PENDING_VERDICTS = {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"}


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now_iso()} | {message}\n")


def parse_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except Exception:
        return default


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def parse_ts_epoch(value: Any) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        pass
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return 0


def resolve_under_root(path_text: str, root: Path) -> Path:
    path = Path(str(path_text or "").strip())
    if path.is_absolute():
        return path
    return root / path


def detect_local_runner_count() -> int:
    patterns = (
        "run_wfa_queue_powered.py --queue",
        "watch_wfa_queue.sh --queue",
        "scripts/optimization/run_wfa_queue.py --queue",
    )
    count = 0
    current = str(os.getpid())
    parent = str(os.getppid())

    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid in {current, parent}:
            continue
        try:
            cmd = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", "ignore")
                .strip()
            )
        except Exception:
            continue
        if not cmd:
            continue
        if "python3 - <<" in cmd or "pgrep -f" in cmd:
            continue
        if any(pattern in cmd for pattern in patterns):
            count += 1
    return count


def read_remote_runner_snapshot(path: Path) -> dict[str, Any]:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {"reachable": False, "runner_count": -1, "load1": 0.0}

    remote = data.get("remote", {})
    if not isinstance(remote, dict):
        remote = {}

    return {
        "reachable": parse_bool(remote.get("reachable"), False),
        "runner_count": parse_int(remote.get("runner_count"), -1),
        "load1": parse_float(remote.get("load1"), 0.0),
        "remote_queue_job_count": parse_int(
            remote.get("remote_queue_job_count"),
            parse_int(remote.get("remote_active_queue_jobs"), parse_int(remote.get("top_level_queue_jobs"), 0)),
        ),
        "remote_active_queue_jobs": parse_int(
            remote.get("remote_active_queue_jobs"),
            parse_int(remote.get("remote_queue_job_count"), parse_int(remote.get("top_level_queue_jobs"), 0)),
        ),
        "top_level_queue_jobs": parse_int(remote.get("top_level_queue_jobs"), 0),
        "watch_queue_count": parse_int(remote.get("watch_queue_count"), 0),
        "remote_child_process_count": parse_int(remote.get("remote_child_process_count"), 0),
        "remote_work_active": parse_bool(remote.get("remote_work_active"), False),
        "cpu_busy_without_queue_job": parse_bool(remote.get("cpu_busy_without_queue_job"), False),
        "ts_epoch": parse_ts_epoch(data.get("ts_epoch") or data.get("ts")),
    }


def read_remote_runtime_snapshot(path: Path) -> dict[str, Any]:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {
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
            "postprocess_active": False,
            "build_index_active": False,
            "active_queues": [],
            "active_remote_queue_rel": "",
            "remote_queue_sync_age_sec": -1,
            "ts_epoch": 0,
        }
    return {
        "reachable": parse_bool(data.get("reachable"), False),
        "load1": parse_float(data.get("load1"), -1.0),
        "remote_queue_job_count": parse_int(
            data.get("remote_queue_job_count"),
            parse_int(data.get("remote_active_queue_jobs"), parse_int(data.get("top_level_queue_jobs"), 0)),
        ),
        "remote_active_queue_jobs": parse_int(
            data.get("remote_active_queue_jobs"),
            parse_int(data.get("remote_queue_job_count"), parse_int(data.get("top_level_queue_jobs"), 0)),
        ),
        "top_level_queue_jobs": parse_int(data.get("top_level_queue_jobs"), 0),
        "watch_queue_count": parse_int(data.get("watch_queue_count"), 0),
        "remote_child_process_count": parse_int(data.get("remote_child_process_count"), 0),
        "remote_runner_count": parse_int(data.get("remote_runner_count"), -1),
        "remote_work_active": parse_bool(data.get("remote_work_active"), False),
        "cpu_busy_without_queue_job": parse_bool(data.get("cpu_busy_without_queue_job"), False),
        "postprocess_active": parse_bool(data.get("postprocess_active"), False),
        "build_index_active": parse_bool(data.get("build_index_active"), False),
        "active_queues": list(data.get("active_queues") or []),
        "active_remote_queue_rel": str(data.get("active_remote_queue_rel") or ""),
        "remote_queue_sync_age_sec": parse_int(data.get("remote_queue_sync_age_sec"), -1),
        "ts_epoch": parse_ts_epoch(data.get("ts_epoch") or data.get("ts")),
    }


def read_ready_queue_buffer(path: Path) -> dict[str, Any]:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {"ready_count": 0, "coverage_verified_ready_count": 0, "candidate_pool_status": ""}
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    coverage_verified_ready_count = 0
    has_coverage_verified_signal = "coverage_verified_ready_count" in data
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "coverage_verified" in entry:
            has_coverage_verified_signal = True
        if parse_bool(entry.get("coverage_verified"), False):
            coverage_verified_ready_count += 1
    if "coverage_verified_ready_count" in data:
        coverage_verified_ready_count = parse_int(
            data.get("coverage_verified_ready_count"),
            coverage_verified_ready_count,
        )
    return {
        "ready_count": parse_int(data.get("ready_count"), len(entries)),
        "coverage_verified_ready_count": (
            coverage_verified_ready_count if has_coverage_verified_signal else None
        ),
        "candidate_pool_status": str(data.get("candidate_pool_status") or "").strip().lower(),
    }


def read_yield_governor_state(path: Path) -> dict[str, Any]:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {
            "hard_block_active": False,
            "hard_block_reason": "",
            "hard_block_until_epoch": 0,
            "zero_coverage_seed_streak": 0,
            "zero_coverage_seed_streak_reason": "",
            "positive_lineage_count": 0,
            "zero_evidence_lineage_count": 0,
            "winner_proximate_positive_lineage_count": 0,
            "broad_search_allowed": None,
            "seed_generation_mode": "",
            "lineages": [],
            "winner_proximate": {},
            "winner_proximate_positive_contains": [],
            "controlled_recovery_active": False,
            "controlled_recovery_reason": "",
            "controlled_recovery_attempts_remaining": 0,
            "controlled_recovery_variants_cap": 0,
        }
    lineages = data.get("lineages", [])
    if not isinstance(lineages, list):
        lineages = []
    winner_proximate = data.get("winner_proximate", {})
    if not isinstance(winner_proximate, dict):
        winner_proximate = {}
    winner_proximate_positive_contains = data.get("winner_proximate_positive_contains")
    if winner_proximate_positive_contains is None:
        winner_proximate_positive_contains = winner_proximate.get("contains")
    if not isinstance(winner_proximate_positive_contains, list):
        winner_proximate_positive_contains = []
    return {
        "hard_block_active": parse_bool(data.get("hard_block_active"), False),
        "hard_block_reason": str(data.get("hard_block_reason") or ""),
        "hard_block_until_epoch": parse_int(data.get("hard_block_until_epoch"), 0),
        "zero_coverage_seed_streak": parse_int(data.get("zero_coverage_seed_streak"), 0),
        "zero_coverage_seed_streak_reason": str(data.get("zero_coverage_seed_streak_reason") or ""),
        "positive_lineage_count": parse_int(data.get("positive_lineage_count"), 0),
        "zero_evidence_lineage_count": parse_int(data.get("zero_evidence_lineage_count"), 0),
        "winner_proximate_positive_lineage_count": parse_int(
            data.get("winner_proximate_positive_lineage_count"),
            0,
        ),
        "broad_search_allowed": data.get("broad_search_allowed"),
        "seed_generation_mode": str(data.get("seed_generation_mode") or ""),
        "lineages": lineages,
        "winner_proximate": winner_proximate,
        "winner_proximate_positive_contains": winner_proximate_positive_contains,
        "controlled_recovery_active": parse_bool(data.get("controlled_recovery_active"), False),
        "controlled_recovery_reason": str(data.get("controlled_recovery_reason") or ""),
        "controlled_recovery_attempts_remaining": parse_int(
            data.get("controlled_recovery_attempts_remaining"),
            0,
        ),
        "controlled_recovery_variants_cap": parse_int(data.get("controlled_recovery_variants_cap"), 0),
    }


def read_queue_seeder_state(path: Path) -> dict[str, Any]:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {
            "covered_window_count": 0,
            "coverage_verified_ready_count": 0,
            "positive_lineage_count": 0,
            "zero_evidence_lineage_count": 0,
            "broad_search_allowed": None,
            "seed_generation_mode": "",
            "recent_seed_quality": {},
            "quality_governor": {},
            "directive": {},
            "winner_proximate_positive_contains": [],
            "controlled_recovery_active": False,
            "controlled_recovery_reason": "",
            "controlled_recovery_attempts_remaining": 0,
            "controlled_recovery_variants_cap": 0,
        }
    covered_window_count = parse_int(data.get("covered_window_count"), 0)
    coverage_verified_ready_count = parse_int(data.get("coverage_verified_ready_count"), 0)
    hygiene = data.get("hygiene", {})
    if isinstance(hygiene, dict):
        covered_window_count = parse_int(hygiene.get("covered_window_count"), covered_window_count)
    window_coverage = data.get("window_coverage", {})
    if isinstance(window_coverage, dict):
        windows = list(window_coverage.get("windows") or [])
        counted_windows = 0
        for entry in windows:
            if isinstance(entry, dict) and parse_bool(entry.get("ok"), False):
                counted_windows += 1
        if counted_windows > 0 or not covered_window_count:
            covered_window_count = counted_windows
    ready_queue_buffer = data.get("ready_queue_buffer", {})
    if isinstance(ready_queue_buffer, dict):
        coverage_verified_ready_count = parse_int(
            ready_queue_buffer.get("coverage_verified_ready_count"),
            coverage_verified_ready_count,
        )
    recent_seed_quality = data.get("recent_seed_quality", {})
    if not isinstance(recent_seed_quality, dict):
        recent_seed_quality = {}
    quality_governor = data.get("quality_governor", {})
    if not isinstance(quality_governor, dict):
        quality_governor = {}
    directive = data.get("directive", {})
    if not isinstance(directive, dict):
        directive = {}
    winner_proximate_positive_contains = data.get("winner_proximate_positive_contains")
    if winner_proximate_positive_contains is None:
        winner_proximate_positive_contains = directive.get("winner_proximate_tokens")
    if not isinstance(winner_proximate_positive_contains, list):
        winner_proximate_positive_contains = []
    return {
        "covered_window_count": covered_window_count,
        "coverage_verified_ready_count": coverage_verified_ready_count,
        "positive_lineage_count": parse_int(data.get("positive_lineage_count"), 0),
        "zero_evidence_lineage_count": parse_int(data.get("zero_evidence_lineage_count"), 0),
        "broad_search_allowed": data.get("broad_search_allowed"),
        "seed_generation_mode": str(data.get("seed_generation_mode") or ""),
        "recent_seed_quality": recent_seed_quality,
        "quality_governor": quality_governor,
        "directive": directive,
        "winner_proximate_positive_contains": winner_proximate_positive_contains,
        "controlled_recovery_active": parse_bool(data.get("controlled_recovery_active"), False),
        "controlled_recovery_reason": str(data.get("controlled_recovery_reason") or ""),
        "controlled_recovery_attempts_remaining": parse_int(
            data.get("controlled_recovery_attempts_remaining"),
            0,
        ),
        "controlled_recovery_variants_cap": parse_int(data.get("controlled_recovery_variants_cap"), 0),
    }


def _count_non_empty_tokens(values: Any) -> int:
    if not isinstance(values, list):
        return 0
    return sum(1 for value in values if str(value or "").strip())


def _normalize_non_empty_tokens(values: Any) -> list[str]:
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


def derive_search_quality_state(
    *,
    yield_governor_state: dict[str, Any],
    queue_seeder_state: dict[str, Any],
) -> dict[str, Any]:
    positive_lineage_count = parse_int(yield_governor_state.get("positive_lineage_count"), 0)
    zero_evidence_lineage_count = parse_int(yield_governor_state.get("zero_evidence_lineage_count"), 0)

    winner_proximate = yield_governor_state.get("winner_proximate", {})
    if not isinstance(winner_proximate, dict):
        winner_proximate = {}
    lineages = yield_governor_state.get("lineages", [])
    if not isinstance(lineages, list):
        lineages = []

    if positive_lineage_count <= 0:
        positive_lineage_count = _count_non_empty_tokens(winner_proximate.get("contains"))
    if positive_lineage_count <= 0:
        directive = queue_seeder_state.get("directive", {})
        if not isinstance(directive, dict):
            directive = {}
        positive_lineage_count = _count_non_empty_tokens(directive.get("winner_proximate_tokens"))
    if positive_lineage_count <= 0 and lineages:
        positive_lineage_count = sum(
            1
            for entry in lineages
            if isinstance(entry, dict)
            and parse_int(entry.get("metrics_present"), 0) > 0
            and parse_int(entry.get("zero_activity"), 0) <= 0
        )

    if zero_evidence_lineage_count <= 0 and lineages:
        zero_evidence_lineage_count = sum(
            1
            for entry in lineages
            if isinstance(entry, dict)
            and (
                parse_int(entry.get("metrics_present"), 0) <= 0
                or parse_int(entry.get("zero_activity"), 0) > 0
            )
        )
    if zero_evidence_lineage_count <= 0:
        zero_evidence_lineage_count = parse_int(queue_seeder_state.get("zero_evidence_lineage_count"), 0)

    winner_proximate_positive_contains = _normalize_non_empty_tokens(
        yield_governor_state.get("winner_proximate_positive_contains"),
    )
    if not winner_proximate_positive_contains:
        winner_proximate_positive_contains = _normalize_non_empty_tokens(
            queue_seeder_state.get("winner_proximate_positive_contains"),
        )
    if not winner_proximate_positive_contains:
        winner_proximate_positive_contains = _normalize_non_empty_tokens(winner_proximate.get("contains"))

    recent_seed_quality = queue_seeder_state.get("recent_seed_quality", {})
    if not isinstance(recent_seed_quality, dict):
        recent_seed_quality = {}
    quality_governor = queue_seeder_state.get("quality_governor", {})
    if not isinstance(quality_governor, dict):
        quality_governor = {}
    directive = queue_seeder_state.get("directive", {})
    if not isinstance(directive, dict):
        directive = {}

    broad_search_allowed = queue_seeder_state.get("broad_search_allowed")
    if broad_search_allowed is None:
        broad_search_allowed = yield_governor_state.get("broad_search_allowed")
    if broad_search_allowed is None:
        broad_search_allowed = not (
            parse_bool(yield_governor_state.get("hard_block_active"), False)
            or parse_bool(recent_seed_quality.get("backlog_suppress"), False)
            or parse_bool(quality_governor.get("repair_mode_effective"), False)
            or parse_bool(directive.get("repair_mode"), False)
        )
    else:
        broad_search_allowed = parse_bool(broad_search_allowed, True)

    seed_generation_mode = str(queue_seeder_state.get("seed_generation_mode") or "").strip()
    if not seed_generation_mode:
        seed_generation_mode = str(yield_governor_state.get("seed_generation_mode") or "").strip()
    if not seed_generation_mode:
        seed_generation_mode = str(directive.get("policy_scale") or "").strip()
    if not seed_generation_mode:
        seed_generation_mode = str(directive.get("mode") or "").strip()
    if not seed_generation_mode and parse_bool(quality_governor.get("repair_mode_effective"), False):
        seed_generation_mode = "repair"

    controlled_recovery_active = parse_bool(yield_governor_state.get("controlled_recovery_active"), False)
    controlled_recovery_reason = str(yield_governor_state.get("controlled_recovery_reason") or "").strip()
    controlled_recovery_attempts_remaining = parse_int(
        yield_governor_state.get("controlled_recovery_attempts_remaining"),
        0,
    )
    controlled_recovery_variants_cap = parse_int(
        yield_governor_state.get("controlled_recovery_variants_cap"),
        0,
    )

    if not controlled_recovery_active:
        controlled_recovery_active = parse_bool(queue_seeder_state.get("controlled_recovery_active"), False)
    if not controlled_recovery_reason:
        controlled_recovery_reason = str(queue_seeder_state.get("controlled_recovery_reason") or "").strip()
    if controlled_recovery_attempts_remaining <= 0:
        controlled_recovery_attempts_remaining = parse_int(
            queue_seeder_state.get("controlled_recovery_attempts_remaining"),
            0,
        )
    if controlled_recovery_variants_cap <= 0:
        controlled_recovery_variants_cap = parse_int(
            queue_seeder_state.get("controlled_recovery_variants_cap"),
            0,
        )

    return {
        "positive_lineage_count": max(0, int(positive_lineage_count)),
        "zero_evidence_lineage_count": max(0, int(zero_evidence_lineage_count)),
        "broad_search_allowed": bool(broad_search_allowed),
        "seed_generation_mode": seed_generation_mode,
        "zero_coverage_seed_streak": parse_int(yield_governor_state.get("zero_coverage_seed_streak"), 0),
        "zero_coverage_seed_streak_reason": str(yield_governor_state.get("zero_coverage_seed_streak_reason") or ""),
        "winner_proximate_positive_contains": winner_proximate_positive_contains,
        "controlled_recovery_active": bool(controlled_recovery_active),
        "controlled_recovery_reason": controlled_recovery_reason,
        "controlled_recovery_attempts_remaining": max(0, int(controlled_recovery_attempts_remaining)),
        "controlled_recovery_variants_cap": max(0, int(controlled_recovery_variants_cap)),
    }


def csv_data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for row in reader if any(str(value or "").strip() for value in row.values()))
    except Exception:
        return 0


def queue_stats(*, aggregate_root: Path, app_root: Path) -> dict[str, Any]:
    totals = Counter()
    per_queue: dict[str, dict[str, int]] = {}
    state_dir = aggregate_root / ".autonomous"
    fullspan_queues = load_fullspan_queue_state(state_dir / "fullspan_decision_state.json")
    orphan_queues = load_orphan_queue_cooldowns(state_dir / "orphan_queues.csv")
    current_epoch = time.time()

    for queue_path in sorted(aggregate_root.glob("*/run_queue.csv")):
        queue_rel = queue_rel_path(queue_path, app_root)

        row_count = 0
        completed = 0
        pending = 0
        running = 0
        stalled = 0
        failed = 0
        executable_rows = 0
        executable_pending = 0
        dispatchable_pending = 0
        hard_rejected_pending = 0

        try:
            rows = list(csv.DictReader(queue_path.open(newline="", encoding="utf-8")))
        except Exception:
            continue

        block_reason = queue_dispatch_block_reason(
            queue_rel=queue_rel,
            fullspan_entry=fullspan_queues.get(queue_rel),
            orphan_entry=orphan_queues.get(queue_rel),
            now_epoch=current_epoch,
        )
        queue_dispatchable = not bool(block_reason)

        for row in rows:
            row_count += 1
            status = normalize_queue_status(row.get("status"))
            config_path = str(row.get("config_path") or "").strip()
            executable = row_counts_executable(status, config_path, app_root)
            dispatchable_row = row_counts_dispatchable_pending(status, config_path, app_root)
            if executable:
                executable_rows += 1

            if status == "completed":
                completed += 1
            if row_counts_pending(status):
                pending += 1
                if executable:
                    executable_pending += 1
                if queue_dispatchable and dispatchable_row:
                    dispatchable_pending += 1
                elif block_reason == "FULLSPAN_REJECT" and dispatchable_row:
                    hard_rejected_pending += 1
            if status == "running":
                running += 1
            elif status == "stalled":
                stalled += 1
            elif status in FAILED_LIKE_STATUSES:
                failed += 1

        per_queue[queue_rel] = {
            "total": row_count,
            "completed": completed,
            "pending": pending,
            "running": running,
            "stalled": stalled,
            "failed": failed,
            "executable_rows": executable_rows,
            "executable_pending": executable_pending,
            "dispatchable_pending": dispatchable_pending,
            "hard_rejected_pending": hard_rejected_pending,
        }

        totals["total"] += row_count
        totals["completed"] += completed
        totals["pending"] += pending
        totals["running"] += running
        totals["stalled"] += stalled
        totals["failed"] += failed
        totals["executable_rows"] += executable_rows
        totals["executable_pending"] += executable_pending
        totals["dispatchable_pending"] += dispatchable_pending
        totals["hard_rejected_pending"] += hard_rejected_pending

    return {
        "totals": dict(totals),
        "per_queue": per_queue,
    }


def count_active_cold_fail_entries(path: Path, *, now: int) -> int:
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        return 0
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return 0
    count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        until_ts = parse_int(entry.get("until_ts"), 0)
        if until_ts > int(now):
            count += 1
    return count


def should_emit(*, state: dict[str, Any], key: str, cooldown_sec: int, now: int) -> bool:
    last_emit = state.setdefault("last_emit", {})
    if not isinstance(last_emit, dict):
        last_emit = {}
        state["last_emit"] = last_emit
    prev = parse_int(last_emit.get(key), 0)
    if now - prev < max(0, int(cooldown_sec)):
        return False
    last_emit[key] = now
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Process SLO/WIP guard for autonomous fullspan loop.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"

    lock_path = state_dir / "process_slo_guard.lock"
    state_path = state_dir / "process_slo_state.json"
    events_path = state_dir / "process_slo_events.jsonl"
    log_path = state_dir / "process_slo_guard.log"
    fullspan_state_path = state_dir / "fullspan_decision_state.json"
    capacity_state_path = state_dir / "capacity_controller_state.json"
    remote_runtime_state_path = state_dir / "remote_runtime_state.json"
    cold_fail_index_path = state_dir / "cold_fail_index.json"
    yield_governor_state_path = state_dir / "yield_governor_state.json"
    ready_buffer_state_path = state_dir / "ready_queue_buffer.json"
    candidate_pool_path = state_dir / "candidate_pool.csv"
    queue_seeder_state_path = state_dir / "queue_seeder.state.json"

    now = now_epoch()
    ts = utc_now_iso()

    wip_search_max = parse_int(os.environ.get("PROCESS_WIP_SEARCH_MAX", "600"), 600)
    wip_confirm_max = parse_int(os.environ.get("PROCESS_WIP_CONFIRM_MAX", "6"), 6)
    sla_strict_pass_sec = parse_int(os.environ.get("PROCESS_SLA_STRICT_PASS_SEC", "21600"), 21600)
    sla_confirm_pending_sec = parse_int(
        os.environ.get("PROCESS_SLA_CONFIRM_PENDING_SEC", os.environ.get("SLA_CONFIRM_PENDING_SEC", "7200")),
        7200,
    )
    sla_no_runner_pending_sec = parse_int(os.environ.get("PROCESS_SLA_NO_RUNNER_PENDING_SEC", "900"), 900)
    stalled_ratio_warn = parse_float(os.environ.get("PROCESS_STALLED_RATIO_WARN", "0.60"), 0.60)
    event_cooldown_sec = parse_int(os.environ.get("PROCESS_SLO_EVENT_COOLDOWN_SEC", "1800"), 1800)
    min_groups = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_GROUPS", "2"), 2)
    min_replies = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_REPLIES", "2"), 2)
    remote_runtime_max_age_sec = parse_int(os.environ.get("REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC", "90"), 90)

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        prev_state = load_json(state_path, {})
        if not isinstance(prev_state, dict):
            prev_state = {}

        process_start_epoch = parse_int(prev_state.get("process_start_epoch"), 0)
        if process_start_epoch <= 0:
            process_start_epoch = now

        qstats = queue_stats(aggregate_root=aggregate_root, app_root=root)
        totals = qstats.get("totals", {}) if isinstance(qstats, dict) else {}

        total_rows = parse_int(totals.get("total"), 0)
        completed_rows = parse_int(totals.get("completed"), 0)
        pending_rows = parse_int(totals.get("pending"), 0)
        running_rows = parse_int(totals.get("running"), 0)
        stalled_rows = parse_int(totals.get("stalled"), 0)
        failed_rows = parse_int(totals.get("failed"), 0)
        executable_rows = parse_int(totals.get("executable_rows"), 0)
        executable_pending_rows = parse_int(totals.get("executable_pending"), 0)
        dispatchable_pending_rows = parse_int(totals.get("dispatchable_pending"), 0)
        hard_rejected_pending_rows = parse_int(totals.get("hard_rejected_pending"), 0)

        fullspan_state = load_json(fullspan_state_path, {})
        yield_governor_state = read_yield_governor_state(yield_governor_state_path)
        ready_buffer_state = read_ready_queue_buffer(ready_buffer_state_path)
        queue_seeder_state = read_queue_seeder_state(queue_seeder_state_path)
        search_quality_state = derive_search_quality_state(
            yield_governor_state=yield_governor_state,
            queue_seeder_state=queue_seeder_state,
        )
        candidate_pool_ready_count = csv_data_row_count(candidate_pool_path)
        queues = fullspan_state.get("queues", {}) if isinstance(fullspan_state, dict) else {}
        if not isinstance(queues, dict):
            queues = {}
        runtime_metrics = fullspan_state.get("runtime_metrics", {}) if isinstance(fullspan_state, dict) else {}
        if not isinstance(runtime_metrics, dict):
            runtime_metrics = {}

        strict_pass_count = 0
        confirm_ready_count = 0
        promote_eligible_count = 0
        confirm_pending_count = 0
        overdue_confirm_queues: list[str] = []
        verdict_counts = Counter()

        for queue_rel, entry in queues.items():
            if not isinstance(entry, dict):
                continue
            verdict = str(entry.get("promotion_verdict") or "ANALYZE").strip().upper() or "ANALYZE"
            verdict_counts[verdict] += 1

            strict_pass = parse_int(entry.get("strict_pass_count"), 0)
            strict_groups = parse_int(entry.get("strict_run_group_count"), 0)
            confirm_count = parse_int(entry.get("confirm_count"), 0)
            confirm_lineage_count = parse_int(entry.get("confirm_verified_lineage_count"), 0)
            contract_hard_pass = bool(entry.get("contract_hard_pass"))

            if strict_pass > 0:
                strict_pass_count += 1

            if verdict in CONFIRM_PENDING_VERDICTS:
                confirm_pending_count += 1
                pending_since = parse_int(entry.get("confirm_pending_since_epoch"), 0)
                if pending_since > 0 and (now - pending_since) >= sla_confirm_pending_sec:
                    overdue_confirm_queues.append(str(queue_rel))

            if verdict == "PROMOTE_ELIGIBLE":
                promote_eligible_count += 1

            if (
                strict_pass > 0
                and strict_groups >= min_groups
                and confirm_count >= min_replies
                and confirm_lineage_count >= min_replies
                and contract_hard_pass
            ):
                confirm_ready_count += 1

        prev_epoch = parse_int(prev_state.get("ts_epoch"), now)
        prev_completed_rows = parse_int(prev_state.get("kpi", {}).get("completed_rows"), completed_rows)
        elapsed_sec = max(1, now - prev_epoch)
        completed_delta = max(0, completed_rows - prev_completed_rows)
        throughput_completed_per_hour = float(completed_delta) * 3600.0 / float(elapsed_sec)

        strict_pass_rate = (float(strict_pass_count) / float(completed_rows)) if completed_rows > 0 else 0.0
        confirm_conversion_rate = (
            float(confirm_ready_count) / float(strict_pass_count) if strict_pass_count > 0 else 0.0
        )
        promote_conversion_rate = (
            float(promote_eligible_count) / float(confirm_ready_count) if confirm_ready_count > 0 else 0.0
        )

        first_strict_pass_epoch = parse_int(prev_state.get("first_strict_pass_epoch"), 0)
        if first_strict_pass_epoch <= 0 and strict_pass_count > 0:
            first_strict_pass_epoch = now

        first_promote_epoch = parse_int(prev_state.get("first_promote_epoch"), 0)
        if first_promote_epoch <= 0 and promote_eligible_count > 0:
            first_promote_epoch = now

        lead_time_to_promote_min = None
        if first_strict_pass_epoch > 0 and first_promote_epoch > 0 and first_promote_epoch >= first_strict_pass_epoch:
            lead_time_to_promote_min = round((first_promote_epoch - first_strict_pass_epoch) / 60.0, 2)

        local_runner_count = detect_local_runner_count()
        remote_runtime_snapshot = read_remote_runtime_snapshot(remote_runtime_state_path)
        remote_snapshot = read_remote_runner_snapshot(capacity_state_path)
        remote_snapshot_age_sec = -1
        remote_runtime_ts = parse_int(remote_runtime_snapshot.get("ts_epoch"), 0)
        if remote_runtime_ts > 0:
            remote_snapshot_age_sec = max(0, now - remote_runtime_ts)
        remote_runtime_fresh = bool(
            remote_snapshot_age_sec >= 0 and remote_snapshot_age_sec <= max(0, int(remote_runtime_max_age_sec))
        )
        if remote_runtime_fresh:
            remote_runner_count = parse_int(remote_runtime_snapshot.get("remote_runner_count"), -1)
            remote_load1 = parse_float(remote_runtime_snapshot.get("load1"), -1.0)
            remote_reachable = bool(remote_runtime_snapshot.get("reachable"))
            remote_queue_job_count = parse_int(
                remote_runtime_snapshot.get("remote_queue_job_count"),
                parse_int(
                    remote_runtime_snapshot.get("remote_active_queue_jobs"),
                    parse_int(remote_runtime_snapshot.get("top_level_queue_jobs"), 0),
                ),
            )
            remote_active_queue_jobs = parse_int(
                remote_runtime_snapshot.get("remote_active_queue_jobs"),
                remote_queue_job_count,
            )
            top_level_queue_jobs = parse_int(remote_runtime_snapshot.get("top_level_queue_jobs"), 0)
            watch_queue_count = parse_int(remote_runtime_snapshot.get("watch_queue_count"), 0)
            remote_child_process_count = parse_int(remote_runtime_snapshot.get("remote_child_process_count"), 0)
            remote_work_active = bool(remote_runtime_snapshot.get("remote_work_active"))
            cpu_busy_without_queue_job = bool(remote_runtime_snapshot.get("cpu_busy_without_queue_job"))
            postprocess_active = bool(remote_runtime_snapshot.get("postprocess_active"))
            build_index_active = bool(remote_runtime_snapshot.get("build_index_active"))
            active_remote_queue_rel = str(remote_runtime_snapshot.get("active_remote_queue_rel") or "")
            remote_queue_sync_age_sec = parse_int(remote_runtime_snapshot.get("remote_queue_sync_age_sec"), -1)
            active_queues = list(remote_runtime_snapshot.get("active_queues") or [])
        else:
            remote_runner_count = parse_int(remote_snapshot.get("runner_count"), -1)
            remote_load1 = parse_float(remote_snapshot.get("load1"), 0.0)
            remote_reachable = bool(remote_snapshot.get("reachable"))
            remote_active_queue_jobs = parse_int(
                remote_snapshot.get("remote_active_queue_jobs"),
                parse_int(
                    runtime_metrics.get("remote_active_queue_jobs"),
                    parse_int(
                        runtime_metrics.get("remote_queue_job_count"),
                        parse_int(runtime_metrics.get("top_level_queue_jobs"), 0),
                    ),
                ),
            )
            remote_queue_job_count = parse_int(
                remote_snapshot.get("remote_queue_job_count"),
                parse_int(
                    remote_snapshot.get("remote_active_queue_jobs"),
                    parse_int(runtime_metrics.get("remote_active_queue_jobs"), 0),
                ),
            )
            top_level_queue_jobs = parse_int(
                remote_snapshot.get("top_level_queue_jobs"),
                parse_int(runtime_metrics.get("top_level_queue_jobs"), remote_active_queue_jobs),
            )
            watch_queue_count = parse_int(remote_snapshot.get("watch_queue_count"), 0)
            remote_child_process_count = parse_int(
                remote_snapshot.get("remote_child_process_count"),
                parse_int(runtime_metrics.get("remote_child_process_count"), remote_runner_count),
            )
            remote_work_active = parse_bool(
                remote_snapshot.get("remote_work_active"),
                parse_bool(runtime_metrics.get("remote_work_active"), False),
            )
            cpu_busy_without_queue_job = parse_bool(
                remote_snapshot.get("cpu_busy_without_queue_job"),
                parse_bool(runtime_metrics.get("cpu_busy_without_queue_job"), False),
            )
            postprocess_active = parse_bool(runtime_metrics.get("postprocess_active"), False)
            build_index_active = parse_bool(runtime_metrics.get("build_index_active"), False)
            active_remote_queue_rel = str(runtime_metrics.get("active_remote_queue_rel") or "")
            remote_queue_sync_age_sec = parse_int(runtime_metrics.get("remote_queue_sync_age_sec"), -1)
            active_queues = []
        ready_buffer_depth = parse_int(runtime_metrics.get("ready_buffer_depth"), 0)
        cold_fail_active_count = parse_int(runtime_metrics.get("cold_fail_active_count"), 0)
        if cold_fail_active_count <= 0:
            cold_fail_active_count = count_active_cold_fail_entries(cold_fail_index_path, now=now)
        if not remote_work_active:
            remote_work_active = bool(
                remote_queue_job_count > 0
                or remote_child_process_count > 0
                or postprocess_active
                or build_index_active
                or (remote_reachable and remote_load1 >= 1.5)
            )
        if not cpu_busy_without_queue_job:
            cpu_busy_without_queue_job = bool(
                remote_reachable
                and remote_queue_job_count <= 0
                and (
                    remote_child_process_count > 0
                    or postprocess_active
                    or build_index_active
                    or remote_load1 >= 1.5
                )
            )
        surrogate_idle_override_count = parse_int(runtime_metrics.get("surrogate_idle_override_count"), 0)
        overlap_dispatch_count = parse_int(runtime_metrics.get("overlap_dispatch_count"), 0)
        vps_duty_cycle_30m = parse_float(runtime_metrics.get("vps_duty_cycle_30m"), 0.0)
        ready_buffer_policy_mismatch_count = parse_int(
            runtime_metrics.get("ready_buffer_policy_mismatch_count"),
            0,
        )
        winner_parent_duplication_rate = parse_float(runtime_metrics.get("winner_parent_duplication_rate"), 0.0)
        fastlane_replay_pending = parse_int(runtime_metrics.get("fastlane_replay_pending"), 0)
        metrics_missing_abort_count_30m = parse_int(runtime_metrics.get("metrics_missing_abort_count_30m"), 0)
        winner_proximate_dispatch_count_30m = parse_int(
            runtime_metrics.get("winner_proximate_dispatch_count_30m"),
            0,
        )
        hot_standby_active = parse_bool(runtime_metrics.get("hot_standby_active"), False)
        infra_gate_status = str(runtime_metrics.get("infra_gate_status") or "").strip()
        infra_gate_reason = str(runtime_metrics.get("infra_gate_reason") or "").strip()
        startup_failure_code = str(runtime_metrics.get("startup_failure_code") or "").strip()
        auto_seed_blocked = parse_bool(
            runtime_metrics.get("auto_seed_blocked"),
            yield_governor_state.get("hard_block_active", False),
        )
        auto_seed_block_reason = str(
            runtime_metrics.get("auto_seed_block_reason")
            or yield_governor_state.get("hard_block_reason")
            or ""
        ).strip()
        if not infra_gate_status and parse_bool(runtime_metrics.get("vps_infra_fail_closed"), False):
            infra_gate_status = "hard_block"
            if not infra_gate_reason:
                infra_gate_reason = "vps_infra_fail_closed"
        coverage_verified_ready_count = parse_int(
            runtime_metrics.get("coverage_verified_ready_count"),
            parse_int(
                ready_buffer_state.get("coverage_verified_ready_count"),
                parse_int(queue_seeder_state.get("coverage_verified_ready_count"), 0),
            ),
        )
        covered_window_count = parse_int(
            runtime_metrics.get("covered_window_count"),
            parse_int(queue_seeder_state.get("covered_window_count"), 0),
        )
        progress_source = "local_queue"
        if remote_runtime_fresh and (
            active_remote_queue_rel
            or remote_queue_sync_age_sec >= 0
            or postprocess_active
            or build_index_active
        ):
            progress_source = "remote_runtime_state"
        elif remote_work_active:
            progress_source = "capacity_or_runtime_metrics"

        idle_with_dispatchable_pending = bool(
            dispatchable_pending_rows > 0
            and local_runner_count <= 0
            and remote_reachable
            and not remote_work_active
        )
        candidate_pool_status = str(ready_buffer_state.get("candidate_pool_status") or "").strip().lower()
        if candidate_pool_ready_count > 0:
            candidate_pool_status = "ready"
        elif dispatchable_pending_rows > 0:
            candidate_pool_status = "empty_error"
        else:
            candidate_pool_status = "empty_expected"

        no_runner_since_epoch = parse_int(prev_state.get("no_runner_since_epoch"), 0)
        if dispatchable_pending_rows > 0 and local_runner_count <= 0:
            if no_runner_since_epoch <= 0:
                no_runner_since_epoch = now
        else:
            no_runner_since_epoch = 0

        no_runner_pending_age_sec = (now - no_runner_since_epoch) if no_runner_since_epoch > 0 else 0
        stalled_ratio = (float(stalled_rows) / float(pending_rows)) if pending_rows > 0 else 0.0

        alerts: list[dict[str, Any]] = []

        if pending_rows > wip_search_max:
            alerts.append(
                {
                    "code": "WIP_SEARCH_OVERFLOW",
                    "severity": "warning",
                    "message": f"pending={pending_rows} > wip_search_max={wip_search_max}",
                }
            )
        if confirm_pending_count > wip_confirm_max:
            alerts.append(
                {
                    "code": "WIP_CONFIRM_OVERFLOW",
                    "severity": "warning",
                    "message": f"confirm_pending={confirm_pending_count} > wip_confirm_max={wip_confirm_max}",
                }
            )
        if strict_pass_count <= 0 and (now - process_start_epoch) >= sla_strict_pass_sec:
            alerts.append(
                {
                    "code": "SLA_STRICT_PASS_BREACH",
                    "severity": "critical",
                    "message": (
                        f"time_to_first_strict_pass={now - process_start_epoch}s >= "
                        f"{sla_strict_pass_sec}s"
                    ),
                }
            )
        if overdue_confirm_queues:
            alerts.append(
                {
                    "code": "SLA_CONFIRM_PENDING_BREACH",
                    "severity": "warning",
                    "message": (
                        f"overdue_confirm_queues={len(overdue_confirm_queues)} "
                        f"(sla={sla_confirm_pending_sec}s)"
                    ),
                    "queues": overdue_confirm_queues[:20],
                }
            )
        if no_runner_pending_age_sec >= sla_no_runner_pending_sec:
            alerts.append(
                {
                    "code": "SLA_NO_RUNNER_PENDING",
                    "severity": "warning",
                    "message": (
                        f"dispatchable_pending={dispatchable_pending_rows} without local_runner for "
                        f"{no_runner_pending_age_sec}s >= {sla_no_runner_pending_sec}s"
                    ),
                }
            )
        if pending_rows > 0 and stalled_ratio >= stalled_ratio_warn:
            alerts.append(
                {
                    "code": "STALLED_RATIO_HIGH",
                    "severity": "warning",
                    "message": (
                        f"stalled_ratio={stalled_ratio:.3f} >= warn={stalled_ratio_warn:.3f} "
                        f"(stalled={stalled_rows}, pending={pending_rows})"
                    ),
                }
            )

        event_state = dict(prev_state)
        emitted = 0
        for alert in alerts:
            code = str(alert.get("code") or "UNKNOWN")
            key = f"process_alert:{code}"
            if should_emit(state=event_state, key=key, cooldown_sec=event_cooldown_sec, now=now):
                append_jsonl(
                    events_path,
                    {
                        "ts": ts,
                        "event": code,
                        "severity": alert.get("severity", "warning"),
                        "payload": alert,
                    },
                )
                emitted += 1

        summary = {
            "ts": ts,
            "ts_epoch": now,
            "process_start_epoch": process_start_epoch,
            "first_strict_pass_epoch": first_strict_pass_epoch,
            "first_promote_epoch": first_promote_epoch,
            "progress_source": progress_source,
            "active_remote_queue_rel": active_remote_queue_rel,
            "remote_queue_sync_age_sec": remote_queue_sync_age_sec,
            "infra_gate_status": infra_gate_status,
            "infra_gate_reason": infra_gate_reason,
            "auto_seed_blocked": auto_seed_blocked,
            "auto_seed_block_reason": auto_seed_block_reason,
            "covered_window_count": covered_window_count,
            "coverage_verified_ready_count": coverage_verified_ready_count,
            "startup_failure_code": startup_failure_code,
            "search_quality": search_quality_state,
            "funnel": {
                "generated": total_rows,
                "executable": executable_rows,
                "completed": completed_rows,
                "strict_pass": strict_pass_count,
                "confirm_ready": confirm_ready_count,
                "promote_eligible": promote_eligible_count,
            },
            "queue": {
                "pending": pending_rows,
                "executable_pending": executable_pending_rows,
                "dispatchable_pending": dispatchable_pending_rows,
                "hard_rejected_pending": hard_rejected_pending_rows,
                "running": running_rows,
                "stalled": stalled_rows,
                "failed": failed_rows,
                "local_runner_count": local_runner_count,
                "remote_runner_count": remote_runner_count,
                "remote_child_process_count": remote_child_process_count,
                "remote_queue_job_count": remote_queue_job_count,
                "remote_active_queue_jobs": remote_active_queue_jobs,
                "top_level_queue_jobs": top_level_queue_jobs,
                "watch_queue_count": watch_queue_count,
                "remote_load1": round(remote_load1, 3),
                "remote_work_active": remote_work_active,
                "remote_snapshot_age_sec": remote_snapshot_age_sec if remote_runtime_fresh else -1,
                "cpu_busy_without_queue_job": cpu_busy_without_queue_job,
                "postprocess_active": postprocess_active,
                "build_index_active": build_index_active,
                "remote_reachable": remote_reachable,
                "idle_with_executable_pending": idle_with_dispatchable_pending,
                "idle_with_dispatchable_pending": idle_with_dispatchable_pending,
                "progress_source": progress_source,
                "active_remote_queue_rel": active_remote_queue_rel,
                "remote_queue_sync_age_sec": remote_queue_sync_age_sec,
                "active_queues": active_queues[:8],
                "ready_buffer_depth": ready_buffer_depth,
                "candidate_pool_status": candidate_pool_status,
                "cold_fail_active_count": cold_fail_active_count,
                "fastlane_replay_pending": fastlane_replay_pending,
                "hot_standby_active": hot_standby_active,
                "coverage_verified_ready_count": coverage_verified_ready_count,
                "covered_window_count": covered_window_count,
                "infra_gate_status": infra_gate_status,
                "infra_gate_reason": infra_gate_reason,
                "auto_seed_blocked": auto_seed_blocked,
                "auto_seed_block_reason": auto_seed_block_reason,
                "startup_failure_code": startup_failure_code,
                "positive_lineage_count": search_quality_state["positive_lineage_count"],
                "zero_evidence_lineage_count": search_quality_state["zero_evidence_lineage_count"],
                "winner_proximate_positive_contains": search_quality_state["winner_proximate_positive_contains"],
                "broad_search_allowed": search_quality_state["broad_search_allowed"],
                "seed_generation_mode": search_quality_state["seed_generation_mode"],
                "zero_coverage_seed_streak": search_quality_state["zero_coverage_seed_streak"],
                "zero_coverage_seed_streak_reason": search_quality_state["zero_coverage_seed_streak_reason"],
                "controlled_recovery_active": search_quality_state["controlled_recovery_active"],
                "controlled_recovery_reason": search_quality_state["controlled_recovery_reason"],
                "controlled_recovery_attempts_remaining": search_quality_state["controlled_recovery_attempts_remaining"],
                "controlled_recovery_variants_cap": search_quality_state["controlled_recovery_variants_cap"],
            },
            "kpi": {
                "completed_rows": completed_rows,
                "completed_delta": completed_delta,
                "throughput_completed_per_hour": round(throughput_completed_per_hour, 3),
                "strict_pass_rate": round(strict_pass_rate, 6),
                "confirm_conversion_rate": round(confirm_conversion_rate, 6),
                "promote_conversion_rate": round(promote_conversion_rate, 6),
                "lead_time_to_promote_min": lead_time_to_promote_min,
                "executable_pending_rows": executable_pending_rows,
                "dispatchable_pending_rows": dispatchable_pending_rows,
                "hard_rejected_pending_rows": hard_rejected_pending_rows,
                "local_runner_count": local_runner_count,
                "remote_runner_count": remote_runner_count,
                "remote_child_process_count": remote_child_process_count,
                "remote_queue_job_count": remote_queue_job_count,
                "remote_active_queue_jobs": remote_active_queue_jobs,
                "top_level_queue_jobs": top_level_queue_jobs,
                "watch_queue_count": watch_queue_count,
                "remote_load1": round(remote_load1, 3),
                "remote_work_active": remote_work_active,
                "remote_snapshot_age_sec": remote_snapshot_age_sec if remote_runtime_fresh else -1,
                "cpu_busy_without_queue_job": cpu_busy_without_queue_job,
                "postprocess_active": postprocess_active,
                "build_index_active": build_index_active,
                "progress_source": progress_source,
                "active_remote_queue_rel": active_remote_queue_rel,
                "remote_queue_sync_age_sec": remote_queue_sync_age_sec,
                "idle_with_executable_pending": idle_with_dispatchable_pending,
                "idle_with_dispatchable_pending": idle_with_dispatchable_pending,
                "candidate_pool_status": candidate_pool_status,
                "vps_duty_cycle_30m": round(vps_duty_cycle_30m, 6),
                "ready_buffer_policy_mismatch_count": ready_buffer_policy_mismatch_count,
                "winner_parent_duplication_rate": round(winner_parent_duplication_rate, 6),
                "fastlane_replay_pending": fastlane_replay_pending,
                "metrics_missing_abort_count_30m": metrics_missing_abort_count_30m,
                "winner_proximate_dispatch_count_30m": winner_proximate_dispatch_count_30m,
                "coverage_verified_ready_count": coverage_verified_ready_count,
                "covered_window_count": covered_window_count,
                "auto_seed_blocked": auto_seed_blocked,
                "startup_failure_code": startup_failure_code,
                "positive_lineage_count": search_quality_state["positive_lineage_count"],
                "zero_evidence_lineage_count": search_quality_state["zero_evidence_lineage_count"],
                "winner_proximate_positive_contains": search_quality_state["winner_proximate_positive_contains"],
                "controlled_recovery_active": search_quality_state["controlled_recovery_active"],
                "controlled_recovery_attempts_remaining": search_quality_state["controlled_recovery_attempts_remaining"],
            },
            "runtime": {
                "ready_buffer_depth": ready_buffer_depth,
                "cold_fail_active_count": cold_fail_active_count,
                "remote_child_process_count": remote_child_process_count,
                "remote_queue_job_count": remote_queue_job_count,
                "remote_active_queue_jobs": remote_active_queue_jobs,
                "top_level_queue_jobs": top_level_queue_jobs,
                "watch_queue_count": watch_queue_count,
                "remote_load1": round(remote_load1, 3),
                "remote_work_active": remote_work_active,
                "remote_snapshot_age_sec": remote_snapshot_age_sec if remote_runtime_fresh else -1,
                "cpu_busy_without_queue_job": cpu_busy_without_queue_job,
                "postprocess_active": postprocess_active,
                "build_index_active": build_index_active,
                "progress_source": progress_source,
                "active_remote_queue_rel": active_remote_queue_rel,
                "remote_queue_sync_age_sec": remote_queue_sync_age_sec,
                "active_queues": active_queues[:8],
                "surrogate_idle_override_count": surrogate_idle_override_count,
                "overlap_dispatch_count": overlap_dispatch_count,
                "candidate_pool_status": candidate_pool_status,
                "vps_duty_cycle_30m": round(vps_duty_cycle_30m, 6),
                "ready_buffer_policy_mismatch_count": ready_buffer_policy_mismatch_count,
                "winner_parent_duplication_rate": round(winner_parent_duplication_rate, 6),
                "fastlane_replay_pending": fastlane_replay_pending,
                "metrics_missing_abort_count_30m": metrics_missing_abort_count_30m,
                "winner_proximate_dispatch_count_30m": winner_proximate_dispatch_count_30m,
                "hot_standby_active": hot_standby_active,
                "coverage_verified_ready_count": coverage_verified_ready_count,
                "covered_window_count": covered_window_count,
                "infra_gate_status": infra_gate_status,
                "infra_gate_reason": infra_gate_reason,
                "auto_seed_blocked": auto_seed_blocked,
                "auto_seed_block_reason": auto_seed_block_reason,
                "startup_failure_code": startup_failure_code,
                "positive_lineage_count": search_quality_state["positive_lineage_count"],
                "zero_evidence_lineage_count": search_quality_state["zero_evidence_lineage_count"],
                "winner_proximate_positive_contains": search_quality_state["winner_proximate_positive_contains"],
                "broad_search_allowed": search_quality_state["broad_search_allowed"],
                "seed_generation_mode": search_quality_state["seed_generation_mode"],
                "zero_coverage_seed_streak": search_quality_state["zero_coverage_seed_streak"],
                "zero_coverage_seed_streak_reason": search_quality_state["zero_coverage_seed_streak_reason"],
                "controlled_recovery_active": search_quality_state["controlled_recovery_active"],
                "controlled_recovery_reason": search_quality_state["controlled_recovery_reason"],
                "controlled_recovery_attempts_remaining": search_quality_state["controlled_recovery_attempts_remaining"],
                "controlled_recovery_variants_cap": search_quality_state["controlled_recovery_variants_cap"],
            },
            "wip": {
                "search_max": wip_search_max,
                "confirm_max": wip_confirm_max,
                "search_pending": pending_rows,
                "confirm_pending": confirm_pending_count,
                "stalled_ratio": round(stalled_ratio, 6),
                "stalled_ratio_warn": stalled_ratio_warn,
            },
            "sla": {
                "strict_pass_sec": sla_strict_pass_sec,
                "confirm_pending_sec": sla_confirm_pending_sec,
                "no_runner_pending_sec": sla_no_runner_pending_sec,
                "no_runner_pending_age_sec": no_runner_pending_age_sec,
            },
            "verdicts": dict(verdict_counts),
            "alerts": alerts,
            "alerts_count": len(alerts),
            "events_emitted": emitted,
            "overdue_confirm_queues": overdue_confirm_queues,
            "last_emit": event_state.get("last_emit", {}),
        }

        if not args.dry_run:
            dump_json(state_path, summary)

        append_log(
            log_path,
            (
                "funnel generated={generated} executable={executable} completed={completed} "
                "strict_pass={strict_pass} confirm_ready={confirm_ready} promote={promote_eligible} "
                "pending={pending} stalled={stalled} local_runner={local_runner} remote_runner={remote_runner} "
                "remote_child_process_count={remote_child_process_count} remote_queue_job_count={remote_queue_job_count} "
                "remote_active_queue_jobs={remote_active_queue_jobs} remote_work_active={remote_work_active} "
                "remote_snapshot_age_sec={remote_snapshot_age_sec} ready_buffer_depth={ready_buffer_depth} "
                "progress_source={progress_source} active_remote_queue_rel={active_remote_queue_rel} "
                "remote_queue_sync_age_sec={remote_queue_sync_age_sec} postprocess_active={postprocess_active} "
                "build_index_active={build_index_active} "
                "infra_gate_status={infra_gate_status} infra_gate_reason={infra_gate_reason} "
                "auto_seed_blocked={auto_seed_blocked} auto_seed_block_reason={auto_seed_block_reason} "
                "covered_window_count={covered_window_count} coverage_verified_ready_count={coverage_verified_ready_count} "
                "startup_failure_code={startup_failure_code} "
                "cold_fail_active_count={cold_fail_active_count} surrogate_idle_override_count={surrogate_idle_override_count} "
                "overlap_dispatch_count={overlap_dispatch_count} cpu_busy_without_queue_job={cpu_busy_without_queue_job} "
                "vps_duty_cycle_30m={vps_duty_cycle_30m:.3f} ready_buffer_policy_mismatch_count={ready_buffer_policy_mismatch_count} "
                "winner_parent_duplication_rate={winner_parent_duplication_rate:.3f} fastlane_replay_pending={fastlane_replay_pending} "
                "metrics_missing_abort_count_30m={metrics_missing_abort_count_30m} winner_proximate_dispatch_count_30m={winner_proximate_dispatch_count_30m} "
                "hot_standby_active={hot_standby_active} "
                "dispatchable_pending={dispatchable_pending} hard_rejected_pending={hard_rejected_pending} "
                "positive_lineage_count={positive_lineage_count} zero_evidence_lineage_count={zero_evidence_lineage_count} "
                "broad_search_allowed={broad_search_allowed} seed_generation_mode={seed_generation_mode} "
                "candidate_pool_status={candidate_pool_status} "
                "idle_with_executable_pending={idle_with_executable_pending} "
                "alerts={alerts} emitted={emitted}"
            ).format(
                generated=summary["funnel"]["generated"],
                executable=summary["funnel"]["executable"],
                completed=summary["funnel"]["completed"],
                strict_pass=summary["funnel"]["strict_pass"],
                confirm_ready=summary["funnel"]["confirm_ready"],
                promote_eligible=summary["funnel"]["promote_eligible"],
                pending=summary["queue"]["pending"],
                stalled=summary["queue"]["stalled"],
                local_runner=summary["queue"]["local_runner_count"],
                remote_runner=summary["queue"]["remote_runner_count"],
                remote_child_process_count=summary["queue"]["remote_child_process_count"],
                remote_queue_job_count=summary["queue"]["remote_queue_job_count"],
                remote_active_queue_jobs=summary["queue"]["remote_active_queue_jobs"],
                remote_work_active=int(bool(summary["queue"]["remote_work_active"])),
                remote_snapshot_age_sec=summary["queue"]["remote_snapshot_age_sec"],
                ready_buffer_depth=summary["queue"]["ready_buffer_depth"],
                progress_source=summary["queue"]["progress_source"],
                active_remote_queue_rel=summary["queue"]["active_remote_queue_rel"] or "none",
                remote_queue_sync_age_sec=summary["queue"]["remote_queue_sync_age_sec"],
                postprocess_active=int(bool(summary["queue"]["postprocess_active"])),
                build_index_active=int(bool(summary["queue"]["build_index_active"])),
                infra_gate_status=summary["queue"]["infra_gate_status"] or "none",
                infra_gate_reason=summary["queue"]["infra_gate_reason"] or "none",
                auto_seed_blocked=int(bool(summary["queue"]["auto_seed_blocked"])),
                auto_seed_block_reason=summary["queue"]["auto_seed_block_reason"] or "none",
                covered_window_count=summary["queue"]["covered_window_count"],
                coverage_verified_ready_count=summary["queue"]["coverage_verified_ready_count"],
                startup_failure_code=summary["queue"]["startup_failure_code"] or "none",
                cold_fail_active_count=summary["queue"]["cold_fail_active_count"],
                surrogate_idle_override_count=summary["runtime"]["surrogate_idle_override_count"],
                overlap_dispatch_count=summary["runtime"]["overlap_dispatch_count"],
                cpu_busy_without_queue_job=int(bool(summary["runtime"]["cpu_busy_without_queue_job"])),
                vps_duty_cycle_30m=summary["runtime"]["vps_duty_cycle_30m"],
                ready_buffer_policy_mismatch_count=summary["runtime"]["ready_buffer_policy_mismatch_count"],
                winner_parent_duplication_rate=summary["runtime"]["winner_parent_duplication_rate"],
                fastlane_replay_pending=summary["runtime"]["fastlane_replay_pending"],
                metrics_missing_abort_count_30m=summary["runtime"]["metrics_missing_abort_count_30m"],
                winner_proximate_dispatch_count_30m=summary["runtime"]["winner_proximate_dispatch_count_30m"],
                hot_standby_active=int(bool(summary["runtime"]["hot_standby_active"])),
                dispatchable_pending=summary["queue"]["dispatchable_pending"],
                hard_rejected_pending=summary["queue"]["hard_rejected_pending"],
                positive_lineage_count=summary["search_quality"]["positive_lineage_count"],
                zero_evidence_lineage_count=summary["search_quality"]["zero_evidence_lineage_count"],
                broad_search_allowed=int(bool(summary["search_quality"]["broad_search_allowed"])),
                seed_generation_mode=summary["search_quality"]["seed_generation_mode"] or "unknown",
                candidate_pool_status=summary["queue"]["candidate_pool_status"] or "none",
                idle_with_executable_pending=int(bool(summary["queue"]["idle_with_executable_pending"])),
                alerts=summary["alerts_count"],
                emitted=summary["events_emitted"],
            ),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
