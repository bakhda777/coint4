#!/usr/bin/env python3
"""Lightweight probe for autonomous WFA state markers.

Purpose:
- Resolve coint4 app root/state dir without relying on a single hardcoded path.
- Return strict/promote/fail-closed and runtime observability markers in deterministic form.
- Optionally materialize process_slo_state.json by running process_slo_guard_agent.py once.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return default


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def detect_app_root_candidates(cli_root: str) -> tuple[list[Path], list[str]]:
    raw_inputs: list[str] = []

    def add_raw(value: str) -> None:
        text = str(value or "").strip()
        if text:
            raw_inputs.append(text)

    if cli_root and cli_root.lower() != "auto":
        add_raw(cli_root)

    for env_key in ("COINT4_APP_ROOT", "ROOT_DIR", "SERVER_WORK_DIR", "SERVER_REPO_DIR"):
        add_raw(os.environ.get(env_key, ""))

    add_raw(str(Path.cwd()))
    script_path = Path(__file__).resolve()
    add_raw(str(script_path.parents[2]))
    add_raw(str(script_path.parents[3]))

    for known in (
        "/opt/coint4/coint4",
        "/opt/coint4",
        "/home/claudeuser/coint4/coint4",
        "/home/claudeuser/coint4",
        "/root/coint4/coint4",
        "/root/coint4",
    ):
        add_raw(known)

    seen: set[str] = set()
    candidates: list[Path] = []
    checked: list[str] = []

    for raw in raw_inputs:
        base = Path(raw).expanduser()
        if not base.is_absolute():
            base = (Path.cwd() / base).resolve()
        else:
            base = base.resolve()

        variants = [base]
        if base.name != "coint4":
            variants.append(base / "coint4")

        for variant in variants:
            key = str(variant)
            if key in seen:
                continue
            seen.add(key)
            checked.append(key)
            candidates.append(variant)

    return candidates, checked


def score_app_root(path: Path) -> int:
    score = 0
    if (path / "scripts" / "optimization" / "autonomous_wfa_driver.sh").is_file():
        score += 30
    if (path / "scripts" / "optimization" / "process_slo_guard_agent.py").is_file():
        score += 20
    if (path / "src").is_dir():
        score += 10
    if (path / "pyproject.toml").is_file():
        score += 5
    if (path / "artifacts" / "wfa" / "aggregate").is_dir():
        score += 5
    if (path / "artifacts" / "wfa" / "aggregate" / ".autonomous").is_dir():
        score += 40
    return score


def resolve_layout(cli_root: str, cli_state_dir: str) -> dict[str, Any]:
    candidates, checked = detect_app_root_candidates(cli_root)

    if cli_state_dir:
        state_dir = Path(cli_state_dir).expanduser()
        if not state_dir.is_absolute():
            state_dir = (Path.cwd() / state_dir).resolve()
        else:
            state_dir = state_dir.resolve()

        inferred_root = state_dir
        for _ in range(4):
            inferred_root = inferred_root.parent
        return {
            "app_root": inferred_root,
            "state_dir": state_dir,
            "checked_candidates": checked,
            "resolution": "explicit_state_dir",
        }

    scored: list[tuple[int, Path]] = []
    for candidate in candidates:
        scored.append((score_app_root(candidate), candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return {
            "app_root": None,
            "state_dir": None,
            "checked_candidates": checked,
            "resolution": "no_candidates",
        }

    best_score, best_root = scored[0]
    if best_score <= 0:
        return {
            "app_root": None,
            "state_dir": None,
            "checked_candidates": checked,
            "resolution": "no_valid_root",
        }

    state_dir = best_root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    return {
        "app_root": best_root,
        "state_dir": state_dir,
        "checked_candidates": checked,
        "resolution": "scored_candidates",
        "best_score": best_score,
    }


def file_meta(path: Path) -> dict[str, Any]:
    exists = path.exists()
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": bool(exists),
    }
    if exists:
        try:
            stat = path.stat()
            payload["size"] = int(stat.st_size)
            payload["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return payload


def parse_utc_epoch(raw_ts: Any) -> int | None:
    text = str(raw_ts or "").strip()
    if not text:
        return None
    try:
        dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def freshness_payload(*, path: Path, ts: Any, now_epoch: int, fresh_sec: int) -> dict[str, Any]:
    ts_epoch = parse_utc_epoch(ts)
    source = "payload_ts"
    if ts_epoch is None:
        try:
            ts_epoch = int(path.stat().st_mtime)
            source = "file_mtime"
        except Exception:
            ts_epoch = None
            source = "unknown"
    age_sec: int | None = None
    if ts_epoch is not None:
        age_sec = max(0, int(now_epoch - ts_epoch))
    return {
        "ts": str(ts or ""),
        "source": source,
        "age_sec": age_sec,
        "fresh_sec": int(fresh_sec),
        "fresh": bool(age_sec is not None and age_sec <= int(fresh_sec)),
    }


def _empty_surrogate_action_stats() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for action in ("SURROGATE_REJECT", "SURROGATE_REFINE", "SURROGATE_ALLOW"):
        out[action] = {
            "count": 0,
            "last_ts": "",
            "last_age_sec": None,
            "present": False,
        }
    return out


def _update_action_stats(stats: dict[str, dict[str, Any]], action: str, ts: str, now_epoch: int) -> None:
    payload = stats.get(action)
    if not isinstance(payload, dict):
        return
    payload["count"] = int(payload.get("count", 0)) + 1
    payload["present"] = True
    prev_age = payload.get("last_age_sec")
    event_epoch = parse_utc_epoch(ts)
    if event_epoch is None:
        if not payload.get("last_ts"):
            payload["last_ts"] = str(ts or "")
        return
    age_sec = max(0, int(now_epoch - event_epoch))
    if prev_age is None or age_sec <= int(prev_age):
        payload["last_ts"] = str(ts or "")
        payload["last_age_sec"] = age_sec


def surrogate_action_stats_from_decision_notes(*, records: list[dict[str, Any]], now_epoch: int) -> dict[str, dict[str, Any]]:
    stats = _empty_surrogate_action_stats()
    for rec in records:
        action = str(rec.get("action") or "").strip().upper()
        if action not in stats:
            continue
        _update_action_stats(stats, action, str(rec.get("ts") or ""), now_epoch)
    return stats


def surrogate_action_stats_from_driver_log(path: Path, *, now_epoch: int, max_lines: int = 5000) -> dict[str, dict[str, Any]]:
    stats = _empty_surrogate_action_stats()
    if not path.exists():
        return stats
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return stats

    action_re = re.compile(r"\baction=(SURROGATE_(?:REJECT|REFINE|ALLOW))\b")
    ts_re = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+\|")
    for line in lines[-max_lines:]:
        match = action_re.search(line)
        if not match:
            continue
        action = str(match.group(1) or "").strip().upper()
        if action not in stats:
            continue
        ts_match = ts_re.search(line)
        event_ts = ts_match.group(1) if ts_match else ""
        _update_action_stats(stats, action, event_ts, now_epoch)
    return stats


def recent_candidate_queues_from_driver_log(path: Path, *, now_epoch: int, max_lines: int = 5000, max_age_sec: int = 180) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    out: list[str] = []
    seen: set[str] = set()
    queue_re = re.compile(r"\bcandidate queue=([^ ]+)")
    ts_re = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+\|")
    for line in reversed(lines[-max_lines:]):
        ts_match = ts_re.search(line)
        if ts_match:
            event_epoch = parse_utc_epoch(ts_match.group(1))
            if event_epoch > 0 and max(0, int(now_epoch - event_epoch)) > int(max_age_sec):
                continue
        match = queue_re.search(line)
        if not match:
            continue
        queue = str(match.group(1) or "").strip()
        if not queue or queue in seen:
            continue
        seen.add(queue)
        out.append(queue)
        if len(out) >= 12:
            break
    return list(reversed(out))


def _runtime_metric_int(runtime: dict[str, Any], key: str) -> int:
    return parse_int(runtime.get(key), 0)


def _gate_decision_counts(gate_state: dict[str, Any]) -> dict[str, int]:
    counts = {"allow": 0, "refine": 0, "reject": 0}
    summary = gate_state.get("summary")
    if isinstance(summary, dict):
        raw_counts = summary.get("decision_counts")
        if isinstance(raw_counts, dict):
            for key in counts:
                counts[key] = parse_int(raw_counts.get(key), 0)
            return counts
    queues = gate_state.get("queues")
    if isinstance(queues, dict):
        for entry in queues.values():
            if not isinstance(entry, dict):
                continue
            decision = str(entry.get("decision") or "").strip().lower()
            if decision in counts:
                counts[decision] += 1
    return counts


def _combine_surrogate_evidence(
    notes_stats: dict[str, dict[str, Any]],
    log_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    combined = _empty_surrogate_action_stats()
    for action in combined:
        notes_item = notes_stats.get(action, {})
        log_item = log_stats.get(action, {})
        notes_count = parse_int(notes_item.get("count"), 0) if isinstance(notes_item, dict) else 0
        log_count = parse_int(log_item.get("count"), 0) if isinstance(log_item, dict) else 0
        notes_age = notes_item.get("last_age_sec") if isinstance(notes_item, dict) else None
        log_age = log_item.get("last_age_sec") if isinstance(log_item, dict) else None
        notes_ts = str(notes_item.get("last_ts") or "") if isinstance(notes_item, dict) else ""
        log_ts = str(log_item.get("last_ts") or "") if isinstance(log_item, dict) else ""
        best_age = None
        best_ts = ""
        if isinstance(notes_age, int):
            best_age = notes_age
            best_ts = notes_ts
        if isinstance(log_age, int) and (best_age is None or log_age < best_age):
            best_age = log_age
            best_ts = log_ts
        combined[action] = {
            "count": int(notes_count + log_count),
            "present": bool(notes_count + log_count > 0),
            "last_ts": best_ts,
            "last_age_sec": best_age,
            "sources": {
                "decision_notes": notes_count,
                "driver_log": log_count,
            },
        }
    return combined


def read_jsonl_tail(path: Path, max_lines: int = 5000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    records: list[dict[str, Any]] = []
    for line in lines[-max_lines:]:
        text = line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def count_fail_closed_in_log(path: Path, max_lines: int = 2000) -> int:
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return 0
    return sum(1 for line in lines[-max_lines:] if "FAIL_CLOSED" in line)


def parse_int_optional(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def dict_path_int(data: Any, path: tuple[str, ...]) -> int | None:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        if key not in cur:
            return None
        cur = cur.get(key)
    return parse_int_optional(cur)


def latest_counter_from_log(path: Path, keys: list[str], *, max_lines: int = 8000) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    for line in reversed(lines[-max_lines:]):
        for key in keys:
            patterns = (
                rf"\b{re.escape(key)}=([-+]?\d+)\b",
                rf'"{re.escape(key)}"\s*:\s*([-+]?\d+)\b',
            )
            for pattern in patterns:
                match = re.search(pattern, line)
                if not match:
                    continue
                value = parse_int_optional(match.group(1))
                if value is None:
                    continue
                return {
                    "value": max(0, int(value)),
                    "source": f"driver_log.{key}",
                    "raw_line": line[-320:],
                }
    return None


def count_token_mentions_in_decision_notes(records: list[dict[str, Any]], *, tokens: list[str]) -> int:
    items = [str(t or "").strip().upper() for t in tokens if str(t or "").strip()]
    if not items:
        return 0
    count = 0
    for rec in records:
        action = str(rec.get("action") or "")
        details = str(rec.get("details") or rec.get("detail") or "")
        reason = str(rec.get("reason") or rec.get("note") or "")
        next_step = str(rec.get("next_step") or "")
        hay = f"{action} {details} {reason} {next_step}".upper()
        if any(token in hay for token in items):
            count += 1
    return count


def count_token_mentions_in_log(path: Path, *, tokens: list[str], max_lines: int = 8000) -> int:
    items = [str(t or "").strip().lower() for t in tokens if str(t or "").strip()]
    if not items:
        return 0
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return 0
    count = 0
    for line in lines[-max_lines:]:
        low = line.lower()
        if any(token in low for token in items):
            count += 1
    return count


def resolve_counter_metric(
    *,
    field: str,
    data_candidates: list[tuple[str, Any, tuple[str, ...]]],
    log_keys: list[str],
    driver_log_path: Path,
    note_records: list[dict[str, Any]],
    mention_tokens: list[str] | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for source_name, data_obj, path in data_candidates:
        value = dict_path_int(data_obj, path)
        if value is None:
            continue
        candidates.append(
            {
                "source": f"{source_name}.{'.'.join(path)}",
                "value": max(0, int(value)),
            }
        )

    latest_from_log = latest_counter_from_log(driver_log_path, log_keys)
    if isinstance(latest_from_log, dict):
        candidates.append(
            {
                "source": str(latest_from_log.get("source") or "driver_log"),
                "value": max(0, parse_int(latest_from_log.get("value"), 0)),
            }
        )

    best_value = 0
    best_source = "default_zero"
    if candidates:
        best = max(candidates, key=lambda item: parse_int(item.get("value"), 0))
        best_value = max(0, parse_int(best.get("value"), 0))
        best_source = str(best.get("source") or "unknown")

    mentions: dict[str, int] = {"decision_notes": 0, "driver_log": 0}
    used_mentions_fallback = False
    if mention_tokens:
        note_hits = count_token_mentions_in_decision_notes(note_records, tokens=mention_tokens)
        log_hits = count_token_mentions_in_log(driver_log_path, tokens=mention_tokens, max_lines=8000)
        mentions = {
            "decision_notes": int(note_hits),
            "driver_log": int(log_hits),
        }
        if best_value <= 0:
            mention_value = max(int(note_hits), int(log_hits))
            if mention_value > 0:
                best_value = mention_value
                best_source = "evidence_mentions"
                used_mentions_fallback = True

    return {
        "field": field,
        "value": int(best_value),
        "present": bool(best_value > 0),
        "source": best_source,
        "candidates": candidates,
        "mentions": mentions,
        "used_mentions_fallback": bool(used_mentions_fallback),
    }


def collect_runtime_observability_markers(
    *,
    runtime_metrics: dict[str, Any],
    process_slo_state: dict[str, Any],
    capacity_state: dict[str, Any],
    directive_overlay: dict[str, Any],
    decision_records: list[dict[str, Any]],
    driver_log_path: Path,
) -> dict[str, Any]:
    data = {
        "runtime": runtime_metrics if isinstance(runtime_metrics, dict) else {},
        "process_slo": process_slo_state if isinstance(process_slo_state, dict) else {},
        "capacity": capacity_state if isinstance(capacity_state, dict) else {},
        "directive": directive_overlay if isinstance(directive_overlay, dict) else {},
    }

    ready_buffer_depth = resolve_counter_metric(
        field="ready_buffer_depth",
        data_candidates=[
            ("runtime", data["runtime"], ("ready_buffer_depth",)),
            ("process_slo", data["process_slo"], ("runtime", "ready_buffer_depth")),
            ("process_slo", data["process_slo"], ("queue", "ready_buffer_depth")),
            ("process_slo", data["process_slo"], ("kpi", "ready_buffer_depth")),
            ("process_slo", data["process_slo"], ("ready_buffer_depth",)),
            ("capacity", data["capacity"], ("backlog", "ready_buffer_depth")),
            ("capacity", data["capacity"], ("runtime", "ready_buffer_depth")),
            ("capacity", data["capacity"], ("ready_buffer_depth",)),
            ("directive", data["directive"], ("ready_buffer_depth",)),
            ("directive", data["directive"], ("contains", "ready_buffer_depth")),
        ],
        log_keys=["ready_buffer_depth", "ready_queue_depth", "ready_buffer"],
        driver_log_path=driver_log_path,
        note_records=decision_records,
    )

    cold_fail_active_count = resolve_counter_metric(
        field="cold_fail_active_count",
        data_candidates=[
            ("runtime", data["runtime"], ("cold_fail_active_count",)),
            ("process_slo", data["process_slo"], ("runtime", "cold_fail_active_count")),
            ("process_slo", data["process_slo"], ("queue", "cold_fail_active_count")),
            ("process_slo", data["process_slo"], ("cold_fail_active_count",)),
            ("capacity", data["capacity"], ("backlog", "cold_fail_active_count")),
            ("capacity", data["capacity"], ("runtime", "cold_fail_active_count")),
            ("capacity", data["capacity"], ("cold_fail_active_count",)),
            ("directive", data["directive"], ("cold_fail_active_count",)),
            ("directive", data["directive"], ("contains", "cold_fail_active_count")),
        ],
        log_keys=["cold_fail_active_count", "cold_tail_active_count", "cold_tail_fail_active_count"],
        driver_log_path=driver_log_path,
        note_records=decision_records,
    )

    top_level_queue_jobs = resolve_counter_metric(
        field="top_level_queue_jobs",
        data_candidates=[
            ("runtime", data["runtime"], ("top_level_queue_jobs",)),
            ("process_slo", data["process_slo"], ("runtime", "top_level_queue_jobs")),
            ("process_slo", data["process_slo"], ("queue", "top_level_queue_jobs")),
            ("process_slo", data["process_slo"], ("kpi", "top_level_queue_jobs")),
            ("capacity", data["capacity"], ("remote", "top_level_queue_jobs")),
            ("directive", data["directive"], ("top_level_queue_jobs",)),
        ],
        log_keys=["top_level_queue_jobs", "remote_active_queue_jobs", "remote_queue_jobs_active"],
        driver_log_path=driver_log_path,
        note_records=decision_records,
    )

    remote_active_queue_jobs = resolve_counter_metric(
        field="remote_active_queue_jobs",
        data_candidates=[
            ("runtime", data["runtime"], ("remote_active_queue_jobs",)),
            ("runtime", data["runtime"], ("remote_queue_job_count",)),
            ("process_slo", data["process_slo"], ("runtime", "remote_active_queue_jobs")),
            ("process_slo", data["process_slo"], ("runtime", "remote_queue_job_count")),
            ("process_slo", data["process_slo"], ("queue", "remote_active_queue_jobs")),
            ("process_slo", data["process_slo"], ("queue", "remote_queue_job_count")),
            ("process_slo", data["process_slo"], ("kpi", "remote_active_queue_jobs")),
            ("process_slo", data["process_slo"], ("kpi", "remote_queue_job_count")),
            ("process_slo", data["process_slo"], ("remote_active_queue_jobs",)),
            ("capacity", data["capacity"], ("remote", "remote_active_queue_jobs")),
            ("capacity", data["capacity"], ("runtime", "remote_active_queue_jobs")),
            ("capacity", data["capacity"], ("remote_active_queue_jobs",)),
            ("directive", data["directive"], ("remote_active_queue_jobs",)),
        ],
        log_keys=["remote_active_queue_jobs", "remote_queue_jobs_active", "remote_runner_count"],
        driver_log_path=driver_log_path,
        note_records=decision_records,
    )

    remote_child_process_count = resolve_counter_metric(
        field="remote_child_process_count",
        data_candidates=[
            ("runtime", data["runtime"], ("remote_child_process_count",)),
            ("process_slo", data["process_slo"], ("runtime", "remote_child_process_count")),
            ("process_slo", data["process_slo"], ("queue", "remote_child_process_count")),
            ("process_slo", data["process_slo"], ("kpi", "remote_child_process_count")),
            ("capacity", data["capacity"], ("remote", "runner_count")),
            ("process_slo", data["process_slo"], ("queue", "remote_runner_count")),
            ("process_slo", data["process_slo"], ("kpi", "remote_runner_count")),
        ],
        log_keys=["remote_child_process_count", "remote_runner_count"],
        driver_log_path=driver_log_path,
        note_records=decision_records,
    )

    cpu_busy_without_queue_job = resolve_counter_metric(
        field="cpu_busy_without_queue_job",
        data_candidates=[
            ("runtime", data["runtime"], ("cpu_busy_without_queue_job",)),
            ("process_slo", data["process_slo"], ("runtime", "cpu_busy_without_queue_job")),
            ("process_slo", data["process_slo"], ("queue", "cpu_busy_without_queue_job")),
            ("process_slo", data["process_slo"], ("kpi", "cpu_busy_without_queue_job")),
        ],
        log_keys=["cpu_busy_without_queue_job", "remote_cpu_busy_without_queue_job"],
        driver_log_path=driver_log_path,
        note_records=decision_records,
    )

    surrogate_idle_override_count = resolve_counter_metric(
        field="surrogate_idle_override_count",
        data_candidates=[
            ("runtime", data["runtime"], ("surrogate_idle_override_count",)),
            ("process_slo", data["process_slo"], ("runtime", "surrogate_idle_override_count")),
            ("process_slo", data["process_slo"], ("surrogate_idle_override_count",)),
            ("capacity", data["capacity"], ("runtime", "surrogate_idle_override_count")),
            ("capacity", data["capacity"], ("surrogate_idle_override_count",)),
            ("directive", data["directive"], ("surrogate_idle_override_count",)),
        ],
        log_keys=["surrogate_idle_override_count", "surrogate_idle_overrides"],
        driver_log_path=driver_log_path,
        note_records=decision_records,
        mention_tokens=["SURROGATE_IDLE_OVERRIDE", "surrogate_idle_override"],
    )

    overlap_dispatch_count = resolve_counter_metric(
        field="overlap_dispatch_count",
        data_candidates=[
            ("runtime", data["runtime"], ("overlap_dispatch_count",)),
            ("process_slo", data["process_slo"], ("runtime", "overlap_dispatch_count")),
            ("process_slo", data["process_slo"], ("overlap_dispatch_count",)),
            ("capacity", data["capacity"], ("runtime", "overlap_dispatch_count")),
            ("capacity", data["capacity"], ("overlap_dispatch_count",)),
            ("directive", data["directive"], ("overlap_dispatch_count",)),
        ],
        log_keys=["overlap_dispatch_count", "overlap_dispatches"],
        driver_log_path=driver_log_path,
        note_records=decision_records,
        mention_tokens=["OVERLAP_DISPATCH", "overlap_dispatch"],
    )

    return {
        "ready_buffer_depth": ready_buffer_depth,
        "cold_fail_active_count": cold_fail_active_count,
        "top_level_queue_jobs": top_level_queue_jobs,
        "remote_child_process_count": remote_child_process_count,
        "remote_active_queue_jobs": remote_active_queue_jobs,
        "cpu_busy_without_queue_job": cpu_busy_without_queue_job,
        "surrogate_idle_override_count": surrogate_idle_override_count,
        "overlap_dispatch_count": overlap_dispatch_count,
    }


def ensure_process_slo_state(
    *,
    mode: str,
    app_root: Path,
    process_slo_state_path: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    result = {
        "mode": mode,
        "attempted": False,
        "ok": True,
        "reason": "not_needed",
        "returncode": 0,
        "duration_sec": 0.0,
    }

    should_run = False
    if mode == "always":
        should_run = True
    elif mode == "auto" and not process_slo_state_path.exists():
        should_run = True

    if not should_run:
        if process_slo_state_path.exists():
            result["reason"] = "already_exists"
        return result

    script = app_root / "scripts" / "optimization" / "process_slo_guard_agent.py"
    if not script.exists():
        result.update({
            "attempted": True,
            "ok": False,
            "reason": "process_slo_guard_missing",
            "returncode": 127,
        })
        return result

    cmd = [sys.executable or "python3", str(script), "--root", str(app_root)]
    result["attempted"] = True
    result["reason"] = "executed"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(app_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
        result["duration_sec"] = round(time.time() - t0, 3)
        result["returncode"] = int(proc.returncode)
        if proc.returncode != 0:
            result["ok"] = False
            result["reason"] = "process_slo_guard_failed"
        if proc.stdout:
            lines = [line for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                result["stdout_tail"] = lines[-5:]
        if proc.stderr:
            lines = [line for line in proc.stderr.splitlines() if line.strip()]
            if lines:
                result["stderr_tail"] = lines[-5:]
    except subprocess.TimeoutExpired:
        result["duration_sec"] = round(time.time() - t0, 3)
        result.update({
            "ok": False,
            "reason": "process_slo_guard_timeout",
            "returncode": 124,
        })

    if result["ok"] and process_slo_state_path.exists():
        result["reason"] = "process_slo_state_ready"

    return result


def collect_markers(
    *,
    fullspan_state_path: Path,
    process_slo_state_path: Path,
    capacity_state_path: Path,
    decision_notes_path: Path,
    driver_log_path: Path,
    gate_surrogate_state_path: Path,
    directive_overlay_path: Path,
) -> dict[str, Any]:
    now_epoch = int(time.time())
    surrogate_fresh_sec = 1800
    directive_fresh_sec = 1800

    fullspan = read_json(fullspan_state_path, {})
    if not isinstance(fullspan, dict):
        fullspan = {}

    queues = fullspan.get("queues", {})
    if not isinstance(queues, dict):
        queues = {}

    runtime = fullspan.get("runtime_metrics", {})
    if not isinstance(runtime, dict):
        runtime = {}

    strict_from_fullspan = 0
    promote_from_fullspan = 0
    fail_closed_from_fullspan = 0
    for entry in queues.values():
        if not isinstance(entry, dict):
            continue
        if parse_int(entry.get("strict_pass_count"), 0) > 0:
            strict_from_fullspan += 1
        if str(entry.get("promotion_verdict") or "").strip().upper() == "PROMOTE_ELIGIBLE":
            promote_from_fullspan += 1
        if str(entry.get("cutover_permission") or "").strip().upper() == "FAIL_CLOSED":
            fail_closed_from_fullspan += 1

    strict_from_runtime = parse_int(runtime.get("strict_fullspan_pass_count"), 0)
    promote_from_runtime = parse_int(runtime.get("promotion_eligible_count"), 0)

    process_slo = read_json(process_slo_state_path, {})
    if not isinstance(process_slo, dict):
        process_slo = {}
    funnel = process_slo.get("funnel", {})
    if not isinstance(funnel, dict):
        funnel = {}

    capacity_state = read_json(capacity_state_path, {})
    if not isinstance(capacity_state, dict):
        capacity_state = {}

    strict_from_process_slo = parse_int(funnel.get("strict_pass"), 0)
    promote_from_process_slo = parse_int(funnel.get("promote_eligible"), 0)

    decision_records = read_jsonl_tail(decision_notes_path, max_lines=8000)
    fail_closed_from_notes = 0
    for rec in decision_records:
        action = str(rec.get("action") or "").strip().upper()
        if "FAIL_CLOSED" in action:
            fail_closed_from_notes += 1

    fail_closed_from_log = count_fail_closed_in_log(driver_log_path, max_lines=3000)

    strict_pass_count = max(strict_from_process_slo, strict_from_runtime, strict_from_fullspan)
    promote_eligible_count = max(promote_from_process_slo, promote_from_runtime, promote_from_fullspan)
    fail_closed_count = max(fail_closed_from_fullspan, fail_closed_from_notes, fail_closed_from_log)
    fail_closed_policy_default = 1 if promote_eligible_count <= 0 else 0

    gate_state = read_json(gate_surrogate_state_path, {})
    if not isinstance(gate_state, dict):
        gate_state = {}
    gate_summary = gate_state.get("summary")
    gate_mode = ""
    if isinstance(gate_summary, dict):
        gate_mode = str(gate_summary.get("mode") or "").strip().lower()
    if not gate_mode:
        hard_fail_policy = gate_state.get("hard_fail_risk_policy")
        enabled = bool(hard_fail_policy.get("enabled")) if isinstance(hard_fail_policy, dict) else False
        gate_mode = "active" if enabled else "neutral"
    gate_decision_counts = _gate_decision_counts(gate_state)
    eligible_gate_queues: set[str] = set()
    raw_gate_queues = gate_state.get("queues")
    if isinstance(raw_gate_queues, dict):
        for queue_key, entry in raw_gate_queues.items():
            if not isinstance(entry, dict):
                continue
            decision = str(entry.get("decision") or "").strip().lower()
            if decision in {"reject", "refine"}:
                eligible_gate_queues.add(str(queue_key))
    gate_freshness = freshness_payload(
        path=gate_surrogate_state_path,
        ts=gate_state.get("ts") if isinstance(gate_state, dict) else "",
        now_epoch=now_epoch,
        fresh_sec=surrogate_fresh_sec,
    )
    gate_inputs = gate_state.get("inputs")
    missing_required_inputs = []
    if isinstance(gate_inputs, dict):
        raw_missing = gate_inputs.get("missing_required_inputs")
        if isinstance(raw_missing, list):
            missing_required_inputs = [str(item).strip() for item in raw_missing if str(item).strip()]

    directive = read_json(directive_overlay_path, {})
    if not isinstance(directive, dict):
        directive = {}
    directive_freshness = freshness_payload(
        path=directive_overlay_path,
        ts=directive.get("ts"),
        now_epoch=now_epoch,
        fresh_sec=directive_fresh_sec,
    )
    directive_gate_ts_freshness = freshness_payload(
        path=directive_overlay_path,
        ts=directive.get("gate_surrogate_ts"),
        now_epoch=now_epoch,
        fresh_sec=directive_fresh_sec,
    )

    decision_surrogate_stats = surrogate_action_stats_from_decision_notes(
        records=decision_records,
        now_epoch=now_epoch,
    )
    log_surrogate_stats = surrogate_action_stats_from_driver_log(
        driver_log_path,
        now_epoch=now_epoch,
        max_lines=5000,
    )
    recent_candidate_queues = recent_candidate_queues_from_driver_log(
        driver_log_path,
        now_epoch=now_epoch,
        max_lines=5000,
    )
    combined_surrogate_stats = _combine_surrogate_evidence(
        decision_surrogate_stats,
        log_surrogate_stats,
    )
    runtime_observability = collect_runtime_observability_markers(
        runtime_metrics=runtime,
        process_slo_state=process_slo,
        capacity_state=capacity_state,
        directive_overlay=directive,
        decision_records=decision_records,
        driver_log_path=driver_log_path,
    )

    runtime_surrogate_reject = _runtime_metric_int(runtime, "surrogate_reject_count")
    runtime_surrogate_refine = _runtime_metric_int(runtime, "surrogate_refine_count")
    runtime_surrogate_allow = _runtime_metric_int(runtime, "surrogate_allow_count")
    eligible_refine_reject = int(gate_decision_counts.get("reject", 0) + gate_decision_counts.get("refine", 0))
    observed_refine_reject = (
        parse_int(combined_surrogate_stats.get("SURROGATE_REJECT", {}).get("count"), 0)
        + parse_int(combined_surrogate_stats.get("SURROGATE_REFINE", {}).get("count"), 0)
        + int(runtime_surrogate_reject)
        + int(runtime_surrogate_refine)
    )
    eligible_runtime_hits = sum(1 for queue in recent_candidate_queues if queue in eligible_gate_queues)
    branch_status = "inconclusive"
    branch_reason = ""
    branch_broken = False
    if not gate_surrogate_state_path.exists():
        branch_status = "missing_gate_state"
        branch_reason = "gate_surrogate_state.json missing"
    elif eligible_refine_reject <= 0:
        branch_status = "no_eligible_case"
        branch_reason = "gate summary has no refine/reject eligible queues"
    elif observed_refine_reject > 0:
        branch_status = "evidence_present"
        branch_reason = "SURROGATE_REJECT/REFINE evidence observed in logs/notes/runtime metrics"
    elif bool(gate_freshness.get("fresh")) and bool(directive_freshness.get("fresh")) and eligible_runtime_hits > 0:
        branch_status = "broken_branch"
        branch_reason = "eligible refine/reject queues present but no surrogate runtime evidence"
        branch_broken = True
    else:
        branch_status = "stale_or_inconclusive"
        branch_reason = "eligible refine/reject queues exist, but driver has not hit them yet or state/directive freshness is stale"

    return {
        "strict_pass": {
            "count": strict_pass_count,
            "present": bool(strict_pass_count > 0),
            "sources": {
                "process_slo_funnel": strict_from_process_slo,
                "fullspan_runtime_metrics": strict_from_runtime,
                "fullspan_queue_entries": strict_from_fullspan,
            },
        },
        "promote_eligible": {
            "count": promote_eligible_count,
            "present": bool(promote_eligible_count > 0),
            "sources": {
                "process_slo_funnel": promote_from_process_slo,
                "fullspan_runtime_metrics": promote_from_runtime,
                "fullspan_queue_entries": promote_from_fullspan,
            },
        },
        "fail_closed": {
            "count": fail_closed_count,
            "present": bool(fail_closed_count > 0 or fail_closed_policy_default > 0),
            "policy_default_present": bool(fail_closed_policy_default > 0),
            "sources": {
                "fullspan_cutover_permission": fail_closed_from_fullspan,
                "decision_notes_actions": fail_closed_from_notes,
                "driver_log_mentions": fail_closed_from_log,
                "policy_default_no_promote_eligible": fail_closed_policy_default,
            },
        },
        "surrogate_runtime": {
            "gate_surrogate": {
                "exists": bool(gate_surrogate_state_path.exists()),
                "mode": gate_mode,
                "decision_counts": gate_decision_counts,
                "freshness": gate_freshness,
                "missing_required_inputs": missing_required_inputs,
            },
            "directive_overlay": {
                "exists": bool(directive_overlay_path.exists()),
                "mode": str(directive.get("mode") or "").strip().lower(),
                "gate_surrogate_mode": str(directive.get("gate_surrogate_mode") or "").strip().lower(),
                "freshness": directive_freshness,
                "gate_surrogate_ts_freshness": directive_gate_ts_freshness,
            },
            "evidence": {
                "decision_notes": decision_surrogate_stats,
                "driver_log": log_surrogate_stats,
                "combined": combined_surrogate_stats,
            },
            "fullspan_runtime_counters": {
                "surrogate_reject_count": runtime_surrogate_reject,
                "surrogate_refine_count": runtime_surrogate_refine,
                "surrogate_allow_count": runtime_surrogate_allow,
            },
            "branch_health": {
                "status": branch_status,
                "reason": branch_reason,
                "broken_branch": bool(branch_broken),
                "eligible_refine_reject_count": int(eligible_refine_reject),
                "observed_refine_reject_evidence": int(observed_refine_reject),
                "recent_candidate_queue_count": int(len(recent_candidate_queues)),
                "eligible_runtime_hits": int(eligible_runtime_hits),
            },
        },
        "runtime_observability": runtime_observability,
    }


def print_text(payload: dict[str, Any]) -> None:
    ok = 1 if bool(payload.get("ok")) else 0
    print(f"PROBE_OK={ok}")
    print(f"PROBE_UTC={payload.get('probe_utc', '')}")
    print(f"RESOLUTION={payload.get('resolution', '')}")

    app_root = payload.get("app_root", "")
    state_dir = payload.get("state_dir", "")
    print(f"APP_ROOT={app_root}")
    print(f"STATE_DIR={state_dir}")

    files = payload.get("files", {}) if isinstance(payload.get("files"), dict) else {}
    print(f"STATE_DIR_EXISTS={1 if bool(payload.get('state_dir_exists')) else 0}")
    print(f"DRIVER_STATE_EXISTS={1 if bool(files.get('driver_state_txt', {}).get('exists')) else 0}")
    print(f"DRIVER_LOG_EXISTS={1 if bool(files.get('driver_log', {}).get('exists')) else 0}")
    print(f"DECISION_NOTES_EXISTS={1 if bool(files.get('decision_notes_jsonl', {}).get('exists')) else 0}")
    print(f"FULLSPAN_STATE_EXISTS={1 if bool(files.get('fullspan_decision_state_json', {}).get('exists')) else 0}")
    print(f"PROCESS_SLO_STATE_EXISTS={1 if bool(files.get('process_slo_state_json', {}).get('exists')) else 0}")
    print(f"CAPACITY_STATE_EXISTS={1 if bool(files.get('capacity_controller_state_json', {}).get('exists')) else 0}")
    print(f"GATE_SURROGATE_STATE_EXISTS={1 if bool(files.get('gate_surrogate_state_json', {}).get('exists')) else 0}")
    print(f"DIRECTIVE_OVERLAY_EXISTS={1 if bool(files.get('search_director_directive_json', {}).get('exists')) else 0}")

    ensure = payload.get("process_slo_ensure", {}) if isinstance(payload.get("process_slo_ensure"), dict) else {}
    print(f"PROCESS_SLO_ENSURE_ATTEMPTED={1 if bool(ensure.get('attempted')) else 0}")
    print(f"PROCESS_SLO_ENSURE_OK={1 if bool(ensure.get('ok', True)) else 0}")
    print(f"PROCESS_SLO_ENSURE_REASON={str(ensure.get('reason', ''))}")

    markers = payload.get("markers", {}) if isinstance(payload.get("markers"), dict) else {}
    strict = markers.get("strict_pass", {}) if isinstance(markers.get("strict_pass"), dict) else {}
    promote = markers.get("promote_eligible", {}) if isinstance(markers.get("promote_eligible"), dict) else {}
    fail_closed = markers.get("fail_closed", {}) if isinstance(markers.get("fail_closed"), dict) else {}
    surrogate = markers.get("surrogate_runtime", {}) if isinstance(markers.get("surrogate_runtime"), dict) else {}
    surrogate_gate = surrogate.get("gate_surrogate", {}) if isinstance(surrogate.get("gate_surrogate"), dict) else {}
    surrogate_gate_fresh = surrogate_gate.get("freshness", {}) if isinstance(surrogate_gate.get("freshness"), dict) else {}
    surrogate_directive = surrogate.get("directive_overlay", {}) if isinstance(surrogate.get("directive_overlay"), dict) else {}
    surrogate_directive_fresh = surrogate_directive.get("freshness", {}) if isinstance(surrogate_directive.get("freshness"), dict) else {}
    surrogate_evidence = surrogate.get("evidence", {}) if isinstance(surrogate.get("evidence"), dict) else {}
    surrogate_combined = surrogate_evidence.get("combined", {}) if isinstance(surrogate_evidence.get("combined"), dict) else {}
    surrogate_counters = surrogate.get("fullspan_runtime_counters", {}) if isinstance(surrogate.get("fullspan_runtime_counters"), dict) else {}
    surrogate_branch = surrogate.get("branch_health", {}) if isinstance(surrogate.get("branch_health"), dict) else {}
    runtime_observability = markers.get("runtime_observability", {}) if isinstance(markers.get("runtime_observability"), dict) else {}
    ready_buffer_depth = runtime_observability.get("ready_buffer_depth", {}) if isinstance(runtime_observability.get("ready_buffer_depth"), dict) else {}
    cold_fail_active_count = runtime_observability.get("cold_fail_active_count", {}) if isinstance(runtime_observability.get("cold_fail_active_count"), dict) else {}
    top_level_queue_jobs = runtime_observability.get("top_level_queue_jobs", {}) if isinstance(runtime_observability.get("top_level_queue_jobs"), dict) else {}
    remote_child_process_count = runtime_observability.get("remote_child_process_count", {}) if isinstance(runtime_observability.get("remote_child_process_count"), dict) else {}
    remote_active_queue_jobs = runtime_observability.get("remote_active_queue_jobs", {}) if isinstance(runtime_observability.get("remote_active_queue_jobs"), dict) else {}
    cpu_busy_without_queue_job = runtime_observability.get("cpu_busy_without_queue_job", {}) if isinstance(runtime_observability.get("cpu_busy_without_queue_job"), dict) else {}
    surrogate_idle_override_count = runtime_observability.get("surrogate_idle_override_count", {}) if isinstance(runtime_observability.get("surrogate_idle_override_count"), dict) else {}
    overlap_dispatch_count = runtime_observability.get("overlap_dispatch_count", {}) if isinstance(runtime_observability.get("overlap_dispatch_count"), dict) else {}

    print(f"STRICT_PASS_COUNT={parse_int(strict.get('count'), 0)}")
    print(f"STRICT_PASS_PRESENT={1 if bool(strict.get('present')) else 0}")
    print(f"PROMOTE_ELIGIBLE_COUNT={parse_int(promote.get('count'), 0)}")
    print(f"PROMOTE_ELIGIBLE_PRESENT={1 if bool(promote.get('present')) else 0}")
    print(f"FAIL_CLOSED_COUNT={parse_int(fail_closed.get('count'), 0)}")
    print(f"FAIL_CLOSED_PRESENT={1 if bool(fail_closed.get('present')) else 0}")
    print(f"FAIL_CLOSED_POLICY_DEFAULT_PRESENT={1 if bool(fail_closed.get('policy_default_present')) else 0}")
    print(f"GATE_SURROGATE_MODE={str(surrogate_gate.get('mode', ''))}")
    print(f"GATE_SURROGATE_FRESH={1 if bool(surrogate_gate_fresh.get('fresh')) else 0}")
    print(f"GATE_SURROGATE_AGE_SEC={parse_int(surrogate_gate_fresh.get('age_sec'), -1)}")
    print(f"GATE_SURROGATE_DECISIONS_ALLOW={parse_int((surrogate_gate.get('decision_counts') or {}).get('allow') if isinstance(surrogate_gate.get('decision_counts'), dict) else 0, 0)}")
    print(f"GATE_SURROGATE_DECISIONS_REFINE={parse_int((surrogate_gate.get('decision_counts') or {}).get('refine') if isinstance(surrogate_gate.get('decision_counts'), dict) else 0, 0)}")
    print(f"GATE_SURROGATE_DECISIONS_REJECT={parse_int((surrogate_gate.get('decision_counts') or {}).get('reject') if isinstance(surrogate_gate.get('decision_counts'), dict) else 0, 0)}")
    print(f"DIRECTIVE_OVERLAY_MODE={str(surrogate_directive.get('mode', ''))}")
    print(f"DIRECTIVE_GATE_SURROGATE_MODE={str(surrogate_directive.get('gate_surrogate_mode', ''))}")
    print(f"DIRECTIVE_OVERLAY_FRESH={1 if bool(surrogate_directive_fresh.get('fresh')) else 0}")
    print(f"DIRECTIVE_OVERLAY_AGE_SEC={parse_int(surrogate_directive_fresh.get('age_sec'), -1)}")
    print(f"SURROGATE_REJECT_EVIDENCE_COUNT={parse_int((surrogate_combined.get('SURROGATE_REJECT') or {}).get('count') if isinstance(surrogate_combined.get('SURROGATE_REJECT'), dict) else 0, 0)}")
    print(f"SURROGATE_REFINE_EVIDENCE_COUNT={parse_int((surrogate_combined.get('SURROGATE_REFINE') or {}).get('count') if isinstance(surrogate_combined.get('SURROGATE_REFINE'), dict) else 0, 0)}")
    print(f"SURROGATE_ALLOW_EVIDENCE_COUNT={parse_int((surrogate_combined.get('SURROGATE_ALLOW') or {}).get('count') if isinstance(surrogate_combined.get('SURROGATE_ALLOW'), dict) else 0, 0)}")
    print(f"SURROGATE_REJECT_COUNTER={parse_int(surrogate_counters.get('surrogate_reject_count'), 0)}")
    print(f"SURROGATE_REFINE_COUNTER={parse_int(surrogate_counters.get('surrogate_refine_count'), 0)}")
    print(f"SURROGATE_ALLOW_COUNTER={parse_int(surrogate_counters.get('surrogate_allow_count'), 0)}")
    print(f"SURROGATE_BRANCH_STATUS={str(surrogate_branch.get('status', ''))}")
    print(f"SURROGATE_BRANCH_BROKEN={1 if bool(surrogate_branch.get('broken_branch')) else 0}")
    print(f"SURROGATE_BRANCH_REASON={str(surrogate_branch.get('reason', ''))}")
    print(f"READY_BUFFER_DEPTH={parse_int(ready_buffer_depth.get('value'), 0)}")
    print(f"COLD_FAIL_ACTIVE_COUNT={parse_int(cold_fail_active_count.get('value'), 0)}")
    print(f"TOP_LEVEL_QUEUE_JOBS={parse_int(top_level_queue_jobs.get('value'), 0)}")
    print(f"REMOTE_CHILD_PROCESS_COUNT={parse_int(remote_child_process_count.get('value'), 0)}")
    print(f"REMOTE_ACTIVE_QUEUE_JOBS={parse_int(remote_active_queue_jobs.get('value'), 0)}")
    print(f"CPU_BUSY_WITHOUT_QUEUE_JOB={parse_int(cpu_busy_without_queue_job.get('value'), 0)}")
    print(f"SURROGATE_IDLE_OVERRIDE_COUNT={parse_int(surrogate_idle_override_count.get('value'), 0)}")
    print(f"OVERLAP_DISPATCH_COUNT={parse_int(overlap_dispatch_count.get('value'), 0)}")



def main() -> int:
    parser = argparse.ArgumentParser(description="Probe autonomous state markers (strict/promote/fail-closed).")
    parser.add_argument("--root", default="auto", help="coint4 app root or 'auto'.")
    parser.add_argument("--state-dir", default="", help="Explicit autonomous state dir path.")
    parser.add_argument(
        "--ensure-process-slo",
        choices=("auto", "always", "never"),
        default="auto",
        help="Ensure process_slo_state exists by running process_slo_guard_agent once when needed.",
    )
    parser.add_argument("--ensure-timeout-sec", type=int, default=30, help="Timeout for process_slo_guard_agent execution.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    parser.add_argument("--pretty", action="store_true", help="Pretty JSON output.")
    args = parser.parse_args()

    layout = resolve_layout(args.root, args.state_dir)
    app_root = layout.get("app_root")
    state_dir = layout.get("state_dir")

    payload: dict[str, Any] = {
        "probe_utc": utc_now_iso(),
        "ok": False,
        "resolution": layout.get("resolution", ""),
        "checked_candidates": layout.get("checked_candidates", []),
    }

    if not isinstance(app_root, Path) or not isinstance(state_dir, Path):
        payload["error"] = "app_root_not_resolved"
        if args.format == "json":
            print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=False))
        else:
            print_text(payload)
        return 2

    app_root = app_root.resolve()
    state_dir = state_dir.resolve()

    driver_state_path = state_dir / "driver_state.txt"
    driver_log_path = state_dir / "driver.log"
    decision_notes_path = state_dir / "decision_notes.jsonl"
    fullspan_state_path = state_dir / "fullspan_decision_state.json"
    process_slo_state_path = state_dir / "process_slo_state.json"
    capacity_state_path = state_dir / "capacity_controller_state.json"
    gate_surrogate_state_path = state_dir / "gate_surrogate_state.json"
    directive_overlay_path = state_dir / "search_director_directive.json"

    ensure_result = ensure_process_slo_state(
        mode=args.ensure_process_slo,
        app_root=app_root,
        process_slo_state_path=process_slo_state_path,
        timeout_sec=args.ensure_timeout_sec,
    )

    markers = collect_markers(
        fullspan_state_path=fullspan_state_path,
        process_slo_state_path=process_slo_state_path,
        capacity_state_path=capacity_state_path,
        decision_notes_path=decision_notes_path,
        driver_log_path=driver_log_path,
        gate_surrogate_state_path=gate_surrogate_state_path,
        directive_overlay_path=directive_overlay_path,
    )

    payload.update(
        {
            "ok": True,
            "app_root": str(app_root),
            "state_dir": str(state_dir),
            "state_dir_exists": bool(state_dir.exists()),
            "process_slo_ensure": ensure_result,
            "markers": markers,
            "files": {
                "driver_state_txt": file_meta(driver_state_path),
                "driver_log": file_meta(driver_log_path),
                "decision_notes_jsonl": file_meta(decision_notes_path),
                "fullspan_decision_state_json": file_meta(fullspan_state_path),
                "process_slo_state_json": file_meta(process_slo_state_path),
                "capacity_controller_state_json": file_meta(capacity_state_path),
                "gate_surrogate_state_json": file_meta(gate_surrogate_state_path),
                "search_director_directive_json": file_meta(directive_overlay_path),
            },
        }
    )

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=False))
    else:
        print_text(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
