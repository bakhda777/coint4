#!/usr/bin/env python3
"""Shared queue status and reject-reason contract for autonomous WFA orchestration."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

PENDING_LIKE_STATUSES = frozenset({"planned", "queued", "running", "stalled", "failed", "error", "active"})
DISPATCHABLE_PENDING_STATUSES = frozenset({"planned", "queued", "failed", "stalled"})
EXECUTABLE_PENDING_STATUSES = frozenset(set(DISPATCHABLE_PENDING_STATUSES) | {"running"})
FAILED_STATUSES = frozenset({"failed", "error"})
HARD_REJECT_QUEUE_VERDICTS = frozenset({"REJECT"})
FAIL_CLOSED_QUEUE_PERMISSIONS = frozenset({"FAIL_CLOSED"})
ZERO_ACTIVITY_REASONS = frozenset(
    {
        "ZERO_OBSERVED_TEST_DAYS",
        "ZERO_COVERAGE",
        "ZERO_TRADES",
        "ZERO_PAIRS",
    }
)

# Backward-compatible aliases for current scripts/tests.
DISPATCHABLE_STATUSES = DISPATCHABLE_PENDING_STATUSES
FAILED_LIKE_STATUSES = FAILED_STATUSES


def normalize_queue_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value or "").strip()
        if not text:
            return default
        number = float(text)
        if not math.isfinite(number):
            return default
        return number
    except Exception:
        return default


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def canonical_reject_reason(
    raw_reason: Any = "",
    *,
    row: dict[str, Any] | None = None,
    metrics_present: Any | None = None,
) -> str:
    text = str(raw_reason or "").strip().upper()
    if text:
        for token in (
            "ZERO_OBSERVED_TEST_DAYS",
            "ZERO_COVERAGE",
            "ZERO_TRADES",
            "ZERO_PAIRS",
            "METRICS_FILE_MISSING",
            "METRICS_MISSING",
            "TRADES_FAIL",
            "PAIRS_FAIL",
            "DD_FAIL",
            "ECONOMIC_FAIL",
            "STEP_FAIL",
            "INSUFFICIENT_WINDOWS",
            "STATUS_NOT_COMPLETED",
            "HOLDOUT_STRESS_MISSING",
            "COVERAGE_UNREACHABLE",
            "MIN_WINDOWS_UNREACHABLE",
            "COVERAGE_BELOW",
        ):
            if token in text:
                return token

    payload = row if isinstance(row, dict) else {}
    metrics_flag = _to_bool(payload.get("metrics_present") if metrics_present is None else metrics_present)
    if not metrics_flag:
        return "METRICS_MISSING"

    observed_test_days = _to_float(payload.get("observed_test_days"), None)
    expected_test_days = _to_float(payload.get("expected_test_days"), None)
    coverage_ratio = _to_float(payload.get("coverage_ratio"), None)
    total_trades = _to_float(payload.get("total_trades"), None)
    total_pairs = _to_float(payload.get("total_pairs_traded"), None)

    if expected_test_days is not None and expected_test_days > 0 and (observed_test_days or 0.0) <= 0.0:
        return "ZERO_OBSERVED_TEST_DAYS"
    if coverage_ratio is not None and coverage_ratio <= 0.0:
        return "ZERO_COVERAGE"
    if total_trades is not None and total_trades <= 0.0:
        return "ZERO_TRADES"
    if total_pairs is not None and total_pairs <= 0.0:
        return "ZERO_PAIRS"
    if metrics_flag:
        return "METRICS_FILE_MISSING"
    return "METRICS_MISSING"


def resolve_queue_config_path(config_path: str | Path, app_root: Path) -> Path:
    path = Path(str(config_path or "").strip())
    return path if path.is_absolute() else app_root / path


def queue_row_has_existing_config(row: dict[str, Any], app_root: Path) -> bool:
    config_path = str(row.get("config_path") or "").strip()
    if not config_path:
        return False
    return resolve_queue_config_path(config_path, app_root).exists()


def queue_row_is_pending_like(row: dict[str, Any]) -> bool:
    return normalize_queue_status(row.get("status")) in PENDING_LIKE_STATUSES


def queue_row_is_dispatchable(row: dict[str, Any]) -> bool:
    return normalize_queue_status(row.get("status")) in DISPATCHABLE_PENDING_STATUSES


def queue_row_is_dispatchable_pending(row: dict[str, Any], app_root: Path) -> bool:
    return queue_row_is_dispatchable(row) and queue_row_has_existing_config(row, app_root)


def queue_row_is_executable_pending(row: dict[str, Any], app_root: Path) -> bool:
    status = normalize_queue_status(row.get("status"))
    if status == "running":
        return True
    return queue_row_is_dispatchable_pending(row, app_root)


def row_counts_pending(status: Any) -> bool:
    return normalize_queue_status(status) in PENDING_LIKE_STATUSES


def row_counts_dispatchable(status: Any) -> bool:
    return normalize_queue_status(status) in DISPATCHABLE_PENDING_STATUSES


def row_counts_dispatchable_pending(status: Any, config_path: str | Path, app_root: Path) -> bool:
    status_text = normalize_queue_status(status)
    if status_text not in DISPATCHABLE_PENDING_STATUSES:
        return False
    return queue_row_has_existing_config({"config_path": str(config_path or "")}, app_root)


def row_counts_executable(status: Any, config_path: str | Path, app_root: Path) -> bool:
    status_text = normalize_queue_status(status)
    if status_text == "running":
        return True
    return row_counts_dispatchable_pending(status_text, config_path, app_root)


def queue_rel_path(queue_path: Path, app_root: Path) -> str:
    try:
        return queue_path.resolve().relative_to(app_root.resolve()).as_posix()
    except Exception:
        return queue_path.as_posix()


def load_orphan_queue_cooldowns(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                queue = str(row.get("queue") or "").strip()
                if not queue:
                    continue
                out[queue] = {
                    "until_ts": float(_to_float(row.get("until_ts"), 0.0) or 0.0),
                    "reason": str(row.get("reason") or "").strip(),
                }
    except Exception:
        return {}
    return out


def load_fullspan_queue_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    queues = payload.get("queues", {}) if isinstance(payload, dict) else {}
    return dict(queues) if isinstance(queues, dict) else {}


def queue_dispatch_block_reason(
    *,
    queue_rel: str,
    fullspan_entry: dict[str, Any] | None = None,
    orphan_entry: dict[str, Any] | None = None,
    now_epoch: float | None = None,
) -> str:
    now = float(now_epoch if now_epoch is not None else time.time())
    entry = fullspan_entry if isinstance(fullspan_entry, dict) else {}
    verdict = str(entry.get("promotion_verdict") or "").strip().upper()
    strict_gate_status = str(entry.get("strict_gate_status") or "").strip().upper()
    strict_pass_count = int(_to_float(entry.get("strict_pass_count"), 0.0) or 0.0)
    contract_hard_pass = _to_bool(entry.get("contract_hard_pass"))
    if (
        verdict in HARD_REJECT_QUEUE_VERDICTS
        or strict_gate_status == "FULLSPAN_PREFILTER_REJECT"
        or (strict_pass_count > 0 and not contract_hard_pass)
    ):
        return "FULLSPAN_REJECT"

    cutover_permission = str(entry.get("cutover_permission") or "").strip().upper()
    if cutover_permission in FAIL_CLOSED_QUEUE_PERMISSIONS:
        return "FAIL_CLOSED"

    orphan = orphan_entry if isinstance(orphan_entry, dict) else {}
    orphan_until_epoch = float(_to_float(orphan.get("until_ts"), 0.0) or 0.0)
    orphan_reason = str(orphan.get("reason") or "").strip().lower()
    if orphan_until_epoch > now:
        if "fail_closed" in orphan_reason:
            return "FAIL_CLOSED"
        return "ORPHAN_COOLDOWN"

    if _to_bool(entry.get("low_yield_fail_closed")):
        return "FAIL_CLOSED"
    for state_key in ("startup_state", "coverage_state", "queue_state"):
        if str(entry.get(state_key) or "").strip().lower() == "fail_closed":
            return "FAIL_CLOSED"
    return ""


def queue_is_dispatchable_queue(
    *,
    queue_rel: str,
    fullspan_entry: dict[str, Any] | None = None,
    orphan_entry: dict[str, Any] | None = None,
    now_epoch: float | None = None,
) -> bool:
    return not bool(
        queue_dispatch_block_reason(
            queue_rel=queue_rel,
            fullspan_entry=fullspan_entry,
            orphan_entry=orphan_entry,
            now_epoch=now_epoch,
        )
    )


def summarize_queue_rows(rows: list[dict[str, Any]], app_root: Path) -> dict[str, int]:
    summary = {
        "total": 0,
        "completed": 0,
        "pending": 0,
        "running": 0,
        "stalled": 0,
        "failed": 0,
        "dispatchable_pending": 0,
        "executable_rows": 0,
        "executable_pending": 0,
        "hard_rejected_pending": 0,
    }
    for row in rows:
        status = normalize_queue_status(row.get("status"))
        summary["total"] += 1

        cfg_exists = queue_row_has_existing_config(row, app_root)
        if cfg_exists or status == "running":
            summary["executable_rows"] += 1

        if status == "completed":
            summary["completed"] += 1
        if status in PENDING_LIKE_STATUSES:
            summary["pending"] += 1
        if status == "running":
            summary["running"] += 1
        elif status == "stalled":
            summary["stalled"] += 1
        elif status in FAILED_STATUSES:
            summary["failed"] += 1

        if row_counts_dispatchable_pending(status, row.get("config_path"), app_root):
            summary["dispatchable_pending"] += 1
        if status == "running" or row_counts_dispatchable_pending(status, row.get("config_path"), app_root):
            summary["executable_pending"] += 1
    return summary
