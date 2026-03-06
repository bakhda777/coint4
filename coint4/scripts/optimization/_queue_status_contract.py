#!/usr/bin/env python3
"""Shared queue status contract for autonomous WFA orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PENDING_LIKE_STATUSES = frozenset({"planned", "queued", "running", "stalled", "failed", "error", "active"})
DISPATCHABLE_PENDING_STATUSES = frozenset({"planned", "queued", "failed", "stalled"})
EXECUTABLE_PENDING_STATUSES = frozenset(set(DISPATCHABLE_PENDING_STATUSES) | {"running"})
FAILED_STATUSES = frozenset({"failed", "error"})

# Backward-compatible aliases for current scripts/tests.
DISPATCHABLE_STATUSES = DISPATCHABLE_PENDING_STATUSES
FAILED_LIKE_STATUSES = FAILED_STATUSES


def normalize_queue_status(value: Any) -> str:
    return str(value or "").strip().lower()


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


def queue_row_is_executable_pending(row: dict[str, Any], app_root: Path) -> bool:
    status = normalize_queue_status(row.get("status"))
    if status == "running":
        return True
    if status not in DISPATCHABLE_PENDING_STATUSES:
        return False
    return queue_row_has_existing_config(row, app_root)


def row_counts_pending(status: Any) -> bool:
    return normalize_queue_status(status) in PENDING_LIKE_STATUSES


def row_counts_dispatchable(status: Any) -> bool:
    return normalize_queue_status(status) in DISPATCHABLE_PENDING_STATUSES


def row_counts_executable(status: Any, config_path: str | Path, app_root: Path) -> bool:
    status_text = normalize_queue_status(status)
    if status_text == "running":
        return True
    if status_text not in DISPATCHABLE_PENDING_STATUSES:
        return False
    return queue_row_has_existing_config({"config_path": str(config_path or "")}, app_root)


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

        if status in DISPATCHABLE_PENDING_STATUSES:
            summary["dispatchable_pending"] += 1
        if status == "running" or (status in DISPATCHABLE_PENDING_STATUSES and cfg_exists):
            summary["executable_pending"] += 1
    return summary
