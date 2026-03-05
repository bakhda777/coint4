#!/usr/bin/env python3
"""Autonomous backlog janitor for WFA queues.

Purpose:
- fail-closed cleanup of non-runnable pending rows
- reduce scheduler noise from deterministic dead entries
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PENDING_STATUSES = {"planned", "queued", "stalled", "failed", "error", "active"}
DETERMINISTIC_CODES = {
    "CONFIG_VALIDATION_ERROR",
    "MAX_VAR_MULTIPLIER_INVALID",
    "MAX_CORRELATION_INVALID",
    "NON_POSITIVE_THRESHOLD",
    "INVALID_PARAM",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def resolve_under_root(path_text: str, root: Path) -> Path:
    path = Path(str(path_text or "").strip())
    if path.is_absolute():
        return path
    return root / path


def derive_run_id(results_dir: str) -> str:
    text = str(results_dir or "").strip().rstrip("/")
    if not text:
        return ""
    return Path(text).name


def load_deterministic_map(path: Path) -> dict[str, str]:
    payload = load_json(path, {})
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return {}

    out: dict[str, str] = {}
    for entry in entries[-800:]:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "").strip().upper()
        if code not in DETERMINISTIC_CODES:
            continue
        run_id = str(entry.get("run_id") or "").strip()
        if run_id:
            out[run_id] = code
            continue
        inferred = derive_run_id(str(entry.get("results_dir") or ""))
        if inferred:
            out[inferred] = code
    return out


def scan_and_clean_queue(
    *,
    queue_path: Path,
    app_root: Path,
    deterministic_by_run_id: dict[str, str],
    dry_run: bool,
) -> tuple[bool, dict[str, int]]:
    counters = {
        "rows_total": 0,
        "pending_rows": 0,
        "config_missing_skipped": 0,
        "deterministic_skipped": 0,
    }
    if not queue_path.exists():
        return False, counters

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ["config_path", "results_dir", "status"])
        for raw in reader:
            row = {str(k): str(v or "").strip() for k, v in (raw or {}).items()}
            rows.append(row)

    changed = False
    for row in rows:
        counters["rows_total"] += 1
        status = str(row.get("status") or "").strip().lower()
        if status not in PENDING_STATUSES:
            continue
        counters["pending_rows"] += 1

        config_path = str(row.get("config_path") or "").strip()
        cfg_ok = False
        if config_path:
            cfg_ok = resolve_under_root(config_path, app_root).exists()
        if not cfg_ok:
            row["status"] = "skipped"
            changed = True
            counters["config_missing_skipped"] += 1
            continue

        if status in {"stalled", "failed", "error"}:
            run_id = derive_run_id(str(row.get("results_dir") or ""))
            code = deterministic_by_run_id.get(run_id, "")
            if code:
                row["status"] = "skipped"
                changed = True
                counters["deterministic_skipped"] += 1

    if changed and not dry_run:
        with queue_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    return changed, counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous backlog janitor agent.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    aggregate_root = root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    state_dir.mkdir(parents=True, exist_ok=True)

    lock_path = state_dir / "backlog_janitor.lock"
    state_path = state_dir / "backlog_janitor_state.json"
    log_path = state_dir / "backlog_janitor.log"
    events_path = state_dir / "backlog_janitor_events.jsonl"
    quarantine_path = state_dir / "deterministic_quarantine.json"

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        deterministic_by_run_id = load_deterministic_map(quarantine_path)

        summary = {
            "ts": utc_now_iso(),
            "status": "ok",
            "dry_run": bool(args.dry_run),
            "queues_scanned": 0,
            "queues_changed": 0,
            "rows_total": 0,
            "pending_rows": 0,
            "config_missing_skipped": 0,
            "deterministic_skipped": 0,
            "deterministic_run_ids": len(deterministic_by_run_id),
        }

        changed_queues: list[str] = []
        for queue_path in sorted(aggregate_root.glob("*/run_queue.csv")):
            if queue_path.parent.name.startswith("."):
                continue
            summary["queues_scanned"] += 1
            changed, counters = scan_and_clean_queue(
                queue_path=queue_path,
                app_root=root,
                deterministic_by_run_id=deterministic_by_run_id,
                dry_run=bool(args.dry_run),
            )
            for key in ("rows_total", "pending_rows", "config_missing_skipped", "deterministic_skipped"):
                summary[key] += int(counters.get(key, 0))
            if changed:
                summary["queues_changed"] += 1
                changed_queues.append(str(queue_path))

        if changed_queues:
            append_jsonl(
                events_path,
                {
                    "ts": summary["ts"],
                    "event": "BACKLOG_JANITOR_APPLIED",
                    "severity": "info",
                    "payload": {
                        "queues_changed": summary["queues_changed"],
                        "config_missing_skipped": summary["config_missing_skipped"],
                        "deterministic_skipped": summary["deterministic_skipped"],
                    },
                    "queues": changed_queues[:100],
                },
            )

        dump_json(state_path, summary)
        append_log(
            log_path,
            (
                "queues_scanned={queues_scanned} queues_changed={queues_changed} "
                "config_missing_skipped={config_missing_skipped} deterministic_skipped={deterministic_skipped} "
                "dry_run={dry_run}"
            ).format(**summary),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
