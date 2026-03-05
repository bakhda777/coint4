#!/usr/bin/env python3
"""Escalate confirm lane when strict-pass waits too long for confirmations."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Confirm SLA escalator.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    fullspan_path = state_dir / "fullspan_decision_state.json"
    out_path = state_dir / "confirm_sla_escalation_state.json"
    lock_path = state_dir / "confirm_sla_escalator.lock"
    log_path = state_dir / "confirm_sla_escalator.log"
    events_path = state_dir / "confirm_sla_escalator_events.jsonl"

    min_groups = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_GROUPS", "2"), 2)
    min_replies = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_REPLIES", "2"), 2)
    sla_confirm_sec = parse_int(os.environ.get("SLA_CONFIRM_PENDING_SEC", "7200"), 7200)
    boost_ttl_sec = parse_int(os.environ.get("CONFIRM_SLA_ESCALATION_TTL_SEC", "900"), 900)

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        state = load_json(fullspan_path, {})
        if not isinstance(state, dict):
            state = {}
        queues = state.get("queues", {})
        if not isinstance(queues, dict):
            queues = {}

        now_epoch = int(time.time())
        overdue: list[dict[str, Any]] = []
        for queue, entry in queues.items():
            if not isinstance(entry, dict):
                continue
            strict_pass_count = parse_int(entry.get("strict_pass_count"), 0)
            strict_run_groups = parse_int(entry.get("strict_run_group_count"), 0)
            confirm_count = parse_int(entry.get("confirm_count"), 0)
            if strict_pass_count <= 0 or strict_run_groups < min_groups or confirm_count >= min_replies:
                continue

            pending_since = parse_int(entry.get("confirm_pending_since_epoch"), 0)
            if pending_since <= 0:
                continue
            age_sec = max(0, now_epoch - pending_since)
            if age_sec < sla_confirm_sec:
                continue

            overdue.append(
                {
                    "queue": queue,
                    "age_sec": age_sec,
                    "strict_pass_count": strict_pass_count,
                    "strict_run_group_count": strict_run_groups,
                    "confirm_count": confirm_count,
                }
            )

        overdue.sort(key=lambda item: int(item.get("age_sec", 0)), reverse=True)

        active = bool(overdue)
        payload = {
            "version": 1,
            "ts": utc_now_iso(),
            "active": active,
            "until_epoch": now_epoch + boost_ttl_sec if active else now_epoch,
            "overdue_count": len(overdue),
            "overdue_queues": overdue[:20],
            "policy": {
                "search_parallel_max": 4 if active else 12,
                "confirm_parallel_min": 3 if active else 2,
                "confirm_parallel_max": 4,
                "confirm_dispatches_per_cycle": 3 if active else 2,
                "confirm_lane_max_active": 3 if active else 2,
                "confirm_lane_max_remote_runners": 8 if active else 6,
            },
            "source": {
                "sla_confirm_sec": sla_confirm_sec,
                "min_groups": min_groups,
                "min_replies": min_replies,
                "boost_ttl_sec": boost_ttl_sec,
            },
        }

        if not args.dry_run:
            dump_json(out_path, payload)

        if active:
            append_jsonl(
                events_path,
                {
                    "ts": utc_now_iso(),
                    "event": "CONFIRM_SLA_ESCALATION_ACTIVE",
                    "overdue_count": len(overdue),
                    "top_queue": overdue[0]["queue"] if overdue else "",
                    "max_age_sec": int(overdue[0]["age_sec"]) if overdue else 0,
                    "until_epoch": payload["until_epoch"],
                },
            )

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | active={int(active)} overdue_count={len(overdue)} "
                f"ttl={boost_ttl_sec}s dry_run={int(bool(args.dry_run))}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
