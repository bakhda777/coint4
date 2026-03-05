#!/usr/bin/env python3
"""Guard confirm replay independence across run_groups.

Fail-closed behavior:
- PROMOTE_ELIGIBLE requires >= N independent confirm run_groups
- if independence is violated, demote to PROMOTE_PENDING_CONFIRM
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
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


def as_groups(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    if "||" in text:
        return [x.strip() for x in text.split("||") if x.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def as_group_lineage(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        group = str(key or "").strip()
        if not group:
            continue
        keys = as_groups(value)
        if keys:
            out[group] = keys
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Confirm diversity guard (independent run_group confirmations).")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    state_path = state_dir / "fullspan_decision_state.json"
    lock_path = state_dir / "confirm_diversity_guard.lock"
    log_path = state_dir / "confirm_diversity_guard.log"
    events_path = state_dir / "confirm_diversity_guard_events.jsonl"

    min_replies = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_REPLIES", "2"), 2)
    min_lineages = parse_int(os.environ.get("FULLSPAN_CONFIRM_MIN_LINEAGES", str(min_replies)), min_replies)

    state_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0

        state = load_json(state_path, {})
        if not isinstance(state, dict):
            state = {}
        queues = state.get("queues", {})
        if not isinstance(queues, dict):
            queues = {}

        changed = 0
        demoted = 0

        for queue, entry in list(queues.items()):
            if not isinstance(entry, dict):
                continue

            verdict_prev = str(entry.get("promotion_verdict") or "ANALYZE").strip().upper() or "ANALYZE"
            top_run_group = str(entry.get("top_run_group") or "").strip()
            verified_groups = as_groups(entry.get("confirm_verified_run_groups"))
            verified_lineage_keys = as_groups(entry.get("confirm_verified_lineage_keys"))
            group_lineage = as_group_lineage(entry.get("confirm_verified_group_lineage_keys"))
            unique_verified = []
            seen = set()
            for group in verified_groups:
                if group not in seen:
                    seen.add(group)
                    unique_verified.append(group)

            independent = [group for group in unique_verified if group and group != top_run_group]
            independent_count = len(independent)
            independent_lineage: list[str] = []
            seen_lineage = set()
            for group in independent:
                for key in group_lineage.get(group, []):
                    lineage_key = str(key or "").strip()
                    if not lineage_key or lineage_key in seen_lineage:
                        continue
                    seen_lineage.add(lineage_key)
                    independent_lineage.append(lineage_key)
            if not independent_lineage:
                for key in verified_lineage_keys:
                    lineage_key = str(key or "").strip()
                    if not lineage_key or lineage_key in seen_lineage:
                        continue
                    seen_lineage.add(lineage_key)
                    independent_lineage.append(lineage_key)
            independent_lineage_count = len(independent_lineage)

            queue_changed = False
            if entry.get("confirm_independent_run_groups") != independent:
                entry["confirm_independent_run_groups"] = independent
                queue_changed = True
            if parse_int(entry.get("confirm_independent_count"), -1) != independent_count:
                entry["confirm_independent_count"] = independent_count
                queue_changed = True
            if entry.get("confirm_independent_lineage_keys") != independent_lineage:
                entry["confirm_independent_lineage_keys"] = independent_lineage
                queue_changed = True
            if parse_int(entry.get("confirm_independent_lineage_count"), -1) != independent_lineage_count:
                entry["confirm_independent_lineage_count"] = independent_lineage_count
                queue_changed = True

            if verdict_prev == "PROMOTE_ELIGIBLE" and (
                independent_count < min_replies or independent_lineage_count < min_lineages
            ):
                entry["promotion_verdict"] = "PROMOTE_PENDING_CONFIRM"
                entry["cutover_permission"] = "FAIL_CLOSED"
                entry["cutover_ready"] = False
                if independent_count < min_replies:
                    entry["rejection_reason"] = "confirm_non_independent"
                else:
                    entry["rejection_reason"] = "confirm_non_independent_lineage"
                demoted += 1
                queue_changed = True
                append_jsonl(
                    events_path,
                    {
                        "ts": utc_now_iso(),
                        "queue": queue,
                        "event": "PROMOTE_DEMOTED_CONFIRM_DIVERSITY",
                        "from": verdict_prev,
                        "to": "PROMOTE_PENDING_CONFIRM",
                        "top_run_group": top_run_group,
                        "verified_run_groups": unique_verified,
                        "independent_run_groups": independent,
                        "independent_lineage_keys": independent_lineage,
                        "required": min_replies,
                        "required_lineages": min_lineages,
                    },
                )

            if queue_changed:
                entry["confirm_diversity_guard_last_update"] = utc_now_iso()
                queues[queue] = entry
                changed += 1

        state["queues"] = queues
        metrics = state.get("runtime_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metrics["confirm_diversity_guard_cycle_count"] = parse_int(metrics.get("confirm_diversity_guard_cycle_count"), 0) + 1
        metrics["confirm_diversity_guard_last_epoch"] = int(datetime.now(timezone.utc).timestamp())
        metrics["confirm_diversity_guard_demoted_total"] = parse_int(metrics.get("confirm_diversity_guard_demoted_total"), 0) + demoted
        state["runtime_metrics"] = metrics

        if changed > 0 and not args.dry_run:
            dump_json(state_path, state)

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | cycle queues={len(queues)} changed={changed} demoted={demoted} dry_run={int(bool(args.dry_run))}\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
