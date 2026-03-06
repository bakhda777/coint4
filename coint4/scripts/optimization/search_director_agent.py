#!/usr/bin/env python3
"""Reason-aware search director for autonomous queue seeding.

Reads strict fullspan reject reasons and emits a directive consumed by
autonomous_queue_seeder.py to bias next batch generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from yield_governor_agent import build_yield_governor_state, dump_json as dump_yield_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def canonical_reason(entry: dict[str, Any]) -> str:
    strict_reason = str(entry.get("strict_gate_reason") or "").strip().upper()
    reject_reason = str(entry.get("rejection_reason") or "").strip().upper()
    merged = strict_reason or reject_reason
    if "DD_FAIL" in merged:
        return "DD_FAIL"
    if "STEP_FAIL" in merged:
        return "STEP_FAIL"
    if "TRADES_FAIL" in merged:
        return "TRADES_FAIL"
    if "PAIRS_FAIL" in merged:
        return "PAIRS_FAIL"
    if "ECONOMIC_FAIL" in merged:
        return "ECONOMIC_FAIL"
    if "METRICS_MISSING" in merged:
        return "METRICS_MISSING"
    return merged or "UNKNOWN"


def _merge_unique(tokens: list[str], extra: list[str], limit: int = 8) -> list[str]:
    out: list[str] = []
    seen = set()
    for token in [*tokens, *extra]:
        value = str(token or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def build_directive(queues: dict[str, Any], yield_state: dict[str, Any] | None = None) -> dict[str, Any]:
    reasons = Counter()
    run_group_hints: list[str] = []
    proximate_tokens: list[str] = []

    for queue, entry in queues.items():
        if not isinstance(entry, dict):
            continue
        verdict = str(entry.get("promotion_verdict") or "").strip().upper()
        strict_pass = parse_int(entry.get("strict_pass_count"), 0)
        if verdict in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM", "PROMOTE_ELIGIBLE"} or strict_pass > 0:
            token = str(entry.get("top_run_group") or "").strip()
            if not token:
                token = Path(str(queue)).parent.name
            if token:
                proximate_tokens.append(token)
        if verdict != "REJECT":
            continue
        reason = canonical_reason(entry)
        reasons[reason] += 1
        top_group = str(entry.get("top_run_group") or "").strip()
        if top_group:
            run_group_hints.append(top_group)
        else:
            queue_group = Path(str(queue)).parent.name
            if queue_group:
                run_group_hints.append(queue_group)

    dominant_reason = "UNKNOWN"
    if reasons:
        dominant_reason = reasons.most_common(1)[0][0]

    contains = []
    seen = set()
    for token in run_group_hints:
        if token and token not in seen:
            seen.add(token)
            contains.append(token)
        if len(contains) >= 3:
            break

    yield_state = yield_state if isinstance(yield_state, dict) else {}
    yield_active = bool(yield_state.get("active"))
    yield_preferred_contains = [str(token).strip() for token in list(yield_state.get("preferred_contains", []) or []) if str(token).strip()]
    winner_proximate = yield_state.get("winner_proximate", {})
    if not isinstance(winner_proximate, dict):
        winner_proximate = {}
    yield_winner_contains = [
        str(token).strip() for token in list(winner_proximate.get("contains", []) or []) if str(token).strip()
    ]
    lane_weights = yield_state.get("lane_weights", {})
    if not isinstance(lane_weights, dict):
        lane_weights = {}
    contains = _merge_unique(proximate_tokens, _merge_unique(yield_winner_contains, yield_preferred_contains), limit=8) or contains

    directive: dict[str, Any] = {
        "version": 1,
        "mode": "neutral",
        "dominant_reason": dominant_reason,
        "contains": contains,
        "policy_scale": "auto",
        "num_variants": 24,
        "max_changed_keys": 4,
        "dedupe_distance": 0.03,
        "extra_cli_args": {},
        "impossibility_pruner": {
            "enabled": False,
            "reason": "",
            "max_changed_keys_cap": 4,
            "dedupe_distance_floor": 0.03,
            "num_variants_cap": 24,
            "policy_scale": "auto",
        },
        "winner_proximate": {
            "enabled": bool(proximate_tokens or yield_winner_contains),
            "contains": _merge_unique(proximate_tokens, yield_winner_contains, limit=8),
            "reason": "strict_pass_or_high_yield_lineage",
        },
        "yield_governor": {
            "active": yield_active,
            "preferred_contains": yield_preferred_contains[:8],
            "cooldown_contains": [str(token).strip() for token in list(yield_state.get("cooldown_contains", []) or []) if str(token).strip()][:12],
            "preferred_operator_ids": [str(token).strip() for token in list(yield_state.get("preferred_operator_ids", []) or []) if str(token).strip()][:8],
            "cooldown_operator_ids": [str(token).strip() for token in list(yield_state.get("cooldown_operator_ids", []) or []) if str(token).strip()][:8],
        },
        "lane_weights": {
            "winner_proximate": max(0, parse_int(lane_weights.get("winner_proximate"), 40)),
            "broad_search": max(0, parse_int(lane_weights.get("broad_search"), 45)),
            "confirm_replay": max(0, parse_int(lane_weights.get("confirm_replay"), 15)),
        },
    }

    if dominant_reason == "DD_FAIL":
        directive.update(
            {
                "mode": "dd_focus",
                "policy_scale": "micro",
                "max_changed_keys": 2,
                "dedupe_distance": 0.05,
                "extra_cli_args": {
                    "--max-dd-pct": "0.12",
                    "--max-tail-pair-share": "0.40",
                    "--max-tail-period-share": "0.55",
                },
                "impossibility_pruner": {
                    "enabled": True,
                    "reason": "DD_FAIL",
                    "max_changed_keys_cap": 2,
                    "dedupe_distance_floor": 0.05,
                    "num_variants_cap": 24,
                    "policy_scale": "micro",
                },
            }
        )
    elif dominant_reason in {"TRADES_FAIL", "PAIRS_FAIL"}:
        directive.update(
            {
                "mode": "breadth_focus",
                "policy_scale": "macro",
                "max_changed_keys": 5,
                "num_variants": 32,
                "dedupe_distance": 0.04,
                "extra_cli_args": {
                    "--max-dd-pct": "0.20",
                },
                "impossibility_pruner": {
                    "enabled": True,
                    "reason": dominant_reason,
                    "max_changed_keys_cap": 5,
                    "dedupe_distance_floor": 0.04,
                    "num_variants_cap": 32,
                    "policy_scale": "macro",
                },
            }
        )
    elif dominant_reason in {"STEP_FAIL", "ECONOMIC_FAIL"}:
        directive.update(
            {
                "mode": "tail_economic_focus",
                "policy_scale": "micro",
                "max_changed_keys": 3,
                "dedupe_distance": 0.05,
                "extra_cli_args": {
                    "--max-dd-pct": "0.18",
                    "--max-tail-pair-share": "0.35",
                    "--max-tail-period-share": "0.50",
                },
                "impossibility_pruner": {
                    "enabled": True,
                    "reason": dominant_reason,
                    "max_changed_keys_cap": 3,
                    "dedupe_distance_floor": 0.05,
                    "num_variants_cap": 24,
                    "policy_scale": "micro",
                },
            }
        )
    elif dominant_reason == "METRICS_MISSING":
        directive.update(
            {
                "mode": "stability_focus",
                "policy_scale": "micro",
                "max_changed_keys": 2,
                "num_variants": 20,
                "impossibility_pruner": {
                    "enabled": True,
                    "reason": "METRICS_MISSING",
                    "max_changed_keys_cap": 2,
                    "dedupe_distance_floor": 0.05,
                    "num_variants_cap": 20,
                    "policy_scale": "micro",
                },
            }
        )

    return directive


def materialize_cold_fail_index(
    *,
    queues: dict[str, Any],
    path: Path,
    ttl_sec: int = 21600,
    policy_version: str = "fullspan_v1",
) -> dict[str, Any]:
    now_ts = int(datetime.now(timezone.utc).timestamp())
    payload = load_json(path, {"ts": "", "policy_version": policy_version, "entries": []})
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        entries = []

    live_entries: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        queue = str(entry.get("queue") or "").strip()
        if not queue:
            continue
        until_ts = parse_int(entry.get("until_ts"), 0)
        if until_ts <= now_ts:
            continue
        live_entries[queue] = dict(entry)

    added = 0
    for queue, entry in queues.items():
        if not isinstance(entry, dict):
            continue
        verdict = str(entry.get("promotion_verdict") or "").strip().upper()
        strict_gate = str(entry.get("strict_gate_status") or "").strip().upper()
        if verdict != "REJECT" and strict_gate != "FULLSPAN_PREFILTER_REJECT":
            continue
        gate_reason = canonical_reason(entry)
        if not gate_reason:
            continue
        existing = live_entries.get(queue, {})
        live_entries[queue] = {
            "queue": str(queue),
            "run_group": str(entry.get("top_run_group") or Path(str(queue)).parent.name).strip(),
            "lineage_uid": str(entry.get("candidate_uid") or "").strip(),
            "gate_reason": gate_reason,
            "source_verdict": verdict or "REJECT",
            "inserted_ts": parse_int(existing.get("inserted_ts"), now_ts),
            "until_ts": max(parse_int(existing.get("until_ts"), 0), now_ts + max(0, int(ttl_sec))),
            "policy_version": policy_version,
        }
        if not existing:
            added += 1

    materialized = {
        "ts": utc_now_iso(),
        "policy_version": policy_version,
        "entries": sorted(live_entries.values(), key=lambda item: str(item.get("queue") or "")),
    }
    dump_json(path, materialized)
    return {
        "path": str(path),
        "active_count": len(materialized["entries"]),
        "added": added,
    }


def _neutral_hard_fail_risk_policy() -> dict[str, Any]:
    return {
        "enabled": False,
        "reject_threshold": 0.75,
        "refine_threshold": 0.45,
    }


def _neutral_repair_mode() -> dict[str, Any]:
    return {
        "enabled": False,
        "validation_neighbor": 0,
        "max_neighbor_attempts": 0,
    }


def _sanitize_lineage_priority(raw: Any, limit: int = 12) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen = set()
    for token in raw:
        value = str(token or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def build_gate_surrogate_overlay(gate_state: Any) -> dict[str, Any]:
    overlay: dict[str, Any] = {
        "hard_fail_risk_policy": _neutral_hard_fail_risk_policy(),
        "lineage_priority": [],
        "repair_mode": _neutral_repair_mode(),
        "gate_surrogate_mode": "neutral",
        "gate_surrogate_ts": "",
    }
    if not isinstance(gate_state, dict):
        return overlay

    raw_policy = gate_state.get("hard_fail_risk_policy", {})
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    enabled = bool(policy.get("enabled"))
    reject_threshold = max(0.0, min(1.0, parse_float(policy.get("reject_threshold"), 0.75)))
    refine_threshold = max(0.0, min(1.0, parse_float(policy.get("refine_threshold"), 0.45)))
    if reject_threshold < refine_threshold:
        reject_threshold, refine_threshold = refine_threshold, reject_threshold

    overlay["hard_fail_risk_policy"] = {
        "enabled": enabled,
        "reject_threshold": round(reject_threshold, 4),
        "refine_threshold": round(refine_threshold, 4),
    }

    overlay["lineage_priority"] = _sanitize_lineage_priority(gate_state.get("lineage_priority", []), limit=12)

    raw_repair = gate_state.get("repair_mode", {})
    repair = raw_repair if isinstance(raw_repair, dict) else {}
    repair_enabled = bool(enabled and repair.get("enabled"))
    overlay["repair_mode"] = {
        "enabled": repair_enabled,
        "validation_neighbor": max(0, parse_int(repair.get("validation_neighbor"), 0)) if repair_enabled else 0,
        "max_neighbor_attempts": max(0, parse_int(repair.get("max_neighbor_attempts"), 0)) if repair_enabled else 0,
    }

    summary = gate_state.get("summary", {})
    summary_mode = ""
    if isinstance(summary, dict):
        summary_mode = str(summary.get("mode") or "").strip().lower()
    overlay["gate_surrogate_mode"] = summary_mode or ("active" if enabled else "neutral")
    overlay["gate_surrogate_ts"] = str(gate_state.get("ts") or "")
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reason-aware search directive from fullspan decision state.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    fullspan_state_path = state_dir / "fullspan_decision_state.json"
    gate_state_path = state_dir / "gate_surrogate_state.json"
    yield_state_path = state_dir / "yield_governor_state.json"
    cold_fail_state_path = state_dir / "cold_fail_index.json"
    directive_path = state_dir / "search_director_directive.json"
    log_path = state_dir / "search_director.log"
    run_index_path = root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"

    fullspan_state = load_json(fullspan_state_path, {})
    queues = fullspan_state.get("queues", {}) if isinstance(fullspan_state, dict) else {}
    if not isinstance(queues, dict):
        queues = {}
    gate_state = load_json(gate_state_path, {})
    yield_state = build_yield_governor_state(
        root=root,
        aggregate_dir=root / "artifacts" / "wfa" / "aggregate",
        run_index_path=run_index_path,
        fullspan_state_path=fullspan_state_path,
        recent_queue_limit=200,
    )

    directive = build_directive(queues, yield_state=yield_state)
    directive.update(build_gate_surrogate_overlay(gate_state))
    directive["ts"] = utc_now_iso()
    directive["source"] = "search_director_agent"
    directive["queue_count"] = len(queues)
    directive["yield_governor_state_path"] = str(yield_state_path)
    directive["cold_fail_index_path"] = str(cold_fail_state_path)
    cold_fail_summary = materialize_cold_fail_index(queues=queues, path=cold_fail_state_path)
    directive["cold_fail_active_count"] = int(cold_fail_summary.get("active_count", 0))

    if not args.dry_run:
        dump_yield_json(yield_state_path, yield_state)
        dump_json(directive_path, directive)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{utc_now_iso()} | mode={directive.get('mode')} dominant_reason={directive.get('dominant_reason')} "
                f"contains={directive.get('contains')} risk_policy={int(bool((directive.get('hard_fail_risk_policy') or {}).get('enabled')))} "
                f"lineage={len(list(directive.get('lineage_priority') or []))} "
                f"repair={int(bool((directive.get('repair_mode') or {}).get('enabled')))} "
                f"winner_proximate={len(list((directive.get('winner_proximate') or {}).get('contains') or []))} "
                f"cold_fail_active={int(cold_fail_summary.get('active_count', 0))}\n"
            )
    else:
        print(json.dumps(directive, ensure_ascii=False, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
