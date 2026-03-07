#!/usr/bin/env python3
"""Reason-aware search director for autonomous queue seeding.

Reads strict fullspan reject reasons and emits a directive consumed by
autonomous_queue_seeder.py to bias next batch generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _search_quality_contract import (
    CANONICAL_ZERO_EVIDENCE_REASONS,
    micro_broad_search_caps,
    normalize_search_quality_state,
)
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


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}


EXPLOIT_FIRST_LANE_WEIGHTS = {
    "winner_proximate": 65,
    "confirm_replay": 20,
    "broad_search": 15,
}


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
    for reason in CANONICAL_ZERO_EVIDENCE_REASONS:
        if reason in merged:
            return reason
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


def _normalize_lane_weights(raw: Any) -> dict[str, int]:
    source = raw if isinstance(raw, dict) else {}
    winner = max(0, parse_int(source.get("winner_proximate"), EXPLOIT_FIRST_LANE_WEIGHTS["winner_proximate"]))
    confirm = max(0, parse_int(source.get("confirm_replay"), EXPLOIT_FIRST_LANE_WEIGHTS["confirm_replay"]))
    broad = max(0, parse_int(source.get("broad_search"), EXPLOIT_FIRST_LANE_WEIGHTS["broad_search"]))
    return {
        "winner_proximate": winner,
        "confirm_replay": confirm,
        "broad_search": broad,
    }


def _stable_policy_hash(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _collect_replay_fastlane(queues: dict[str, Any], *, yield_state: dict[str, Any]) -> dict[str, Any]:
    contains: list[str] = []
    pending_confirm_queues: list[str] = []
    replay_ready_count = 0

    for queue, raw_entry in queues.items():
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        verdict = str(entry.get("promotion_verdict") or "").strip().upper()
        strict_pass = max(0, parse_int(entry.get("strict_pass_count"), 0))
        strict_groups = max(0, parse_int(entry.get("strict_run_group_count"), 0))
        confirm_count = max(0, parse_int(entry.get("confirm_count"), 0))
        token = str(entry.get("top_run_group") or "").strip() or Path(str(queue)).parent.name
        if verdict in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"}:
            pending_confirm_queues.append(str(queue))
            if token:
                contains.append(token)
        if strict_pass > 0 and strict_groups >= 2 and confirm_count < 2:
            replay_ready_count += 1
            if token:
                contains.append(token)

    yield_replay = yield_state.get("replay_fastlane", {}) if isinstance(yield_state, dict) else {}
    if not isinstance(yield_replay, dict):
        yield_replay = {}
    yield_contains = [str(token).strip() for token in list(yield_replay.get("contains", []) or []) if str(token).strip()]
    yield_pending = [str(queue).strip() for queue in list(yield_replay.get("pending_confirm_queues", []) or []) if str(queue).strip()]
    replay_ready_count = max(replay_ready_count, max(0, parse_int(yield_replay.get("replay_ready_count"), 0)))

    return {
        "enabled": bool(pending_confirm_queues or yield_pending or replay_ready_count > 0),
        "contains": _merge_unique(contains, yield_contains, limit=8),
        "pending_confirm_queues": _merge_unique(pending_confirm_queues, yield_pending, limit=16),
        "replay_ready_count": replay_ready_count,
        "reason": "strict_pass_pending_confirm_replay",
        "min_confirm_run_groups": 2,
        "min_confirm_replays": 2,
    }


def _build_planner_policy_inputs(directive: dict[str, Any]) -> dict[str, Any]:
    winner = directive.get("winner_proximate", {})
    if not isinstance(winner, dict):
        winner = {}
    replay = directive.get("replay_fastlane", {})
    if not isinstance(replay, dict):
        replay = {}
    yield_governor = directive.get("yield_governor", {})
    if not isinstance(yield_governor, dict):
        yield_governor = {}
    risk_policy = directive.get("hard_fail_risk_policy", {})
    if not isinstance(risk_policy, dict):
        risk_policy = {}
    repair_mode = directive.get("repair_mode", {})
    if not isinstance(repair_mode, dict):
        repair_mode = {}
    search_quality = directive.get("search_quality", {})
    if not isinstance(search_quality, dict):
        search_quality = {}
    return {
        "version": max(1, parse_int(directive.get("version"), 1)),
        "policy_family": "exploit_first",
        "mode": str(directive.get("mode") or ""),
        "dominant_reason": str(directive.get("dominant_reason") or ""),
        "contains": [str(token).strip() for token in list(directive.get("contains", []) or []) if str(token).strip()][:12],
        "policy_scale": str(directive.get("policy_scale") or "auto"),
        "num_variants": max(1, parse_int(directive.get("num_variants"), 24)),
        "max_changed_keys": max(1, parse_int(directive.get("max_changed_keys"), 4)),
        "dedupe_distance": round(max(0.0, parse_float(directive.get("dedupe_distance"), 0.03)), 6),
        "lane_weights": _normalize_lane_weights(directive.get("lane_weights")),
        "winner_proximate": {
            "enabled": bool(winner.get("enabled")),
            "contains": [str(token).strip() for token in list(winner.get("contains", []) or []) if str(token).strip()][:8],
        },
        "replay_fastlane": {
            "enabled": bool(replay.get("enabled")),
            "contains": [str(token).strip() for token in list(replay.get("contains", []) or []) if str(token).strip()][:8],
            "pending_confirm_queues": [str(queue).strip() for queue in list(replay.get("pending_confirm_queues", []) or []) if str(queue).strip()][:16],
            "replay_ready_count": max(0, parse_int(replay.get("replay_ready_count"), 0)),
        },
        "search_quality": {
            "positive_lineage_count": max(0, parse_int(search_quality.get("positive_lineage_count"), 0)),
            "zero_evidence_lineage_count": max(0, parse_int(search_quality.get("zero_evidence_lineage_count"), 0)),
            "winner_proximate_positive_lineage_count": max(
                0,
                parse_int(search_quality.get("winner_proximate_positive_lineage_count"), 0),
            ),
            "winner_proximate_positive_contains": [
                str(token).strip()
                for token in list(search_quality.get("winner_proximate_positive_contains", []) or [])
                if str(token).strip()
            ][:8],
            "controlled_recovery_contains": [
                str(token).strip()
                for token in list(search_quality.get("controlled_recovery_contains", []) or [])
                if str(token).strip()
            ][:8],
            "broad_search_allowed": bool(search_quality.get("broad_search_allowed")),
            "seed_generation_mode": str(search_quality.get("seed_generation_mode") or "broad_search_micro").strip(),
            "controlled_recovery_active": bool(search_quality.get("controlled_recovery_active")),
            "controlled_recovery_reason": str(search_quality.get("controlled_recovery_reason") or "").strip(),
            "controlled_recovery_attempts_remaining": max(
                0,
                parse_int(search_quality.get("controlled_recovery_attempts_remaining"), 0),
            ),
            "controlled_recovery_variants_cap": max(
                1,
                parse_int(search_quality.get("controlled_recovery_variants_cap"), 8),
            ),
        },
        "yield_governor": {
            "active": bool(yield_governor.get("active")),
            "preferred_contains": [str(token).strip() for token in list(yield_governor.get("preferred_contains", []) or []) if str(token).strip()][:12],
            "cooldown_contains": [str(token).strip() for token in list(yield_governor.get("cooldown_contains", []) or []) if str(token).strip()][:12],
            "preferred_operator_ids": [str(token).strip() for token in list(yield_governor.get("preferred_operator_ids", []) or []) if str(token).strip()][:8],
            "cooldown_operator_ids": [str(token).strip() for token in list(yield_governor.get("cooldown_operator_ids", []) or []) if str(token).strip()][:8],
        },
        "hard_fail_risk_policy": {
            "enabled": bool(risk_policy.get("enabled")),
            "reject_threshold": round(max(0.0, parse_float(risk_policy.get("reject_threshold"), 0.75)), 4),
            "refine_threshold": round(max(0.0, parse_float(risk_policy.get("refine_threshold"), 0.45)), 4),
        },
        "lineage_priority": [str(token).strip() for token in list(directive.get("lineage_priority", []) or []) if str(token).strip()][:12],
        "repair_mode": {
            "enabled": bool(repair_mode.get("enabled")),
            "validation_neighbor": max(0, parse_int(repair_mode.get("validation_neighbor"), 0)),
            "max_neighbor_attempts": max(0, parse_int(repair_mode.get("max_neighbor_attempts"), 0)),
        },
    }


def _attach_policy_metadata(directive: dict[str, Any]) -> dict[str, Any]:
    policy_inputs = _build_planner_policy_inputs(directive)
    policy_hash = _stable_policy_hash(policy_inputs)
    directive["planner-policy-inputs"] = policy_inputs
    directive["policy-hash"] = policy_hash
    # Keep underscore aliases for backward-compatible consumers.
    directive["planner_policy_inputs"] = policy_inputs
    directive["policy_hash"] = policy_hash
    return directive


def build_directive(queues: dict[str, Any], yield_state: dict[str, Any] | None = None) -> dict[str, Any]:
    micro_caps = micro_broad_search_caps()
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
    search_quality = normalize_search_quality_state(
        yield_state,
        winner_proximate_contains=_merge_unique(proximate_tokens, yield_winner_contains, limit=8),
    )
    lane_weights = _normalize_lane_weights(yield_state.get("lane_weights", {}))
    if not bool(search_quality.get("broad_search_allowed")):
        lane_weights["broad_search"] = 0
    replay_fastlane = _collect_replay_fastlane(queues, yield_state=yield_state)
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
            **micro_caps,
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
        "replay_fastlane": replay_fastlane,
        "search_quality": search_quality,
        "positive_lineage_count": int(search_quality.get("positive_lineage_count", 0) or 0),
        "zero_evidence_lineage_count": int(search_quality.get("zero_evidence_lineage_count", 0) or 0),
        "winner_proximate_positive_lineage_count": int(
            search_quality.get("winner_proximate_positive_lineage_count", 0) or 0
        ),
        "winner_proximate_positive_contains": list(search_quality.get("winner_proximate_positive_contains", []) or []),
        "controlled_recovery_contains": list(search_quality.get("controlled_recovery_contains", []) or []),
        "broad_search_allowed": bool(search_quality.get("broad_search_allowed")),
        "seed_generation_mode": str(search_quality.get("seed_generation_mode") or "broad_search_micro").strip(),
        "controlled_recovery_active": bool(search_quality.get("controlled_recovery_active")),
        "controlled_recovery_reason": str(search_quality.get("controlled_recovery_reason") or ""),
        "controlled_recovery_attempts_remaining": int(
            search_quality.get("controlled_recovery_attempts_remaining", 0) or 0
        ),
        "controlled_recovery_variants_cap": int(search_quality.get("controlled_recovery_variants_cap", 0) or 0),
        "lane_weights": lane_weights,
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
                    **micro_caps,
                },
            }
        )
    elif dominant_reason in {"METRICS_MISSING", *CANONICAL_ZERO_EVIDENCE_REASONS}:
        directive.update(
            {
                "mode": "stability_focus",
                "policy_scale": "micro",
                "max_changed_keys": int(micro_caps["max_changed_keys_cap"]),
                "num_variants": 20,
                "dedupe_distance": float(micro_caps["dedupe_distance_floor"]),
                "impossibility_pruner": {
                    "enabled": True,
                    "reason": dominant_reason,
                    **micro_caps,
                },
            }
        )

    if str(search_quality.get("seed_generation_mode") or "") == "broad_search_micro":
        directive["policy_scale"] = "micro"
        directive["max_changed_keys"] = min(
            max(1, parse_int(directive.get("max_changed_keys"), micro_caps["max_changed_keys_cap"])),
            int(micro_caps["max_changed_keys_cap"]),
        )
        directive["num_variants"] = min(
            max(1, parse_int(directive.get("num_variants"), micro_caps["num_variants_cap"])),
            int(micro_caps["num_variants_cap"]),
        )
        directive["dedupe_distance"] = max(
            float(micro_caps["dedupe_distance_floor"]),
            parse_float(directive.get("dedupe_distance"), float(micro_caps["dedupe_distance_floor"])),
        )

    if bool(search_quality.get("controlled_recovery_active")):
        controlled_contains = [
            str(token).strip()
            for token in list(
                search_quality.get("controlled_recovery_contains")
                or search_quality.get("winner_proximate_positive_contains", [])
                or []
            )
            if str(token).strip()
        ][:8]
        controlled_variants_cap = max(1, parse_int(search_quality.get("controlled_recovery_variants_cap"), 8))
        lane_weights = _normalize_lane_weights(directive.get("lane_weights"))
        lane_weights["winner_proximate"] = max(int(lane_weights.get("winner_proximate", 0)), 100)
        lane_weights["confirm_replay"] = 0
        lane_weights["broad_search"] = 0
        directive.update(
            {
                "mode": "controlled_recovery",
                "contains": controlled_contains,
                "policy_scale": "micro",
                "num_variants": controlled_variants_cap,
                "max_changed_keys": min(
                    max(1, parse_int(directive.get("max_changed_keys"), int(micro_caps["max_changed_keys_cap"]))),
                    int(micro_caps["max_changed_keys_cap"]),
                ),
                "dedupe_distance": max(
                    0.08,
                    parse_float(directive.get("dedupe_distance"), 0.08),
                ),
                "winner_proximate": {
                    "enabled": bool(controlled_contains),
                    "contains": controlled_contains,
                    "reason": "controlled_recovery_positive_winner_lineage",
                },
                "repair_mode": {
                    "enabled": True,
                    "validation_neighbor": 1,
                    "max_neighbor_attempts": min(controlled_variants_cap, 8),
                },
                "lineage_priority": controlled_contains,
                "lane_weights": lane_weights,
            }
        )

    return _attach_policy_metadata(directive)


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
    existing_yield_state = load_json(yield_state_path, {})
    if not isinstance(existing_yield_state, dict):
        existing_yield_state = {}
    yield_state = build_yield_governor_state(
        root=root,
        aggregate_dir=root / "artifacts" / "wfa" / "aggregate",
        run_index_path=run_index_path,
        fullspan_state_path=fullspan_state_path,
        recent_queue_limit=200,
        hard_block_active=parse_bool(existing_yield_state.get("hard_block_active")),
        hard_block_reason=str(existing_yield_state.get("hard_block_reason") or ""),
        existing_state=existing_yield_state,
    )
    for key in (
        "hard_block_active",
        "hard_block_reason",
        "hard_block_until_epoch",
        "zero_coverage_seed_streak",
        "zero_coverage_seed_streak_reason",
    ):
        if key in existing_yield_state:
            yield_state[key] = existing_yield_state[key]
    if "winner_proximate_positive_contains" in existing_yield_state and not yield_state.get("winner_proximate_positive_contains"):
        yield_state["winner_proximate_positive_contains"] = list(existing_yield_state.get("winner_proximate_positive_contains") or [])
    if "controlled_recovery_contains" in existing_yield_state and not yield_state.get("controlled_recovery_contains"):
        yield_state["controlled_recovery_contains"] = list(existing_yield_state.get("controlled_recovery_contains") or [])
    if isinstance(yield_state.get("search_quality"), dict):
        for key in (
            "winner_proximate_positive_contains",
            "controlled_recovery_contains",
            "controlled_recovery_active",
            "controlled_recovery_reason",
            "controlled_recovery_attempts_remaining",
            "controlled_recovery_variants_cap",
        ):
            if key in yield_state:
                yield_state["search_quality"][key] = yield_state[key]

    directive = build_directive(queues, yield_state=yield_state)
    directive.update(build_gate_surrogate_overlay(gate_state))
    directive["ts"] = utc_now_iso()
    directive["source"] = "search_director_agent"
    directive["queue_count"] = len(queues)
    directive["yield_governor_state_path"] = str(yield_state_path)
    directive["cold_fail_index_path"] = str(cold_fail_state_path)
    cold_fail_summary = materialize_cold_fail_index(queues=queues, path=cold_fail_state_path)
    directive["cold_fail_active_count"] = int(cold_fail_summary.get("active_count", 0))
    directive = _attach_policy_metadata(directive)

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
