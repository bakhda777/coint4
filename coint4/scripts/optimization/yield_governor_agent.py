#!/usr/bin/env python3
"""Build fail-safe yield governor state for autonomous queue seeding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _search_quality_contract import (
    CONTROLLED_RECOVERY_MAX_BATCHES,
    CONTROLLED_RECOVERY_REASON,
    CONTROLLED_RECOVERY_VARIANTS_CAP,
    DETERMINISTIC_QUARANTINE_CODES,
    build_controlled_recovery_state,
    build_search_quality_state,
    canonical_zero_evidence_reason,
    controlled_recovery_rearm_eligible,
    has_positive_coverage_trade_evidence,
    micro_broad_search_caps,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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


def _normalize_tokens(values: Any, *, limit: int = 8) -> list[str]:
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",")]
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
        if len(normalized) >= max(1, int(limit)):
            break
    return normalized


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def canonical_hard_fail_reason(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return "METRICS_MISSING"
    if not parse_bool(row.get("metrics_present")):
        return "METRICS_MISSING"
    if parse_float(row.get("total_trades"), 0.0) < 200.0:
        return "TRADES_FAIL"
    if parse_float(row.get("total_pairs_traded"), 0.0) < 20.0:
        return "PAIRS_FAIL"
    if abs(parse_float(row.get("max_drawdown_on_equity"), 0.0)) > 0.20:
        return "DD_FAIL"
    if parse_float(row.get("total_pnl"), 0.0) < 0.0:
        return "ECONOMIC_FAIL"
    worst_step = row.get("tail_loss_worst_period_pnl")
    if worst_step in {None, ""}:
        worst_step = row.get("tail_loss_worst_pair_pnl")
    if parse_float(worst_step, 0.0) < -200.0:
        return "STEP_FAIL"
    return ""


def _decode_metadata_json(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = str(row.get("metadata_json") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _row_lineage_uid(row: Mapping[str, Any], queue_group: str) -> str:
    explicit = str(row.get("lineage_uid") or "").strip()
    if explicit:
        return explicit
    meta = _decode_metadata_json(row)
    explicit = str(meta.get("lineage_uid") or "").strip()
    if explicit:
        return explicit
    return queue_group


def _row_operator_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("operator_id") or "").strip()
    if explicit:
        return explicit
    meta = _decode_metadata_json(row)
    explicit = str(meta.get("operator_id") or "").strip()
    if explicit:
        return explicit
    return "unknown"


def _row_run_group_token(queue_group: str, run_index_row: Mapping[str, Any] | None) -> str:
    if isinstance(run_index_row, Mapping):
        token = str(run_index_row.get("run_group") or "").strip()
        if token:
            return token
    return queue_group


def _classify_yield(*, completed: int, metrics_present: int, zero_activity: int, hard_fail: int) -> tuple[str, float]:
    if completed <= 0:
        return "neutral", 0.0
    metrics_rate = float(metrics_present) / float(completed)
    zero_rate = float(zero_activity) / float(completed)
    hard_fail_rate = float(hard_fail) / float(completed)
    sample_bonus = min(float(completed), 24.0) / 24.0
    score = (metrics_rate * 2.0) - zero_rate - (hard_fail_rate * 0.75) + (sample_bonus * 0.25)
    if completed >= 8 and (zero_rate >= 0.75 or metrics_rate <= 0.20 or hard_fail_rate >= 0.80):
        return "cooldown", round(score, 6)
    if completed >= 8 and metrics_rate >= 0.60 and zero_rate <= 0.25 and hard_fail_rate <= 0.40:
        return "boost", round(score, 6)
    return "neutral", round(score, 6)


EXPLOIT_FIRST_LANE_WEIGHTS = {
    "winner_proximate": 65,
    "confirm_replay": 20,
    "broad_search": 15,
}


def _stable_policy_hash(payload: Mapping[str, Any]) -> str:
    body = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _normalize_lane_weights(weights: Mapping[str, Any] | None = None) -> dict[str, int]:
    raw = weights if isinstance(weights, Mapping) else {}
    winner = max(0, parse_int(raw.get("winner_proximate"), EXPLOIT_FIRST_LANE_WEIGHTS["winner_proximate"]))
    confirm = max(0, parse_int(raw.get("confirm_replay"), EXPLOIT_FIRST_LANE_WEIGHTS["confirm_replay"]))
    broad = max(0, parse_int(raw.get("broad_search"), EXPLOIT_FIRST_LANE_WEIGHTS["broad_search"]))
    return {
        "winner_proximate": winner,
        "confirm_replay": confirm,
        "broad_search": broad,
    }


def _collect_replay_fastlane(queues_state: Mapping[str, Any]) -> dict[str, Any]:
    contains: list[str] = []
    pending_confirm_queues: list[str] = []
    strict_pass_queues: list[str] = []
    replay_ready = 0

    for queue_rel, raw_entry in queues_state.items():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        verdict = str(entry.get("promotion_verdict") or "").strip().upper()
        strict_pass = max(0, parse_int(entry.get("strict_pass_count"), 0))
        strict_groups = max(0, parse_int(entry.get("strict_run_group_count"), 0))
        confirm_count = max(0, parse_int(entry.get("confirm_count"), 0))
        token = str(entry.get("top_run_group") or "").strip() or Path(str(queue_rel)).parent.name
        if token and token not in contains:
            contains.append(token)
        if strict_pass > 0 and str(queue_rel) not in strict_pass_queues:
            strict_pass_queues.append(str(queue_rel))
        is_confirm_pending = verdict in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM"}
        if is_confirm_pending and str(queue_rel) not in pending_confirm_queues:
            pending_confirm_queues.append(str(queue_rel))
        if is_confirm_pending and strict_groups >= 2 and confirm_count < 2:
            replay_ready += 1
    return {
        "enabled": bool(pending_confirm_queues or replay_ready > 0),
        "contains": contains[:8],
        "pending_confirm_queues": pending_confirm_queues[:16],
        "strict_pass_queues": strict_pass_queues[:16],
        "replay_ready_count": int(replay_ready),
        "source": "fullspan_decision_state",
    }


def _load_recent_quarantine_by_group(path: Path, *, limit: int = 200) -> dict[str, dict[str, int]]:
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return {}
    grouped: dict[str, dict[str, int]] = {}
    for entry in entries[-max(1, int(limit)) :]:
        if not isinstance(entry, dict):
            continue
        queue = str(entry.get("queue") or "").strip()
        code = str(entry.get("code") or "").strip().upper()
        if not queue or not code:
            continue
        run_group = Path(queue).parent.name
        counters = grouped.setdefault(run_group, {})
        counters[code] = int(counters.get(code, 0) or 0) + 1
    return grouped


def _filter_controlled_recovery_contains(
    tokens: list[str],
    *,
    quarantine_by_group: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    grouped = quarantine_by_group if isinstance(quarantine_by_group, Mapping) else {}
    filtered: list[str] = []
    for token in tokens:
        token_text = str(token or "").strip()
        if not token_text:
            continue
        counts = grouped.get(token_text, {})
        if isinstance(counts, Mapping) and any(
            parse_int(counts.get(code), 0) > 0 for code in sorted(DETERMINISTIC_QUARANTINE_CODES)
        ):
            continue
        filtered.append(token_text)
    return filtered


def _build_planner_policy_inputs(
    *,
    lane_weights: Mapping[str, Any],
    preferred_contains: list[str],
    cooldown_contains: list[str],
    preferred_operator_ids: list[str],
    cooldown_operator_ids: list[str],
    winner_proximate: Mapping[str, Any],
    replay_fastlane: Mapping[str, Any],
    search_quality: Mapping[str, Any],
    sample_size: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "v2",
        "policy_family": "exploit_first",
        "lane_weights": _normalize_lane_weights(lane_weights),
        "winner_proximate": {
            "enabled": bool(winner_proximate.get("enabled")),
            "contains": [str(token).strip() for token in list(winner_proximate.get("contains", []) or []) if str(token).strip()][:8],
        },
        "replay_fastlane": {
            "enabled": bool(replay_fastlane.get("enabled")),
            "contains": [str(token).strip() for token in list(replay_fastlane.get("contains", []) or []) if str(token).strip()][:8],
            "replay_ready_count": max(0, parse_int(replay_fastlane.get("replay_ready_count"), 0)),
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
        "preferred_contains": [str(token).strip() for token in preferred_contains if str(token).strip()][:12],
        "cooldown_contains": [str(token).strip() for token in cooldown_contains if str(token).strip()][:12],
        "preferred_operator_ids": [str(token).strip() for token in preferred_operator_ids if str(token).strip()][:8],
        "cooldown_operator_ids": [str(token).strip() for token in cooldown_operator_ids if str(token).strip()][:8],
        "sample_size": {str(k): max(0, parse_int(v, 0)) for k, v in dict(sample_size).items()},
    }


def build_yield_governor_state(
    *,
    root: Path,
    aggregate_dir: Path,
    run_index_path: Path,
    fullspan_state_path: Path,
    recent_queue_limit: int = 200,
    hard_block_active: bool = False,
    hard_block_reason: str = "",
    existing_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_index_map: dict[str, dict[str, str]] = {}
    if run_index_path.exists():
        try:
            with run_index_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    run_id = str(row.get("run_id") or "").strip()
                    if run_id:
                        run_index_map[run_id] = {k: (v or "").strip() for k, v in row.items()}
        except Exception:
            run_index_map = {}

    fullspan_state = load_json(fullspan_state_path, {})
    queues_state = fullspan_state.get("queues", {}) if isinstance(fullspan_state, dict) else {}
    if not isinstance(queues_state, dict):
        queues_state = {}
    quarantine_by_group = _load_recent_quarantine_by_group(aggregate_dir / ".autonomous" / "deterministic_quarantine.json")

    queue_paths = sorted(
        (path for path in aggregate_dir.glob("*/run_queue.csv") if not path.parent.name.startswith(".")),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )[: max(1, int(recent_queue_limit))]

    lineage_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "completed": 0,
            "metrics_present": 0,
            "zero_activity": 0,
            "hard_fail": 0,
            "tokens": set(),
            "positive_evidence_count": 0,
            "zero_reason_counts": defaultdict(int),
        }
    )
    operator_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"completed": 0, "metrics_present": 0, "zero_activity": 0, "hard_fail": 0}
    )

    completed_rows_total = 0
    for queue_path in queue_paths:
        queue_group = queue_path.parent.name
        try:
            rows = list(csv.DictReader(queue_path.open("r", encoding="utf-8", newline="")))
        except Exception:
            continue
        for row in rows:
            if str(row.get("status") or "").strip().lower() != "completed":
                continue
            results_dir = str(row.get("results_dir") or "").strip()
            run_id = Path(results_dir).name.strip() if results_dir else ""
            run_index_row = run_index_map.get(run_id)
            lineage_uid = _row_lineage_uid(row, queue_group)
            operator_id = _row_operator_id(row)
            token = _row_run_group_token(queue_group, run_index_row)
            metrics_present = int(parse_bool((run_index_row or {}).get("metrics_present")))
            zero_activity = 0
            if run_index_row:
                trades = parse_float(run_index_row.get("total_trades"), 0.0)
                pairs = parse_float(run_index_row.get("total_pairs_traded"), 0.0)
                zero_activity = int((not metrics_present) or trades <= 0.0 or pairs <= 0.0)
            else:
                zero_activity = 1
            hard_fail = int(bool(canonical_hard_fail_reason(run_index_row)))

            lineage_entry = lineage_stats[lineage_uid]
            lineage_entry["completed"] += 1
            lineage_entry["metrics_present"] += metrics_present
            lineage_entry["zero_activity"] += zero_activity
            lineage_entry["hard_fail"] += hard_fail
            lineage_entry["tokens"].add(token)
            if run_index_row and has_positive_coverage_trade_evidence(run_index_row):
                lineage_entry["positive_evidence_count"] += 1
            else:
                reason = canonical_zero_evidence_reason(run_index_row, require_metrics_present=False)
                if reason != "METRICS_PRESENT":
                    lineage_entry["zero_reason_counts"][reason] += 1

            operator_entry = operator_stats[operator_id]
            operator_entry["completed"] += 1
            operator_entry["metrics_present"] += metrics_present
            operator_entry["zero_activity"] += zero_activity
            operator_entry["hard_fail"] += hard_fail

            completed_rows_total += 1

    lineages: list[dict[str, Any]] = []
    for lineage_uid, stats in lineage_stats.items():
        status, yield_score = _classify_yield(
            completed=int(stats["completed"]),
            metrics_present=int(stats["metrics_present"]),
            zero_activity=int(stats["zero_activity"]),
            hard_fail=int(stats["hard_fail"]),
        )
        positive_evidence_count = max(0, parse_int(stats.get("positive_evidence_count"), 0))
        zero_reason_counts = {
            str(code).strip(): max(0, parse_int(count, 0))
            for code, count in dict(stats.get("zero_reason_counts") or {}).items()
            if str(code).strip() and parse_int(count, 0) > 0
        }
        dominant_zero_reason = ""
        for reason in (
            "ZERO_OBSERVED_TEST_DAYS",
            "ZERO_COVERAGE",
            "ZERO_TRADES",
            "ZERO_PAIRS",
            "METRICS_MISSING",
        ):
            if zero_reason_counts.get(reason, 0) > 0:
                dominant_zero_reason = reason
                break
        zero_evidence = bool(int(stats["completed"]) > 0 and positive_evidence_count <= 0)
        lineages.append(
            {
                "lineage_uid": lineage_uid,
                "status": status,
                "yield_score": yield_score,
                "completed": int(stats["completed"]),
                "metrics_present": int(stats["metrics_present"]),
                "zero_activity": int(stats["zero_activity"]),
                "hard_fail": int(stats["hard_fail"]),
                "tokens": sorted(str(token) for token in stats["tokens"] if str(token).strip()),
                "positive_evidence_count": int(positive_evidence_count),
                "has_positive_evidence": bool(positive_evidence_count > 0),
                "zero_evidence": bool(zero_evidence),
                "dominant_zero_reason": dominant_zero_reason,
                "zero_reason_counts": zero_reason_counts,
            }
        )
    lineages.sort(key=lambda item: (float(item["yield_score"]), int(item["completed"])), reverse=True)

    operators: list[dict[str, Any]] = []
    for operator_id, stats in operator_stats.items():
        status, yield_score = _classify_yield(
            completed=int(stats["completed"]),
            metrics_present=int(stats["metrics_present"]),
            zero_activity=int(stats["zero_activity"]),
            hard_fail=int(stats["hard_fail"]),
        )
        operators.append(
            {
                "operator_id": operator_id,
                "status": status,
                "yield_score": yield_score,
                "completed": int(stats["completed"]),
                "metrics_present": int(stats["metrics_present"]),
                "zero_activity": int(stats["zero_activity"]),
                "hard_fail": int(stats["hard_fail"]),
            }
        )
    operators.sort(key=lambda item: (float(item["yield_score"]), int(item["completed"])), reverse=True)

    winner_tokens: list[str] = []
    for queue_rel, entry in queues_state.items():
        if not isinstance(entry, Mapping):
            continue
        verdict = str(entry.get("promotion_verdict") or "").strip().upper()
        strict_pass = int(parse_float(entry.get("strict_pass_count"), 0.0))
        if verdict not in {"PROMOTE_PENDING_CONFIRM", "PROMOTE_DEFER_CONFIRM", "PROMOTE_ELIGIBLE"} and strict_pass <= 0:
            continue
        token = str(entry.get("top_run_group") or "").strip()
        if not token:
            token = Path(str(queue_rel)).parent.name
        if token and token not in winner_tokens:
            winner_tokens.append(token)
        if len(winner_tokens) >= 6:
            break

    preferred_contains: list[str] = list(winner_tokens)
    if len(preferred_contains) < 6:
        for entry in lineages:
            if str(entry.get("status")) != "boost":
                continue
            for token in list(entry.get("tokens") or []):
                if token not in preferred_contains:
                    preferred_contains.append(token)
                if len(preferred_contains) >= 6:
                    break
            if len(preferred_contains) >= 6:
                break

    cooldown_contains: list[str] = []
    for entry in lineages:
        if str(entry.get("status")) != "cooldown":
            continue
        for token in list(entry.get("tokens") or []):
            if token not in cooldown_contains:
                cooldown_contains.append(token)
            if len(cooldown_contains) >= 12:
                break
        if len(cooldown_contains) >= 12:
            break

    preferred_operator_ids = [str(entry["operator_id"]) for entry in operators if str(entry.get("status")) == "boost"][:6]
    cooldown_operator_ids = [str(entry["operator_id"]) for entry in operators if str(entry.get("status")) == "cooldown"][:6]
    replay_fastlane = _collect_replay_fastlane(queues_state)
    lane_weights = _normalize_lane_weights()
    sample_size = {
        "queue_files": len(queue_paths),
        "completed_rows": completed_rows_total,
        "lineages": len(lineages),
        "operators": len(operators),
    }

    preferred_contains_set = set(preferred_contains)
    winner_proximate_positive_lineage_count = sum(
        1
        for entry in lineages
        if bool(entry.get("has_positive_evidence"))
        and any(token in preferred_contains_set for token in list(entry.get("tokens") or []))
    )
    winner_proximate_positive_contains: list[str] = []
    for token in preferred_contains:
        token_text = str(token or "").strip()
        if not token_text:
            continue
        if any(
            bool(entry.get("has_positive_evidence")) and token_text in list(entry.get("tokens") or [])
            for entry in lineages
        ):
            winner_proximate_positive_contains.append(token_text)
        if len(winner_proximate_positive_contains) >= 8:
            break
    if winner_tokens and winner_proximate_positive_lineage_count <= 0:
        winner_proximate_positive_lineage_count = 1
    controlled_recovery_contains = _filter_controlled_recovery_contains(
        winner_proximate_positive_contains,
        quarantine_by_group=quarantine_by_group,
    )
    search_quality = build_search_quality_state(
        positive_lineage_count=sum(1 for entry in lineages if bool(entry.get("has_positive_evidence"))),
        zero_evidence_lineage_count=sum(1 for entry in lineages if bool(entry.get("zero_evidence"))),
        winner_proximate_positive_lineage_count=winner_proximate_positive_lineage_count,
        winner_proximate_positive_contains=winner_proximate_positive_contains,
        controlled_recovery_contains=controlled_recovery_contains,
        hard_block_active=bool(hard_block_active),
        hard_block_reason=str(hard_block_reason or ""),
        existing_state=existing_state,
    )
    planner_policy_inputs = _build_planner_policy_inputs(
        lane_weights=lane_weights,
        preferred_contains=preferred_contains,
        cooldown_contains=cooldown_contains,
        preferred_operator_ids=preferred_operator_ids,
        cooldown_operator_ids=cooldown_operator_ids,
        winner_proximate={
            "enabled": bool(preferred_contains),
            "contains": preferred_contains,
        },
        replay_fastlane=replay_fastlane,
        search_quality=search_quality,
        sample_size=sample_size,
    )
    policy_hash = _stable_policy_hash(planner_policy_inputs)
    search_quality = dict(planner_policy_inputs.get("search_quality") or search_quality)
    controlled_recovery = build_controlled_recovery_state(
        hard_block_active=bool(hard_block_active),
        hard_block_reason=str(hard_block_reason or ""),
        winner_proximate_positive_contains=winner_proximate_positive_contains,
        controlled_recovery_contains=controlled_recovery_contains,
        existing_state=existing_state,
    )

    micro_caps = micro_broad_search_caps()
    return {
        "ts": utc_now_iso(),
        "schema_version": "v2",
        "active": True,
        "sample_size": sample_size,
        "preferred_contains": preferred_contains,
        "cooldown_contains": cooldown_contains,
        "preferred_operator_ids": preferred_operator_ids,
        "cooldown_operator_ids": cooldown_operator_ids,
        "winner_proximate": {
            "enabled": bool(preferred_contains),
            "contains": preferred_contains,
            "reason": "strict_pass_or_high_yield_lineage",
        },
        "hard_block_active": bool(hard_block_active),
        "hard_block_reason": str(hard_block_reason or ""),
        "hard_block_until_epoch": int(parse_int(existing_state.get("hard_block_until_epoch"), 0)),
        "zero_coverage_seed_streak": int(parse_int(existing_state.get("zero_coverage_seed_streak"), 0)),
        "zero_coverage_seed_streak_reason": str(existing_state.get("zero_coverage_seed_streak_reason") or ""),
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
        "controlled_recovery_active": bool(controlled_recovery.get("controlled_recovery_active")),
        "controlled_recovery_reason": str(controlled_recovery.get("controlled_recovery_reason") or ""),
        "controlled_recovery_attempts_remaining": int(
            controlled_recovery.get("controlled_recovery_attempts_remaining", 0) or 0
        ),
        "controlled_recovery_variants_cap": int(controlled_recovery.get("controlled_recovery_variants_cap", 0) or 0),
        "lane_weights": lane_weights,
        "policy_overrides": {
            "num_variants_cap": int(micro_caps["num_variants_cap"]),
            "policy_scale": str(micro_caps["policy_scale"]),
        },
        "lineages": lineages[:48],
        "operators": operators[:24],
        "run_index_path": safe_rel(run_index_path, root),
        "fullspan_state_path": safe_rel(fullspan_state_path, root),
        "planner-policy-inputs": planner_policy_inputs,
        "policy-hash": policy_hash,
        "planner_policy_inputs": planner_policy_inputs,
        "policy_hash": policy_hash,
    }


def rearm_controlled_recovery_if_eligible(
    *,
    state_path: Path,
    process_slo_state_path: Path,
    attempts: int = CONTROLLED_RECOVERY_MAX_BATCHES,
    variants_cap: int = CONTROLLED_RECOVERY_VARIANTS_CAP,
) -> dict[str, Any]:
    payload = load_json(state_path, {})
    if not isinstance(payload, dict):
        payload = {}
    process_slo = load_json(process_slo_state_path, {})
    if not isinstance(process_slo, dict):
        process_slo = {}
    queue = process_slo.get("queue", {})
    if not isinstance(queue, dict):
        queue = {}
    search_quality = process_slo.get("search_quality", {})
    if not isinstance(search_quality, dict):
        search_quality = {}

    winner_tokens = _normalize_tokens(
        search_quality.get("controlled_recovery_contains")
        or payload.get("controlled_recovery_contains")
        or search_quality.get("winner_proximate_positive_contains")
        or payload.get("winner_proximate_positive_contains")
        or [],
    )
    dispatchable_pending = parse_int(queue.get("dispatchable_pending"), 0)
    candidate_pool_status = str(queue.get("candidate_pool_status") or "").strip().lower()
    attempts_remaining = parse_int(payload.get("controlled_recovery_attempts_remaining"), 0)
    hard_block_active = parse_bool(payload.get("hard_block_active"))
    hard_block_reason = str(payload.get("hard_block_reason") or "").strip()

    eligible = controlled_recovery_rearm_eligible(
        hard_block_active=hard_block_active,
        hard_block_reason=hard_block_reason,
        winner_proximate_positive_contains=winner_tokens,
        controlled_recovery_contains=winner_tokens,
        attempts_remaining=attempts_remaining,
        dispatchable_pending=dispatchable_pending,
        candidate_pool_status=candidate_pool_status,
    )
    result = {
        "eligible": bool(eligible),
        "updated": False,
        "dispatchable_pending": int(dispatchable_pending),
        "candidate_pool_status": candidate_pool_status,
        "winner_proximate_positive_contains": list(winner_tokens),
        "controlled_recovery_contains": list(winner_tokens),
        "attempts_remaining_before": int(attempts_remaining),
    }
    if not eligible:
        result["reason"] = "rearm_ineligible"
        return result

    attempts_target = max(1, int(attempts))
    variants_target = max(1, int(variants_cap))
    payload.update(
        {
            "controlled_recovery_active": True,
            "controlled_recovery_reason": CONTROLLED_RECOVERY_REASON,
            "controlled_recovery_attempts_remaining": attempts_target,
            "controlled_recovery_variants_cap": variants_target,
            "winner_proximate_positive_contains": list(winner_tokens),
            "controlled_recovery_contains": list(winner_tokens),
            "ts": utc_now_iso(),
        }
    )
    nested_search_quality = payload.get("search_quality", {})
    if not isinstance(nested_search_quality, dict):
        nested_search_quality = {}
    nested_search_quality.update(
        {
            "winner_proximate_positive_contains": list(winner_tokens),
            "controlled_recovery_contains": list(winner_tokens),
            "controlled_recovery_active": True,
            "controlled_recovery_reason": CONTROLLED_RECOVERY_REASON,
            "controlled_recovery_attempts_remaining": attempts_target,
            "controlled_recovery_variants_cap": variants_target,
        }
    )
    payload["search_quality"] = nested_search_quality
    dump_json(state_path, payload)
    result.update(
        {
            "updated": True,
            "reason": "controlled_recovery_rearmed",
            "attempts_remaining_after": attempts_target,
            "variants_cap": variants_target,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-safe yield governor state for autonomous queue seeding.")
    parser.add_argument("--root", default="", help="App root (`coint4/`).")
    parser.add_argument("--aggregate-dir", default="artifacts/wfa/aggregate")
    parser.add_argument("--run-index", default="artifacts/wfa/aggregate/rollup/run_index.csv")
    parser.add_argument("--fullspan-state", default="artifacts/wfa/aggregate/.autonomous/fullspan_decision_state.json")
    parser.add_argument("--output", default="artifacts/wfa/aggregate/.autonomous/yield_governor_state.json")
    parser.add_argument(
        "--process-slo-state",
        default="artifacts/wfa/aggregate/.autonomous/process_slo_state.json",
    )
    parser.add_argument("--recent-queue-limit", type=int, default=200)
    parser.add_argument("--rearm-controlled-recovery", action="store_true")
    parser.add_argument("--rearm-attempts", type=int, default=CONTROLLED_RECOVERY_MAX_BATCHES)
    parser.add_argument("--rearm-variants-cap", type=int, default=CONTROLLED_RECOVERY_VARIANTS_CAP)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if str(args.root or "").strip() else Path(__file__).resolve().parents[2]
    aggregate_dir = Path(args.aggregate_dir) if Path(args.aggregate_dir).is_absolute() else root / str(args.aggregate_dir)
    run_index_path = Path(args.run_index) if Path(args.run_index).is_absolute() else root / str(args.run_index)
    fullspan_state_path = Path(args.fullspan_state) if Path(args.fullspan_state).is_absolute() else root / str(args.fullspan_state)
    output_path = Path(args.output) if Path(args.output).is_absolute() else root / str(args.output)
    process_slo_state_path = (
        Path(args.process_slo_state) if Path(args.process_slo_state).is_absolute() else root / str(args.process_slo_state)
    )
    existing_state = load_json(output_path, {})
    if not isinstance(existing_state, dict):
        existing_state = {}

    payload = build_yield_governor_state(
        root=root,
        aggregate_dir=aggregate_dir,
        run_index_path=run_index_path,
        fullspan_state_path=fullspan_state_path,
        recent_queue_limit=int(args.recent_queue_limit),
        hard_block_active=parse_bool(existing_state.get("hard_block_active")),
        hard_block_reason=str(existing_state.get("hard_block_reason") or ""),
        existing_state=existing_state,
    )
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        dump_json(output_path, payload)
    if args.rearm_controlled_recovery:
        result = rearm_controlled_recovery_if_eligible(
            state_path=output_path,
            process_slo_state_path=process_slo_state_path,
            attempts=max(1, int(args.rearm_attempts)),
            variants_cap=max(1, int(args.rearm_variants_cap)),
        )
        if args.dry_run:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
