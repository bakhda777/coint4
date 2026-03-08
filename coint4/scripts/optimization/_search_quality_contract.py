from __future__ import annotations

from collections import Counter
import time
from typing import Any, Iterable, Mapping


MICRO_BROAD_SEARCH_CAPS = {
    "max_changed_keys_cap": 3,
    "dedupe_distance_floor": 0.04,
    "num_variants_cap": 48,
    "policy_scale": "micro",
}

CONTROLLED_RECOVERY_REASON = "zero_coverage_seed_streak_with_positive_lineage"
CONTROLLED_RECOVERY_HARD_BLOCK_REASON = "zero_coverage_seed_streak"
CONTROLLED_RECOVERY_VARIANTS_CAP = 8
CONTROLLED_RECOVERY_MAX_BATCHES = 2
CONTROLLED_RECOVERY_EMPTY_POOL_STATUS = "empty_expected"
CONTROLLED_BROAD_REARM_DELAY_SEC = 600

CANONICAL_ZERO_EVIDENCE_REASONS = (
    "ZERO_OBSERVED_TEST_DAYS",
    "ZERO_COVERAGE",
    "ZERO_TRADES",
    "ZERO_PAIRS",
)

DETERMINISTIC_QUARANTINE_CODES = frozenset(
    {
        "MAX_VAR_MULTIPLIER_INVALID",
        "CONFIG_VALIDATION_ERROR",
    }
)


def _to_bool(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def _lookup_metric(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = _to_float(row.get(key), None)
        if value is not None:
            return value
        text = str(row.get(key) if row.get(key) is not None else "").strip()
        if text:
            return None
    return None


def micro_broad_search_caps() -> dict[str, Any]:
    return dict(MICRO_BROAD_SEARCH_CAPS)


def _normalize_tokens(value: Iterable[Any] | Any, *, limit: int = 8) -> list[str]:
    items: list[Any]
    if isinstance(value, str):
        text = str(value).strip()
        if not text:
            return []
        items = [part.strip() for part in text.split(",")] if "," in text else [text]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        return []
    out: list[str] = []
    seen = set()
    for item in items:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _explicit_tokens_or_none(value: Iterable[Any] | Any | None, *, limit: int = 8) -> list[str] | None:
    if value is None:
        return None
    return _normalize_tokens(value, limit=limit)


def build_controlled_recovery_state(
    *,
    hard_block_active: bool,
    hard_block_reason: str,
    winner_proximate_positive_contains: Iterable[Any] | None = None,
    controlled_recovery_contains: Iterable[Any] | None = None,
    existing_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    winner_contains = _normalize_tokens(winner_proximate_positive_contains or [], limit=8)
    existing = existing_state if isinstance(existing_state, Mapping) else {}
    nested = existing.get("search_quality", {}) if isinstance(existing.get("search_quality"), Mapping) else {}
    controlled_contains = _explicit_tokens_or_none(controlled_recovery_contains, limit=8)
    if controlled_contains is None:
        controlled_contains = _explicit_tokens_or_none(nested.get("controlled_recovery_contains"), limit=8)
    if controlled_contains is None:
        controlled_contains = _explicit_tokens_or_none(existing.get("controlled_recovery_contains"), limit=8)
    if controlled_contains is None:
        controlled_contains = list(winner_contains)
    eligible = (
        bool(hard_block_active)
        and str(hard_block_reason or "").strip() == CONTROLLED_RECOVERY_HARD_BLOCK_REASON
        and bool(controlled_contains)
    )
    attempts_raw = existing.get("controlled_recovery_attempts_remaining")
    attempts_key_present = "controlled_recovery_attempts_remaining" in existing
    if not attempts_key_present and isinstance(nested, Mapping):
        attempts_key_present = "controlled_recovery_attempts_remaining" in nested
        attempts_raw = nested.get("controlled_recovery_attempts_remaining")
    attempts_remaining = _to_int(
        attempts_raw,
        CONTROLLED_RECOVERY_MAX_BATCHES,
    )
    if not attempts_key_present:
        attempts_remaining = CONTROLLED_RECOVERY_MAX_BATCHES
    attempts_remaining = max(0, attempts_remaining)
    variants_raw = existing.get("controlled_recovery_variants_cap")
    if "controlled_recovery_variants_cap" not in existing and isinstance(nested, Mapping):
        variants_raw = nested.get("controlled_recovery_variants_cap")
    variants_cap = max(
        1,
        _to_int(variants_raw, CONTROLLED_RECOVERY_VARIANTS_CAP),
    )
    reason_raw = existing.get("controlled_recovery_reason")
    if (reason_raw is None or str(reason_raw).strip() == "") and isinstance(nested, Mapping):
        reason_raw = nested.get("controlled_recovery_reason")
    reason = str(reason_raw or CONTROLLED_RECOVERY_REASON).strip() or CONTROLLED_RECOVERY_REASON
    active = bool(eligible and attempts_remaining > 0)
    nested_rearm_after = 0
    if isinstance(nested, Mapping):
        nested_rearm_after = _to_int(nested.get("controlled_broad_rearm_after_epoch"), 0)
    existing_rearm_after = max(
        0,
        _to_int(existing.get("controlled_broad_rearm_after_epoch"), 0),
        nested_rearm_after,
    )
    controlled_broad_rearm_after_epoch = 0
    if eligible and attempts_remaining <= 0:
        now_epoch = max(0, int(time.time()))
        target_rearm_after = now_epoch + CONTROLLED_BROAD_REARM_DELAY_SEC
        if existing_rearm_after > 0:
            controlled_broad_rearm_after_epoch = int(min(existing_rearm_after, target_rearm_after))
        else:
            controlled_broad_rearm_after_epoch = int(target_rearm_after)
    return {
        "winner_proximate_positive_contains": winner_contains,
        "controlled_recovery_contains": controlled_contains,
        "controlled_recovery_active": bool(active),
        "controlled_recovery_reason": reason if eligible else "",
        "controlled_recovery_attempts_remaining": int(attempts_remaining if eligible else 0),
        "controlled_recovery_variants_cap": int(variants_cap),
        "controlled_broad_rearm_after_epoch": int(controlled_broad_rearm_after_epoch),
    }


def controlled_recovery_backlog_ready(
    *,
    dispatchable_pending: Any,
    candidate_pool_status: Any,
) -> bool:
    return (
        _to_int(dispatchable_pending, 0) <= 0
        and str(candidate_pool_status or "").strip().lower() == CONTROLLED_RECOVERY_EMPTY_POOL_STATUS
    )


def controlled_recovery_rearm_eligible(
    *,
    hard_block_active: bool,
    hard_block_reason: str,
    winner_proximate_positive_contains: Iterable[Any] | None = None,
    controlled_recovery_contains: Iterable[Any] | None = None,
    attempts_remaining: Any = 0,
    dispatchable_pending: Any = 0,
    candidate_pool_status: Any = "",
) -> bool:
    contains = _explicit_tokens_or_none(controlled_recovery_contains, limit=8)
    if contains is None:
        contains = _normalize_tokens(winner_proximate_positive_contains or [], limit=8)
    return bool(
        hard_block_active
        and str(hard_block_reason or "").strip() == CONTROLLED_RECOVERY_HARD_BLOCK_REASON
        and bool(contains)
        and _to_int(attempts_remaining, 0) <= 0
        and controlled_recovery_backlog_ready(
            dispatchable_pending=dispatchable_pending,
            candidate_pool_status=candidate_pool_status,
        )
    )


def canonical_zero_evidence_reason(
    row: Mapping[str, Any] | None,
    *,
    require_metrics_present: bool = False,
) -> str:
    if not row:
        return "METRICS_MISSING"

    metrics_present = _to_bool(row.get("metrics_present"))
    if require_metrics_present and not metrics_present:
        return "METRICS_MISSING"

    observed_test_days = _lookup_metric(row, "observed_test_days")
    if observed_test_days is not None and observed_test_days <= 0.0:
        return "ZERO_OBSERVED_TEST_DAYS"

    coverage_ratio = _lookup_metric(row, "coverage_ratio", "coverage")
    if coverage_ratio is not None and coverage_ratio <= 0.0:
        return "ZERO_COVERAGE"

    total_trades = _lookup_metric(row, "total_trades")
    if total_trades is not None and total_trades <= 0.0:
        return "ZERO_TRADES"

    total_pairs = _lookup_metric(row, "total_pairs_traded", "total_pairs")
    if total_pairs is not None and total_pairs <= 0.0:
        return "ZERO_PAIRS"

    return "METRICS_PRESENT"


def has_positive_coverage_trade_evidence(row: Mapping[str, Any] | None) -> bool:
    if not row:
        return False
    if "metrics_present" in row and not _to_bool(row.get("metrics_present")):
        return False
    observed_test_days = _lookup_metric(row, "observed_test_days")
    coverage_ratio = _lookup_metric(row, "coverage_ratio", "coverage")
    total_trades = _lookup_metric(row, "total_trades")
    total_pairs = _lookup_metric(row, "total_pairs_traded", "total_pairs")
    return bool(
        observed_test_days is not None
        and observed_test_days > 0.0
        and coverage_ratio is not None
        and coverage_ratio > 0.0
        and total_trades is not None
        and total_trades > 0.0
        and total_pairs is not None
        and total_pairs > 0.0
    )


def summarize_recent_zero_evidence(rows: Iterable[Mapping[str, Any]], *, limit: int = 12) -> dict[str, Any]:
    recent = list(rows)[-max(1, int(limit)) :]
    reason_counts: Counter[str] = Counter()
    positive_count = 0
    for row in recent:
        if has_positive_coverage_trade_evidence(row):
            positive_count += 1
        reason = canonical_zero_evidence_reason(row)
        if reason in CANONICAL_ZERO_EVIDENCE_REASONS:
            reason_counts[reason] += 1
    dominant_reason = ""
    for reason in CANONICAL_ZERO_EVIDENCE_REASONS:
        if reason_counts.get(reason, 0) > 0:
            dominant_reason = reason
            break
    if not dominant_reason and recent and positive_count <= 0:
        dominant_reason = "NO_RECENT_POSITIVE_COVERAGE_TRADE_EVIDENCE"
    return {
        "rows_analyzed": len(recent),
        "positive_coverage_trade_count": int(positive_count),
        "has_positive_coverage_trade_evidence": bool(positive_count > 0),
        "dominant_zero_reason": dominant_reason,
        "zero_reason_counts": dict(reason_counts),
    }


def build_search_quality_state(
    *,
    positive_lineage_count: int,
    zero_evidence_lineage_count: int,
    winner_proximate_positive_lineage_count: int = 0,
    winner_proximate_positive_contains: Iterable[Any] | None = None,
    controlled_recovery_contains: Iterable[Any] | None = None,
    broad_search_allowed: bool | None = None,
    seed_generation_mode: str = "",
    hard_block_active: bool = False,
    hard_block_reason: str = "",
    existing_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    positive = max(0, int(positive_lineage_count))
    zero_evidence = max(0, int(zero_evidence_lineage_count))
    winner_positive = max(0, int(winner_proximate_positive_lineage_count))
    winner_positive_contains = _normalize_tokens(winner_proximate_positive_contains or [], limit=8)
    broad_allowed = bool(broad_search_allowed) if broad_search_allowed is not None else winner_positive <= 0
    mode = str(seed_generation_mode or "").strip().lower()
    if mode not in {"winner_proximate_only", "broad_search_micro"}:
        mode = "broad_search_micro" if broad_allowed else "winner_proximate_only"
    controlled_recovery = build_controlled_recovery_state(
        hard_block_active=hard_block_active,
        hard_block_reason=hard_block_reason,
        winner_proximate_positive_contains=winner_positive_contains,
        controlled_recovery_contains=controlled_recovery_contains,
        existing_state=existing_state,
    )
    return {
        "positive_lineage_count": positive,
        "zero_evidence_lineage_count": zero_evidence,
        "winner_proximate_positive_lineage_count": winner_positive,
        "winner_proximate_positive_contains": winner_positive_contains,
        "controlled_recovery_contains": list(controlled_recovery["controlled_recovery_contains"]),
        "broad_search_allowed": bool(broad_allowed),
        "seed_generation_mode": mode,
        "controlled_recovery_active": bool(controlled_recovery["controlled_recovery_active"]),
        "controlled_recovery_reason": str(controlled_recovery["controlled_recovery_reason"]),
        "controlled_recovery_attempts_remaining": int(controlled_recovery["controlled_recovery_attempts_remaining"]),
        "controlled_recovery_variants_cap": int(controlled_recovery["controlled_recovery_variants_cap"]),
        "controlled_broad_rearm_after_epoch": int(controlled_recovery["controlled_broad_rearm_after_epoch"]),
    }


def normalize_search_quality_state(
    payload: Mapping[str, Any] | None,
    *,
    winner_proximate_contains: Iterable[Any] | None = None,
) -> dict[str, Any]:
    source = payload if isinstance(payload, Mapping) else {}
    nested = source.get("search_quality", {}) if isinstance(source.get("search_quality"), Mapping) else {}
    winner_tokens = [
        str(token).strip()
        for token in list(winner_proximate_contains or [])
        if str(token).strip()
    ]
    positive = max(
        0,
        _to_int(
            nested.get(
                "positive_lineage_count",
                source.get("positive_lineage_count", 0),
            ),
            0,
        ),
    )
    zero_evidence = max(
        0,
        _to_int(
            nested.get(
                "zero_evidence_lineage_count",
                source.get("zero_evidence_lineage_count", 0),
            ),
            0,
        ),
    )
    default_winner_positive = 1 if positive > 0 and winner_tokens else 0
    winner_positive = max(
        0,
        _to_int(
            nested.get(
                "winner_proximate_positive_lineage_count",
                source.get("winner_proximate_positive_lineage_count", default_winner_positive),
            ),
            default_winner_positive,
        ),
    )
    winner_positive_contains = _normalize_tokens(
        nested.get(
            "winner_proximate_positive_contains",
            source.get("winner_proximate_positive_contains", []),
        ),
        limit=8,
    )
    controlled_recovery_raw: Any | None = None
    if "controlled_recovery_contains" in nested:
        controlled_recovery_raw = nested.get("controlled_recovery_contains")
    elif "controlled_recovery_contains" in source:
        controlled_recovery_raw = source.get("controlled_recovery_contains")
    controlled_recovery_contains = _explicit_tokens_or_none(controlled_recovery_raw, limit=8)
    broad_raw = nested.get("broad_search_allowed")
    if broad_raw is None:
        broad_raw = source.get("broad_search_allowed")
    broad_allowed = bool(broad_raw) if broad_raw is not None else winner_positive <= 0
    mode = str(nested.get("seed_generation_mode") or source.get("seed_generation_mode") or "").strip().lower()
    return build_search_quality_state(
        positive_lineage_count=positive,
        zero_evidence_lineage_count=zero_evidence,
        winner_proximate_positive_lineage_count=winner_positive,
        winner_proximate_positive_contains=winner_positive_contains,
        controlled_recovery_contains=controlled_recovery_contains,
        broad_search_allowed=broad_allowed,
        seed_generation_mode=mode,
        hard_block_active=_to_bool(source.get("hard_block_active")),
        hard_block_reason=str(source.get("hard_block_reason") or "").strip(),
        existing_state=source,
    )
