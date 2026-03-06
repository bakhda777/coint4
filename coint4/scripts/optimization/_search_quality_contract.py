from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


MICRO_BROAD_SEARCH_CAPS = {
    "max_changed_keys_cap": 3,
    "dedupe_distance_floor": 0.04,
    "num_variants_cap": 48,
    "policy_scale": "micro",
}

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
