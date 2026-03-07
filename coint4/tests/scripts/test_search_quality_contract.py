from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "_search_quality_contract.py"
SPEC = importlib.util.spec_from_file_location("_search_quality_contract", SCRIPT_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_micro_broad_search_caps_are_locked() -> None:
    caps = module.micro_broad_search_caps()

    assert caps == {
        "max_changed_keys_cap": 3,
        "dedupe_distance_floor": 0.04,
        "num_variants_cap": 48,
        "policy_scale": "micro",
    }


def test_canonical_zero_evidence_reason_prioritizes_observed_coverage_trades_pairs() -> None:
    assert module.canonical_zero_evidence_reason({"metrics_present": "true", "observed_test_days": "0"}) == "ZERO_OBSERVED_TEST_DAYS"
    assert module.canonical_zero_evidence_reason({"metrics_present": "true", "coverage_ratio": "0"}) == "ZERO_COVERAGE"
    assert module.canonical_zero_evidence_reason({"metrics_present": "true", "total_trades": "0"}) == "ZERO_TRADES"
    assert module.canonical_zero_evidence_reason({"metrics_present": "true", "total_pairs_traded": "0"}) == "ZERO_PAIRS"


def test_positive_coverage_trade_evidence_requires_all_positive_fields() -> None:
    row = {
        "metrics_present": "true",
        "observed_test_days": "75",
        "coverage_ratio": "1.0",
        "total_trades": "12",
        "total_pairs_traded": "4",
    }

    assert module.has_positive_coverage_trade_evidence(row) is True
    assert module.has_positive_coverage_trade_evidence({**row, "coverage_ratio": "0"}) is False
    assert module.has_positive_coverage_trade_evidence({**row, "total_pairs_traded": ""}) is False


def test_summarize_recent_zero_evidence_prefers_canonical_reason_when_no_positive_rows() -> None:
    rows = [
        {"metrics_present": "true", "observed_test_days": "0", "coverage_ratio": "0", "total_trades": "", "total_pairs_traded": ""},
        {"metrics_present": "true", "coverage_ratio": "0", "total_trades": "0", "total_pairs_traded": "0"},
    ]

    summary = module.summarize_recent_zero_evidence(rows)

    assert summary["has_positive_coverage_trade_evidence"] is False
    assert summary["dominant_zero_reason"] == "ZERO_OBSERVED_TEST_DAYS"


def test_build_search_quality_state_blocks_broad_search_when_winner_positive_exists() -> None:
    payload = module.build_search_quality_state(
        positive_lineage_count=3,
        zero_evidence_lineage_count=5,
        winner_proximate_positive_lineage_count=2,
    )

    assert payload["positive_lineage_count"] == 3
    assert payload["zero_evidence_lineage_count"] == 5
    assert payload["winner_proximate_positive_lineage_count"] == 2
    assert payload["broad_search_allowed"] is False
    assert payload["seed_generation_mode"] == "winner_proximate_only"


def test_normalize_search_quality_state_infers_winner_positive_from_counts_and_tokens() -> None:
    payload = module.normalize_search_quality_state(
        {
            "positive_lineage_count": 1,
            "zero_evidence_lineage_count": 4,
        },
        winner_proximate_contains=["strict_rg"],
    )

    assert payload["positive_lineage_count"] == 1
    assert payload["zero_evidence_lineage_count"] == 4
    assert payload["winner_proximate_positive_lineage_count"] == 1
    assert payload["broad_search_allowed"] is False
    assert payload["seed_generation_mode"] == "winner_proximate_only"


def test_build_search_quality_state_enables_controlled_recovery_for_positive_zero_coverage_block() -> None:
    payload = module.build_search_quality_state(
        positive_lineage_count=3,
        zero_evidence_lineage_count=5,
        winner_proximate_positive_lineage_count=1,
        winner_proximate_positive_contains=["strict_rg"],
        hard_block_active=True,
        hard_block_reason="zero_coverage_seed_streak",
    )

    assert payload["winner_proximate_positive_contains"] == ["strict_rg"]
    assert payload["controlled_recovery_contains"] == ["strict_rg"]
    assert payload["controlled_recovery_active"] is True
    assert payload["controlled_recovery_reason"] == "zero_coverage_seed_streak_with_positive_lineage"
    assert payload["controlled_recovery_attempts_remaining"] == 2
    assert payload["controlled_recovery_variants_cap"] == 8


def test_normalize_search_quality_state_does_not_activate_controlled_recovery_without_explicit_positive_tokens() -> None:
    payload = module.normalize_search_quality_state(
        {
            "positive_lineage_count": 1,
            "zero_evidence_lineage_count": 4,
            "winner_proximate_positive_lineage_count": 1,
            "hard_block_active": True,
            "hard_block_reason": "zero_coverage_seed_streak",
        },
        winner_proximate_contains=["strict_rg"],
    )

    assert payload["winner_proximate_positive_lineage_count"] == 1
    assert payload["winner_proximate_positive_contains"] == []
    assert payload["controlled_recovery_contains"] == []
    assert payload["controlled_recovery_active"] is False
    assert payload["controlled_recovery_attempts_remaining"] == 0


def test_build_search_quality_state_uses_explicit_controlled_recovery_subset() -> None:
    payload = module.build_search_quality_state(
        positive_lineage_count=2,
        zero_evidence_lineage_count=3,
        winner_proximate_positive_lineage_count=2,
        winner_proximate_positive_contains=["strict_rg", "poisoned_rg"],
        controlled_recovery_contains=["strict_rg"],
        hard_block_active=True,
        hard_block_reason="zero_coverage_seed_streak",
    )

    assert payload["winner_proximate_positive_contains"] == ["strict_rg", "poisoned_rg"]
    assert payload["controlled_recovery_contains"] == ["strict_rg"]
    assert payload["controlled_recovery_active"] is True
