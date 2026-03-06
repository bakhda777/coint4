from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "search_director_agent.py"
SPEC = importlib.util.spec_from_file_location("search_director_agent", SCRIPT_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_build_directive_prefers_winner_proximate_and_yield_tokens() -> None:
    queues = {
        "artifacts/wfa/aggregate/autonomous_seed_a/run_queue.csv": {
            "promotion_verdict": "PROMOTE_PENDING_CONFIRM",
            "strict_pass_count": 1,
            "top_run_group": "strict_rg",
            "rejection_reason": "",
            "strict_gate_reason": "",
        },
        "artifacts/wfa/aggregate/autonomous_seed_b/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "METRICS_MISSING",
        },
    }
    yield_state = {
        "active": True,
        "preferred_contains": ["yield_rg"],
        "winner_proximate": {"enabled": True, "contains": ["yield_strict_rg"], "reason": "strict_pass_or_high_yield_lineage"},
        "lane_weights": {"winner_proximate": 65, "broad_search": 15, "confirm_replay": 20},
        "cooldown_contains": ["cold_rg"],
        "preferred_operator_ids": ["op_good"],
        "cooldown_operator_ids": ["op_bad"],
    }

    directive = module.build_directive(queues, yield_state=yield_state)

    assert directive["contains"][:3] == ["strict_rg", "yield_strict_rg", "yield_rg"]
    assert directive["winner_proximate"]["enabled"] is True
    assert directive["replay_fastlane"]["enabled"] is True
    assert directive["yield_governor"]["active"] is True
    assert directive["lane_weights"] == {
        "winner_proximate": 65,
        "confirm_replay": 20,
        "broad_search": 15,
    }
    assert len(str(directive["policy-hash"])) == 64
    assert directive["policy-hash"] == directive["policy_hash"]
    assert directive["planner-policy-inputs"]["lane_weights"]["winner_proximate"] == 65
    assert directive["planner-policy-inputs"] == directive["planner_policy_inputs"]


def test_build_directive_disables_broad_search_when_positive_winner_lineage_exists() -> None:
    queues = {
        "artifacts/wfa/aggregate/autonomous_seed_a/run_queue.csv": {
            "promotion_verdict": "PROMOTE_PENDING_CONFIRM",
            "strict_pass_count": 1,
            "top_run_group": "strict_rg",
            "rejection_reason": "",
            "strict_gate_reason": "",
        }
    }
    yield_state = {
        "active": True,
        "preferred_contains": ["strict_rg"],
        "winner_proximate": {"enabled": True, "contains": ["strict_rg"], "reason": "strict_pass_or_high_yield_lineage"},
        "search_quality": {
            "positive_lineage_count": 2,
            "zero_evidence_lineage_count": 3,
            "winner_proximate_positive_lineage_count": 1,
            "winner_proximate_positive_contains": ["strict_rg"],
            "broad_search_allowed": False,
            "seed_generation_mode": "winner_proximate_only",
            "controlled_recovery_active": True,
            "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
            "controlled_recovery_attempts_remaining": 2,
            "controlled_recovery_variants_cap": 8,
        },
        "lane_weights": {"winner_proximate": 65, "broad_search": 15, "confirm_replay": 20},
        "hard_block_active": True,
        "hard_block_reason": "zero_coverage_seed_streak",
    }

    directive = module.build_directive(queues, yield_state=yield_state)

    assert directive["search_quality"]["positive_lineage_count"] == 2
    assert directive["search_quality"]["winner_proximate_positive_lineage_count"] == 1
    assert directive["search_quality"]["winner_proximate_positive_contains"] == ["strict_rg"]
    assert directive["broad_search_allowed"] is False
    assert directive["seed_generation_mode"] == "winner_proximate_only"
    assert directive["mode"] == "controlled_recovery"
    assert directive["contains"] == ["strict_rg"]
    assert directive["num_variants"] == 8
    assert directive["repair_mode"]["enabled"] is True
    assert directive["lane_weights"]["broad_search"] == 0
    assert directive["lane_weights"]["confirm_replay"] == 0
    assert directive["planner-policy-inputs"]["search_quality"]["broad_search_allowed"] is False
    assert directive["planner-policy-inputs"]["search_quality"]["controlled_recovery_active"] is True


def test_materialize_cold_fail_index_backfills_rejects(tmp_path: Path) -> None:
    cold_path = tmp_path / "cold_fail_index.json"
    queues = {
        "artifacts/wfa/aggregate/q1/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "LOW_YIELD_HOMOGENEOUS_METRICS_MISSING",
            "top_run_group": "rg1",
            "candidate_uid": "cand-1",
        },
        "artifacts/wfa/aggregate/q2/run_queue.csv": {
            "promotion_verdict": "ANALYZE",
            "rejection_reason": "",
        },
    }

    summary = module.materialize_cold_fail_index(queues=queues, path=cold_path, ttl_sec=3600)
    payload = json.loads(cold_path.read_text(encoding="utf-8"))

    assert summary["active_count"] == 1
    assert summary["added"] == 1
    assert payload["entries"][0]["queue"] == "artifacts/wfa/aggregate/q1/run_queue.csv"
    assert payload["entries"][0]["gate_reason"] == "METRICS_MISSING"


def test_build_directive_preserves_zero_coverage_reasons_and_micro_caps() -> None:
    queues = {
        "artifacts/wfa/aggregate/q1/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "ZERO_COVERAGE",
            "strict_gate_reason": "",
        }
    }

    directive = module.build_directive(queues, yield_state={})

    assert directive["dominant_reason"] == "ZERO_COVERAGE"
    assert directive["mode"] == "stability_focus"
    assert directive["max_changed_keys"] == 3
    assert directive["dedupe_distance"] == 0.04
    assert directive["impossibility_pruner"]["reason"] == "ZERO_COVERAGE"
    assert directive["impossibility_pruner"]["max_changed_keys_cap"] == 3
    assert directive["impossibility_pruner"]["dedupe_distance_floor"] == 0.04
    assert directive["impossibility_pruner"]["num_variants_cap"] == 48
    assert directive["impossibility_pruner"]["policy_scale"] == "micro"
    assert directive["search_quality"]["broad_search_allowed"] is True
    assert directive["search_quality"]["seed_generation_mode"] == "broad_search_micro"
