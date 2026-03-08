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


def test_existing_yield_state_does_not_force_stale_controlled_recovery_fields(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    state_dir = app_root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    state_dir.mkdir(parents=True, exist_ok=True)
    fullspan_state_path = state_dir / "fullspan_decision_state.json"
    fullspan_state_path.write_text(json.dumps({"queues": {}}, ensure_ascii=False), encoding="utf-8")
    run_index_path = app_root / "artifacts" / "wfa" / "aggregate" / "rollup" / "run_index.csv"
    run_index_path.parent.mkdir(parents=True, exist_ok=True)
    run_index_path.write_text(
        "run_id,run_group,metrics_present,observed_test_days,coverage_ratio,total_trades,total_pairs_traded\n",
        encoding="utf-8",
    )
    gate_state_path = state_dir / "gate_surrogate_state.json"
    gate_state_path.write_text(json.dumps({"queues": {}}, ensure_ascii=False), encoding="utf-8")
    yield_state_path = state_dir / "yield_governor_state.json"
    yield_state_path.write_text(
        json.dumps(
            {
                "hard_block_active": False,
                "hard_block_reason": "",
                "controlled_recovery_active": True,
                "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
                "controlled_recovery_attempts_remaining": 2,
                "controlled_recovery_variants_cap": 8,
                "winner_proximate_positive_contains": ["stale_rg"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = state_dir / "search_director_directive.json"

    import sys as _sys

    prev_argv = list(_sys.argv)
    try:
        _sys.argv = ["search_director_agent.py", "--root", str(app_root)]
        assert module.main() == 0
    finally:
        _sys.argv = prev_argv

    directive = json.loads(output_path.read_text(encoding="utf-8"))
    assert directive["search_quality"]["controlled_recovery_active"] is False
    assert directive["search_quality"]["controlled_recovery_attempts_remaining"] == 0


def test_build_directive_activates_controlled_broad_recovery_after_stagnation() -> None:
    queues = {
        "artifacts/wfa/aggregate/autonomous_seed_q/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "METRICS_MISSING",
        }
    }
    yield_state = {
        "active": True,
        "preferred_contains": ["broad_anchor"],
        "search_quality": {
            "positive_lineage_count": 3,
            "zero_evidence_lineage_count": 9,
            "winner_proximate_positive_lineage_count": 0,
            "winner_proximate_positive_contains": ["positive_anchor"],
            "controlled_recovery_contains": [],
            "broad_search_allowed": False,
            "seed_generation_mode": "winner_proximate_only",
            "controlled_recovery_active": False,
            "controlled_recovery_reason": "",
            "controlled_recovery_attempts_remaining": 0,
            "controlled_recovery_variants_cap": 8,
        },
        "lane_weights": {"winner_proximate": 65, "broad_search": 15, "confirm_replay": 20},
    }
    runtime_metrics = {
        "candidate_pool_status": "empty_expected",
        "ready_buffer_depth": 0,
        "global_pending_dispatchable": 0,
        "remote_active_queue_jobs": 0,
        "completed_with_metrics_stagnant_sec": 2400,
        "no_progress_breaker_streak": 2,
        "last_seed_trigger_epoch": int(module.datetime.now(module.timezone.utc).timestamp()) - 7200,
    }

    directive = module.build_directive(queues, yield_state=yield_state, runtime_metrics=runtime_metrics)

    assert directive["mode"] == "controlled_broad_recovery"
    assert directive["broad_search_allowed"] is True
    assert directive["seed_generation_mode"] == "controlled_broad_micro"
    assert directive["controlled_broad_active"] is True
    assert directive["lane_weights"] == {
        "winner_proximate": 0,
        "confirm_replay": 0,
        "broad_search": 100,
    }
    assert directive["num_variants"] == 24
    assert directive["num_variants_floor"] == 16
    assert directive["max_changed_keys"] == 3
    assert directive["dedupe_distance"] == 0.04
    assert directive["search_quality"]["controlled_broad_active"] is True
    assert directive["search_quality"]["controlled_broad_contains"] == ["positive_anchor", "broad_anchor"]
    assert directive["planner-policy-inputs"]["controlled_broad_recovery"]["enabled"] is True


def test_build_directive_does_not_activate_controlled_broad_with_pending_backlog() -> None:
    queues = {
        "artifacts/wfa/aggregate/autonomous_seed_q/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "METRICS_MISSING",
        }
    }
    yield_state = {
        "active": True,
        "preferred_contains": ["broad_anchor"],
        "search_quality": {
            "positive_lineage_count": 3,
            "zero_evidence_lineage_count": 9,
            "winner_proximate_positive_lineage_count": 0,
            "winner_proximate_positive_contains": ["positive_anchor"],
            "broad_search_allowed": False,
            "seed_generation_mode": "winner_proximate_only",
            "controlled_recovery_active": False,
        },
    }
    runtime_metrics = {
        "candidate_pool_status": "empty_expected",
        "ready_buffer_depth": 0,
        "global_pending_dispatchable": 5,
        "remote_active_queue_jobs": 0,
        "completed_with_metrics_stagnant_sec": 2400,
        "no_progress_breaker_streak": 2,
        "last_seed_trigger_epoch": int(module.datetime.now(module.timezone.utc).timestamp()) - 7200,
    }

    directive = module.build_directive(queues, yield_state=yield_state, runtime_metrics=runtime_metrics)

    assert directive["mode"] != "controlled_broad_recovery"
    assert directive["controlled_broad_active"] is False
    assert directive["search_quality"].get("controlled_broad_active", False) is False
    assert directive["planner-policy-inputs"]["controlled_broad_recovery"]["enabled"] is False


def test_build_directive_does_not_activate_controlled_broad_inside_cooldown() -> None:
    queues = {
        "artifacts/wfa/aggregate/autonomous_seed_q/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "METRICS_MISSING",
        }
    }
    yield_state = {
        "active": True,
        "preferred_contains": ["broad_anchor"],
        "search_quality": {
            "positive_lineage_count": 3,
            "zero_evidence_lineage_count": 9,
            "winner_proximate_positive_lineage_count": 0,
            "winner_proximate_positive_contains": ["positive_anchor"],
            "broad_search_allowed": False,
            "seed_generation_mode": "winner_proximate_only",
            "controlled_recovery_active": False,
        },
    }
    runtime_metrics = {
        "candidate_pool_status": "empty_expected",
        "ready_buffer_depth": 0,
        "global_pending_dispatchable": 0,
        "remote_active_queue_jobs": 0,
        "completed_with_metrics_stagnant_sec": 2400,
        "no_progress_breaker_streak": 2,
        "last_seed_trigger_epoch": int(module.datetime.now(module.timezone.utc).timestamp()) - 300,
    }

    directive = module.build_directive(queues, yield_state=yield_state, runtime_metrics=runtime_metrics)

    assert directive["mode"] != "controlled_broad_recovery"
    assert directive["controlled_broad_active"] is False
    assert directive["search_quality"].get("controlled_broad_active", False) is False
    assert directive["planner-policy-inputs"]["controlled_broad_recovery"]["enabled"] is False


def test_build_directive_activates_controlled_broad_when_rearm_epoch_is_due() -> None:
    now_epoch = int(module.datetime.now(module.timezone.utc).timestamp())
    queues = {
        "artifacts/wfa/aggregate/autonomous_seed_q/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "METRICS_MISSING",
        }
    }
    yield_state = {
        "active": True,
        "preferred_contains": ["broad_anchor"],
        "hard_block_active": True,
        "hard_block_reason": "zero_coverage_seed_streak",
        "search_quality": {
            "positive_lineage_count": 3,
            "zero_evidence_lineage_count": 9,
            "winner_proximate_positive_lineage_count": 1,
            "winner_proximate_positive_contains": ["positive_anchor"],
            "controlled_recovery_contains": ["positive_anchor"],
            "broad_search_allowed": False,
            "seed_generation_mode": "winner_proximate_only",
            "controlled_recovery_active": False,
            "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
            "controlled_recovery_attempts_remaining": 0,
            "controlled_recovery_variants_cap": 8,
            "controlled_broad_rearm_after_epoch": now_epoch - 1,
        },
        "lane_weights": {"winner_proximate": 65, "broad_search": 15, "confirm_replay": 20},
    }
    runtime_metrics = {
        "candidate_pool_status": "empty_expected",
        "ready_buffer_depth": 0,
        "global_pending_dispatchable": 0,
        "remote_active_queue_jobs": 0,
        "completed_with_metrics_stagnant_sec": 0,
        "no_progress_breaker_streak": 0,
        "last_seed_trigger_epoch": now_epoch - 300,
    }

    directive = module.build_directive(queues, yield_state=yield_state, runtime_metrics=runtime_metrics)

    assert directive["mode"] == "controlled_broad_recovery"
    assert directive["controlled_broad_active"] is True
    assert directive["controlled_broad_reason"] == "controlled_broad_rearm_after_exhausted_recovery"
    assert directive["controlled_broad_cooldown_sec"] == 600
    assert directive["search_quality"]["controlled_broad_rearm_after_epoch"] == now_epoch - 1
    assert directive["planner-policy-inputs"]["search_quality"]["controlled_broad_rearm_after_epoch"] == now_epoch - 1
    assert directive["planner-policy-inputs"]["controlled_broad_recovery"]["rearm_after_epoch"] == now_epoch - 1
    assert directive["planner-policy-inputs"]["controlled_broad_recovery"]["cooldown_sec"] == 600


def test_build_directive_activates_controlled_broad_for_empty_expected_degraded() -> None:
    now_epoch = int(module.datetime.now(module.timezone.utc).timestamp())
    queues = {
        "artifacts/wfa/aggregate/autonomous_seed_q/run_queue.csv": {
            "promotion_verdict": "REJECT",
            "rejection_reason": "METRICS_MISSING",
        }
    }
    yield_state = {
        "active": True,
        "preferred_contains": ["broad_anchor"],
        "hard_block_active": True,
        "hard_block_reason": "zero_coverage_seed_streak",
        "search_quality": {
            "positive_lineage_count": 3,
            "zero_evidence_lineage_count": 9,
            "winner_proximate_positive_lineage_count": 1,
            "winner_proximate_positive_contains": ["positive_anchor"],
            "controlled_recovery_contains": ["positive_anchor"],
            "broad_search_allowed": False,
            "seed_generation_mode": "winner_proximate_only",
            "controlled_recovery_active": False,
            "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
            "controlled_recovery_attempts_remaining": 0,
            "controlled_recovery_variants_cap": 8,
            "controlled_broad_rearm_after_epoch": now_epoch - 1,
        },
    }
    runtime_metrics = {
        "candidate_pool_status": "empty_expected_degraded",
        "ready_buffer_depth": 0,
        "global_pending_dispatchable": 0,
        "remote_active_queue_jobs": 0,
        "completed_with_metrics_stagnant_sec": 0,
        "no_progress_breaker_streak": 0,
        "last_seed_trigger_epoch": now_epoch - 300,
    }

    directive = module.build_directive(queues, yield_state=yield_state, runtime_metrics=runtime_metrics)

    assert directive["mode"] == "controlled_broad_recovery"
    assert directive["controlled_broad_active"] is True
