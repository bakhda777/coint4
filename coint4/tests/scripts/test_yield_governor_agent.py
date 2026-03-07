from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "yield_governor_agent.py"
SPEC = importlib.util.spec_from_file_location("yield_governor_agent", SCRIPT_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_yield_governor_state_prefers_strict_and_high_yield(tmp_path: Path) -> None:
    root = tmp_path
    aggregate_dir = root / "artifacts" / "wfa" / "aggregate"
    run_index_path = aggregate_dir / "rollup" / "run_index.csv"
    fullspan_state_path = aggregate_dir / ".autonomous" / "fullspan_decision_state.json"

    queue_dir = aggregate_dir / "autonomous_seed_demo"
    _write_csv(
        queue_dir / "run_queue.csv",
        [
            {
                "config_path": "configs/demo.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_autonomous_seed_demo_v001",
                "status": "completed",
                "lineage_uid": "lineage-good",
                "operator_id": "op_good",
                "metadata_json": "",
            }
        ],
    )
    _write_csv(
        run_index_path,
        [
            {
                "run_id": "holdout_autonomous_seed_demo_v001",
                "run_group": "strict_rg",
                "metrics_present": "true",
                "observed_test_days": "75",
                "coverage_ratio": "1.0",
                "total_trades": "500",
                "total_pairs_traded": "30",
                "max_drawdown_on_equity": "0.05",
                "total_pnl": "100",
                "tail_loss_worst_period_pnl": "-50",
            }
        ],
    )
    fullspan_state_path.parent.mkdir(parents=True, exist_ok=True)
    fullspan_state_path.write_text(
        '{"queues":{"artifacts/wfa/aggregate/demo/run_queue.csv":{"promotion_verdict":"PROMOTE_PENDING_CONFIRM","strict_pass_count":1,"top_run_group":"strict_rg"}}}',
        encoding="utf-8",
    )

    payload = module.build_yield_governor_state(
        root=root,
        aggregate_dir=aggregate_dir,
        run_index_path=run_index_path,
        fullspan_state_path=fullspan_state_path,
        recent_queue_limit=20,
        hard_block_active=True,
        hard_block_reason="zero_coverage_seed_streak",
        existing_state={"controlled_recovery_attempts_remaining": 2},
    )

    assert payload["active"] is True
    assert payload["winner_proximate"]["contains"][0] == "strict_rg"
    assert "strict_rg" in payload["preferred_contains"]
    assert payload["replay_fastlane"]["enabled"] is True
    assert "strict_rg" in payload["replay_fastlane"]["contains"]
    assert payload["positive_lineage_count"] == 1
    assert payload["zero_evidence_lineage_count"] == 0
    assert payload["winner_proximate_positive_lineage_count"] == 1
    assert payload["winner_proximate_positive_contains"] == ["strict_rg"]
    assert payload["controlled_recovery_contains"] == ["strict_rg"]
    assert payload["broad_search_allowed"] is False
    assert payload["seed_generation_mode"] == "winner_proximate_only"
    assert payload["controlled_recovery_active"] is True
    assert payload["controlled_recovery_reason"] == "zero_coverage_seed_streak_with_positive_lineage"
    assert payload["controlled_recovery_attempts_remaining"] == 2
    assert payload["controlled_recovery_variants_cap"] == 8
    assert payload["lane_weights"] == {
        "winner_proximate": 65,
        "confirm_replay": 20,
        "broad_search": 15,
    }
    assert len(str(payload["policy-hash"])) == 64
    assert payload["policy-hash"] == payload["policy_hash"]
    assert payload["planner-policy-inputs"]["policy_family"] == "exploit_first"
    assert payload["planner-policy-inputs"]["lane_weights"]["winner_proximate"] == 65
    assert payload["planner-policy-inputs"]["search_quality"]["positive_lineage_count"] == 1
    assert payload["planner-policy-inputs"]["search_quality"]["broad_search_allowed"] is False
    assert payload["planner-policy-inputs"]["search_quality"]["winner_proximate_positive_contains"] == ["strict_rg"]
    assert payload["planner-policy-inputs"]["search_quality"]["controlled_recovery_contains"] == ["strict_rg"]
    assert payload["planner-policy-inputs"]["search_quality"]["controlled_recovery_active"] is True
    assert payload["planner-policy-inputs"] == payload["planner_policy_inputs"]


def test_rearm_controlled_recovery_if_eligible_updates_state(tmp_path: Path) -> None:
    state_path = tmp_path / "yield_governor_state.json"
    state_path.write_text(
        json.dumps(
            {
                "hard_block_active": True,
                "hard_block_reason": "zero_coverage_seed_streak",
                "controlled_recovery_active": False,
                "controlled_recovery_attempts_remaining": 0,
                "controlled_recovery_variants_cap": 8,
                "winner_proximate_positive_contains": ["strict_rg"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    process_slo_state_path = tmp_path / "process_slo_state.json"
    process_slo_state_path.write_text(
        json.dumps(
            {
                "queue": {
                    "dispatchable_pending": 0,
                    "candidate_pool_status": "empty_expected",
                },
                "search_quality": {
                    "winner_proximate_positive_contains": ["strict_rg"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = module.rearm_controlled_recovery_if_eligible(
        state_path=state_path,
        process_slo_state_path=process_slo_state_path,
    )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert result["eligible"] is True
    assert result["updated"] is True
    assert payload["controlled_recovery_active"] is True
    assert payload["controlled_recovery_attempts_remaining"] == 2
    assert payload["controlled_recovery_contains"] == ["strict_rg"]
    assert payload["search_quality"]["controlled_recovery_active"] is True


def test_rearm_controlled_recovery_if_eligible_rejects_non_empty_expected_pool(tmp_path: Path) -> None:
    state_path = tmp_path / "yield_governor_state.json"
    state_path.write_text(
        json.dumps(
            {
                "hard_block_active": True,
                "hard_block_reason": "zero_coverage_seed_streak",
                "controlled_recovery_active": False,
                "controlled_recovery_attempts_remaining": 0,
                "winner_proximate_positive_contains": ["strict_rg"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    process_slo_state_path = tmp_path / "process_slo_state.json"
    process_slo_state_path.write_text(
        json.dumps(
            {
                "queue": {
                    "dispatchable_pending": 0,
                    "candidate_pool_status": "empty_error",
                },
                "search_quality": {
                    "winner_proximate_positive_contains": ["strict_rg"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = module.rearm_controlled_recovery_if_eligible(
        state_path=state_path,
        process_slo_state_path=process_slo_state_path,
    )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert result["eligible"] is False
    assert result["updated"] is False
    assert payload["controlled_recovery_attempts_remaining"] == 0


def test_build_yield_governor_state_filters_controlled_recovery_contains_by_deterministic_quarantine(tmp_path: Path) -> None:
    root = tmp_path
    aggregate_dir = root / "artifacts" / "wfa" / "aggregate"
    run_index_path = aggregate_dir / "rollup" / "run_index.csv"
    fullspan_state_path = aggregate_dir / ".autonomous" / "fullspan_decision_state.json"
    quarantine_path = aggregate_dir / ".autonomous" / "deterministic_quarantine.json"

    queue_dir = aggregate_dir / "autonomous_seed_demo"
    _write_csv(
        queue_dir / "run_queue.csv",
        [
            {
                "config_path": "configs/demo.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_autonomous_seed_demo_v001",
                "status": "completed",
                "lineage_uid": "lineage-good",
                "operator_id": "op_good",
                "metadata_json": "",
            }
        ],
    )
    _write_csv(
        run_index_path,
        [
            {
                "run_id": "holdout_autonomous_seed_demo_v001",
                "run_group": "strict_rg",
                "metrics_present": "true",
                "observed_test_days": "75",
                "coverage_ratio": "1.0",
                "total_trades": "500",
                "total_pairs_traded": "30",
                "max_drawdown_on_equity": "0.05",
                "total_pnl": "100",
                "tail_loss_worst_period_pnl": "-50",
            }
        ],
    )
    fullspan_state_path.parent.mkdir(parents=True, exist_ok=True)
    fullspan_state_path.write_text(
        '{"queues":{"artifacts/wfa/aggregate/demo/run_queue.csv":{"promotion_verdict":"PROMOTE_PENDING_CONFIRM","strict_pass_count":1,"top_run_group":"strict_rg"}}}',
        encoding="utf-8",
    )
    quarantine_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "queue": "artifacts/wfa/aggregate/strict_rg/run_queue.csv",
                        "code": "MAX_VAR_MULTIPLIER_INVALID",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = module.build_yield_governor_state(
        root=root,
        aggregate_dir=aggregate_dir,
        run_index_path=run_index_path,
        fullspan_state_path=fullspan_state_path,
        recent_queue_limit=20,
        hard_block_active=True,
        hard_block_reason="zero_coverage_seed_streak",
        existing_state={"controlled_recovery_attempts_remaining": 2},
    )

    assert payload["winner_proximate_positive_contains"] == ["strict_rg"]
    assert payload["controlled_recovery_contains"] == []
    assert payload["controlled_recovery_active"] is False
