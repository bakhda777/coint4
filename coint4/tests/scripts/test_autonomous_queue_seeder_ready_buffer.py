from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "autonomous_queue_seeder.py"
SPEC = importlib.util.spec_from_file_location("autonomous_queue_seeder", SCRIPT_PATH)
assert SPEC and SPEC.loader
autonomous_queue_seeder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(autonomous_queue_seeder)


def test_load_ready_queue_buffer_missing_file_is_fail_safe(tmp_path: Path) -> None:
    state = autonomous_queue_seeder._load_ready_queue_buffer(tmp_path / "ready_queue_buffer.json")

    assert state["exists"] is False
    assert state["status"] == "missing"
    assert state["seed_needed"] is False
    assert state["ready_depth"] == 0


def test_load_ready_queue_buffer_invalid_json_is_fail_safe(tmp_path: Path) -> None:
    state_path = tmp_path / "ready_queue_buffer.json"
    state_path.write_text("{broken-json", encoding="utf-8")

    state = autonomous_queue_seeder._load_ready_queue_buffer(state_path)

    assert state["exists"] is True
    assert state["status"] == "invalid_json"
    assert state["seed_needed"] is False


def test_load_ready_queue_buffer_invalid_entries_is_fail_safe(tmp_path: Path) -> None:
    state_path = tmp_path / "ready_queue_buffer.json"
    state_path.write_text(
        json.dumps({"target_depth": 6, "refill_threshold": 4, "entries": "wrong-type"}, ensure_ascii=False),
        encoding="utf-8",
    )

    state = autonomous_queue_seeder._load_ready_queue_buffer(state_path)

    assert state["exists"] is True
    assert state["status"] == "invalid_entries"
    assert state["seed_needed"] is False


def test_load_ready_queue_buffer_triggers_seed_when_below_refill_threshold(tmp_path: Path) -> None:
    state_path = tmp_path / "ready_queue_buffer.json"
    state_path.write_text(
        json.dumps(
            {
                "target_depth": 8,
                "refill_threshold": 5,
                "entries": [{"queue": "q1"}, {"queue": "q2"}, {"queue": "q3"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = autonomous_queue_seeder._load_ready_queue_buffer(state_path)

    assert state["status"] == "ok"
    assert state["ready_depth"] == 3
    assert state["effective_refill_threshold"] == 5
    assert state["seed_needed"] is True
    assert state["reason"] == "below_refill_threshold"


def test_load_ready_queue_buffer_uses_target_depth_when_refill_threshold_missing(tmp_path: Path) -> None:
    state_path = tmp_path / "ready_queue_buffer.json"
    state_path.write_text(
        json.dumps(
            {
                "target_depth": 4,
                "entries": [{"queue": "q1"}, {"queue": "q2"}, {"queue": "q3"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = autonomous_queue_seeder._load_ready_queue_buffer(state_path)

    assert state["status"] == "ok"
    assert state["refill_threshold"] is None
    assert state["effective_refill_threshold"] == 4
    assert state["ready_depth"] == 3
    assert state["seed_needed"] is True


def test_load_ready_queue_buffer_counts_only_coverage_verified_entries(tmp_path: Path) -> None:
    state_path = tmp_path / "ready_queue_buffer.json"
    state_path.write_text(
        json.dumps(
            {
                "target_depth": 4,
                "refill_threshold": 2,
                "entries": [
                    {"queue": "q1", "coverage_verified": True},
                    {"queue": "q2", "coverage_verified": False},
                    {"queue": "q3"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = autonomous_queue_seeder._load_ready_queue_buffer(state_path)

    assert state["ready_depth_total"] == 3
    assert state["coverage_verified_ready_count"] == 2
    assert state["ready_depth"] == 2
    assert state["seed_needed"] is False


def test_evaluate_seed_needed_includes_ready_buffer_reason_with_healthy_global_backlog() -> None:
    seed_needed, reasons = autonomous_queue_seeder._evaluate_seed_needed(
        executable_pending=128,
        pending_threshold=48,
        runnable_queue_count=7,
        ready_buffer_state={"seed_needed": True},
    )

    assert seed_needed is True
    assert reasons == ["ready_buffer_below_refill_threshold"]


def test_load_yield_governor_state_is_fail_safe_and_extracts_fastlane(tmp_path: Path) -> None:
    state_path = tmp_path / "yield_governor_state.json"
    state_path.write_text(
        json.dumps(
            {
                "active": True,
                "preferred_contains": ["rg_fast", "rg_broad"],
                "cooldown_contains": ["rg_cold"],
                "winner_proximate": {"enabled": True, "contains": ["rg_fast"], "reason": "strict_pass"},
                "replay_fastlane": {"enabled": True, "contains": ["confirm_rg"], "replay_ready_count": 2},
                "search_quality": {
                    "positive_lineage_count": 3,
                    "zero_evidence_lineage_count": 4,
                    "winner_proximate_positive_lineage_count": 1,
                    "winner_proximate_positive_contains": ["rg_fast"],
                    "broad_search_allowed": False,
                    "seed_generation_mode": "winner_proximate_only",
                    "controlled_recovery_active": True,
                    "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
                    "controlled_recovery_attempts_remaining": 2,
                    "controlled_recovery_variants_cap": 8,
                },
                "lane_weights": {"winner_proximate": 40, "broad_search": 45, "confirm_replay": 15},
                "policy_overrides": {"policy_scale": "micro", "num_variants_cap": 64},
                "hard_block_active": True,
                "hard_block_reason": "zero_coverage_seed_streak",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = autonomous_queue_seeder._load_yield_governor_state(state_path)

    assert state["exists"] is True
    assert state["status"] == "ok"
    assert state["active"] is True
    assert state["hard_block_active"] is True
    assert state["hard_block_until_epoch"] == 0
    assert state["zero_coverage_seed_streak"] == 0
    assert state["preferred_contains"] == ["rg_fast", "rg_broad"]
    assert state["winner_proximate"]["contains"] == ["rg_fast"]
    assert state["replay_fastlane"]["contains"] == ["confirm_rg"]
    assert state["replay_fastlane"]["replay_ready_count"] == 2
    assert state["confirm_replay"]["contains"] == ["confirm_rg"]
    assert state["confirm_replay_contains"] == ["confirm_rg"]
    assert state["search_quality"]["positive_lineage_count"] == 3
    assert state["search_quality"]["zero_evidence_lineage_count"] == 4
    assert state["search_quality"]["winner_proximate_positive_lineage_count"] == 1
    assert state["winner_proximate_positive_contains"] == ["rg_fast"]
    assert state["broad_search_allowed"] is False
    assert state["seed_generation_mode"] == "winner_proximate_only"
    assert state["controlled_recovery_active"] is True
    assert state["controlled_recovery_reason"] == "zero_coverage_seed_streak_with_positive_lineage"
    assert state["controlled_recovery_attempts_remaining"] == 2
    assert state["controlled_recovery_variants_cap"] == 8
    assert state["lane_weights"]["winner_proximate"] == 40


def test_derive_planner_focus_separates_winner_tokens_from_generic_anchor() -> None:
    focus = autonomous_queue_seeder._derive_planner_focus(
        user_contains=[],
        directive_contains=["strict_rg", "yield_rg"],
        directive_winner_contains=["strict_rg"],
        yield_governor={
            "preferred_contains": ["yield_rg", "broad_rg"],
            "winner_proximate": {"contains": ["strict_rg", "strict_rg_alt"]},
        },
        controller_group="autonomous_queue_seeder",
    )

    assert focus["generic_contains"] == ["strict_rg"]
    assert focus["winner_proximate_tokens"] == ["strict_rg", "strict_rg_alt"]
    assert focus["preferred_any_contains"] == ["strict_rg", "strict_rg_alt", "yield_rg", "broad_rg"]
    assert focus["anchor_source"] == "winner_proximate_anchor"


def test_select_seed_lane_prefers_winner_then_rotates_after_streak() -> None:
    first = autonomous_queue_seeder._select_seed_lane(
        winner_proximate_tokens=["strict_rg", "strict_alt"],
        preferred_any_contains=["yield_rg"],
        generic_contains=["autonomous_queue_seeder"],
        yield_governor={"lane_weights": {"winner_proximate": 60, "broad_search": 30, "confirm_replay": 10}},
        previous_state={},
    )
    assert first["selected_lane"] == "winner_proximate"
    assert first["contains"] == ["strict_rg"]

    second = autonomous_queue_seeder._select_seed_lane(
        winner_proximate_tokens=["strict_rg", "strict_alt"],
        preferred_any_contains=["yield_rg"],
        generic_contains=["autonomous_queue_seeder"],
        yield_governor={"lane_weights": {"winner_proximate": 60, "broad_search": 30, "confirm_replay": 10}},
        previous_state={"lane_selection": {"selected_lane": "winner_proximate", "lane_streak": 2, "token_rotation": 0}},
    )
    assert second["selected_lane"] == "broad_search"
    assert second["contains"] == ["yield_rg"]


def test_select_seed_lane_keeps_broad_search_blocked_when_positive_winner_exists() -> None:
    selection = autonomous_queue_seeder._select_seed_lane(
        winner_proximate_tokens=["strict_rg", "strict_alt"],
        preferred_any_contains=["yield_rg"],
        generic_contains=["autonomous_queue_seeder"],
        yield_governor={
            "search_quality": {
                "positive_lineage_count": 2,
                "zero_evidence_lineage_count": 5,
                "winner_proximate_positive_lineage_count": 1,
                "broad_search_allowed": False,
                "seed_generation_mode": "winner_proximate_only",
            },
            "lane_weights": {"winner_proximate": 60, "broad_search": 30, "confirm_replay": 10},
        },
        previous_state={"lane_selection": {"selected_lane": "winner_proximate", "lane_streak": 2, "token_rotation": 0}},
    )

    assert selection["selected_lane"] == "winner_proximate"
    assert selection["search_quality"]["broad_search_allowed"] is False
    assert selection["search_quality"]["seed_generation_mode"] == "winner_proximate_only"


def test_select_seed_lane_rotates_controlled_recovery_winner_anchor() -> None:
    first = autonomous_queue_seeder._select_seed_lane(
        winner_proximate_tokens=["strict_rg", "strict_alt"],
        preferred_any_contains=["strict_rg", "strict_alt"],
        generic_contains=["strict_rg", "strict_alt"],
        yield_governor={
            "search_quality": {
                "positive_lineage_count": 2,
                "zero_evidence_lineage_count": 5,
                "winner_proximate_positive_lineage_count": 2,
                "winner_proximate_positive_contains": ["strict_rg", "strict_alt"],
                "broad_search_allowed": False,
                "seed_generation_mode": "winner_proximate_only",
                "controlled_recovery_active": True,
                "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
                "controlled_recovery_attempts_remaining": 2,
                "controlled_recovery_variants_cap": 8,
            },
            "lane_weights": {"winner_proximate": 100, "broad_search": 0, "confirm_replay": 0},
            "replay_fastlane": {"contains": []},
            "confirm_replay": {"contains": []},
            "confirm_replay_contains": [],
            "hard_block_active": True,
            "hard_block_reason": "zero_coverage_seed_streak",
        },
        previous_state={},
    )

    second = autonomous_queue_seeder._select_seed_lane(
        winner_proximate_tokens=["strict_rg", "strict_alt"],
        preferred_any_contains=["strict_rg", "strict_alt"],
        generic_contains=["strict_rg", "strict_alt"],
        yield_governor={
            "search_quality": {
                "positive_lineage_count": 2,
                "zero_evidence_lineage_count": 5,
                "winner_proximate_positive_lineage_count": 2,
                "winner_proximate_positive_contains": ["strict_rg", "strict_alt"],
                "broad_search_allowed": False,
                "seed_generation_mode": "winner_proximate_only",
                "controlled_recovery_active": True,
                "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
                "controlled_recovery_attempts_remaining": 2,
                "controlled_recovery_variants_cap": 8,
            },
            "lane_weights": {"winner_proximate": 100, "broad_search": 0, "confirm_replay": 0},
            "replay_fastlane": {"contains": []},
            "confirm_replay": {"contains": []},
            "confirm_replay_contains": [],
            "hard_block_active": True,
            "hard_block_reason": "zero_coverage_seed_streak",
        },
        previous_state={"lane_selection": {"selected_lane": "winner_proximate", "lane_streak": 1, "token_rotation": 0}},
    )

    assert first["selected_lane"] == "winner_proximate"
    assert first["contains"] == ["strict_rg"]
    assert first["search_quality"]["controlled_recovery_active"] is True
    assert second["selected_lane"] == "winner_proximate"
    assert second["contains"] == ["strict_alt"]


def test_planner_repair_mode_args_supplies_validation_neighbor() -> None:
    assert autonomous_queue_seeder._planner_repair_mode_args(
        enabled=True,
        supported_planner_args={"--repair-mode"},
    ) == ["--repair-mode", "validation_neighbor"]
    assert autonomous_queue_seeder._planner_repair_mode_args(
        enabled=False,
        supported_planner_args={"--repair-mode"},
    ) == []


def test_queue_policy_sidecar_and_metadata_include_controlled_recovery_fields(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    queue_path = app_root / "artifacts" / "wfa" / "aggregate" / "autonomous_seed_demo" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "run_name,config_path,status",
                "valid,configs/valid_holdout.yaml,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    queue_policy_path = autonomous_queue_seeder._write_queue_policy_sidecar(
        queue_path=queue_path,
        app_root=app_root,
        planner_policy_hash="policy_deadbeef",
        selected_lane="winner_proximate",
        selected_lane_index=0,
        token_rotation=1,
        parent_rotation_offset=1,
        parent_diversity_depth=5,
        confirm_replay_hints=[],
        decision_payload={},
        coverage_verified=True,
        coverage_reason="coverage_verified",
        ready_buffer_excluded=False,
        seed_feasibility_status="ok",
        seed_feasibility_reason="",
        lineage_positive_evidence=True,
        recovery_mode="controlled",
        recovery_reason="zero_coverage_seed_streak",
        recovery_lineage_anchor="strict_rg",
    )

    autonomous_queue_seeder._decorate_queue_metadata(
        queue_path=queue_path,
        planner_policy_hash="policy_deadbeef",
        queue_policy_path=queue_policy_path,
        app_root=app_root,
        coverage_verified=True,
        coverage_reason="coverage_verified",
        ready_buffer_excluded=False,
        seed_feasibility_status="ok",
        seed_feasibility_reason="",
        lineage_positive_evidence=True,
        recovery_mode="controlled",
        recovery_reason="zero_coverage_seed_streak",
        recovery_lineage_anchor="strict_rg",
    )

    queue_policy = json.loads(queue_policy_path.read_text(encoding="utf-8"))
    rows = autonomous_queue_seeder._load_queue_rows(queue_path)
    metadata = json.loads(rows[0]["metadata_json"])

    assert queue_policy["recovery_mode"] == "controlled"
    assert queue_policy["recovery_reason"] == "zero_coverage_seed_streak"
    assert queue_policy["recovery_lineage_anchor"] == "strict_rg"
    assert metadata["recovery_mode"] == "controlled"
    assert metadata["recovery_reason"] == "zero_coverage_seed_streak"
    assert metadata["recovery_lineage_anchor"] == "strict_rg"


def test_hygiene_seed_queues_preserves_fresh_controlled_recovery_identity(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    aggregate_dir = app_root / "artifacts" / "wfa" / "aggregate"
    queue_dir = aggregate_dir / "autonomous_seed_20260307_094316"
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_path = queue_dir / "run_queue.csv"
    queue_path.write_text(
        "\n".join(
            [
                "run_name,config_path,status,metadata_json",
                'valid,configs/valid_holdout.yaml,planned,"{""planner_policy_hash"":""policy_deadbeef"",""recovery_mode"":""controlled"",""recovery_reason"":""zero_coverage_seed_streak"",""recovery_lineage_anchor"":""strict_rg"",""lineage_positive_evidence"":true}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    valid_cfg = app_root / "configs" / "valid_holdout.yaml"
    valid_cfg.parent.mkdir(parents=True, exist_ok=True)
    valid_cfg.write_text(
        "metadata:\n  evo_hash: evo_a\n  lineage_uid: line_a\nwalk_forward:\n  start_date: 2025-07-01\n  end_date: 2025-07-31\n",
        encoding="utf-8",
    )
    (app_root / "data_downloaded" / "year=2025" / "month=07").mkdir(parents=True, exist_ok=True)
    autonomous_queue_seeder._write_queue_policy_sidecar(
        queue_path=queue_path,
        app_root=app_root,
        planner_policy_hash="policy_deadbeef",
        selected_lane="winner_proximate",
        selected_lane_index=0,
        token_rotation=2,
        parent_rotation_offset=3,
        parent_diversity_depth=5,
        confirm_replay_hints=["confirm_rg"],
        decision_payload={
            "planner_hashes": {"planner_hash": "planner_cafebabe"},
            "lane_selection": {
                "seed_lane": "winner_proximate",
                "seed_lane_index": 0,
                "confirm_replay_hints": ["confirm_rg"],
            },
            "parent_diversification": {"rotation_offset": 3, "depth": 5},
            "parent_resolution": {"winner_proximate_tokens": ["strict_rg"]},
        },
        coverage_verified=True,
        coverage_reason="coverage_verified",
        ready_buffer_excluded=False,
        seed_feasibility_status="ok",
        seed_feasibility_reason="",
        lineage_positive_evidence=True,
        recovery_mode="controlled",
        recovery_reason="zero_coverage_seed_streak",
        recovery_lineage_anchor="strict_rg",
    )

    hygiene = autonomous_queue_seeder._hygiene_seed_queues(
        aggregate_dir=aggregate_dir,
        app_root=app_root,
        run_group_prefix="autonomous_seed",
        orphan_path=aggregate_dir / ".autonomous" / "orphan_queues.csv",
    )

    assert hygiene["reviewed"] == 1
    assert hygiene["orphaned"] == 0
    queue_policy = json.loads((queue_dir / "queue_policy.json").read_text(encoding="utf-8"))
    rows = autonomous_queue_seeder._load_queue_rows(queue_path)
    metadata = json.loads(rows[0]["metadata_json"])

    assert queue_policy["planner_policy_hash"] == "policy_deadbeef"
    assert queue_policy["seed_lane"] == "winner_proximate"
    assert queue_policy["recovery_mode"] == "controlled"
    assert queue_policy["recovery_reason"] == "zero_coverage_seed_streak"
    assert queue_policy["recovery_lineage_anchor"] == "strict_rg"
    assert queue_policy["winner_proximate_tokens"] == ["strict_rg"]
    assert metadata["planner_policy_hash"] == "policy_deadbeef"
    assert metadata["recovery_mode"] == "controlled"
    assert metadata["recovery_reason"] == "zero_coverage_seed_streak"
    assert metadata["recovery_lineage_anchor"] == "strict_rg"


def test_load_yield_governor_state_backfills_replay_fastlane_from_legacy_confirm_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "yield_governor_state.json"
    state_path.write_text(
        json.dumps(
            {
                "active": True,
                "confirm_replay": {"enabled": True, "contains": ["legacy_rg"], "replay_ready_count": 1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = autonomous_queue_seeder._load_yield_governor_state(state_path)

    assert state["replay_fastlane"]["enabled"] is True
    assert state["replay_fastlane"]["contains"] == ["legacy_rg"]
    assert state["replay_fastlane"]["replay_ready_count"] == 1
    assert state["replay_fastlane"]["source"] == "legacy_confirm_replay"
    assert state["confirm_replay_contains"] == ["legacy_rg"]


def test_load_yield_governor_state_reads_hard_block_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "yield_governor_state.json"
    state_path.write_text(
        json.dumps(
            {
                "active": True,
                "hard_block_active": True,
                "hard_block_reason": "zero_coverage_seed_streak",
                "hard_block_until_epoch": 1772809999,
                "zero_coverage_seed_streak": 3,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = autonomous_queue_seeder._load_yield_governor_state(state_path)

    assert state["hard_block_active"] is True
    assert state["hard_block_reason"] == "zero_coverage_seed_streak"
    assert state["hard_block_until_epoch"] == 1772809999
    assert state["zero_coverage_seed_streak"] == 3


def test_select_seed_lane_prefers_replay_fastlane_before_winner_fallback() -> None:
    selection = autonomous_queue_seeder._select_seed_lane(
        winner_proximate_tokens=["winner_rg"],
        preferred_any_contains=["yield_rg"],
        generic_contains=["autonomous_queue_seeder"],
        yield_governor={
            "lane_weights": {"winner_proximate": 10, "broad_search": 5, "confirm_replay": 90},
            "replay_fastlane": {"contains": ["confirm_rg"]},
            "confirm_replay": {"contains": ["legacy_rg"]},
            "confirm_replay_contains": ["legacy_contains_rg"],
        },
        previous_state={},
    )

    assert selection["selected_lane"] == "confirm_replay"
    assert selection["confirm_replay_hints"] == ["confirm_rg"]
    assert selection["confirm_replay_source"] == "yield_replay_fastlane"
    assert selection["contains"] == ["confirm_rg"]


def test_select_seed_lane_prefers_directive_replay_fastlane_over_yield_state() -> None:
    selection = autonomous_queue_seeder._select_seed_lane(
        winner_proximate_tokens=["winner_rg"],
        preferred_any_contains=["yield_rg"],
        generic_contains=["autonomous_queue_seeder"],
        directive_replay_fastlane_tokens=["directive_confirm_rg"],
        yield_governor={
            "lane_weights": {"winner_proximate": 10, "broad_search": 5, "confirm_replay": 90},
            "replay_fastlane": {"contains": ["yield_confirm_rg"]},
        },
        previous_state={},
    )

    assert selection["selected_lane"] == "confirm_replay"
    assert selection["confirm_replay_hints"] == ["directive_confirm_rg"]
    assert selection["confirm_replay_source"] == "directive_replay_fastlane"


def test_stable_hash_is_deterministic() -> None:
    left = autonomous_queue_seeder._stable_hash({"a": 1, "b": ["x", "y"]}, prefix="policy", size=12)
    right = autonomous_queue_seeder._stable_hash({"b": ["x", "y"], "a": 1}, prefix="policy", size=12)
    assert left == right
    assert left.startswith("policy_")


def test_coverage_gate_rejects_missing_oos_months(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    config_path = app_root / "configs" / "sample.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "walk_forward:\n  start_date: 2025-07-01\n  end_date: 2025-09-30\n",
        encoding="utf-8",
    )
    (app_root / "data_downloaded" / "year=2025" / "month=07").mkdir(parents=True, exist_ok=True)

    gate = autonomous_queue_seeder._coverage_gate_for_config(config_path=config_path, app_root=app_root)

    assert gate["ok"] is False
    assert gate["reason"] == "missing_data_coverage"
    assert gate["missing_months"] == ["2025-08", "2025-09"]


def test_assess_window_data_coverage_fail_closed_on_missing_months(tmp_path: Path) -> None:
    data_root = tmp_path / "app" / "data_downloaded"
    (data_root / "year=2025" / "month=07").mkdir(parents=True, exist_ok=True)
    (data_root / "year=2025" / "month=08").mkdir(parents=True, exist_ok=True)

    coverage = autonomous_queue_seeder._assess_window_data_coverage(
        data_root,
        ["2025-07-01,2025-08-31", "2025-09-01,2025-10-31"],
    )

    assert coverage["ok"] is False
    assert coverage["reason"] == "missing_data_coverage"
    assert coverage["covered_window_count"] == 1
    assert coverage["uncovered_window_count"] == 1
    assert coverage["missing_months"] == ["2025-09", "2025-10"]
    assert coverage["windows"][0]["ok"] is True
    assert coverage["windows"][1]["ok"] is False


def test_select_covered_recovery_windows_prefers_latest_positive_covered_window(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    data_root = app_root / "data_downloaded"
    for token in ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07"]:
        year, month = token.split("-")
        (data_root / f"year={year}" / f"month={month}").mkdir(parents=True, exist_ok=True)

    cfg_dir = app_root / "configs" / "evolution" / "anchor"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    older_cfg = cfg_dir / "older.yaml"
    older_cfg.write_text(
        "walk_forward:\n  start_date: 2024-05-01\n  end_date: 2025-06-30\n",
        encoding="utf-8",
    )
    uncovered_cfg = cfg_dir / "uncovered.yaml"
    uncovered_cfg.write_text(
        "walk_forward:\n  start_date: 2025-07-01\n  end_date: 2025-12-31\n",
        encoding="utf-8",
    )

    windows = autonomous_queue_seeder._select_covered_recovery_windows(
        candidate_groups=["winner_anchor"],
        run_index_groups={
            "winner_anchor": [
                {
                    "config_path": "configs/evolution/anchor/uncovered.yaml",
                    "metrics_present": "1",
                    "observed_test_days": "75",
                    "coverage_ratio": "1.0",
                    "total_trades": "100",
                    "total_pairs_traded": "10",
                },
                {
                    "config_path": "configs/evolution/anchor/older.yaml",
                    "metrics_present": "1",
                    "observed_test_days": "75",
                    "coverage_ratio": "1.0",
                    "total_trades": "200",
                    "total_pairs_traded": "12",
                },
            ]
        },
        app_root=app_root,
        limit=1,
    )

    assert windows == ["2024-05-01,2025-06-30"]


def test_prune_seed_queue_filters_missing_coverage_and_duplicates(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    queue_path = app_root / "artifacts" / "wfa" / "aggregate" / "group_a" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    configs_dir = app_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    valid_cfg = configs_dir / "valid_holdout.yaml"
    valid_cfg.write_text(
        "metadata:\n  evo_hash: evo_a\n  lineage_uid: line_a\nwalk_forward:\n  start_date: 2025-07-01\n  end_date: 2025-07-31\n",
        encoding="utf-8",
    )
    dup_cfg = configs_dir / "dup_holdout.yaml"
    dup_cfg.write_text(valid_cfg.read_text(encoding="utf-8"), encoding="utf-8")
    missing_cfg = configs_dir / "missing.yaml"
    missing_cfg.write_text(
        "metadata:\n  evo_hash: evo_b\n  lineage_uid: line_b\nwalk_forward:\n  start_date: 2025-08-01\n  end_date: 2025-09-30\n",
        encoding="utf-8",
    )
    (app_root / "data_downloaded" / "year=2025" / "month=07").mkdir(parents=True, exist_ok=True)

    queue_path.write_text(
        "\n".join(
            [
                "run_name,config_path,status",
                "valid,configs/valid_holdout.yaml,planned",
                "dup,configs/dup_holdout.yaml,planned",
                "missing,configs/missing.yaml,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = autonomous_queue_seeder._prune_seed_queue(queue_path=queue_path, app_root=app_root)
    rows = autonomous_queue_seeder._load_queue_rows(queue_path)

    assert stats["rows_before"] == 3
    assert stats["rows_after"] == 1
    assert stats["coverage_rejected"] == 1
    assert stats["dedupe_rejected"] == 1
    assert stats["missing_months"] == ["2025-08", "2025-09"]
    assert [row["run_name"] for row in rows] == ["valid"]


def test_prune_seed_queue_marks_full_prune_as_blocked_rows(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    queue_path = app_root / "artifacts" / "wfa" / "aggregate" / "group_b" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = app_root / "configs" / "missing_holdout.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "walk_forward:\n  start_date: 2025-08-01\n  end_date: 2025-09-30\n",
        encoding="utf-8",
    )
    queue_path.write_text(
        "\n".join(
            [
                "run_name,config_path,status",
                "missing_a,configs/missing_holdout.yaml,planned",
                "missing_b,configs/missing_holdout.yaml,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = autonomous_queue_seeder._prune_seed_queue(queue_path=queue_path, app_root=app_root)
    rows = autonomous_queue_seeder._load_queue_rows(queue_path)

    assert stats["rows_before"] == 2
    assert stats["rows_after"] == 0
    assert stats["coverage_rejected"] == 2
    assert stats["blocked_rows_written"] == 2
    assert stats["block_reason"] == "coverage_fail_closed"
    assert [row["status"] for row in rows] == ["blocked", "blocked"]
    assert [row["note"] for row in rows] == ["coverage_fail_closed", "coverage_fail_closed"]


def test_prune_seed_queue_materializes_header_only_queue_as_blocked_placeholder(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    queue_path = app_root / "artifacts" / "wfa" / "aggregate" / "group_empty" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("config_path,results_dir,status\n", encoding="utf-8")

    stats = autonomous_queue_seeder._prune_seed_queue(queue_path=queue_path, app_root=app_root)
    rows = autonomous_queue_seeder._load_queue_rows(queue_path)

    assert stats["rows_before"] == 0
    assert stats["rows_after"] == 0
    assert stats["blocked_rows_written"] == 1
    assert stats["block_reason"] == "queue_pruned_empty"
    assert len(rows) == 1
    assert rows[0]["status"] == "blocked"
    assert rows[0]["note"] == "queue_pruned_empty"


def test_hygiene_seed_queues_orphans_zero_coverage_history(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    aggregate_dir = app_root / "artifacts" / "wfa" / "aggregate"
    queue_dir = aggregate_dir / "autonomous_seed_20260306_122359"
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_path = queue_dir / "run_queue.csv"
    queue_path.write_text(
        "\n".join(
            [
                "run_name,config_path,status",
                "valid,configs/valid_holdout.yaml,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    valid_cfg = app_root / "configs" / "valid_holdout.yaml"
    valid_cfg.parent.mkdir(parents=True, exist_ok=True)
    valid_cfg.write_text(
        "metadata:\n  evo_hash: evo_a\n  lineage_uid: line_a\nwalk_forward:\n  start_date: 2025-07-01\n  end_date: 2025-07-31\n",
        encoding="utf-8",
    )
    (app_root / "data_downloaded" / "year=2025" / "month=07").mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "rollup").mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "rollup" / "run_index.csv").write_text(
        "\n".join(
            [
                "run_group,coverage_ratio,total_trades,total_pnl",
                "autonomous_seed_20260306_122359,0,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (queue_dir / "rank_result.json").write_text(
        json.dumps(
            {
                "details": "RANK_OK_FALLBACK_STRICT_BINDING:min_windows,min_trades,min_pairs,coverage_below",
                "strict_diag": {
                    "variants_passing_all": 0,
                    "binding_gates": ["min_windows", "min_trades", "coverage_below"],
                    "rejects": {"min_trades": 1, "coverage_below": 1},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    hygiene = autonomous_queue_seeder._hygiene_seed_queues(
        aggregate_dir=aggregate_dir,
        app_root=app_root,
        run_group_prefix="autonomous_seed",
        orphan_path=aggregate_dir / ".autonomous" / "orphan_queues.csv",
    )

    assert hygiene["reviewed"] == 1
    assert hygiene["orphaned"] == 1
    assert hygiene["zero_coverage_rejected"] == 1
    assert hygiene["covered_window_count"] == 0
    assert hygiene["queues"][0]["orphan_reason"] == "ZERO_COVERAGE"

    queue_policy = json.loads((queue_dir / "queue_policy.json").read_text(encoding="utf-8"))
    assert queue_policy["coverage_verified"] is False
    assert queue_policy["coverage_reason"] == "ZERO_COVERAGE"
    assert queue_policy["ready_buffer_excluded"] is True

    rows = autonomous_queue_seeder._load_queue_rows(queue_path)
    metadata = json.loads(rows[0]["metadata_json"])
    assert metadata["coverage_verified"] is False
    assert metadata["coverage_reason"] == "ZERO_COVERAGE"
    assert metadata["ready_buffer_excluded"] is True


def test_persist_yield_governor_state_merges_hard_block_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "yield_governor_state.json"
    state_path.write_text(json.dumps({"active": True, "preferred_contains": ["rg_fast"]}, ensure_ascii=False), encoding="utf-8")

    autonomous_queue_seeder._persist_yield_governor_state(
        state_path,
        {
            "zero_coverage_seed_streak": 2,
            "hard_block_active": True,
            "hard_block_reason": "zero_coverage_seed_streak",
            "hard_block_until_epoch": 1772809999,
        },
    )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["active"] is True
    assert payload["preferred_contains"] == ["rg_fast"]
    assert payload["zero_coverage_seed_streak"] == 2
    assert payload["hard_block_active"] is True
    assert payload["hard_block_reason"] == "zero_coverage_seed_streak"
    assert payload["hard_block_until_epoch"] == 1772809999
    assert isinstance(payload["ts"], str) and payload["ts"]


def test_recent_zero_yield_signal_activates_on_zero_activity_streak(tmp_path: Path) -> None:
    rank_results_dir = tmp_path / "artifacts" / "optimization_state" / "rank_results"
    rank_results_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        (rank_results_dir / f"autonomous_seed_{idx}_latest.json").write_text(
            json.dumps(
                {
                    "coverage": 0.0,
                    "observed_test_days": 0,
                    "total_trades": 0,
                    "details": "RANK_OK_FALLBACK_STRICT_BINDING:min_windows,min_trades,min_pairs,coverage_below",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    signal = autonomous_queue_seeder._recent_zero_yield_signal(rank_results_dir)

    assert signal["active"] is True
    assert signal["analyzed"] == 4
    assert signal["zeroish"] == 4
    assert signal["strict_binding"] == 4


def test_assess_recent_seed_quality_detects_zero_coverage_streak(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    aggregate_dir = app_root / "artifacts" / "wfa" / "aggregate"
    rollup_dir = aggregate_dir / "rollup"
    rollup_dir.mkdir(parents=True, exist_ok=True)
    run_index = rollup_dir / "run_index.csv"
    run_index.write_text(
        "\n".join(
            [
                "run_group,coverage_ratio,observed_test_days,total_trades,total_pairs_traded,total_pnl",
                "autonomous_seed_20260306_120953,0,0,0,0,0",
                "autonomous_seed_20260306_122359,0,0,0,0,0",
                "autonomous_seed_20260306_123454,1,30,5,3,10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for group, variants in [
        ("autonomous_seed_20260306_120953", 0),
        ("autonomous_seed_20260306_122359", 0),
        ("autonomous_seed_20260306_123454", 1),
    ]:
        group_dir = aggregate_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "rank_result.json").write_text(
            json.dumps({"strict_diag": {"variants_passing_all": variants}}, ensure_ascii=False),
            encoding="utf-8",
        )

    quality = autonomous_queue_seeder._assess_recent_seed_quality(
        app_root=app_root,
        run_group_prefix="autonomous_seed",
        run_index_path=run_index,
    )

    assert quality["groups_analyzed"] == 3
    assert quality["zero_coverage_seed_streak"] == 0
    assert quality["covered_window_count"] == 1
    assert quality["hard_block_recommended"] is False

    run_index.write_text(
        "\n".join(
            [
                "run_group,coverage_ratio,observed_test_days,total_trades,total_pairs_traded,total_pnl",
                "autonomous_seed_20260306_122359,0,0,0,0,0",
                "autonomous_seed_20260306_123454,0,0,0,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for group in ["autonomous_seed_20260306_122359", "autonomous_seed_20260306_123454"]:
        (aggregate_dir / group / "rank_result.json").write_text(
            json.dumps({"strict_diag": {"variants_passing_all": 0}}, ensure_ascii=False),
            encoding="utf-8",
        )

    quality = autonomous_queue_seeder._assess_recent_seed_quality(
        app_root=app_root,
        run_group_prefix="autonomous_seed",
        run_index_path=run_index,
    )

    assert quality["zero_coverage_seed_streak"] == 2
    assert quality["backlog_suppress"] is True
    assert quality["hard_block_recommended"] is True


def test_load_search_blacklist_uses_micro_defaults_for_missing_caps(tmp_path: Path) -> None:
    blacklist_path = tmp_path / "search_policy_blacklist.json"
    blacklist_path.write_text(
        json.dumps({"active": True, "stats": {"dominant_code": "MAX_VAR_MULTIPLIER_INVALID", "total_coded": 8}}, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = autonomous_queue_seeder._load_search_blacklist(blacklist_path)

    assert payload["active"] is True
    assert payload["max_changed_keys_cap"] == 3
    assert payload["dedupe_distance_floor"] == 0.04
    assert payload["num_variants_cap"] == 48
    assert payload["policy_scale"] == "micro"


def test_prune_seed_queue_blocks_recent_zero_evidence_lineage_before_runnable(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    queue_path = app_root / "artifacts" / "wfa" / "aggregate" / "autonomous_seed_demo" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "run_name,config_path,status",
                "valid,configs/valid_holdout.yaml,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    valid_cfg = app_root / "configs" / "valid_holdout.yaml"
    valid_cfg.parent.mkdir(parents=True, exist_ok=True)
    valid_cfg.write_text(
        "metadata:\n  evo_hash: evo_a\n  lineage_uid: line_a\nwalk_forward:\n  start_date: 2025-07-01\n  end_date: 2025-07-31\n",
        encoding="utf-8",
    )
    (app_root / "data_downloaded" / "year=2025" / "month=07").mkdir(parents=True, exist_ok=True)

    stats = autonomous_queue_seeder._prune_seed_queue(
        queue_path=queue_path,
        app_root=app_root,
        decision_payload={"primary_parent": {"run_group": "parent_rg"}},
        run_index_groups={
            "parent_rg": [
                {
                    "metrics_present": "True",
                    "observed_test_days": "0",
                    "coverage_ratio": "0",
                    "total_trades": "",
                    "total_pairs_traded": "",
                }
            ]
        },
        quarantine_by_group={},
    )
    rows = autonomous_queue_seeder._load_queue_rows(queue_path)

    assert stats["rows_before"] == 1
    assert stats["rows_after"] == 0
    assert stats["block_reason"] == "ZERO_OBSERVED_TEST_DAYS"
    assert stats["matched_run_groups"] == ["parent_rg"]
    assert rows[0]["status"] == "blocked"
    assert rows[0]["note"] == "ZERO_OBSERVED_TEST_DAYS"


def test_prune_seed_queue_blocks_deterministic_quarantine_lineage_before_runnable(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    queue_path = app_root / "artifacts" / "wfa" / "aggregate" / "autonomous_seed_demo" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "run_name,config_path,status",
                "valid,configs/valid_holdout.yaml,planned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    valid_cfg = app_root / "configs" / "valid_holdout.yaml"
    valid_cfg.parent.mkdir(parents=True, exist_ok=True)
    valid_cfg.write_text(
        "metadata:\n  evo_hash: evo_a\n  lineage_uid: line_a\nwalk_forward:\n  start_date: 2025-07-01\n  end_date: 2025-07-31\n",
        encoding="utf-8",
    )
    (app_root / "data_downloaded" / "year=2025" / "month=07").mkdir(parents=True, exist_ok=True)

    stats = autonomous_queue_seeder._prune_seed_queue(
        queue_path=queue_path,
        app_root=app_root,
        decision_payload={"primary_parent": {"run_group": "parent_rg"}},
        run_index_groups={"parent_rg": []},
        quarantine_by_group={"parent_rg": {"MAX_VAR_MULTIPLIER_INVALID": 3}},
    )
    rows = autonomous_queue_seeder._load_queue_rows(queue_path)

    assert stats["rows_after"] == 0
    assert stats["block_reason"] == "MAX_VAR_MULTIPLIER_INVALID"
    assert stats["quarantine_counts"] == {"MAX_VAR_MULTIPLIER_INVALID": 3}
    assert rows[0]["status"] == "blocked"
    assert rows[0]["note"] == "MAX_VAR_MULTIPLIER_INVALID"
