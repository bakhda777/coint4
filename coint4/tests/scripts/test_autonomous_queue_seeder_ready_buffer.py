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
                "lane_weights": {"winner_proximate": 40, "broad_search": 45, "confirm_replay": 15},
                "policy_overrides": {"policy_scale": "micro", "num_variants_cap": 64},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = autonomous_queue_seeder._load_yield_governor_state(state_path)

    assert state["exists"] is True
    assert state["status"] == "ok"
    assert state["active"] is True
    assert state["preferred_contains"] == ["rg_fast", "rg_broad"]
    assert state["winner_proximate"]["contains"] == ["rg_fast"]
    assert state["replay_fastlane"]["contains"] == ["confirm_rg"]
    assert state["replay_fastlane"]["replay_ready_count"] == 2
    assert state["confirm_replay"]["contains"] == ["confirm_rg"]
    assert state["confirm_replay_contains"] == ["confirm_rg"]
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
