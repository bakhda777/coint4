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
