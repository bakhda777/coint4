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
