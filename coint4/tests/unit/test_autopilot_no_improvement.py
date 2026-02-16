import importlib.util
import sys
from pathlib import Path

import pytest


def _load_autopilot_module():
    app_root = Path(__file__).resolve().parents[2]
    module_path = app_root / "scripts/optimization/autopilot_budget1000.py"
    assert module_path.exists()

    spec = importlib.util.spec_from_file_location("autopilot_budget1000", module_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_normalize_stop_state_backfills_fields() -> None:
    mod = _load_autopilot_module()
    state = {
        "current_best": {"score": "1.23"},
        "no_improvement_streak": "2",
        "stop_reason": "",
    }

    mod._normalize_stop_state(state)

    assert state["best_score"] == pytest.approx(1.23)
    assert state["no_improvement_streak"] == 2
    assert state["stop_reason"] is None


def test_apply_no_improvement_round_stops_on_threshold() -> None:
    mod = _load_autopilot_module()
    state = {
        "done": False,
        "no_improvement_streak": 1,
        "stop_reason": None,
    }

    stop = mod._apply_no_improvement_round(
        state=state,
        improved_in_round=False,
        no_improvement_rounds=2,
        min_improvement=0.02,
    )

    assert stop is True
    assert state["done"] is True
    assert state["no_improvement_streak"] == 2
    assert "no_improvement_streak_reached" in state["stop_reason"]
    assert "rounds=2" in state["stop_reason"]


def test_apply_no_improvement_round_continues_before_threshold() -> None:
    mod = _load_autopilot_module()
    state = {
        "done": False,
        "no_improvement_streak": 0,
        "stop_reason": None,
    }

    stop = mod._apply_no_improvement_round(
        state=state,
        improved_in_round=False,
        no_improvement_rounds=2,
        min_improvement=0.02,
    )

    assert stop is False
    assert state["done"] is False
    assert state["no_improvement_streak"] == 1
    assert state["stop_reason"] is None


def test_apply_no_improvement_round_resets_streak_on_improvement() -> None:
    mod = _load_autopilot_module()
    state = {
        "done": False,
        "no_improvement_streak": 3,
        "stop_reason": "old",
    }

    stop = mod._apply_no_improvement_round(
        state=state,
        improved_in_round=True,
        no_improvement_rounds=3,
        min_improvement=0.02,
    )

    assert stop is False
    assert state["done"] is False
    assert state["no_improvement_streak"] == 0
    assert state["stop_reason"] == "old"
