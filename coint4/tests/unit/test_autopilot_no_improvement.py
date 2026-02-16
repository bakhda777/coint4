import importlib.util
import json
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


def test_metric_delta_and_round_decision() -> None:
    mod = _load_autopilot_module()
    before = {"score": 1.0, "worst_robust_sharpe": 2.0, "worst_dd_pct": 0.20}
    after = {"score": 1.2, "worst_robust_sharpe": 2.3, "worst_dd_pct": 0.15}

    assert mod._metric_delta(before=before, after=after, key="score") == pytest.approx(0.2)
    assert mod._metric_delta(before=before, after=after, key="worst_robust_sharpe") == pytest.approx(0.3)
    assert mod._metric_delta(before=before, after=after, key="worst_dd_pct") == pytest.approx(-0.05)
    assert mod._metric_delta(before=None, after=after, key="score") is None

    continue_decision = mod._round_decision(
        current_round=1,
        max_rounds=3,
        stopped_by_no_improvement=False,
        stop_reason=None,
    )
    assert continue_decision == {
        "action": "continue",
        "reason": "continue_to_next_round",
        "next_round": 2,
    }

    stop_decision = mod._round_decision(
        current_round=3,
        max_rounds=3,
        stopped_by_no_improvement=False,
        stop_reason=None,
    )
    assert stop_decision["action"] == "stop"
    assert "max_rounds_reached" in stop_decision["reason"]
    assert stop_decision["next_round"] is None


def test_write_round_analysis_creates_json_and_md(tmp_path: Path) -> None:
    mod = _load_autopilot_module()
    payload = {
        "generated_at_utc": "2026-02-16T00:00:00Z",
        "controller_group": "test_controller",
        "round": 2,
        "best_before_round": {
            "run_group": "rg_before",
            "variant_id": "v1",
            "score": 1.0,
            "worst_robust_sharpe": 2.0,
            "worst_dd_pct": 0.2,
            "sample_config_path": "configs/a.yaml",
        },
        "best_after_round": {
            "run_group": "rg_after",
            "variant_id": "v2",
            "score": 1.1,
            "worst_robust_sharpe": 2.1,
            "worst_dd_pct": 0.18,
            "sample_config_path": "configs/b.yaml",
        },
        "delta_score": 0.1,
        "delta_worst_robust_sharpe": 0.1,
        "delta_worst_dd_pct": -0.02,
        "decision": {"action": "continue", "reason": "continue_to_next_round", "next_round": 3},
    }

    mod._write_round_analysis(controller_dir=tmp_path, payload=payload)

    json_path = tmp_path / "round_analysis" / "round_02.json"
    md_path = tmp_path / "round_analysis" / "round_02.md"
    assert json_path.exists()
    assert md_path.exists()

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["round"] == 2
    assert saved["delta_score"] == pytest.approx(0.1)
    assert saved["decision"]["next_round"] == 3

    md_text = md_path.read_text(encoding="utf-8")
    assert "Round 02 analysis" in md_text
    assert "delta_score" in md_text
    assert "continue_to_next_round" in md_text
