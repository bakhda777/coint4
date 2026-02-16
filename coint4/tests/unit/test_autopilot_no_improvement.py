import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


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


def test_branch_candidate_offsets_local_refine_and_fallback() -> None:
    mod = _load_autopilot_module()
    assert mod._branch_candidate_offsets(candidates=[-4, -2, 0, 2, 4], branch="local_refine") == [-2, 0, 2]
    assert mod._branch_candidate_offsets(candidates=[-4, -2, 0, 2, 4], branch="fallback_knob") == [-4, -2, 0, 2, 4]


def test_derive_next_queue_plan_local_refine_when_previous_improved() -> None:
    mod = _load_autopilot_module()
    plan = mod._derive_next_queue_plan(
        next_round=3,
        knobs_count=4,
        previous_result={
            "round": 2,
            "improved": True,
            "knob_index": 2,
        },
    )
    assert plan["round"] == 3
    assert plan["branch"] == "local_refine"
    assert plan["knob_index"] == 2


def test_derive_next_queue_plan_fallback_when_previous_not_improved() -> None:
    mod = _load_autopilot_module()
    plan = mod._derive_next_queue_plan(
        next_round=3,
        knobs_count=4,
        previous_result={
            "round": 2,
            "improved": False,
            "knob_index": 2,
        },
    )
    assert plan["round"] == 3
    assert plan["branch"] == "fallback_knob"
    assert plan["knob_index"] != 2


def test_is_valid_config_combo_rejects_invalid_pairs() -> None:
    mod = _load_autopilot_module()
    cfg = {
        "walk_forward": {"start_date": "2024-05-01", "end_date": "2023-04-30"},
        "portfolio": {"risk_per_position_pct": 0.015},
        "backtest": {
            "zscore_entry_threshold": 0.8,
            "zscore_exit": 0.9,
            "max_var_multiplier": 1.01,
            "pair_stop_loss_usd": 4.0,
            "portfolio_daily_stop_pct": 0.02,
        },
    }
    assert mod._is_valid_config_combo(cfg) is False

    cfg["walk_forward"] = {"start_date": "2023-04-30", "end_date": "2024-05-01"}
    cfg["backtest"]["zscore_exit"] = 0.05
    assert mod._is_valid_config_combo(cfg) is True


def test_collect_seen_config_signatures_uses_run_group_prefix(tmp_path: Path) -> None:
    mod = _load_autopilot_module()
    app_root = tmp_path / "app"
    queue_dir = app_root / "artifacts" / "wfa" / "aggregate" / "grp_r01_risk"
    cfg_dir = app_root / "configs" / "x"
    queue_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    cfg_a = {"walk_forward": {"start_date": "2023-01-01", "end_date": "2023-12-31"}, "backtest": {"zscore_exit": 0.05}}
    cfg_b = {"walk_forward": {"start_date": "2023-01-01", "end_date": "2023-12-31"}, "backtest": {"zscore_exit": 0.07}}
    (cfg_dir / "a.yaml").write_text(yaml.safe_dump(cfg_a), encoding="utf-8")
    (cfg_dir / "b.yaml").write_text(yaml.safe_dump(cfg_b), encoding="utf-8")

    queue_path = queue_dir / "run_queue.csv"
    queue_path.write_text(
        "config_path,results_dir,status\n"
        "configs/x/a.yaml,artifacts/wfa/runs_clean/grp/a,planned\n"
        "configs/x/b.yaml,artifacts/wfa/runs_clean/grp/b,planned\n",
        encoding="utf-8",
    )

    seen = mod._collect_seen_config_signatures(
        app_root=app_root,
        queue_dir_rel="artifacts/wfa/aggregate",
        run_group_prefix="grp",
    )
    assert len(seen) == 2
