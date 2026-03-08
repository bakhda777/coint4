from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_script_module(tmp_name: str, script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / script_name
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(tmp_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[tmp_name] = module
    spec.loader.exec_module(module)
    return module


def test_fullspan_contract_defaults_match_hard_gates() -> None:
    module = _load_script_module("fullspan_contract_regression_guard", "fullspan_contract.py")
    thresholds = module.FullspanThresholds()
    policy = module.fullspan_policy_defaults()

    assert thresholds.min_trades == 200.0
    assert thresholds.min_pairs == 20.0
    assert thresholds.max_dd_pct == 0.20
    assert thresholds.min_pnl == 0.0
    assert thresholds.initial_capital == 1000.0
    assert thresholds.max_worst_step_loss_pct == 0.20
    assert policy["min_windows"] == 1
    assert policy["min_coverage_ratio"] == 0.95
    assert module.PRIMARY_RANKING_KEY == "score_fullspan_v1"
    assert module.DIAGNOSTIC_RANKING_KEY == "avg_robust_sharpe"
    assert module.CONTRACT_NAME == "strict_fullspan_holdout_stress_v1"


def _strict_gate_row_base() -> dict[str, str]:
    return {
        "metrics_present": "true",
        "total_trades": "200",
        "total_pairs_traded": "20",
        "max_drawdown_on_equity": "0.20",
        "total_pnl": "0",
        "tail_loss_worst_period_pnl": "-200",
    }


@pytest.mark.parametrize(
    ("patch", "expected_reason"),
    [
        ({"metrics_present": "false"}, "METRICS_MISSING"),
        ({"observed_test_days": "0"}, "ZERO_OBSERVED_TEST_DAYS"),
        ({"coverage_ratio": "0"}, "ZERO_COVERAGE"),
        ({"total_trades": "0"}, "ZERO_TRADES"),
        ({"total_pairs_traded": "0"}, "ZERO_PAIRS"),
        ({"total_trades": "199"}, "TRADES_FAIL"),
        ({"total_pairs_traded": "19"}, "PAIRS_FAIL"),
        ({"max_drawdown_on_equity": "0.21"}, "DD_FAIL"),
        ({"total_pnl": "-0.01"}, "ECONOMIC_FAIL"),
        ({"tail_loss_worst_period_pnl": "-201"}, "STEP_FAIL"),
    ],
)
def test_fullspan_contract_row_gates_hard_thresholds(patch: dict[str, str], expected_reason: str) -> None:
    module = _load_script_module("fullspan_contract_row_guard", "fullspan_contract.py")
    thresholds = module.FullspanThresholds()
    row = _strict_gate_row_base()
    row.update(patch)

    result = module.evaluate_row_hard_gates(row, thresholds)
    assert result.passed is False
    assert result.reason == expected_reason


@pytest.mark.parametrize(
    "missing_key",
    [
        "total_trades",
        "total_pairs_traded",
        "max_drawdown_on_equity",
        "total_pnl",
        "tail_loss_worst_period_pnl",
    ],
)
def test_fullspan_contract_row_gates_missing_required_metrics_fail_closed(missing_key: str) -> None:
    module = _load_script_module("fullspan_contract_row_missing_guard", "fullspan_contract.py")
    thresholds = module.FullspanThresholds()

    row = _strict_gate_row_base()
    row.pop(missing_key, None)

    result = module.evaluate_row_hard_gates(row, thresholds)
    assert result.passed is False
    assert result.reason == "METRICS_MISSING"


def test_fullspan_contract_row_gates_boundary_values_pass() -> None:
    module = _load_script_module("fullspan_contract_row_boundary_guard", "fullspan_contract.py")
    thresholds = module.FullspanThresholds()

    result = module.evaluate_row_hard_gates(_strict_gate_row_base(), thresholds)
    assert result.passed is True
    assert result.reason == "PASS"


def test_fullspan_cycle_summary_primary_key_is_locked() -> None:
    contract_module = _load_script_module("fullspan_contract_primary_key_guard", "fullspan_contract.py")
    module = _load_script_module("fullspan_decision_cycle_guard", "run_fullspan_decision_cycle.py")
    args = SimpleNamespace(queue=["artifacts/wfa/aggregate/demo/run_queue.csv"], contains=["demo"])
    strict_override = {
        "pass_count": 1,
        "run_group_count": 1,
        "run_groups": ["demo_group"],
        "rows": [],
        "top_run_group": "demo_group",
        "top_variant": "variant_demo",
        "top_config": "configs/demo.yaml",
        "top_score": "1.0",
        "rejection_reason_line": "",
        "exit_code": 0,
        "status": "pass",
    }
    summary = module._build_summary(
        args,
        strict_rc=0,
        strict_out="",
        diag_rc=1,
        diag_out="",
        diag_skipped_reason="strict_pass_fastlane",
        strict_summary_override=strict_override,
    )

    assert summary["winner_contract"] == contract_module.CONTRACT_NAME
    assert summary["primary_ranking_key"] == contract_module.PRIMARY_RANKING_KEY
    assert summary["diagnostic_ranking_key"] == contract_module.DIAGNOSTIC_RANKING_KEY
    assert summary["strict"]["ranking_primary_key"] == contract_module.PRIMARY_RANKING_KEY
    assert summary["strict"]["diagnostic_key"] == contract_module.DIAGNOSTIC_RANKING_KEY


def test_gatekeeper_and_diversity_guard_keep_fail_closed_contract() -> None:
    root = Path(__file__).resolve().parents[2] / "scripts" / "optimization"
    gatekeeper_src = (root / "promotion_gatekeeper_agent.py").read_text(encoding="utf-8")
    diversity_src = (root / "confirm_diversity_guard_agent.py").read_text(encoding="utf-8")

    required_gatekeeper_snippets = [
        "strict_pass_count > 0",
        "strict_run_groups >= cfg.min_groups",
        "confirm_count >= cfg.min_replays",
        "contract_pass",
        'updated["ranking_primary_key"] = "score_fullspan_v1"',
    ]
    for snippet in required_gatekeeper_snippets:
        assert snippet in gatekeeper_src

    assert '"ranking_primary_key"] = "avg_robust_sharpe"' not in gatekeeper_src
    assert "confirm_independent_lineage_count" in diversity_src
    assert "confirm_non_independent_lineage" in diversity_src


def test_dominant_rejection_reason_prefers_counted_contract_reason() -> None:
    module = _load_script_module("fullspan_contract_reject_reason_guard", "fullspan_contract.py")

    reason = module.dominant_rejection_reason(
        "strict_contract_fail(TRADES_FAIL:1,PAIRS_FAIL:2)",
        reject_reasons={"TRADES_FAIL": 1, "PAIRS_FAIL": 2},
    )

    assert reason == "PAIRS_FAIL"


def test_dominant_rejection_reason_preserves_strict_diag_tokens() -> None:
    module = _load_script_module("fullspan_contract_diag_reason_guard", "fullspan_contract.py")

    assert module.dominant_rejection_reason("strict_hard_fail: coverage_below") == "coverage_below"
    assert module.dominant_rejection_reason("RANK_OK_FALLBACK_STRICT_BINDING:min_windows,min_pairs") == "min_windows"


def test_dominant_rejection_reason_preserves_canonical_zero_evidence_tokens() -> None:
    module = _load_script_module("fullspan_contract_zero_reason_guard", "fullspan_contract.py")

    assert module.dominant_rejection_reason("strict_contract_fail(ZERO_COVERAGE:2,TRADES_FAIL:1)") == "ZERO_COVERAGE"


def test_discover_variant_candidates_uses_run_index_pairs_directly() -> None:
    module = _load_script_module("fullspan_contract_candidate_guard", "fullspan_contract.py")

    candidates = module.discover_variant_candidates(
        run_index_rows=[
            {
                "run_group": "demo_group",
                "run_id": "holdout_variant_alpha_oos20240101_20240331",
                "config_path": "configs/alpha.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_alpha",
                "status": "completed",
            },
            {
                "run_group": "demo_group",
                "run_id": "stress_variant_alpha_oos20240101_20240331",
                "config_path": "configs/alpha.yaml",
                "results_dir": "artifacts/wfa/runs/demo/stress_alpha",
                "status": "completed",
            },
            {
                "run_group": "demo_group",
                "run_id": "holdout_variant_beta_oos20240101_20240331",
                "config_path": "configs/beta.yaml",
                "results_dir": "artifacts/wfa/runs/demo/holdout_beta",
                "status": "completed",
            },
        ],
        contains=["demo_group"],
    )

    assert candidates == [
        {
            "run_group": "demo_group",
            "variant_id": "variant_alpha",
            "sample_config": "configs/alpha.yaml",
            "row_count": 2,
            "holdout_window_count": 1,
            "stress_window_count": 1,
            "paired_window_count": 1,
        },
        {
            "run_group": "demo_group",
            "variant_id": "variant_beta",
            "sample_config": "configs/beta.yaml",
            "row_count": 1,
            "holdout_window_count": 1,
            "stress_window_count": 0,
            "paired_window_count": 0,
        },
    ]
