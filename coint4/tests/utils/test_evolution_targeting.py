from __future__ import annotations

from coint2.ops.evolution_targeting import (
    FailureThresholds,
    build_variant_diagnostics,
    infer_failure_mode,
    select_operator_plan,
)


def _holdout_rows(
    *,
    run_group: str = "rg",
    variant: str = "v1",
    oos: str = "20220101_20221231",
    holdout_sharpe: float = 1.4,
    holdout_dd: float = -0.08,
    trades: float = 260,
    pairs: float = 32,
    zero_pair: float = 0.0,
    tail_pair: float = 0.2,
    tail_period: float = 0.25,
) -> list[dict[str, str]]:
    base_id = f"{run_group}_{variant}_oos{oos}"
    return [
        {
            "run_id": f"holdout_{base_id}",
            "run_group": run_group,
            "config_path": f"configs/evolution/{run_group}/{variant}.yaml",
            "results_dir": f"artifacts/wfa/runs/{run_group}/holdout_{base_id}",
            "status": "completed",
            "metrics_present": "true",
            "sharpe_ratio_abs": str(holdout_sharpe),
            "max_drawdown_on_equity": str(holdout_dd),
            "total_trades": str(trades),
            "total_pairs_traded": str(pairs),
            "wf_zero_pair_steps_pct": str(zero_pair),
            "tail_loss_worst_pair_share": str(tail_pair),
            "tail_loss_worst_period_share": str(tail_period),
        },
    ]


def test_build_variant_diagnostics_computes_robust_values() -> None:
    rows = _holdout_rows()
    diagnostics = build_variant_diagnostics(rows, contains=["rg"], include_noncompleted=False)
    assert len(diagnostics) == 1
    top = diagnostics[0]
    assert top.variant_id.endswith("_v1")
    assert top.worst_robust_sharpe == 1.4
    assert top.worst_dd_pct == 0.08
    assert top.worst_trades == 260
    assert top.worst_pairs == 32


def test_infer_failure_mode_zero_pair_has_priority() -> None:
    rows = _holdout_rows(pairs=2, zero_pair=0.4)
    diagnostics = build_variant_diagnostics(rows, contains=["rg"], include_noncompleted=False)
    failure = infer_failure_mode(diagnostics[0], thresholds=FailureThresholds())
    assert failure.failure_mode == "zero_pair"


def test_infer_failure_mode_trades_when_low_activity() -> None:
    rows = _holdout_rows(trades=50, pairs=30, zero_pair=0.0)
    diagnostics = build_variant_diagnostics(rows, contains=["rg"], include_noncompleted=False)
    failure = infer_failure_mode(diagnostics[0], thresholds=FailureThresholds())
    assert failure.failure_mode == "trades"


def test_select_operator_plan_uses_rule_budget() -> None:
    plan = select_operator_plan("tail", variants_count=12)
    assert plan.operator_kind == "mutate_step_v1"
    assert plan.budget >= 16
    assert "keys" in plan.params
