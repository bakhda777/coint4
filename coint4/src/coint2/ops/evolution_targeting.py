"""Targeted mutation diagnostics and operator selection for evolution batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

FailureMode = Literal["dd", "trades", "zero_pair", "tail", "balanced", "cold_start"]


@dataclass(frozen=True, slots=True)
class FailureThresholds:
    min_windows: int = 3
    min_trades: float = 200.0
    min_pairs: float = 20.0
    max_dd_pct: float = 0.14
    max_zero_pair_steps_pct: float = 0.2
    max_tail_pair_share: float = 0.45
    max_tail_period_share: float = 0.6


@dataclass(frozen=True, slots=True)
class VariantDiagnostics:
    run_group: str
    variant_id: str
    sample_config_path: str
    windows: int
    worst_robust_sharpe: float
    worst_dd_pct: float
    worst_trades: float
    worst_pairs: float
    worst_zero_pair_steps_pct: float | None
    worst_tail_pair_share: float | None
    worst_tail_period_share: float | None


@dataclass(frozen=True, slots=True)
class FailureAssessment:
    failure_mode: FailureMode
    triggers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TargetedOperatorPlan:
    failure_mode: FailureMode
    operator_kind: str
    budget: int
    params: dict[str, Any]
    hypothesis: str


@dataclass(frozen=True, slots=True)
class _Entry:
    run_id: str
    run_group: str
    config_path: str
    results_dir: str
    status: str
    metrics_present: bool
    sharpe: float | None
    dd_pct: float | None
    trades: float | None
    pairs: float | None
    wf_zero_pair_steps_pct: float | None
    tail_loss_worst_pair_share: float | None
    tail_loss_worst_period_share: float | None


@dataclass(frozen=True, slots=True)
class _WindowMetrics:
    run_group: str
    variant_id: str
    sample_config_path: str
    robust_sharpe: float
    robust_dd_pct: float
    robust_trades: float
    robust_pairs: float
    robust_zero_pair_steps_pct: float | None
    robust_tail_pair_share: float | None
    robust_tail_period_share: float | None


TARGETED_MUTATION_RULES: dict[FailureMode, dict[str, Any]] = {
    "dd": {
        "operator_kind": "mutate_step_v1",
        "keys": [
            "portfolio.risk_per_position_pct",
            "portfolio.max_active_positions",
            "backtest.portfolio_daily_stop_pct",
            "backtest.max_var_multiplier",
            "backtest.pair_stop_loss_usd",
            "backtest.pair_stop_loss_zscore",
            "backtest.stop_loss_multiplier",
            "backtest.time_stop_multiplier",
        ],
        "hypothesis": "DD выше порога: tighten risk и stop-параметры.",
        "budget_multiplier": 6,
        "min_budget": 16,
    },
    "trades": {
        "operator_kind": "mutate_step_v1",
        "keys": [
            "backtest.zscore_entry_threshold",
            "backtest.zscore_exit",
            "backtest.rolling_window",
            "backtest.cooldown_hours",
            "backtest.min_spread_move_sigma",
        ],
        "hypothesis": "Сделок недостаточно: скорректировать сигнальные/частотные knobs.",
        "budget_multiplier": 6,
        "min_budget": 16,
    },
    "zero_pair": {
        "operator_kind": "coordinate_sweep_v1",
        "keys": [
            "pair_selection.max_pairs",
            "pair_selection.min_correlation",
            "pair_selection.coint_pvalue_threshold",
            "pair_selection.lookback_days",
            "pair_selection.ssd_top_n",
            "pair_selection.pvalue_top_n",
            "pair_selection.kpss_pvalue_threshold",
            "pair_selection.max_hurst_exponent",
            "pair_selection.min_mean_crossings",
            "pair_selection.max_half_life_days",
            "pair_selection.enable_pair_tradeability_filter",
            "pair_selection.min_volume_usd_24h",
            "pair_selection.min_days_live",
            "pair_selection.max_funding_rate_abs",
            "pair_selection.max_tick_size_pct",
            "data_processing.min_history_ratio",
            "filter_params.max_hurst_exponent",
            "filter_params.min_mean_crossings",
            "filter_params.max_half_life_days",
        ],
        "hypothesis": "Пустые WF шаги/узкая breadth: смягчить фильтры и расширить пары.",
        "budget_multiplier": 3,
        "min_budget": 12,
        "max_keys": 2,
    },
    "tail": {
        "operator_kind": "mutate_step_v1",
        "keys": [
            "portfolio.max_active_positions",
            "backtest.pair_stop_loss_usd",
            "backtest.portfolio_daily_stop_pct",
            "pair_selection.max_pairs",
        ],
        "hypothesis": "Tail-концентрация завышена: ужесточить tail-guard и концентрацию позиций.",
        "budget_multiplier": 6,
        "min_budget": 16,
    },
    "balanced": {
        "operator_kind": "mutate_step_v1",
        "keys": [
            "portfolio.risk_per_position_pct",
            "backtest.max_var_multiplier",
            "backtest.zscore_entry_threshold",
            "pair_selection.max_pairs",
            "pair_selection.min_correlation",
        ],
        "hypothesis": "Явного провала нет: мягкая локальная мутация по mix риск/сигнал/breadth.",
        "budget_multiplier": 6,
        "min_budget": 16,
    },
    "cold_start": {
        "operator_kind": "random_restart_v1",
        "keys": [
            "portfolio.risk_per_position_pct",
            "backtest.zscore_entry_threshold",
            "pair_selection.max_pairs",
        ],
        "hypothesis": "Истории недостаточно: random restart по core knobs (risk/signal/breadth).",
        "budget_multiplier": 6,
        "min_budget": 16,
    },
}


def build_variant_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    contains: Sequence[str],
    include_noncompleted: bool,
) -> list[VariantDiagnostics]:
    entries: list[_Entry] = []
    for row in rows:
        run_id = str(row.get("run_id") or "").strip()
        run_group = str(row.get("run_group") or "").strip()
        kind, _base_id = _kind_and_base_id(run_id)
        if kind == "stress":
            continue
        meta = " | ".join(
            (
                run_group,
                run_id,
                str(row.get("config_path") or "").strip(),
                str(row.get("results_dir") or "").strip(),
            )
        )
        if contains and not _matches_all(meta, contains):
            continue
        entries.append(
            _Entry(
                run_id=run_id,
                run_group=run_group,
                config_path=str(row.get("config_path") or "").strip(),
                results_dir=str(row.get("results_dir") or "").strip(),
                status=str(row.get("status") or "").strip(),
                metrics_present=_to_bool(row.get("metrics_present")),
                sharpe=_to_float(row.get("sharpe_ratio_abs")),
                dd_pct=_to_float(row.get("max_drawdown_on_equity")),
                trades=_to_float(row.get("total_trades")),
                pairs=_to_float(row.get("total_pairs_traded")),
                wf_zero_pair_steps_pct=_to_float(row.get("wf_zero_pair_steps_pct")),
                tail_loss_worst_pair_share=_to_float(row.get("tail_loss_worst_pair_share")),
                tail_loss_worst_period_share=_to_float(row.get("tail_loss_worst_period_share")),
            )
        )

    by_variant: dict[str, list[_WindowMetrics]] = {}
    for entry in entries:
        kind, base_id = _kind_and_base_id(entry.run_id)
        if kind == "stress":
            continue

        if not include_noncompleted and entry.status.lower() != "completed":
            continue
        if not entry.metrics_present:
            continue
        if entry.sharpe is None or entry.dd_pct is None:
            continue
        if entry.trades is None or entry.pairs is None:
            continue

        variant_id = _variant_id(base_id)
        by_variant.setdefault(variant_id, []).append(
            _WindowMetrics(
                run_group=entry.run_group,
                variant_id=variant_id,
                sample_config_path=entry.config_path,
                robust_sharpe=float(entry.sharpe),
                robust_dd_pct=abs(float(entry.dd_pct)),
                robust_trades=float(entry.trades),
                robust_pairs=float(entry.pairs),
                robust_zero_pair_steps_pct=entry.wf_zero_pair_steps_pct,
                robust_tail_pair_share=entry.tail_loss_worst_pair_share,
                robust_tail_period_share=entry.tail_loss_worst_period_share,
            )
        )

    diagnostics: list[VariantDiagnostics] = []
    for _variant_id_val, items in by_variant.items():
        items_sorted = sorted(items, key=lambda item: item.robust_sharpe, reverse=True)
        worst_robust_sharpe = min(item.robust_sharpe for item in items_sorted)
        worst_dd_pct = max(item.robust_dd_pct for item in items_sorted)
        worst_trades = min(item.robust_trades for item in items_sorted)
        worst_pairs = min(item.robust_pairs for item in items_sorted)
        worst_zero_pair_steps_pct = _reduce_optional_max(item.robust_zero_pair_steps_pct for item in items_sorted)
        worst_tail_pair_share = _reduce_optional_max(item.robust_tail_pair_share for item in items_sorted)
        worst_tail_period_share = _reduce_optional_max(item.robust_tail_period_share for item in items_sorted)

        first = items_sorted[0]
        diagnostics.append(
            VariantDiagnostics(
                run_group=first.run_group,
                variant_id=first.variant_id,
                sample_config_path=first.sample_config_path,
                windows=len(items_sorted),
                worst_robust_sharpe=float(worst_robust_sharpe),
                worst_dd_pct=float(worst_dd_pct),
                worst_trades=float(worst_trades),
                worst_pairs=float(worst_pairs),
                worst_zero_pair_steps_pct=worst_zero_pair_steps_pct,
                worst_tail_pair_share=worst_tail_pair_share,
                worst_tail_period_share=worst_tail_period_share,
            )
        )

    diagnostics.sort(key=lambda item: item.worst_robust_sharpe, reverse=True)
    return diagnostics


def infer_failure_mode(
    diagnostics: VariantDiagnostics | None,
    *,
    thresholds: FailureThresholds,
) -> FailureAssessment:
    if diagnostics is None:
        return FailureAssessment(
            failure_mode="cold_start",
            triggers=("no historical completed holdout rows matched filter",),
        )

    triggers: list[str] = []
    dd_breach = False
    if float(diagnostics.worst_dd_pct) > float(thresholds.max_dd_pct):
        triggers.append(
            f"worst_dd_pct={float(diagnostics.worst_dd_pct):.4f} > max_dd_pct={float(thresholds.max_dd_pct):.4f}"
        )
        dd_breach = True

    zero_pair_breach = False
    if float(diagnostics.worst_pairs) < float(thresholds.min_pairs):
        triggers.append(
            f"worst_pairs={float(diagnostics.worst_pairs):.2f} < min_pairs={float(thresholds.min_pairs):.2f}"
        )
        zero_pair_breach = True
    if diagnostics.worst_zero_pair_steps_pct is not None and float(diagnostics.worst_zero_pair_steps_pct) > float(
        thresholds.max_zero_pair_steps_pct
    ):
        triggers.append(
            "worst_zero_pair_steps_pct="
            f"{float(diagnostics.worst_zero_pair_steps_pct):.4f} > "
            f"max_zero_pair_steps_pct={float(thresholds.max_zero_pair_steps_pct):.4f}"
        )
        zero_pair_breach = True
    if zero_pair_breach:
        return FailureAssessment(failure_mode="zero_pair", triggers=tuple(triggers))

    if float(diagnostics.worst_trades) < float(thresholds.min_trades):
        triggers.append(
            f"worst_trades={float(diagnostics.worst_trades):.2f} < min_trades={float(thresholds.min_trades):.2f}"
        )
        return FailureAssessment(failure_mode="trades", triggers=tuple(triggers))

    tail_triggers: list[str] = []
    if diagnostics.worst_tail_pair_share is not None and float(diagnostics.worst_tail_pair_share) > float(
        thresholds.max_tail_pair_share
    ):
        tail_triggers.append(
            "worst_tail_pair_share="
            f"{float(diagnostics.worst_tail_pair_share):.4f} > "
            f"max_tail_pair_share={float(thresholds.max_tail_pair_share):.4f}"
        )
    if diagnostics.worst_tail_period_share is not None and float(diagnostics.worst_tail_period_share) > float(
        thresholds.max_tail_period_share
    ):
        tail_triggers.append(
            "worst_tail_period_share="
            f"{float(diagnostics.worst_tail_period_share):.4f} > "
            f"max_tail_period_share={float(thresholds.max_tail_period_share):.4f}"
        )
    if tail_triggers:
        return FailureAssessment(failure_mode="tail", triggers=tuple([*triggers, *tail_triggers]))

    if int(diagnostics.windows) < int(thresholds.min_windows):
        triggers.append(f"windows={int(diagnostics.windows)} < min_windows={int(thresholds.min_windows)}")
        return FailureAssessment(failure_mode="cold_start", triggers=tuple(triggers))

    if dd_breach:
        return FailureAssessment(failure_mode="dd", triggers=tuple(triggers))

    return FailureAssessment(failure_mode="balanced", triggers=("all hard checks passed",))


def select_operator_plan(
    failure_mode: FailureMode,
    *,
    variants_count: int,
) -> TargetedOperatorPlan:
    rule = TARGETED_MUTATION_RULES.get(failure_mode, TARGETED_MUTATION_RULES["balanced"])
    budget_multiplier = int(rule.get("budget_multiplier") or 1)
    min_budget = int(rule.get("min_budget") or 1)
    budget = max(int(variants_count) * budget_multiplier, min_budget)
    params: dict[str, Any] = {"keys": list(rule.get("keys") or [])}
    max_keys = rule.get("max_keys")
    if max_keys is not None:
        params["max_keys"] = int(max_keys)
    return TargetedOperatorPlan(
        failure_mode=failure_mode,
        operator_kind=str(rule.get("operator_kind") or "mutate_step_v1"),
        budget=budget,
        params=params,
        hypothesis=str(rule.get("hypothesis") or "").strip(),
    )


def _kind_and_base_id(run_id: str) -> tuple[str | None, str]:
    text = str(run_id or "").strip()
    if text.startswith("holdout_"):
        return "holdout", text[len("holdout_") :]
    if text.startswith("stress_"):
        return "stress", text[len("stress_") :]
    return None, text


def _variant_id(base_id: str) -> str:
    import re

    return re.sub(r"_oos(\d{8})_(\d{8})", "", str(base_id or ""))


def _matches_all(text: str, needles: Iterable[str]) -> bool:
    hay = str(text or "").lower()
    for needle in needles:
        token = str(needle or "").strip().lower()
        if token and token not in hay:
            return False
    return True


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not (out == out and abs(out) != float("inf")):
        return None
    return out


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return max(float(a), float(b))


def _reduce_optional_max(values: Iterable[float | None]) -> float | None:
    outs = [float(value) for value in values if value is not None]
    if not outs:
        return None
    return max(outs)
