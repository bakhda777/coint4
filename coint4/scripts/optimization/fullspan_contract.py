#!/usr/bin/env python3
"""Canonical strict fullspan contract helpers.

Single-source contract used by autonomous orchestrators.
Fail-closed defaults are used for missing/non-finite metrics.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_OOS_RE = re.compile(r"_oos(\d{8})_(\d{8})")
CONTRACT_NAME = "strict_fullspan_holdout_stress_v1"
PRIMARY_RANKING_KEY = "score_fullspan_v1"
DIAGNOSTIC_RANKING_KEY = "avg_robust_sharpe"


@dataclass(frozen=True)
class FullspanThresholds:
    min_trades: float = 200.0
    min_pairs: float = 20.0
    max_dd_pct: float = 0.20
    min_pnl: float = 0.0
    initial_capital: float = 1000.0
    max_worst_step_loss_pct: float = 0.20


@dataclass(frozen=True)
class RowGateResult:
    passed: bool
    reason: str
    metrics_present: bool
    total_trades: float
    total_pairs_traded: float
    worst_dd_pct: float
    worst_robust_pnl: float
    worst_step_pnl: float


@dataclass(frozen=True)
class VariantContractResult:
    passed: bool
    reason: str
    run_group: str
    variant_id: str
    windows_total: int
    windows_passed: int
    windows_required: int
    score_fullspan_v1: float | None
    avg_robust_sharpe: float | None
    worst_robust_sharpe: float | None
    worst_dd_pct: float | None
    worst_robust_pnl: float | None
    worst_step_pnl: float | None
    q_step_pnl: float | None
    min_total_trades: float | None
    min_total_pairs_traded: float | None
    sample_config: str


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return default
        out = float(text)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _to_bool(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() in _TRUE_VALUES


def _quantile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(v for v in values if v is not None and math.isfinite(float(v)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    q = min(1.0, max(0.0, float(q)))
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * weight)


def canonical_base_id(run_id: str) -> str:
    text = str(run_id or "").strip()
    if text.startswith("holdout_"):
        return text[len("holdout_") :]
    if text.startswith("stress_"):
        return text[len("stress_") :]
    return text


def variant_id_from_base(base_id: str) -> str:
    return _OOS_RE.sub("", str(base_id or "").strip())


def row_worst_step_pnl(row: dict[str, Any]) -> float | None:
    period = _to_float(row.get("tail_loss_worst_period_pnl"), None)
    pair = _to_float(row.get("tail_loss_worst_pair_pnl"), None)
    if period is None and pair is None:
        return None
    if period is None:
        return pair
    if pair is None:
        return period
    return min(float(period), float(pair))


def _metrics_missing_result(*, metrics_present: bool) -> RowGateResult:
    return RowGateResult(
        passed=False,
        reason="METRICS_MISSING",
        metrics_present=bool(metrics_present),
        total_trades=0.0,
        total_pairs_traded=0.0,
        worst_dd_pct=1.0,
        worst_robust_pnl=-float("inf"),
        worst_step_pnl=-float("inf"),
    )


def evaluate_row_hard_gates(row: dict[str, Any] | None, thresholds: FullspanThresholds) -> RowGateResult:
    if not row:
        return _metrics_missing_result(metrics_present=False)

    metrics_present = _to_bool(row.get("metrics_present"))
    if not metrics_present:
        return _metrics_missing_result(metrics_present=False)

    total_trades_raw = _to_float(row.get("total_trades"), None)
    total_pairs_raw = _to_float(row.get("total_pairs_traded"), None)
    worst_dd_raw = _to_float(row.get("max_drawdown_on_equity"), None)
    robust_pnl_raw = _to_float(row.get("total_pnl"), None)
    step_pnl = row_worst_step_pnl(row)
    if (
        total_trades_raw is None
        or total_pairs_raw is None
        or worst_dd_raw is None
        or robust_pnl_raw is None
        or step_pnl is None
    ):
        return _metrics_missing_result(metrics_present=True)

    total_trades = float(total_trades_raw)
    total_pairs = float(total_pairs_raw)
    worst_dd = abs(float(worst_dd_raw))
    robust_pnl = float(robust_pnl_raw)
    step_pnl = float(step_pnl)

    if total_trades < float(thresholds.min_trades):
        return RowGateResult(False, "TRADES_FAIL", True, total_trades, total_pairs, worst_dd, robust_pnl, step_pnl)
    if total_pairs < float(thresholds.min_pairs):
        return RowGateResult(False, "PAIRS_FAIL", True, total_trades, total_pairs, worst_dd, robust_pnl, step_pnl)
    if worst_dd > float(thresholds.max_dd_pct):
        return RowGateResult(False, "DD_FAIL", True, total_trades, total_pairs, worst_dd, robust_pnl, step_pnl)
    if robust_pnl < float(thresholds.min_pnl):
        return RowGateResult(False, "ECONOMIC_FAIL", True, total_trades, total_pairs, worst_dd, robust_pnl, step_pnl)
    if step_pnl < (-float(thresholds.max_worst_step_loss_pct) * float(thresholds.initial_capital)):
        return RowGateResult(False, "STEP_FAIL", True, total_trades, total_pairs, worst_dd, robust_pnl, step_pnl)
    return RowGateResult(True, "PASS", True, total_trades, total_pairs, worst_dd, robust_pnl, step_pnl)


def score_fullspan_v1(
    *,
    worst_robust_sharpe: float,
    q_step_pnl: float,
    worst_step_pnl: float,
    initial_capital: float,
    tail_q_soft_loss_pct: float = 0.03,
    tail_worst_soft_loss_pct: float = 0.10,
    tail_q_penalty: float = 2.0,
    tail_worst_penalty: float = 1.0,
) -> float:
    q_loss = max(0.0, (-float(q_step_pnl) / float(initial_capital)) - float(tail_q_soft_loss_pct))
    worst_loss = max(0.0, (-float(worst_step_pnl) / float(initial_capital)) - float(tail_worst_soft_loss_pct))
    return float(worst_robust_sharpe) - float(tail_q_penalty) * q_loss - float(tail_worst_penalty) * worst_loss


def _run_kind(run_id: str) -> str | None:
    text = str(run_id or "").strip()
    if text.startswith("holdout_"):
        return "holdout"
    if text.startswith("stress_"):
        return "stress"
    return None


def _load_run_index(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def evaluate_variant_contract(
    *,
    run_index_path: Path,
    run_group: str,
    variant_id: str,
    thresholds: FullspanThresholds,
    min_windows: int = 1,
    tail_quantile: float = 0.20,
    tail_q_soft_loss_pct: float = 0.03,
    tail_worst_soft_loss_pct: float = 0.10,
    tail_q_penalty: float = 2.0,
    tail_worst_penalty: float = 1.0,
) -> VariantContractResult:
    rg = str(run_group or "").strip()
    variant = str(variant_id or "").strip()
    if not rg or not variant:
        return VariantContractResult(
            passed=False,
            reason="METRICS_MISSING",
            run_group=rg,
            variant_id=variant,
            windows_total=0,
            windows_passed=0,
            windows_required=int(min_windows),
            score_fullspan_v1=None,
            avg_robust_sharpe=None,
            worst_robust_sharpe=None,
            worst_dd_pct=None,
            worst_robust_pnl=None,
            worst_step_pnl=None,
            q_step_pnl=None,
            min_total_trades=None,
            min_total_pairs_traded=None,
            sample_config="",
        )

    rows = _load_run_index(Path(run_index_path))
    by_base: dict[str, dict[str, dict[str, Any] | None]] = {}
    sample_cfg = ""

    for row in rows:
        if str(row.get("run_group") or "").strip() != rg:
            continue
        run_id = str(row.get("run_id") or "").strip()
        kind = _run_kind(run_id)
        if kind not in {"holdout", "stress"}:
            continue
        base_id = canonical_base_id(run_id)
        if variant_id_from_base(base_id) != variant:
            continue
        bundle = by_base.setdefault(base_id, {"holdout": None, "stress": None})
        bundle[kind] = row
        if not sample_cfg:
            sample_cfg = str(row.get("config_path") or "").strip()

    if not by_base:
        return VariantContractResult(
            passed=False,
            reason="METRICS_MISSING",
            run_group=rg,
            variant_id=variant,
            windows_total=0,
            windows_passed=0,
            windows_required=int(min_windows),
            score_fullspan_v1=None,
            avg_robust_sharpe=None,
            worst_robust_sharpe=None,
            worst_dd_pct=None,
            worst_robust_pnl=None,
            worst_step_pnl=None,
            q_step_pnl=None,
            min_total_trades=None,
            min_total_pairs_traded=None,
            sample_config=sample_cfg,
        )

    robust_sharpes: list[float] = []
    robust_dds: list[float] = []
    robust_pnls: list[float] = []
    robust_steps: list[float] = []
    robust_trades: list[float] = []
    robust_pairs: list[float] = []
    windows_passed = 0

    for base_id in sorted(by_base):
        bundle = by_base[base_id]
        holdout = bundle.get("holdout")
        stress = bundle.get("stress")
        if holdout is None or stress is None:
            return VariantContractResult(
                passed=False,
                reason="HOLDOUT_STRESS_MISSING",
                run_group=rg,
                variant_id=variant,
                windows_total=len(by_base),
                windows_passed=windows_passed,
                windows_required=int(min_windows),
                score_fullspan_v1=None,
                avg_robust_sharpe=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                worst_robust_pnl=None,
                worst_step_pnl=None,
                q_step_pnl=None,
                min_total_trades=None,
                min_total_pairs_traded=None,
                sample_config=sample_cfg,
            )

        if str(holdout.get("status") or "").strip().lower() != "completed" or str(stress.get("status") or "").strip().lower() != "completed":
            return VariantContractResult(
                passed=False,
                reason="STATUS_NOT_COMPLETED",
                run_group=rg,
                variant_id=variant,
                windows_total=len(by_base),
                windows_passed=windows_passed,
                windows_required=int(min_windows),
                score_fullspan_v1=None,
                avg_robust_sharpe=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                worst_robust_pnl=None,
                worst_step_pnl=None,
                q_step_pnl=None,
                min_total_trades=None,
                min_total_pairs_traded=None,
                sample_config=sample_cfg,
            )

        hold_gate = evaluate_row_hard_gates(holdout, thresholds)
        if not hold_gate.passed:
            return VariantContractResult(
                passed=False,
                reason=hold_gate.reason,
                run_group=rg,
                variant_id=variant,
                windows_total=len(by_base),
                windows_passed=windows_passed,
                windows_required=int(min_windows),
                score_fullspan_v1=None,
                avg_robust_sharpe=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                worst_robust_pnl=None,
                worst_step_pnl=None,
                q_step_pnl=None,
                min_total_trades=None,
                min_total_pairs_traded=None,
                sample_config=sample_cfg,
            )
        stress_gate = evaluate_row_hard_gates(stress, thresholds)
        if not stress_gate.passed:
            return VariantContractResult(
                passed=False,
                reason=stress_gate.reason,
                run_group=rg,
                variant_id=variant,
                windows_total=len(by_base),
                windows_passed=windows_passed,
                windows_required=int(min_windows),
                score_fullspan_v1=None,
                avg_robust_sharpe=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                worst_robust_pnl=None,
                worst_step_pnl=None,
                q_step_pnl=None,
                min_total_trades=None,
                min_total_pairs_traded=None,
                sample_config=sample_cfg,
            )

        h_sharpe = _to_float(holdout.get("sharpe_ratio_abs"), None)
        s_sharpe = _to_float(stress.get("sharpe_ratio_abs"), None)
        if h_sharpe is None or s_sharpe is None:
            return VariantContractResult(
                passed=False,
                reason="METRICS_MISSING",
                run_group=rg,
                variant_id=variant,
                windows_total=len(by_base),
                windows_passed=windows_passed,
                windows_required=int(min_windows),
                score_fullspan_v1=None,
                avg_robust_sharpe=None,
                worst_robust_sharpe=None,
                worst_dd_pct=None,
                worst_robust_pnl=None,
                worst_step_pnl=None,
                q_step_pnl=None,
                min_total_trades=None,
                min_total_pairs_traded=None,
                sample_config=sample_cfg,
            )

        robust_sharpes.append(min(float(h_sharpe), float(s_sharpe)))
        robust_dds.append(max(float(hold_gate.worst_dd_pct), float(stress_gate.worst_dd_pct)))
        robust_pnls.append(min(float(hold_gate.worst_robust_pnl), float(stress_gate.worst_robust_pnl)))
        robust_steps.append(min(float(hold_gate.worst_step_pnl), float(stress_gate.worst_step_pnl)))
        robust_trades.append(min(float(hold_gate.total_trades), float(stress_gate.total_trades)))
        robust_pairs.append(min(float(hold_gate.total_pairs_traded), float(stress_gate.total_pairs_traded)))
        windows_passed += 1

    if windows_passed < int(min_windows):
        return VariantContractResult(
            passed=False,
            reason="INSUFFICIENT_WINDOWS",
            run_group=rg,
            variant_id=variant,
            windows_total=len(by_base),
            windows_passed=windows_passed,
            windows_required=int(min_windows),
            score_fullspan_v1=None,
            avg_robust_sharpe=None,
            worst_robust_sharpe=None,
            worst_dd_pct=None,
            worst_robust_pnl=None,
            worst_step_pnl=None,
            q_step_pnl=None,
            min_total_trades=min(robust_trades) if robust_trades else None,
            min_total_pairs_traded=min(robust_pairs) if robust_pairs else None,
            sample_config=sample_cfg,
        )

    min_trades = min(robust_trades)
    min_pairs = min(robust_pairs)
    worst_dd = max(robust_dds)
    worst_pnl = min(robust_pnls)
    worst_step = min(robust_steps)

    if min_trades < float(thresholds.min_trades):
        reason = "TRADES_FAIL"
        ok = False
    elif min_pairs < float(thresholds.min_pairs):
        reason = "PAIRS_FAIL"
        ok = False
    elif worst_dd > float(thresholds.max_dd_pct):
        reason = "DD_FAIL"
        ok = False
    elif worst_pnl < float(thresholds.min_pnl):
        reason = "ECONOMIC_FAIL"
        ok = False
    elif worst_step < (-float(thresholds.max_worst_step_loss_pct) * float(thresholds.initial_capital)):
        reason = "STEP_FAIL"
        ok = False
    else:
        reason = "PASS"
        ok = True

    q_step = _quantile(robust_steps, float(tail_quantile))
    if q_step is None:
        q_step = worst_step

    worst_sharpe = min(robust_sharpes)
    avg_sharpe = sum(robust_sharpes) / len(robust_sharpes)

    score = None
    if ok:
        score = score_fullspan_v1(
            worst_robust_sharpe=float(worst_sharpe),
            q_step_pnl=float(q_step),
            worst_step_pnl=float(worst_step),
            initial_capital=float(thresholds.initial_capital),
            tail_q_soft_loss_pct=float(tail_q_soft_loss_pct),
            tail_worst_soft_loss_pct=float(tail_worst_soft_loss_pct),
            tail_q_penalty=float(tail_q_penalty),
            tail_worst_penalty=float(tail_worst_penalty),
        )

    return VariantContractResult(
        passed=bool(ok),
        reason=reason,
        run_group=rg,
        variant_id=variant,
        windows_total=len(by_base),
        windows_passed=windows_passed,
        windows_required=int(min_windows),
        score_fullspan_v1=float(score) if score is not None else None,
        avg_robust_sharpe=float(avg_sharpe),
        worst_robust_sharpe=float(worst_sharpe),
        worst_dd_pct=float(worst_dd),
        worst_robust_pnl=float(worst_pnl),
        worst_step_pnl=float(worst_step),
        q_step_pnl=float(q_step),
        min_total_trades=float(min_trades),
        min_total_pairs_traded=float(min_pairs),
        sample_config=sample_cfg,
    )
