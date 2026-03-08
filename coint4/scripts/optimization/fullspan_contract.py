#!/usr/bin/env python3
"""Canonical strict fullspan contract helpers.

Single-source contract used by autonomous orchestrators.
Fail-closed defaults are used for missing/non-finite metrics.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _search_quality_contract import CANONICAL_ZERO_EVIDENCE_REASONS, canonical_zero_evidence_reason


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_OOS_RE = re.compile(r"_oos(\d{8})_(\d{8})")
CONTRACT_NAME = "strict_fullspan_holdout_stress_v1"
PRIMARY_RANKING_KEY = "score_fullspan_v1"
DIAGNOSTIC_RANKING_KEY = "avg_robust_sharpe"
DEFAULT_FULLSPAN_MIN_WINDOWS = 1
DEFAULT_FULLSPAN_MIN_COVERAGE_RATIO = 0.95
DEFAULT_FULLSPAN_STRICT_TOP = 200
DEFAULT_FULLSPAN_RESEARCH_TOP = 10
DEFAULT_FULLSPAN_STRICT_TAIL_WORST_GATE_PCT = 0.20
DEFAULT_FULLSPAN_DIAGNOSTIC_TAIL_WORST_GATE_PCT = 0.21
DEFAULT_FULLSPAN_TAIL_QUANTILE = 0.20
DEFAULT_FULLSPAN_TAIL_Q_SOFT_LOSS_PCT = 0.03
DEFAULT_FULLSPAN_TAIL_WORST_SOFT_LOSS_PCT = 0.10
DEFAULT_FULLSPAN_TAIL_Q_PENALTY = 2.0
DEFAULT_FULLSPAN_TAIL_WORST_PENALTY = 1.0


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


def _to_int(value: Any, default: int) -> int:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return int(default)
        return int(float(text))
    except Exception:
        return int(default)


def fullspan_policy_defaults(*, initial_capital: float = 1000.0) -> dict[str, float | int]:
    capital = float(initial_capital)
    return {
        "min_windows": int(DEFAULT_FULLSPAN_MIN_WINDOWS),
        "min_trades": 200.0,
        "min_pairs": 20.0,
        "max_dd_pct": 0.20,
        "min_pnl": 0.0,
        "initial_capital": capital,
        "max_worst_step_loss_pct": 0.20,
        "min_coverage_ratio": float(DEFAULT_FULLSPAN_MIN_COVERAGE_RATIO),
        "strict_top": int(DEFAULT_FULLSPAN_STRICT_TOP),
        "research_top": int(DEFAULT_FULLSPAN_RESEARCH_TOP),
        "strict_tail_worst_gate_pct": float(DEFAULT_FULLSPAN_STRICT_TAIL_WORST_GATE_PCT),
        "diagnostic_tail_worst_gate_pct": float(DEFAULT_FULLSPAN_DIAGNOSTIC_TAIL_WORST_GATE_PCT),
        "tail_quantile": float(DEFAULT_FULLSPAN_TAIL_QUANTILE),
        "tail_q_soft_loss_pct": float(DEFAULT_FULLSPAN_TAIL_Q_SOFT_LOSS_PCT),
        "tail_worst_soft_loss_pct": float(DEFAULT_FULLSPAN_TAIL_WORST_SOFT_LOSS_PCT),
        "tail_q_penalty": float(DEFAULT_FULLSPAN_TAIL_Q_PENALTY),
        "tail_worst_penalty": float(DEFAULT_FULLSPAN_TAIL_WORST_PENALTY),
    }


def load_fullspan_policy_from_env(
    env: Mapping[str, Any] | None = None,
    *,
    initial_capital: float = 1000.0,
) -> dict[str, float | int]:
    env_map = os.environ if env is None else env
    defaults = fullspan_policy_defaults(initial_capital=initial_capital)
    policy = dict(defaults)
    env_mapping: dict[str, tuple[str, float | int]] = {
        "FULLSPAN_MIN_WINDOWS": ("min_windows", defaults["min_windows"]),
        "FULLSPAN_MIN_TRADES": ("min_trades", defaults["min_trades"]),
        "FULLSPAN_MIN_PAIRS": ("min_pairs", defaults["min_pairs"]),
        "FULLSPAN_MAX_DD_PCT": ("max_dd_pct", defaults["max_dd_pct"]),
        "FULLSPAN_MIN_PNL": ("min_pnl", defaults["min_pnl"]),
        "FULLSPAN_INITIAL_CAPITAL": ("initial_capital", defaults["initial_capital"]),
        "FULLSPAN_MAX_WORST_STEP_LOSS_PCT": ("max_worst_step_loss_pct", defaults["max_worst_step_loss_pct"]),
        "FULLSPAN_MIN_COVERAGE_RATIO": ("min_coverage_ratio", defaults["min_coverage_ratio"]),
        "FULLSPAN_STRICT_TOP": ("strict_top", defaults["strict_top"]),
        "FULLSPAN_RESEARCH_TOP": ("research_top", defaults["research_top"]),
        "FULLSPAN_STRICT_TAIL_WORST_GATE_PCT": (
            "strict_tail_worst_gate_pct",
            defaults["strict_tail_worst_gate_pct"],
        ),
        "FULLSPAN_DIAGNOSTIC_TAIL_WORST_GATE_PCT": (
            "diagnostic_tail_worst_gate_pct",
            defaults["diagnostic_tail_worst_gate_pct"],
        ),
        "FULLSPAN_TAIL_QUANTILE": ("tail_quantile", defaults["tail_quantile"]),
        "FULLSPAN_TAIL_Q_SOFT_LOSS_PCT": ("tail_q_soft_loss_pct", defaults["tail_q_soft_loss_pct"]),
        "FULLSPAN_TAIL_WORST_SOFT_LOSS_PCT": (
            "tail_worst_soft_loss_pct",
            defaults["tail_worst_soft_loss_pct"],
        ),
        "FULLSPAN_TAIL_Q_PENALTY": ("tail_q_penalty", defaults["tail_q_penalty"]),
        "FULLSPAN_TAIL_WORST_PENALTY": ("tail_worst_penalty", defaults["tail_worst_penalty"]),
    }
    int_keys = {"min_windows", "strict_top", "research_top"}
    for env_key, (policy_key, default_value) in env_mapping.items():
        raw_value = env_map.get(env_key)
        if raw_value is None:
            continue
        if policy_key in int_keys:
            policy[policy_key] = _to_int(raw_value, int(default_value))
        else:
            policy[policy_key] = float(_to_float(raw_value, float(default_value)) or float(default_value))
    return policy


def fullspan_thresholds_from_policy(policy: Mapping[str, Any] | None = None) -> FullspanThresholds:
    data = dict(policy or fullspan_policy_defaults())
    return FullspanThresholds(
        min_trades=float(_to_float(data.get("min_trades"), 200.0) or 200.0),
        min_pairs=float(_to_float(data.get("min_pairs"), 20.0) or 20.0),
        max_dd_pct=float(_to_float(data.get("max_dd_pct"), 0.20) or 0.20),
        min_pnl=float(_to_float(data.get("min_pnl"), 0.0) or 0.0),
        initial_capital=float(_to_float(data.get("initial_capital"), 1000.0) or 1000.0),
        max_worst_step_loss_pct=float(
            _to_float(data.get("max_worst_step_loss_pct"), 0.20) or 0.20
        ),
    )


def dominant_rejection_reason(
    raw_reason: str | None,
    *,
    reject_reasons: Mapping[str, Any] | None = None,
) -> str:
    if isinstance(reject_reasons, Mapping) and reject_reasons:
        ranked: list[tuple[int, str]] = []
        for key, value in reject_reasons.items():
            token = str(key or "").strip()
            if not token:
                continue
            ranked.append((_to_int(value, 0), token))
        if ranked:
            ranked.sort(key=lambda item: (-item[0], item[1]))
            return ranked[0][1]
    text = str(raw_reason or "").strip()
    if not text:
        return ""
    match = re.search(r"strict_contract_fail\((.*?)\)", text)
    if match:
        entries = [part.strip() for part in match.group(1).split(",") if part.strip()]
        if entries:
            first = entries[0]
            return first.split(":", 1)[0].strip()
    for token in (
        "coverage_below",
        "min_windows",
        "min_trades",
        "min_pairs",
        "max_dd_pct",
        "min_pnl",
        "TRADES_FAIL",
        "PAIRS_FAIL",
        "DD_FAIL",
        "ECONOMIC_FAIL",
        "STEP_FAIL",
        "INSUFFICIENT_WINDOWS",
        "STATUS_NOT_COMPLETED",
        "HOLDOUT_STRESS_MISSING",
        *CANONICAL_ZERO_EVIDENCE_REASONS,
        "METRICS_MISSING",
    ):
        if token in text:
            return token
    if ":" in text:
        tail = text.rsplit(":", 1)[-1].strip()
        if tail:
            return tail
    return text


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


def _matches_all(text: str, contains: Iterable[str] | None) -> bool:
    needles = [str(item or "").strip().lower() for item in (contains or []) if str(item or "").strip()]
    if not needles:
        return True
    lowered = str(text or "").lower()
    return all(needle in lowered for needle in needles)


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


def _metrics_missing_result(*, metrics_present: bool, reason: str = "METRICS_MISSING") -> RowGateResult:
    return RowGateResult(
        passed=False,
        reason=str(reason or "METRICS_MISSING"),
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

    zero_evidence_reason = canonical_zero_evidence_reason(row, require_metrics_present=True)
    if zero_evidence_reason in CANONICAL_ZERO_EVIDENCE_REASONS:
        return _metrics_missing_result(metrics_present=True, reason=zero_evidence_reason)

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


def load_run_index_rows(path: Path) -> list[dict[str, str]]:
    return _load_run_index(Path(path))


def discover_variant_candidates(
    *,
    run_index_path: Path | None = None,
    run_index_rows: Iterable[Mapping[str, Any]] | None = None,
    contains: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if run_index_rows is None:
        if run_index_path is None:
            raise ValueError("run_index_path is required when run_index_rows is not provided")
        rows: list[Mapping[str, Any]] = _load_run_index(Path(run_index_path))
    else:
        rows = list(run_index_rows)

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        run_group = str(row.get("run_group") or "").strip()
        run_id = str(row.get("run_id") or "").strip()
        if not run_group or not run_id:
            continue
        kind = _run_kind(run_id)
        if kind not in {"holdout", "stress"}:
            continue
        base_id = canonical_base_id(run_id)
        variant_id = variant_id_from_base(base_id)
        if not variant_id:
            continue

        key = (run_group, variant_id)
        candidate = grouped.setdefault(
            key,
            {
                "run_group": run_group,
                "variant_id": variant_id,
                "sample_config": "",
                "holdout_windows": set(),
                "stress_windows": set(),
                "meta": [],
                "row_count": 0,
            },
        )
        candidate["row_count"] = int(candidate["row_count"]) + 1
        if kind == "holdout":
            candidate["holdout_windows"].add(base_id)
        else:
            candidate["stress_windows"].add(base_id)
        config_path = str(row.get("config_path") or "").strip()
        if config_path and not candidate["sample_config"]:
            candidate["sample_config"] = config_path
        candidate["meta"].append(
            " | ".join(
                part
                for part in (
                    run_group,
                    variant_id,
                    base_id,
                    run_id,
                    config_path,
                    str(row.get("results_dir") or "").strip(),
                    str(row.get("status") or "").strip(),
                )
                if part
            )
        )

    candidates: list[dict[str, Any]] = []
    for candidate in grouped.values():
        if not _matches_all(" || ".join(candidate["meta"]), contains):
            continue
        holdout_windows = sorted(candidate["holdout_windows"])
        stress_windows = sorted(candidate["stress_windows"])
        paired = sorted(set(holdout_windows) & set(stress_windows))
        candidates.append(
            {
                "run_group": str(candidate["run_group"]),
                "variant_id": str(candidate["variant_id"]),
                "sample_config": str(candidate["sample_config"]),
                "row_count": int(candidate["row_count"]),
                "holdout_window_count": len(holdout_windows),
                "stress_window_count": len(stress_windows),
                "paired_window_count": len(paired),
            }
        )

    candidates.sort(key=lambda item: (str(item["run_group"]), str(item["variant_id"])))
    return candidates


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
    run_index_rows: Iterable[Mapping[str, Any]] | None = None,
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

    if run_index_rows is None:
        rows: list[Mapping[str, Any]] = _load_run_index(Path(run_index_path))
    else:
        rows = list(run_index_rows)
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
