#!/usr/bin/env python3
"""Rank WFA runs across multiple OOS windows (holdout-only).

This script consumes the rollup run index and selects holdout runs (run_id
starts with `holdout_`, or has no kind prefix) within the same run_group.

It then aggregates *across OOS windows* for the same variant. OOS windows are
detected by the filename tag pattern: `_oosYYYYMMDD_YYYYMMDD`.

Default objective:
  robust_sharpe_window = holdout_sharpe
  score = min_window(robust_sharpe_window)

Alternative objectives (`--score-mode`):
  - worst:     score = min_window(robust_sharpe_window)
  - quantile:  score = quantile(robust_sharpe_window, q), where q=`--quantile-q`
  - hybrid:    score = w_worst * worst + w_q * q_score + w_avg * avg
               with weights from `--hybrid-*-weight` (normalized to sum=1)

Evaluator protocol v2 (`--evaluator-protocol v2`):
  - primary rank: Pareto front over objective vector
    [worst_robust_sh, q_robust_sh, avg_robust_sh, worst_dd_pct(min), worst_pnl]
  - tie-break inside front: weighted utility decomposition (normalized objective terms)
  - emitted in table columns: pareto_front, dominated_by, decomposition

Optional fullspan contract (`--fullspan-policy-v1`):
  score_fullspan_v1 = worst_robust_sharpe
      - tail_q_penalty * max(0, (-q_tail / initial_capital) - tail_q_soft_loss_pct)
      - tail_worst_penalty * max(0, (-worst_tail / initial_capital) - tail_worst_soft_loss_pct)
where `q_tail` / `worst_tail` are computed from holdout daily PnL.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_OOS_RE = re.compile(r"_oos(\d{8})_(\d{8})")


@dataclass(frozen=True)
class _Entry:
    run_id: str
    run_group: str
    config_path: str
    results_dir: str
    status: str
    metrics_present: bool
    sharpe: Optional[float]
    pnl: Optional[float]
    dd_abs: Optional[float]
    dd_pct: Optional[float]
    psr: Optional[float]
    dsr: Optional[float]
    trades: Optional[float]
    pairs: Optional[float]
    costs: Optional[float]
    coverage_ratio: Optional[float]
    tail_loss_worst_pair_share: Optional[float]
    tail_loss_worst_period_share: Optional[float]


@dataclass(frozen=True)
class _WindowStats:
    window: str
    robust_sharpe: float
    dd_pct: float
    robust_coverage_ratio: Optional[float]
    robust_psr: Optional[float]
    robust_dsr: Optional[float]
    robust_pnl: Optional[float]
    robust_tail_pair_share: Optional[float]
    robust_tail_period_share: Optional[float]
    tail_samples: Tuple[float, ...]
    holdout: _Entry


@dataclass(frozen=True)
class _RankRow:
    score: float
    score_mode: str
    worst_robust: float
    q_robust: float
    worst_dd: float
    avg_robust: float
    avg_dd: float
    worst_psr: Optional[float]
    worst_dsr: Optional[float]
    windows: int
    variant: str
    sample_cfg: str
    run_group: str
    worst_pnl: Optional[float]
    avg_pnl: Optional[float]
    worst_tail_pair_share: Optional[float]
    worst_tail_period_share: Optional[float]
    worst_step_pnl: Optional[float]
    q_step_pnl: Optional[float]
    pareto_front: Optional[int] = None
    dominated_by: Optional[int] = None
    decomposition: Optional[str] = None


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: str) -> Optional[float]:
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _kind_and_base_id(run_id: str) -> Tuple[Optional[str], str]:
    if run_id.startswith("holdout_"):
        return "holdout", run_id[len("holdout_") :]
    if run_id.startswith("stress_"):
        return "stress", run_id[len("stress_") :]
    return None, run_id


def _matches_all(text: str, needles: Iterable[str]) -> bool:
    hay = text.lower()
    for needle in needles:
        if needle.lower() not in hay:
            return False
    return True


def _parse_window(base_id: str) -> str:
    m = _OOS_RE.search(base_id)
    if not m:
        return "-"
    return f"{m.group(1)}-{m.group(2)}"


def _variant_id(base_id: str) -> str:
    return _OOS_RE.sub("", base_id)


def _fmt(value: Optional[float], *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _quantile(values: Iterable[float], q: float) -> Optional[float]:
    ordered = sorted(values)
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


def _load_daily_pnl_map(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields:
            return {}

        date_key = fields[0]
        pnl_key: Optional[str] = None
        for field in fields:
            if "pnl" in field.lower():
                pnl_key = field
                break
        if pnl_key is None and len(fields) >= 2:
            pnl_key = fields[1]
        if pnl_key is None:
            return {}

        out: Dict[str, float] = {}
        for row in reader:
            date = str(row.get(date_key) or "").strip()
            if not date:
                continue
            pnl = _to_float(str(row.get(pnl_key) or ""))
            if pnl is None:
                continue
            out[date] = float(pnl)
        return out


def _load_daily_tail(project_root: Path, run: _Entry) -> Tuple[float, ...]:
    daily = _load_daily_pnl_map(_resolve_path(project_root, run.results_dir) / "daily_pnl.csv")
    if not daily:
        return ()
    # Keep order stable for quantile computations.
    return tuple(daily[day] for day in sorted(daily))


def _fullspan_score(
    *,
    worst_robust_sharpe: float,
    q_step_pnl: float,
    worst_step_pnl: float,
    initial_capital: float,
    tail_q_soft_loss_pct: float,
    tail_worst_soft_loss_pct: float,
    tail_q_penalty: float,
    tail_worst_penalty: float,
) -> float:
    q_loss = max(0.0, (-q_step_pnl / initial_capital) - tail_q_soft_loss_pct)
    worst_loss = max(0.0, (-worst_step_pnl / initial_capital) - tail_worst_soft_loss_pct)
    return float(worst_robust_sharpe) - float(tail_q_penalty) * q_loss - float(tail_worst_penalty) * worst_loss


def _hybrid_weights(args: argparse.Namespace) -> Tuple[float, float, float]:
    w_worst = max(0.0, float(args.hybrid_worst_weight))
    w_q = max(0.0, float(args.hybrid_quantile_weight))
    w_avg = max(0.0, float(args.hybrid_avg_weight))
    total = w_worst + w_q + w_avg
    if total <= 0.0:
        raise SystemExit("hybrid weights must have positive sum")
    return (w_worst / total, w_q / total, w_avg / total)


def _print_concentration_rejects(
    *,
    max_tail_pair_share: Optional[float],
    max_tail_period_share: Optional[float],
    pair_missing: int,
    pair_above_max: int,
    period_missing: int,
    period_above_max: int,
) -> None:
    if pair_missing <= 0 and pair_above_max <= 0 and period_missing <= 0 and period_above_max <= 0:
        return

    parts: List[str] = []
    if max_tail_pair_share is not None:
        parts.append(f"pair_missing={pair_missing}")
        parts.append(f"pair_above_max={pair_above_max}")
        parts.append(f"max_pair_share={float(max_tail_pair_share):.3f}")
    if max_tail_period_share is not None:
        parts.append(f"period_missing={period_missing}")
        parts.append(f"period_above_max={period_above_max}")
        parts.append(f"max_period_share={float(max_tail_period_share):.3f}")
    if parts:
        print("Concentration gate rejections: " + ", ".join(parts) + ".")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank holdout WFA runs across OOS windows (multi-window Sharpe)")
    parser.add_argument(
        "--run-index",
        default="artifacts/wfa/aggregate/rollup/run_index.csv",
        help="Path to rollup run_index.csv (relative to project root).",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Substring filter (repeatable). All must match combined run metadata.",
    )
    parser.add_argument("--top", type=int, default=20, help="How many variants to print.")
    parser.add_argument(
        "--include-noncompleted",
        action="store_true",
        help="Include variants where at least one window is not 'completed'.",
    )
    parser.add_argument("--min-windows", type=int, default=3, help="Require at least this many windows per variant.")
    parser.add_argument("--min-trades", type=int, default=200, help="Require min(total_trades) across windows >= this.")
    parser.add_argument("--min-pairs", type=int, default=20, help="Require min(total_pairs_traded) across windows >= this.")
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.95,
        help=(
            "Gate: worst-window coverage_ratio must be >= this "
            "(default: 0.95). Missing/non-finite coverage fails closed."
        ),
    )
    parser.add_argument(
        "--max-dd-pct",
        type=float,
        default=0.20,
        help="Gate: max window drawdown on equity (abs) must be <= this (default: 0.20 = 20%%).",
    )
    parser.add_argument(
        "--min-pnl",
        type=float,
        default=0.0,
        help="Gate: worst window PnL (holdout total_pnl) must be >= this (default: 0.0).",
    )
    parser.add_argument(
        "--min-psr",
        type=float,
        default=None,
        help="Gate: worst-window robust PSR must be >= this threshold (disabled when omitted).",
    )
    parser.add_argument(
        "--min-dsr",
        type=float,
        default=None,
        help="Gate: worst-window robust DSR must be >= this threshold (disabled when omitted).",
    )
    parser.add_argument(
        "--max-tail-pair-share",
        type=float,
        default=None,
        help=(
            "Gate: worst robust tail-loss share contributed by one pair "
            "must be <= this in [0, 1] (disabled when omitted)."
        ),
    )
    parser.add_argument(
        "--max-tail-period-share",
        type=float,
        default=None,
        help=(
            "Gate: worst robust tail-loss share contributed by one WF period "
            "must be <= this in [0, 1] (disabled when omitted)."
        ),
    )
    parser.add_argument(
        "--score-mode",
        choices=("worst", "quantile", "hybrid"),
        default="worst",
        help=(
            "Ranking objective for multi-window robust Sharpe: "
            "worst=min(window robust_sharpe), "
            "quantile=Q_q(window robust_sharpe), "
            "hybrid=w_worst*worst + w_q*Q_q + w_avg*avg."
        ),
    )
    parser.add_argument(
        "--evaluator-protocol",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "Ranking protocol: v1 uses scalar score_mode objective, "
            "v2 uses formal multi-objective evaluator (Pareto + utility decomposition)."
        ),
    )
    parser.add_argument(
        "--quantile-q",
        type=float,
        default=0.20,
        help="Lower-tail quantile q in [0,1] for quantile/hybrid score modes (default: 0.20).",
    )
    parser.add_argument(
        "--hybrid-worst-weight",
        type=float,
        default=0.50,
        help="Hybrid weight for worst robust Sharpe (default: 0.50).",
    )
    parser.add_argument(
        "--hybrid-quantile-weight",
        type=float,
        default=0.30,
        help="Hybrid weight for quantile robust Sharpe (default: 0.30).",
    )
    parser.add_argument(
        "--hybrid-avg-weight",
        type=float,
        default=0.20,
        help="Hybrid weight for average robust Sharpe (default: 0.20).",
    )
    parser.add_argument(
        "--v2-weight-worst-robust",
        type=float,
        default=0.45,
        help="Evaluator v2 utility weight for worst robust Sharpe objective.",
    )
    parser.add_argument(
        "--v2-weight-quantile-robust",
        type=float,
        default=0.20,
        help="Evaluator v2 utility weight for quantile robust Sharpe objective.",
    )
    parser.add_argument(
        "--v2-weight-avg-robust",
        type=float,
        default=0.10,
        help="Evaluator v2 utility weight for average robust Sharpe objective.",
    )
    parser.add_argument(
        "--v2-weight-worst-dd",
        type=float,
        default=0.20,
        help="Evaluator v2 utility weight for worst drawdown objective (minimize).",
    )
    parser.add_argument(
        "--v2-weight-worst-pnl",
        type=float,
        default=0.05,
        help="Evaluator v2 utility weight for worst robust PnL objective.",
    )
    parser.add_argument(
        "--v2-decomposition-top",
        type=int,
        default=3,
        help="How many strongest objective terms to include in v2 decomposition output.",
    )
    parser.add_argument(
        "--fullspan-policy-v1",
        action="store_true",
        help=(
            "Enable canonical fullspan scoring: worst robust Sharpe with tail penalties from robust daily PnL "
            "plus hard gate on worst-step loss."
        ),
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000.0,
        help="Initial capital used to normalize tail losses (default: 1000).",
    )
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.20,
        help="Tail quantile q in [0, 1] for fullspan score (default: 0.20).",
    )
    parser.add_argument(
        "--tail-q-soft-loss-pct",
        type=float,
        default=0.03,
        help="Soft threshold for quantile loss as share of capital (default: 0.03).",
    )
    parser.add_argument(
        "--tail-worst-soft-loss-pct",
        type=float,
        default=0.10,
        help="Soft threshold for worst-step loss as share of capital (default: 0.10).",
    )
    parser.add_argument(
        "--tail-q-penalty",
        type=float,
        default=2.0,
        help="Penalty multiplier for quantile tail overflow (default: 2.0).",
    )
    parser.add_argument(
        "--tail-worst-penalty",
        type=float,
        default=1.0,
        help="Penalty multiplier for worst-step tail overflow (default: 1.0).",
    )
    parser.add_argument(
        "--tail-worst-gate-pct",
        type=float,
        default=0.20,
        help="Hard gate for worst robust daily PnL as share of capital (default: 0.20).",
    )
    args = parser.parse_args()

    if args.initial_capital <= 0:
        raise SystemExit("--initial-capital must be > 0")
    if not (0.0 <= args.quantile_q <= 1.0):
        raise SystemExit("--quantile-q must be in [0, 1]")
    if not (0.0 <= args.tail_quantile <= 1.0):
        raise SystemExit("--tail-quantile must be in [0, 1]")
    if args.min_psr is not None and not (0.0 <= float(args.min_psr) <= 1.0):
        raise SystemExit("--min-psr must be in [0, 1]")
    if args.min_coverage_ratio is not None and not (0.0 <= float(args.min_coverage_ratio) <= 1.0):
        raise SystemExit("--min-coverage-ratio must be in [0, 1]")
    if args.max_tail_pair_share is not None and not (0.0 <= float(args.max_tail_pair_share) <= 1.0):
        raise SystemExit("--max-tail-pair-share must be in [0, 1]")
    if args.max_tail_period_share is not None and not (0.0 <= float(args.max_tail_period_share) <= 1.0):
        raise SystemExit("--max-tail-period-share must be in [0, 1]")
    if args.v2_decomposition_top < 1:
        raise SystemExit("--v2-decomposition-top must be >= 1")
    if args.fullspan_policy_v1 and str(args.evaluator_protocol) == "v2":
        raise SystemExit("--evaluator-protocol v2 cannot be combined with --fullspan-policy-v1")
    hybrid_worst_weight, hybrid_q_weight, hybrid_avg_weight = _hybrid_weights(args)

    project_root = Path(__file__).resolve().parents[2]
    run_index_path = _resolve_path(project_root, args.run_index)
    if not run_index_path.exists():
        raise SystemExit(f"run index not found: {run_index_path}")

    entries: List[_Entry] = []
    with run_index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                _Entry(
                    run_id=(row.get("run_id") or "").strip(),
                    run_group=(row.get("run_group") or "").strip(),
                    results_dir=(row.get("results_dir") or "").strip(),
                    config_path=(row.get("config_path") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    metrics_present=_to_bool(row.get("metrics_present") or ""),
                    sharpe=_to_float(row.get("sharpe_ratio_abs") or ""),
                    pnl=_to_float(row.get("total_pnl") or ""),
                    dd_abs=_to_float(row.get("max_drawdown_abs") or ""),
                    dd_pct=_to_float(row.get("max_drawdown_on_equity") or ""),
                    psr=_to_float(row.get("psr") or ""),
                    dsr=_to_float(row.get("dsr") or ""),
                    trades=_to_float(row.get("total_trades") or ""),
                    pairs=_to_float(row.get("total_pairs_traded") or ""),
                    costs=_to_float(row.get("total_costs") or ""),
                    coverage_ratio=_to_float(row.get("coverage_ratio") or ""),
                    tail_loss_worst_pair_share=_to_float(row.get("tail_loss_worst_pair_share") or ""),
                    tail_loss_worst_period_share=_to_float(row.get("tail_loss_worst_period_share") or ""),
                )
            )

    # Step 1: collect holdout runs (stress rows are ignored).
    holdouts: Dict[Tuple[str, str], _Entry] = {}
    for entry in entries:
        kind, base_id = _kind_and_base_id(entry.run_id)
        if kind == "stress":
            continue

        meta = " | ".join([entry.run_group, base_id, entry.run_id, entry.config_path, entry.results_dir, entry.status])
        if args.contains and not _matches_all(meta, args.contains):
            continue

        holdouts[(entry.run_group, base_id)] = entry

    # Step 2: compute per-window metrics and aggregate by variant.
    windows_by_variant: Dict[Tuple[str, str], List[_WindowStats]] = {}
    for (run_group, base_id), holdout in holdouts.items():
        if not holdout.metrics_present:
            continue
        if holdout.sharpe is None or holdout.dd_pct is None:
            continue
        if not args.include_noncompleted and holdout.status.lower() != "completed":
            continue

        robust_sharpe = float(holdout.sharpe)
        dd_pct = float(abs(holdout.dd_pct))
        robust_psr = holdout.psr
        robust_dsr = holdout.dsr
        robust_pnl = holdout.pnl

        robust_coverage = None
        if holdout.coverage_ratio is not None:
            cov = float(holdout.coverage_ratio)
            if math.isfinite(cov):
                robust_coverage = cov

        robust_tail_pair_share = holdout.tail_loss_worst_pair_share
        robust_tail_period_share = holdout.tail_loss_worst_period_share

        tail_samples: Tuple[float, ...] = ()
        if args.fullspan_policy_v1:
            tail_samples = _load_daily_tail(project_root, holdout)

        window = _parse_window(base_id)
        variant = _variant_id(base_id)
        windows_by_variant.setdefault((run_group, variant), []).append(
            _WindowStats(
                window=window,
                robust_sharpe=float(robust_sharpe),
                dd_pct=float(dd_pct),
                robust_coverage_ratio=robust_coverage,
                robust_psr=robust_psr,
                robust_dsr=robust_dsr,
                robust_pnl=robust_pnl,
                robust_tail_pair_share=robust_tail_pair_share,
                robust_tail_period_share=robust_tail_period_share,
                tail_samples=tail_samples,
                holdout=holdout,
            )
        )

    rows: List[_RankRow] = []
    skipped_tail_missing = 0
    skipped_tail_gate = 0
    skipped_coverage_missing = 0
    skipped_coverage_gate = 0
    coverage_rejects: List[Tuple[float, str, str, str, int, Optional[float]]] = []
    skipped_pair_concentration_missing = 0
    skipped_pair_concentration_gate = 0
    skipped_period_concentration_missing = 0
    skipped_period_concentration_gate = 0
    for (run_group, variant), items in windows_by_variant.items():
        if len(items) < max(1, args.min_windows):
            continue

        worst_dd = max(item.dd_pct for item in items)
        if worst_dd > max(0.0, args.max_dd_pct):
            continue

        min_trades = min(float(item.holdout.trades or 0.0) for item in items)
        min_pairs = min(float(item.holdout.pairs or 0.0) for item in items)
        if min_trades < args.min_trades or min_pairs < args.min_pairs:
            continue

        worst_robust = min(item.robust_sharpe for item in items)
        q_robust = _quantile((item.robust_sharpe for item in items), args.quantile_q)
        if q_robust is None:
            continue

        if args.min_coverage_ratio is not None:
            coverage_values = [
                float(item.robust_coverage_ratio)
                for item in items
                if item.robust_coverage_ratio is not None and math.isfinite(float(item.robust_coverage_ratio))
            ]
            worst_coverage = min(coverage_values) if coverage_values else None
            if worst_coverage is None or len(coverage_values) != len(items):
                skipped_coverage_missing += 1
                coverage_rejects.append(
                    (
                        float(worst_robust),
                        str(run_group),
                        str(variant),
                        str(items[0].holdout.config_path),
                        int(len(items)),
                        worst_coverage,
                    )
                )
                continue
            if float(worst_coverage) < float(args.min_coverage_ratio):
                skipped_coverage_gate += 1
                coverage_rejects.append(
                    (
                        float(worst_robust),
                        str(run_group),
                        str(variant),
                        str(items[0].holdout.config_path),
                        int(len(items)),
                        float(worst_coverage),
                    )
                )
                continue

        avg_robust = sum(item.robust_sharpe for item in items) / len(items)
        avg_dd = sum(item.dd_pct for item in items) / len(items)
        psr_values = [item.robust_psr for item in items if item.robust_psr is not None]
        worst_psr = min(psr_values) if psr_values else None
        if args.min_psr is not None:
            if worst_psr is None or len(psr_values) != len(items) or worst_psr < float(args.min_psr):
                continue

        dsr_values = [item.robust_dsr for item in items if item.robust_dsr is not None]
        worst_dsr = min(dsr_values) if dsr_values else None
        if args.min_dsr is not None:
            if worst_dsr is None or len(dsr_values) != len(items) or worst_dsr < float(args.min_dsr):
                continue

        pnls = [item.robust_pnl for item in items if item.robust_pnl is not None]
        worst_pnl = min(pnls) if pnls else None
        avg_pnl = (sum(pnls) / len(pnls)) if pnls else None
        if worst_pnl is None or worst_pnl < args.min_pnl:
            continue
        pair_tail_shares = [item.robust_tail_pair_share for item in items if item.robust_tail_pair_share is not None]
        worst_tail_pair_share = max(pair_tail_shares) if pair_tail_shares else None
        if args.max_tail_pair_share is not None:
            if worst_tail_pair_share is None or len(pair_tail_shares) != len(items):
                skipped_pair_concentration_missing += 1
                continue
            if worst_tail_pair_share > float(args.max_tail_pair_share):
                skipped_pair_concentration_gate += 1
                continue
        period_tail_shares = [
            item.robust_tail_period_share for item in items if item.robust_tail_period_share is not None
        ]
        worst_tail_period_share = max(period_tail_shares) if period_tail_shares else None
        if args.max_tail_period_share is not None:
            if worst_tail_period_share is None or len(period_tail_shares) != len(items):
                skipped_period_concentration_missing += 1
                continue
            if worst_tail_period_share > float(args.max_tail_period_share):
                skipped_period_concentration_gate += 1
                continue

        if args.score_mode == "worst":
            score = float(worst_robust)
        elif args.score_mode == "quantile":
            score = float(q_robust)
        else:
            score = (
                hybrid_worst_weight * float(worst_robust)
                + hybrid_q_weight * float(q_robust)
                + hybrid_avg_weight * float(avg_robust)
            )
        worst_step_pnl: Optional[float] = None
        q_step_pnl: Optional[float] = None

        if args.fullspan_policy_v1:
            tail_values = [sample for item in items for sample in item.tail_samples]
            if not tail_values:
                skipped_tail_missing += 1
                continue
            worst_step_pnl = min(tail_values)
            q_step_pnl = _quantile(tail_values, args.tail_quantile)
            if q_step_pnl is None:
                skipped_tail_missing += 1
                continue
            if worst_step_pnl < (-args.tail_worst_gate_pct * args.initial_capital):
                skipped_tail_gate += 1
                continue
            score = _fullspan_score(
                worst_robust_sharpe=worst_robust,
                q_step_pnl=q_step_pnl,
                worst_step_pnl=worst_step_pnl,
                initial_capital=float(args.initial_capital),
                tail_q_soft_loss_pct=float(args.tail_q_soft_loss_pct),
                tail_worst_soft_loss_pct=float(args.tail_worst_soft_loss_pct),
                tail_q_penalty=float(args.tail_q_penalty),
                tail_worst_penalty=float(args.tail_worst_penalty),
            )

        sample_cfg = items[0].holdout.config_path
        rows.append(
            _RankRow(
                score=float(score),
                score_mode="fullspan_v1" if args.fullspan_policy_v1 else str(args.score_mode),
                worst_robust=float(worst_robust),
                q_robust=float(q_robust),
                worst_dd=float(worst_dd),
                avg_robust=float(avg_robust),
                avg_dd=float(avg_dd),
                worst_psr=worst_psr,
                worst_dsr=worst_dsr,
                windows=len(items),
                variant=variant,
                sample_cfg=sample_cfg,
                run_group=run_group,
                worst_pnl=worst_pnl,
                avg_pnl=avg_pnl,
                worst_tail_pair_share=worst_tail_pair_share,
                worst_tail_period_share=worst_tail_period_share,
                worst_step_pnl=worst_step_pnl,
                q_step_pnl=q_step_pnl,
            )
        )

    if str(args.evaluator_protocol) == "v2" and rows:
        from coint2.ops.evaluator import (
            CandidateEvaluationInput,
            ObjectiveSpec,
            format_decomposition,
            rank_candidates_v2,
        )

        v2_specs = [
            ObjectiveSpec(name="worst_robust_sh", direction="maximize", weight=float(args.v2_weight_worst_robust)),
            ObjectiveSpec(name="q_robust_sh", direction="maximize", weight=float(args.v2_weight_quantile_robust)),
            ObjectiveSpec(name="avg_robust_sh", direction="maximize", weight=float(args.v2_weight_avg_robust)),
            ObjectiveSpec(name="worst_dd_pct", direction="minimize", weight=float(args.v2_weight_worst_dd)),
            ObjectiveSpec(name="worst_pnl", direction="maximize", weight=float(args.v2_weight_worst_pnl)),
        ]

        inputs = [
            CandidateEvaluationInput(
                candidate_id=f"{row.run_group}::{row.variant}",
                objectives={
                    "worst_robust_sh": float(row.worst_robust),
                    "q_robust_sh": float(row.q_robust),
                    "avg_robust_sh": float(row.avg_robust),
                    "worst_dd_pct": float(row.worst_dd),
                    "worst_pnl": row.worst_pnl,
                },
            )
            for row in rows
        ]
        ranked = rank_candidates_v2(inputs, objective_specs=v2_specs)
        ranked_map = {item.candidate_id: item for item in ranked}
        rows = [
            replace(
                row,
                score=float(ranked_map[f"{row.run_group}::{row.variant}"].utility_score),
                score_mode="evaluator_v2",
                pareto_front=int(ranked_map[f"{row.run_group}::{row.variant}"].pareto_front),
                dominated_by=int(ranked_map[f"{row.run_group}::{row.variant}"].dominated_by),
                decomposition=format_decomposition(
                    ranked_map[f"{row.run_group}::{row.variant}"].decomposition,
                    top_n=int(args.v2_decomposition_top),
                ),
            )
            for row in rows
        ]

    if str(args.evaluator_protocol) == "v2":
        rows.sort(
            key=lambda row: (
                row.pareto_front if row.pareto_front is not None else 10**9,
                -row.score,
                -row.worst_robust,
                -row.q_robust,
                -row.avg_robust,
                row.worst_dd,
                row.variant,
            )
        )
    elif args.fullspan_policy_v1:
        # Canonical tie-break for fullspan: worst_robust_pnl -> worst_dd_pct -> avg_robust_sharpe.
        rows.sort(
            key=lambda row: (
                row.score,
                row.worst_pnl if row.worst_pnl is not None else float("-inf"),
                -row.worst_dd,
                row.avg_robust,
            ),
            reverse=True,
        )
    else:
        rows.sort(
            key=lambda row: (
                row.score,
                row.worst_robust,
                row.q_robust,
                row.avg_robust,
                row.worst_psr if row.worst_psr is not None else float("-inf"),
                row.worst_dsr if row.worst_dsr is not None else float("-inf"),
                -row.worst_dd,
            ),
            reverse=True,
        )

    rows = rows[: max(1, args.top)]

    if not rows:
        if args.min_coverage_ratio is not None and (skipped_coverage_missing or skipped_coverage_gate):
            print(
                "Coverage gate rejections: "
                f"missing={skipped_coverage_missing}, below_min={skipped_coverage_gate}, "
                f"min_coverage_ratio={float(args.min_coverage_ratio):.3f}."
            )
        _print_concentration_rejects(
            max_tail_pair_share=args.max_tail_pair_share,
            max_tail_period_share=args.max_tail_period_share,
            pair_missing=skipped_pair_concentration_missing,
            pair_above_max=skipped_pair_concentration_gate,
            period_missing=skipped_period_concentration_missing,
            period_above_max=skipped_period_concentration_gate,
        )
        if args.fullspan_policy_v1:
            print(
                "No variants matched fullspan policy v1 "
                f"(missing_tail={skipped_tail_missing}, worst_step_gate_failed={skipped_tail_gate})."
            )
            return 1
        print("No variants matched (check filters/gates or ensure rollup index is up to date).")
        return 1

    q_robust_label = f"q{int(round(args.quantile_q * 100)):02d}_robust_sh"
    q_step_label = f"q{int(round(args.tail_quantile * 100)):02d}_step_pnl"
    print(
        "| rank | score_mode | score | worst_robust_sh | {q_robust_label} | avg_robust_sh | worst_dd_pct | avg_dd_pct | worst_psr | worst_dsr | windows | variant_id | sample_config | run_group | worst_pnl | avg_pnl | worst_pair_tail_share | worst_period_tail_share | worst_step_pnl | {q_step_label} | pareto_front | dominated_by | decomposition |".format(
            q_robust_label=q_robust_label,
            q_step_label=q_step_label,
        )
    )
    print(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for idx, row in enumerate(rows, 1):
        print(
            "| {rank} | {mode} | {score} | {worst} | {qrobust} | {avg} | {wdd} | {add} | {wpsr} | {wdsr} | {n} | {variant} | {cfg} | {group} | {wpnl} | {apnl} | {wpair} | {wperiod} | {wstep} | {qstep} | {pareto} | {dominated_by} | {decomposition} |".format(
                rank=idx,
                mode=row.score_mode,
                score=_fmt(row.score, digits=3),
                worst=_fmt(row.worst_robust, digits=3),
                qrobust=_fmt(row.q_robust, digits=3),
                wdd=_fmt(row.worst_dd, digits=3),
                avg=_fmt(row.avg_robust, digits=3),
                add=_fmt(row.avg_dd, digits=3),
                wpsr=_fmt(row.worst_psr, digits=3),
                wdsr=_fmt(row.worst_dsr, digits=3),
                n=row.windows,
                variant=row.variant,
                cfg=row.sample_cfg,
                group=row.run_group,
                wpnl=_fmt(row.worst_pnl, digits=2),
                apnl=_fmt(row.avg_pnl, digits=2),
                wpair=_fmt(row.worst_tail_pair_share, digits=3),
                wperiod=_fmt(row.worst_tail_period_share, digits=3),
                wstep=_fmt(row.worst_step_pnl, digits=2),
                qstep=_fmt(row.q_step_pnl, digits=2),
                pareto=row.pareto_front if row.pareto_front is not None else "-",
                dominated_by=row.dominated_by if row.dominated_by is not None else "-",
                decomposition=(row.decomposition or "-"),
            )
        )
    _print_concentration_rejects(
        max_tail_pair_share=args.max_tail_pair_share,
        max_tail_period_share=args.max_tail_period_share,
        pair_missing=skipped_pair_concentration_missing,
        pair_above_max=skipped_pair_concentration_gate,
        period_missing=skipped_period_concentration_missing,
        period_above_max=skipped_period_concentration_gate,
    )
    if args.min_coverage_ratio is not None and (skipped_coverage_missing or skipped_coverage_gate):
        print(
            "Coverage gate rejections: "
            f"missing={skipped_coverage_missing}, below_min={skipped_coverage_gate}, "
            f"min_coverage_ratio={float(args.min_coverage_ratio):.3f}."
        )
        if coverage_rejects:
            print("Top coverage rejects (by worst_robust_sh):")
            for worst_robust, run_group, variant, cfg, windows, worst_cov in sorted(
                coverage_rejects, key=lambda row: row[0], reverse=True
            )[:10]:
                cov_str = "-" if worst_cov is None or not math.isfinite(float(worst_cov)) else f"{float(worst_cov):.3f}"
                print(
                    f"- {variant} | worst_robust_sh={worst_robust:.3f} "
                    f"worst_coverage_ratio={cov_str} windows={windows} | {run_group} | {cfg}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
