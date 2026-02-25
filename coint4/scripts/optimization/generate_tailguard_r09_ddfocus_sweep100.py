#!/usr/bin/env python3
"""Generate r09 dd-focus sweep (100 variants) around the current best fullspan baseline (r08 v10).

Goal
----
Fast-loop on the worst drawdown regime window (dd-focus), using a larger, structured
search space (100 variants = 10 tradeability profiles x 10 quality profiles).
Then we can confirm the top few variants on fullspan.

Design goals
------------
- Do NOT touch risk/stop/z/dstop/maxpos and other backtest dynamics.
- Update ONLY:
  - walk_forward.start_date / walk_forward.end_date (dd-focus window)
  - tradeability knobs (pair_selection.* microstructure gates)
  - quality knobs (filter_params.* + optional KPSS via pair_selection.kpss_pvalue_threshold)
- Produce:
  configs/budget1000_autopilot/<run_group>/*.yaml
  artifacts/wfa/aggregate/<run_group>/run_queue.csv
  artifacts/wfa/aggregate/<run_group>/search_space.(csv|md)

Run from app-root (coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/generate_tailguard_r09_ddfocus_sweep100.py
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from coint2.ops.run_queue import RunQueueEntry, write_run_queue


DD_START = "2023-06-29"
DD_END = "2024-06-27"

STRESS_OVERRIDES: dict[str, Any] = {
    "backtest.commission_pct": 0.0006,
    "backtest.commission_rate_per_leg": 0.0006,
    "backtest.slippage_pct": 0.001,
    "backtest.slippage_stress_multiplier": 2.0,
}


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node: Dict[str, Any] = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def get_nested(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    parts = dotted_key.split(".")
    node: Any = cfg
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@dataclass(frozen=True)
class Profile:
    profile_id: str
    hypothesis: str
    overrides: dict[str, Any]


def _tradeability_profiles() -> list[Profile]:
    # NOTE: guardrail floors liquidity to >=300k and caps bid-ask/funding at high values.
    # These profiles intentionally stay above the liquidity floor and tighten bid-ask/funding/tick.
    return [
        Profile("t01_base", "Tradeability: base (r08 v10)", {}),
        Profile(
            "t02_loosen",
            "Tradeability: broaden (liq=300k, bid-ask=0.20%)",
            {"pair_selection.liquidity_usd_daily": 300_000, "pair_selection.max_bid_ask_pct": 0.002},
        ),
        Profile("t03_liq300k", "Tradeability: liquidity 300k (keep bid-ask baseline)", {"pair_selection.liquidity_usd_daily": 300_000}),
        Profile("t04_liq500k", "Tradeability: liquidity 500k", {"pair_selection.liquidity_usd_daily": 500_000}),
        Profile("t05_liq750k", "Tradeability: liquidity 750k", {"pair_selection.liquidity_usd_daily": 750_000}),
        Profile("t06_bidask0p0008", "Tradeability: bid-ask 0.08%", {"pair_selection.max_bid_ask_pct": 0.0008}),
        Profile("t07_bidask0p0006", "Tradeability: bid-ask 0.06%", {"pair_selection.max_bid_ask_pct": 0.0006}),
        Profile("t08_dayslive365", "Tradeability: min_days_live 365", {"pair_selection.min_days_live": 365}),
        Profile("t09_tick0p0004", "Tradeability: tick_size_pct 0.04%", {"pair_selection.max_tick_size_pct": 0.0004}),
        Profile(
            "t10_funding0p0007",
            "Tradeability: funding caps 0.07% (avg+abs)",
            {"pair_selection.max_avg_funding_pct": 0.0007, "pair_selection.max_funding_rate_abs": 0.0007},
        ),
    ]


def _quality_profiles() -> list[Profile]:
    return [
        Profile("q01_base", "Quality: base (r08 v10)", {}),
        Profile("q02_hurst0p70", "Quality: max_hurst_exponent=0.70", {"filter_params.max_hurst_exponent": 0.70}),
        Profile("q03_hurst0p74", "Quality: max_hurst_exponent=0.74", {"filter_params.max_hurst_exponent": 0.74}),
        Profile("q04_cross3", "Quality: min_mean_crossings=3", {"filter_params.min_mean_crossings": 3}),
        Profile("q05_hl40", "Quality: max_half_life_days=40", {"filter_params.max_half_life_days": 40}),
        Profile("q06_beta_tight", "Quality: beta range tighter (0.02..40)", {"filter_params.min_beta": 0.02, "filter_params.max_beta": 40.0}),
        Profile("q07_kpss_on", "Quality: enable KPSS (p>=0.05)", {"pair_selection.kpss_pvalue_threshold": 0.05}),
        Profile("q08_beta_drift0p20", "Quality: beta drift ratio <= 0.20", {"filter_params.max_beta_drift_ratio": 0.20}),
        Profile("q09_ecm_t2p0", "Quality: ECM alpha t-stat <= -2.0", {"filter_params.ecm_alpha_tstat_threshold": 2.0}),
        Profile(
            "q10_ecm2p5_betadrift0p15",
            "Quality: ECM<=-2.5 + beta drift<=0.15",
            {"filter_params.ecm_alpha_tstat_threshold": 2.5, "filter_params.max_beta_drift_ratio": 0.15},
        ),
    ]


def _iter_variants(trades: list[Profile], quals: list[Profile]) -> Iterable[tuple[int, Profile, Profile, str, dict[str, Any]]]:
    # Deterministic 10x10 grid -> 100 variants.
    n = 0
    for t in trades:
        for q in quals:
            n += 1
            variant_id = f"v{n:03d}_{t.profile_id}_{q.profile_id}"
            hypothesis = f"{t.hypothesis} + {q.hypothesis}"
            overrides = dict(t.overrides)
            overrides.update(q.overrides)
            yield n, t, q, hypothesis, overrides


def _render_search_space_md(
    *,
    run_group: str,
    base_path: Path,
    dd_start: str,
    dd_end: str,
    trade_profiles: list[Profile],
    quality_profiles: list[Profile],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("# tailguard_r09 dd-focus sweep100\n")
    lines.append(f"Date: {now}")
    lines.append(f"Run group: `{run_group}`")
    lines.append("Goal: 100 вариантов (10x10) на dd-focus окне; варьируем только tradeability+quality.")
    lines.append("")
    lines.append("## DD-Focus window\n")
    lines.append(f"- walk_forward.start_date: `{dd_start}`")
    lines.append(f"- walk_forward.end_date: `{dd_end}`")
    lines.append("")
    lines.append("## Base config\n")
    lines.append(f"- Base (holdout): `{base_path.as_posix()}`")
    lines.append("")
    lines.append("## Tradeability profiles (T)\n")
    for p in trade_profiles:
        lines.append(f"- `{p.profile_id}`: {p.hypothesis}")
    lines.append("")
    lines.append("## Quality profiles (Q)\n")
    for p in quality_profiles:
        lines.append(f"- `{p.profile_id}`: {p.hypothesis}")
    lines.append("")
    lines.append("## Variants\n")
    lines.append("")
    lines.append("Variants are the cartesian product T x Q (10 x 10 = 100).")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-group",
        default="20260225_tailguard_r09_ddfocus_sweep100",
        help="Run group name (used in output dirs).",
    )
    ap.add_argument(
        "--base",
        default=(
            "configs/budget1000_autopilot/20260224_tailguard_r08_tradeability_quality_sweep/"
            "holdout_tailguard_r08_fullspan_v10_h2_quality_hurst0p72.yaml"
        ),
        help="Base holdout config (relative to app root).",
    )
    ap.add_argument("--dd-start", default=DD_START, help="walk_forward.start_date (test start)")
    ap.add_argument("--dd-end", default=DD_END, help="walk_forward.end_date (test end)")
    ap.add_argument(
        "--out-config-dir",
        default="configs/budget1000_autopilot",
        help="Base directory for generated configs (relative to app root).",
    )
    ap.add_argument(
        "--out-queue-dir",
        default="artifacts/wfa/aggregate",
        help="Base directory for run_queue/search_space (relative to app root).",
    )
    ap.add_argument(
        "--runs-dir",
        default="artifacts/wfa/runs",
        help="Base directory for results_dir in run_queue entries (relative to app root).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    app_root = Path(__file__).resolve().parents[2]
    run_group = str(args.run_group).strip()
    if not run_group:
        raise SystemExit("--run-group is empty")

    base_path = (app_root / str(args.base)).resolve()
    if not base_path.exists():
        raise SystemExit(f"Base config not found: {base_path}")
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base_cfg, dict):
        raise SystemExit(f"Base config invalid (expected mapping): {base_path}")

    dd_start = str(args.dd_start).strip()
    dd_end = str(args.dd_end).strip()
    if not dd_start or not dd_end:
        raise SystemExit("dd-start/dd-end is empty")
    if dd_start >= dd_end:
        raise SystemExit(f"Invalid dd window: start={dd_start} end={dd_end}")

    out_config_dir = (app_root / str(args.out_config_dir) / run_group).resolve()
    out_queue_dir = (app_root / str(args.out_queue_dir) / run_group).resolve()
    runs_dir = str(args.runs_dir).strip().rstrip("/")

    trade_profiles = _tradeability_profiles()
    quality_profiles = _quality_profiles()
    if len(trade_profiles) != 10 or len(quality_profiles) != 10:
        raise SystemExit("Expected exactly 10 tradeability + 10 quality profiles")

    entries: list[RunQueueEntry] = []
    space_rows: list[dict[str, Any]] = []

    if not args.dry_run:
        out_config_dir.mkdir(parents=True, exist_ok=True)
        out_queue_dir.mkdir(parents=True, exist_ok=True)

    for idx, trade_p, qual_p, hypothesis, overrides in _iter_variants(trade_profiles, quality_profiles):
        holdout_cfg = copy.deepcopy(base_cfg)
        # dd-focus window (test range); train starts at start_date - training_period_days.
        set_nested(holdout_cfg, "walk_forward.start_date", dd_start)
        set_nested(holdout_cfg, "walk_forward.end_date", dd_end)
        # Apply overrides (tradeability + quality only).
        for k, v in overrides.items():
            set_nested(holdout_cfg, k, v)

        variant_id = f"v{idx:03d}_{trade_p.profile_id}_{qual_p.profile_id}"
        holdout_name = f"holdout_tailguard_r09_ddfocus_{variant_id}"
        holdout_cfg_path = out_config_dir / f"{holdout_name}.yaml"
        holdout_results = f"{runs_dir}/{run_group}/{holdout_name}"

        if not args.dry_run:
            holdout_cfg_path.write_text(
                yaml.dump(holdout_cfg, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )

        entries.append(
            RunQueueEntry(
                config_path=str(holdout_cfg_path.relative_to(app_root)),
                results_dir=holdout_results,
                status="planned",
            )
        )

        stress_cfg = copy.deepcopy(holdout_cfg)
        for k, v in STRESS_OVERRIDES.items():
            set_nested(stress_cfg, k, v)

        stress_name = f"stress_tailguard_r09_ddfocus_{variant_id}"
        stress_cfg_path = out_config_dir / f"{stress_name}.yaml"
        stress_results = f"{runs_dir}/{run_group}/{stress_name}"

        if not args.dry_run:
            stress_cfg_path.write_text(
                yaml.dump(stress_cfg, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )

        entries.append(
            RunQueueEntry(
                config_path=str(stress_cfg_path.relative_to(app_root)),
                results_dir=stress_results,
                status="planned",
            )
        )

        space_rows.append(
            {
                "variant_id": variant_id,
                "trade_profile": trade_p.profile_id,
                "quality_profile": qual_p.profile_id,
                "hypothesis": hypothesis,
                "start_date": get_nested(holdout_cfg, "walk_forward.start_date"),
                "end_date": get_nested(holdout_cfg, "walk_forward.end_date"),
                # Tradeability knobs
                "liquidity_usd_daily": get_nested(holdout_cfg, "pair_selection.liquidity_usd_daily"),
                "max_bid_ask_pct": get_nested(holdout_cfg, "pair_selection.max_bid_ask_pct"),
                "max_avg_funding_pct": get_nested(holdout_cfg, "pair_selection.max_avg_funding_pct"),
                "min_days_live": get_nested(holdout_cfg, "pair_selection.min_days_live"),
                "max_funding_rate_abs": get_nested(holdout_cfg, "pair_selection.max_funding_rate_abs"),
                "max_tick_size_pct": get_nested(holdout_cfg, "pair_selection.max_tick_size_pct"),
                "require_market_metrics": get_nested(holdout_cfg, "pair_selection.require_market_metrics"),
                "require_same_quote": get_nested(holdout_cfg, "pair_selection.require_same_quote"),
                "kpss_pvalue_threshold": get_nested(holdout_cfg, "pair_selection.kpss_pvalue_threshold"),
                # Quality knobs
                "min_mean_crossings": get_nested(holdout_cfg, "filter_params.min_mean_crossings"),
                "max_half_life_days": get_nested(holdout_cfg, "filter_params.max_half_life_days"),
                "max_hurst_exponent": get_nested(holdout_cfg, "filter_params.max_hurst_exponent"),
                "min_beta": get_nested(holdout_cfg, "filter_params.min_beta"),
                "max_beta": get_nested(holdout_cfg, "filter_params.max_beta"),
                "max_beta_drift_ratio": get_nested(holdout_cfg, "filter_params.max_beta_drift_ratio"),
                "ecm_alpha_tstat_threshold": get_nested(
                    holdout_cfg, "filter_params.ecm_alpha_tstat_threshold"
                ),
            }
        )

    queue_path = out_queue_dir / "run_queue.csv"
    space_csv = out_queue_dir / "search_space.csv"
    space_md = out_queue_dir / "search_space.md"

    if args.dry_run:
        print(f"[dry-run] would write {len(entries)} queue entries into: {queue_path}")
        print(f"[dry-run] would write configs into: {out_config_dir}")
        return 0

    write_run_queue(queue_path, entries)
    with space_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(space_rows[0].keys()))
        writer.writeheader()
        writer.writerows(space_rows)

    space_md.write_text(
        _render_search_space_md(
            run_group=run_group,
            base_path=base_path.relative_to(app_root),
            dd_start=dd_start,
            dd_end=dd_end,
            trade_profiles=trade_profiles,
            quality_profiles=quality_profiles,
        ),
        encoding="utf-8",
    )

    print(f"Wrote configs: {out_config_dir.relative_to(app_root)}")
    print(f"Wrote queue:   {queue_path.relative_to(app_root)} ({len(entries)} entries)")
    print(f"Wrote space:   {space_csv.relative_to(app_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

