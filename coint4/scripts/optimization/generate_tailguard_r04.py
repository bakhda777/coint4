#!/usr/bin/env python3
"""Generate curated r04 configs (tradeability/quality/stability) from r03 top-1 base.

Design goals:
- Do NOT touch r03 risk/stop/z/dstop/maxpos axes.
- Vary only:
  H1 tradeability tighten (mild/med/hard + 2 diag + baseline control)
  H2 quality tighten (mild/med/hard + 1 diag kpss)
  H3 stability tighten (mild/med + 1 hard)
- Produce:
  configs/budget1000_autopilot/<run_group>/*.yaml
  artifacts/wfa/aggregate/<run_group>/run_queue.csv
  artifacts/wfa/aggregate/<run_group>/search_space.(csv|md)

Run from app-root (coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/generate_tailguard_r04.py
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from coint2.ops.run_queue import RunQueueEntry, write_run_queue


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
class Variant:
    variant_id: str
    hypothesis: str
    overrides: dict[str, Any]


def _variant_space() -> list[Variant]:
    v: list[Variant] = []

    # v01: baseline control (r03 top-1, replay under current code)
    v.append(Variant("v01_control", "control_replay_r03_top1", {}))

    # H1: tradeability tighten sets (based on Bybit snapshot metrics)
    # Keep liquidity 300-400k to avoid collapsing below min_pairs>=20.
    v.append(
        Variant(
            "v02_h1_trade_mild_A",
            "H1_tradeability_tighten",
            {
                "pair_selection.require_market_metrics": True,
                "pair_selection.require_same_quote": True,
                "pair_selection.liquidity_usd_daily": 300_000,
                "pair_selection.max_bid_ask_pct": 0.002,
                "pair_selection.min_days_live": 180,
                "pair_selection.max_funding_rate_abs": 0.001,
                "pair_selection.max_tick_size_pct": 0.0005,
                # Keep this redundant knob off; liquidity gate already covers turnover.
                "pair_selection.min_volume_usd_24h": 0.0,
            },
        )
    )
    v.append(
        Variant(
            "v03_h1_trade_balanced_B",
            "H1_tradeability_tighten",
            {
                "pair_selection.require_market_metrics": True,
                "pair_selection.require_same_quote": True,
                "pair_selection.liquidity_usd_daily": 400_000,
                "pair_selection.max_bid_ask_pct": 0.001,
                "pair_selection.min_days_live": 180,
                "pair_selection.max_funding_rate_abs": 0.001,
                "pair_selection.max_tick_size_pct": 0.0005,
                "pair_selection.min_volume_usd_24h": 0.0,
            },
        )
    )
    v.append(
        Variant(
            "v04_h1_trade_hard_C",
            "H1_tradeability_tighten",
            {
                "pair_selection.require_market_metrics": True,
                "pair_selection.require_same_quote": True,
                "pair_selection.liquidity_usd_daily": 400_000,
                "pair_selection.max_bid_ask_pct": 0.001,
                "pair_selection.min_days_live": 365,
                "pair_selection.max_funding_rate_abs": 0.001,
                "pair_selection.max_tick_size_pct": 0.0005,
                "pair_selection.min_volume_usd_24h": 0.0,
            },
        )
    )
    v.append(
        Variant(
            "v05_h1_diag_bidask_only",
            "H1_tradeability_diag_one_filter",
            {
                # Only tighten bid-ask; keep other tradeability knobs as in base.
                "pair_selection.max_bid_ask_pct": 0.001,
            },
        )
    )
    v.append(
        Variant(
            "v06_h1_diag_require_metrics_only",
            "H1_tradeability_diag_one_filter",
            {
                # Only enforce snapshot presence; other knobs stay as in base.
                "pair_selection.require_market_metrics": True,
            },
        )
    )

    # H2: quality tighten (mean_crossings/half-life/hurst/beta + optional KPSS)
    v.append(
        Variant(
            "v07_h2_quality_mild",
            "H2_quality_tighten",
            {
                "filter_params.min_mean_crossings": 2,
                "filter_params.max_half_life_days": 45,
                "filter_params.max_hurst_exponent": 0.75,
                "filter_params.min_beta": 0.01,
                "filter_params.max_beta": 50.0,
                "pair_selection.kpss_pvalue_threshold": 1.0,  # keep off
            },
        )
    )
    v.append(
        Variant(
            "v08_h2_quality_med",
            "H2_quality_tighten",
            {
                "filter_params.min_mean_crossings": 4,
                "filter_params.max_half_life_days": 30,
                "filter_params.max_hurst_exponent": 0.70,
                "filter_params.min_beta": 0.05,
                "filter_params.max_beta": 25.0,
                "pair_selection.kpss_pvalue_threshold": 0.05,
            },
        )
    )
    v.append(
        Variant(
            "v09_h2_quality_hard",
            "H2_quality_tighten",
            {
                "filter_params.min_mean_crossings": 6,
                "filter_params.max_half_life_days": 20,
                "filter_params.max_hurst_exponent": 0.65,
                "filter_params.min_beta": 0.05,
                "filter_params.max_beta": 15.0,
                "pair_selection.kpss_pvalue_threshold": 0.10,
            },
        )
    )
    v.append(
        Variant(
            "v10_h2_diag_kpss_only",
            "H2_quality_diag_one_filter",
            {
                "pair_selection.kpss_pvalue_threshold": 0.05,
            },
        )
    )

    # H3: pair stability tighten across WFA steps (trailing streak)
    v.append(
        Variant(
            "v11_h3_stability_mild",
            "H3_stability_tighten",
            {
                "pair_selection.pair_stability_window_steps": 3,
                "pair_selection.pair_stability_min_steps": 3,
            },
        )
    )
    v.append(
        Variant(
            "v12_h3_stability_med",
            "H3_stability_tighten",
            {
                "pair_selection.pair_stability_window_steps": 4,
                "pair_selection.pair_stability_min_steps": 4,
            },
        )
    )
    v.append(
        Variant(
            "v13_h3_stability_hard",
            "H3_stability_tighten",
            {
                "pair_selection.pair_stability_window_steps": 5,
                "pair_selection.pair_stability_min_steps": 5,
            },
        )
    )

    return v


def _render_search_space_md(*, run_group: str, base_path: Path, variants: list[Variant]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append(f"# tailguard_r04 search space\n")
    lines.append(f"Date: {now}")
    lines.append(f"Run group: `{run_group}`")
    lines.append("Goal: поднять `worst_robust_sh` за счёт фильтров торгуемости/качества/стабильности, не трогая risk/stop/z/dstop/maxpos оси r03.")
    lines.append("")
    lines.append("## Base config")
    lines.append("")
    lines.append(f"- Base: `{base_path.as_posix()}`")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append(f"- Count: `{len(variants)}` (each variant generates holdout+stress)")
    lines.append("- Design: one-axis-at-a-time (curated; no cartesian explosion)")
    lines.append("")
    lines.append("## Axes")
    lines.append("")
    lines.append("- H1 tradeability: liquidity/bid-ask/tick/funding/days_live + snapshot presence + quote-match")
    lines.append("- H2 quality: mean crossings / half-life / hurst / beta range + optional KPSS")
    lines.append("- H3 stability: trailing-streak presence across WFA steps")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Liquidity floor kept at 300-400k to avoid collapsing below `min_pairs>=20`.")
    lines.append("- Missing Bybit snapshot fields are fail-closed in selection filter (bid-ask=1, funding=1, tick=1, days_live=0, turnover=0).")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-group",
        default="20260223_tailguard_r04",
        help="Run group name (used in output dirs).",
    )
    ap.add_argument(
        "--base",
        default=(
            "configs/budget1000_autopilot/20260222_tailguard_r03/"
            "holdout_tailguard_risk0p0055_risk0p0055_slusd1p81_max_var_multiplier1p0065_mp21_corr0p335_pv0p39_"
            "risk0p0055_slusd1p81_max_var_multiplier1p0065_mp21_corr0p335_pv0p39_z1p15_exit0p08_dstop0p02_maxpos18.yaml"
        ),
        help="Base holdout config (relative to app root).",
    )
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

    out_config_dir = (app_root / str(args.out_config_dir) / run_group).resolve()
    out_queue_dir = (app_root / str(args.out_queue_dir) / run_group).resolve()
    runs_dir = str(args.runs_dir).strip().rstrip("/")

    variants = _variant_space()

    entries: list[RunQueueEntry] = []
    space_rows: list[dict[str, Any]] = []

    if not args.dry_run:
        out_config_dir.mkdir(parents=True, exist_ok=True)
        out_queue_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        holdout_cfg = copy.deepcopy(base_cfg)
        for k, v in variant.overrides.items():
            set_nested(holdout_cfg, k, v)

        holdout_name = f"holdout_tailguard_r04_{variant.variant_id}"
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

        stress_name = f"stress_tailguard_r04_{variant.variant_id}"
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

        # Record final (holdout) parameters for the search space table.
        space_rows.append(
            {
                "variant_id": variant.variant_id,
                "hypothesis": variant.hypothesis,
                "liquidity_usd_daily": get_nested(holdout_cfg, "pair_selection.liquidity_usd_daily"),
                "max_bid_ask_pct": get_nested(holdout_cfg, "pair_selection.max_bid_ask_pct"),
                "max_avg_funding_pct": get_nested(holdout_cfg, "pair_selection.max_avg_funding_pct"),
                "require_market_metrics": get_nested(holdout_cfg, "pair_selection.require_market_metrics"),
                "require_same_quote": get_nested(holdout_cfg, "pair_selection.require_same_quote"),
                "min_volume_usd_24h": get_nested(holdout_cfg, "pair_selection.min_volume_usd_24h"),
                "min_days_live": get_nested(holdout_cfg, "pair_selection.min_days_live"),
                "max_funding_rate_abs": get_nested(holdout_cfg, "pair_selection.max_funding_rate_abs"),
                "max_tick_size_pct": get_nested(holdout_cfg, "pair_selection.max_tick_size_pct"),
                "min_mean_crossings": get_nested(holdout_cfg, "filter_params.min_mean_crossings"),
                "min_half_life_days": get_nested(holdout_cfg, "filter_params.min_half_life_days"),
                "max_half_life_days": get_nested(holdout_cfg, "filter_params.max_half_life_days"),
                "max_hurst_exponent": get_nested(holdout_cfg, "filter_params.max_hurst_exponent"),
                "min_beta": get_nested(holdout_cfg, "filter_params.min_beta"),
                "max_beta": get_nested(holdout_cfg, "filter_params.max_beta"),
                "kpss_pvalue_threshold": get_nested(holdout_cfg, "pair_selection.kpss_pvalue_threshold"),
                "pair_stability_window_steps": get_nested(holdout_cfg, "pair_selection.pair_stability_window_steps"),
                "pair_stability_min_steps": get_nested(holdout_cfg, "pair_selection.pair_stability_min_steps"),
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

    # search_space.csv
    with space_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(space_rows[0].keys()))
        writer.writeheader()
        writer.writerows(space_rows)

    # search_space.md
    space_md.write_text(
        _render_search_space_md(run_group=run_group, base_path=base_path.relative_to(app_root), variants=variants),
        encoding="utf-8",
    )

    print(f"Wrote configs: {out_config_dir.relative_to(app_root)}")
    print(f"Wrote queue:   {queue_path.relative_to(app_root)} ({len(entries)} entries)")
    print(f"Wrote space:   {space_csv.relative_to(app_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

