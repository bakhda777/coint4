#!/usr/bin/env python3
"""Generate curated r08 fullspan configs: tradeability+quality sweep around r07 baseline.

Motivation
----------
After fixing WFA "zero windows" + coverage gating, we want a *curated* (not cartesian)
search around the current best-investable fullspan baseline (r07 v02).

Design goals
------------
- Do NOT touch risk/stop/z/dstop/maxpos and other backtest dynamics.
- Vary ONLY:
  - Tradeability thresholds (`pair_selection.*` microstructure gates)
  - Quality thresholds (`filter_params.*` + optional KPSS gate)
- Produce:
  configs/budget1000_autopilot/<run_group>/*.yaml
  artifacts/wfa/aggregate/<run_group>/run_queue.csv
  artifacts/wfa/aggregate/<run_group>/search_space.(csv|md)

Run from app-root (coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/generate_tailguard_r08_tradeability_quality_sweep.py
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

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

    v.append(Variant("v01_control_r07v02", "control_replay_r07v02_fullspan", {}))

    # H1: tradeability (microstructure) knobs.
    v.append(
        Variant(
            "v02_h1_trade_loosen_A",
            "H1_tradeability_broaden: liquidity 300k + bid-ask 0.20% (diversify pairs; may add noise)",
            {
                "pair_selection.liquidity_usd_daily": 300_000,
                "pair_selection.max_bid_ask_pct": 0.002,
            },
        )
    )
    v.append(
        Variant(
            "v03_h1_trade_liq500k",
            "H1_tradeability_tighten: liquidity 500k (filter more illiquid tails)",
            {"pair_selection.liquidity_usd_daily": 500_000},
        )
    )
    v.append(
        Variant(
            "v04_h1_trade_bidask0p0008",
            "H1_tradeability_tighten: bid-ask <= 0.08% (reduce microstructure bleed)",
            {"pair_selection.max_bid_ask_pct": 0.0008},
        )
    )
    v.append(
        Variant(
            "v05_h1_trade_dayslive365",
            "H1_tradeability_tighten: min_days_live=365 (avoid newly listed / regime-unstable symbols)",
            {"pair_selection.min_days_live": 365},
        )
    )
    v.append(
        Variant(
            "v06_h1_trade_funding0p0007",
            "H1_tradeability_tighten: funding caps 0.07% (reduce carry bleed / tail shocks)",
            {
                "pair_selection.max_avg_funding_pct": 0.0007,
                "pair_selection.max_funding_rate_abs": 0.0007,
            },
        )
    )
    v.append(
        Variant(
            "v07_h1_trade_tick0p0004",
            "H1_tradeability_tighten: tick_size_pct<=0.04% (avoid coarse pricing)",
            {"pair_selection.max_tick_size_pct": 0.0004},
        )
    )

    # H2: quality knobs (keep moderate; avoid collapsing to 1-2 pairs).
    v.append(
        Variant(
            "v08_h2_quality_cross3",
            "H2_quality_tighten: min_mean_crossings=3",
            {"filter_params.min_mean_crossings": 3},
        )
    )
    v.append(
        Variant(
            "v09_h2_quality_hl40",
            "H2_quality_tighten: max_half_life_days=40",
            {"filter_params.max_half_life_days": 40},
        )
    )
    v.append(
        Variant(
            "v10_h2_quality_hurst0p72",
            "H2_quality_tighten: max_hurst_exponent=0.72",
            {"filter_params.max_hurst_exponent": 0.72},
        )
    )
    v.append(
        Variant(
            "v11_h2_quality_beta_tight",
            "H2_quality_tighten: beta range tighter (avoid extreme betas)",
            {"filter_params.min_beta": 0.02, "filter_params.max_beta": 40.0},
        )
    )
    v.append(
        Variant(
            "v12_h2_quality_kpss_on",
            "H2_quality_tighten: enable KPSS pvalue>=0.05",
            {"pair_selection.kpss_pvalue_threshold": 0.05},
        )
    )

    # Combined bets (still curated, avoid full cartesian sweep).
    v.append(
        Variant(
            "v13_h12_liq500k_cross3",
            "H1+H2_combo: liquidity 500k + min_mean_crossings=3",
            {
                "pair_selection.liquidity_usd_daily": 500_000,
                "filter_params.min_mean_crossings": 3,
            },
        )
    )
    v.append(
        Variant(
            "v14_h12_bidask0p0008_beta",
            "H1+H2_combo: bid-ask 0.08% + beta tight",
            {
                "pair_selection.max_bid_ask_pct": 0.0008,
                "filter_params.min_beta": 0.02,
                "filter_params.max_beta": 40.0,
            },
        )
    )

    return v


def _render_search_space_md(*, run_group: str, base_path: Path, variants: list[Variant]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("# tailguard_r08 tradeability+quality sweep (fullspan)\n")
    lines.append(f"Date: {now}")
    lines.append(f"Run group: `{run_group}`")
    lines.append("")
    lines.append("## Base config\n")
    lines.append(f"- Base (holdout): `{base_path.as_posix()}`")
    lines.append("")
    lines.append("## Variants\n")
    for var in variants:
        lines.append(f"- `{var.variant_id}`: {var.hypothesis}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-group",
        default="20260224_tailguard_r08_tradeability_quality_sweep",
        help="Run group name (used in output dirs).",
    )
    ap.add_argument(
        "--base",
        default=(
            "configs/budget1000_autopilot/20260223_tailguard_r07_fullspan_confirm_top3/"
            "holdout_tailguard_r07_fullspan_v02_from_r06v03_trade_balanced_B.yaml"
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

        holdout_name = f"holdout_tailguard_r08_fullspan_{variant.variant_id}"
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

        stress_name = f"stress_tailguard_r08_fullspan_{variant.variant_id}"
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
                "variant_id": variant.variant_id,
                "hypothesis": variant.hypothesis,
                "start_date": get_nested(holdout_cfg, "walk_forward.start_date"),
                "end_date": get_nested(holdout_cfg, "walk_forward.end_date"),
                "pairs_file": get_nested(holdout_cfg, "walk_forward.pairs_file"),
                # Tradeability knobs
                "liquidity_usd_daily": get_nested(holdout_cfg, "pair_selection.liquidity_usd_daily"),
                "max_bid_ask_pct": get_nested(holdout_cfg, "pair_selection.max_bid_ask_pct"),
                "max_avg_funding_pct": get_nested(holdout_cfg, "pair_selection.max_avg_funding_pct"),
                "min_days_live": get_nested(holdout_cfg, "pair_selection.min_days_live"),
                "max_funding_rate_abs": get_nested(holdout_cfg, "pair_selection.max_funding_rate_abs"),
                "max_tick_size_pct": get_nested(holdout_cfg, "pair_selection.max_tick_size_pct"),
                # Quality knobs
                "min_mean_crossings": get_nested(holdout_cfg, "filter_params.min_mean_crossings"),
                "max_half_life_days": get_nested(holdout_cfg, "filter_params.max_half_life_days"),
                "max_hurst_exponent": get_nested(holdout_cfg, "filter_params.max_hurst_exponent"),
                "min_beta": get_nested(holdout_cfg, "filter_params.min_beta"),
                "max_beta": get_nested(holdout_cfg, "filter_params.max_beta"),
                "kpss_pvalue_threshold": get_nested(holdout_cfg, "pair_selection.kpss_pvalue_threshold"),
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
            variants=variants,
        ),
        encoding="utf-8",
    )

    print(f"Wrote configs: {out_config_dir.relative_to(app_root)}")
    print(f"Wrote queue:   {queue_path.relative_to(app_root)} ({len(entries)} entries)")
    print(f"Wrote space:   {space_csv.relative_to(app_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

