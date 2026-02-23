#!/usr/bin/env python3
"""Generate curated r06 dd-focus configs on a wider fixed universe.

Why r06:
- r05b proved dd-focus strict gating failed due to width: min_pairs topped at 19.
- r06 fixes this by using a wider `walk_forward.pairs_file` + disabling pair-stability
  (which can zero-out steps) while keeping risk/stop axes intact.

Design goals:
- Do NOT touch risk/stop/z/dstop/maxpos axes (inherit from base r04 v07).
- Keep dd-focus window fixed: 2023-06-29 .. 2024-06-27 (test period).
- Use a wide fixed universe file (>=500 pairs) to make min_pairs>=20 achievable.
- Vary only tradeability/quality around v07 baseline (same curated space as r05).

Run from app-root (coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/generate_tailguard_r06_ddfocus_wideuniverse.py
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


def del_nested(cfg: Dict[str, Any], dotted_key: str) -> None:
    parts = dotted_key.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


@dataclass(frozen=True)
class Variant:
    variant_id: str
    hypothesis: str
    overrides: dict[str, Any]


def _variant_space() -> list[Variant]:
    v: list[Variant] = []

    # v01: baseline control (r04 v07 replay, but on dd-focus window)
    v.append(Variant("v01_control_v07", "control_replay_r04_v07_ddfocus", {}))

    # H1: tradeability tighten (meaningful microstructure thresholds; not guardrail-clamped)
    v.append(
        Variant(
            "v02_h1_trade_mild_A",
            "H1_tradeability_tighten",
            {
                "pair_selection.require_market_metrics": True,
                "pair_selection.require_same_quote": True,
                "pair_selection.liquidity_usd_daily": 300_000,
                "pair_selection.max_bid_ask_pct": 0.002,
                "pair_selection.max_avg_funding_pct": 0.001,
                "pair_selection.min_days_live": 180,
                "pair_selection.max_funding_rate_abs": 0.001,
                "pair_selection.max_tick_size_pct": 0.0005,
                # Redundant with liquidity_usd_daily; keep off for clarity.
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
                "pair_selection.max_avg_funding_pct": 0.001,
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
                "pair_selection.max_avg_funding_pct": 0.001,
                "pair_selection.min_days_live": 365,
                "pair_selection.max_funding_rate_abs": 0.001,
                "pair_selection.max_tick_size_pct": 0.0005,
                "pair_selection.min_volume_usd_24h": 0.0,
            },
        )
    )

    # Diagnostics: one-filter-at-a-time (start from control to measure marginal effect)
    v.append(
        Variant(
            "v05_h1_diag_bidask_only",
            "H1_tradeability_diag_one_filter",
            {"pair_selection.max_bid_ask_pct": 0.001},
        )
    )
    v.append(
        Variant(
            "v06_h1_diag_tick_only",
            "H1_tradeability_diag_one_filter",
            {"pair_selection.max_tick_size_pct": 0.0005},
        )
    )
    v.append(
        Variant(
            "v07_h1_diag_dayslive_only",
            "H1_tradeability_diag_one_filter",
            {"pair_selection.min_days_live": 180},
        )
    )

    # H2: quality refinements around v07 (moderate; avoid the v08/v09 collapse)
    v.append(
        Variant(
            "v08_h2_quality_cross3",
            "H2_quality_tighten",
            {"filter_params.min_mean_crossings": 3},
        )
    )
    v.append(
        Variant(
            "v09_h2_quality_hl40",
            "H2_quality_tighten",
            {"filter_params.max_half_life_days": 40},
        )
    )
    v.append(
        Variant(
            "v10_h2_quality_hurst0p72",
            "H2_quality_tighten",
            {"filter_params.max_hurst_exponent": 0.72},
        )
    )
    v.append(
        Variant(
            "v11_h2_quality_beta_tight",
            "H2_quality_tighten",
            {"filter_params.min_beta": 0.02, "filter_params.max_beta": 40.0},
        )
    )
    v.append(
        Variant(
            "v12_h2_quality_kpss_on",
            "H2_quality_tighten",
            {"pair_selection.kpss_pvalue_threshold": 0.05},
        )
    )

    # Small combined bets (still curated; no cartesian sweep)
    v.append(
        Variant(
            "v13_h12_tradeA_cross3",
            "H1+H2_combo",
            {
                # tradeability mild A
                "pair_selection.require_market_metrics": True,
                "pair_selection.require_same_quote": True,
                "pair_selection.liquidity_usd_daily": 300_000,
                "pair_selection.max_bid_ask_pct": 0.002,
                "pair_selection.max_avg_funding_pct": 0.001,
                "pair_selection.min_days_live": 180,
                "pair_selection.max_funding_rate_abs": 0.001,
                "pair_selection.max_tick_size_pct": 0.0005,
                "pair_selection.min_volume_usd_24h": 0.0,
                # quality: small tighten
                "filter_params.min_mean_crossings": 3,
            },
        )
    )
    v.append(
        Variant(
            "v14_h12_tradeB_beta",
            "H1+H2_combo",
            {
                # tradeability balanced B
                "pair_selection.require_market_metrics": True,
                "pair_selection.require_same_quote": True,
                "pair_selection.liquidity_usd_daily": 400_000,
                "pair_selection.max_bid_ask_pct": 0.001,
                "pair_selection.max_avg_funding_pct": 0.001,
                "pair_selection.min_days_live": 180,
                "pair_selection.max_funding_rate_abs": 0.001,
                "pair_selection.max_tick_size_pct": 0.0005,
                "pair_selection.min_volume_usd_24h": 0.0,
                # quality: small tighten
                "filter_params.min_beta": 0.02,
                "filter_params.max_beta": 40.0,
            },
        )
    )

    return v


def _render_search_space_md(
    *,
    run_group: str,
    base_path: Path,
    dd_start: str,
    dd_end: str,
    pairs_file: str,
    max_pairs: int,
    variants: list[Variant],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("# tailguard_r06 dd-focus (wide universe) search space\n")
    lines.append(f"Date: {now}")
    lines.append(f"Run group: `{run_group}`")
    lines.append("Goal: dd-focus fast-loop with strict width gates (min_pairs>=20) enabled by wide pairs_file.")
    lines.append("")
    lines.append("## DD-Focus window")
    lines.append("")
    lines.append(f"- walk_forward.start_date: `{dd_start}` (test start)")
    lines.append(f"- walk_forward.end_date: `{dd_end}` (test end)")
    lines.append("")
    lines.append("## Universe / selection guard")
    lines.append("")
    lines.append(f"- walk_forward.pairs_file: `{pairs_file}`")
    lines.append(f"- pair_selection.max_pairs: `{max_pairs}` (cap per WF step; not risk)")
    lines.append("- pair_selection.pair_stability_*: disabled (avoid zero-pair steps)")
    lines.append("")
    lines.append("## Base config")
    lines.append("")
    lines.append(f"- Base (holdout): `{base_path.as_posix()}`")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    for var in variants:
        lines.append(f"- `{var.variant_id}`: {var.hypothesis}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-group",
        default="20260223_tailguard_r06_ddfocus_wideuniverse",
        help="Run group name (used in output dirs).",
    )
    ap.add_argument(
        "--base",
        default=(
            "configs/budget1000_autopilot/20260223_tailguard_r04/"
            "holdout_tailguard_r04_v07_h2_quality_mild.yaml"
        ),
        help="Base holdout config (relative to app root).",
    )
    ap.add_argument("--dd-start", default="2023-06-29", help="walk_forward.start_date (test start).")
    ap.add_argument("--dd-end", default="2024-06-27", help="walk_forward.end_date (test end).")
    ap.add_argument(
        "--pairs-file",
        default="configs/universe/ddfocus_wide_pairs_universe_v01.yaml",
        help="Fixed pairs universe file (relative to app root).",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=60,
        help="pair_selection.max_pairs (cap per WF step).",
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

    dd_start = str(args.dd_start).strip()
    dd_end = str(args.dd_end).strip()
    if not dd_start or not dd_end:
        raise SystemExit("--dd-start/--dd-end must be non-empty")

    base_path = (app_root / str(args.base)).resolve()
    if not base_path.exists():
        raise SystemExit(f"Base config not found: {base_path}")
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base_cfg, dict):
        raise SystemExit(f"Base config invalid (expected mapping): {base_path}")

    pairs_file = str(args.pairs_file).strip()
    if not pairs_file:
        raise SystemExit("--pairs-file is empty")
    pairs_file_path = (app_root / pairs_file).resolve()
    if not pairs_file_path.exists():
        raise SystemExit(f"pairs_file not found: {pairs_file_path}")

    max_pairs = int(args.max_pairs)
    if max_pairs <= 0:
        raise SystemExit("--max-pairs must be > 0")

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
        set_nested(holdout_cfg, "walk_forward.start_date", dd_start)
        set_nested(holdout_cfg, "walk_forward.end_date", dd_end)
        set_nested(holdout_cfg, "walk_forward.pairs_file", pairs_file)
        set_nested(holdout_cfg, "pair_selection.max_pairs", max_pairs)

        # Disable pair-stability for dd-focus loop (avoid zero-pair steps).
        del_nested(holdout_cfg, "pair_selection.pair_stability_window_steps")
        del_nested(holdout_cfg, "pair_selection.pair_stability_min_steps")

        for k, v in variant.overrides.items():
            set_nested(holdout_cfg, k, v)

        holdout_name = f"holdout_tailguard_r06_ddfocus_{variant.variant_id}"
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

        stress_name = f"stress_tailguard_r06_ddfocus_{variant.variant_id}"
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
                "dd_start": dd_start,
                "dd_end": dd_end,
                "pairs_file": pairs_file,
                "max_pairs": max_pairs,
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
            pairs_file=pairs_file,
            max_pairs=max_pairs,
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
