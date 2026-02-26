#!/usr/bin/env python3
"""Generate r12 dd-focus sweep (50 variants, 100 runs) to test:

1) Pair ranking mode (pair_selection.rank_mode)
2) Entry signal ranking under max_active_positions (portfolio.entry_rank_mode + alpha)

We keep the rest of the strategy identical to the chosen baseline config.

Outputs
-------
  configs/budget1000_autopilot/<run_group>/*.yaml
  artifacts/wfa/aggregate/<run_group>/run_queue.csv
  artifacts/wfa/aggregate/<run_group>/search_space.(csv|md)

Run from app-root (coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/generate_tailguard_r12_ddfocus_rank_entry_sweep50.py
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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


def _fmt_p(val: float, *, decimals: int) -> str:
    """Format float as a compact tag like 1p15 / 0p05."""
    s = f"{val:.{decimals}f}"
    return s.replace(".", "p")


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


def _alpha_grid() -> list[float]:
    # 24 values -> with 2 rank-modes gives 48 + 2 controls = 50 variants.
    return [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
        1.10,
        1.25,
        1.40,
        1.60,
    ]


def _iter_variants() -> Iterable[Variant]:
    # Controls (no entry weighting): baseline and rank-mode-only.
    yield Variant(
        variant_id="v001_rankstd_abs",
        hypothesis="Control: rank_mode=spread_std; entry_rank=abs_signal (baseline)",
        overrides={
            "pair_selection.rank_mode": "spread_std",
            "portfolio.entry_rank_mode": "abs_signal",
            "portfolio.entry_pair_quality_alpha": 0.0,
        },
    )
    yield Variant(
        variant_id="v002_rankcmp_abs",
        hypothesis="Pair ranking only: rank_mode=composite_v1; entry_rank=abs_signal",
        overrides={
            "pair_selection.rank_mode": "composite_v1",
            "portfolio.entry_rank_mode": "abs_signal",
            "portfolio.entry_pair_quality_alpha": 0.0,
        },
    )

    alphas = _alpha_grid()
    for rank_mode, rank_tag in [("spread_std", "rankstd"), ("composite_v1", "rankcmp")]:
        for alpha in alphas:
            # 0.05..1.60 with 2 decimals is enough; tag keeps stable ordering.
            a_tag = _fmt_p(float(alpha), decimals=2)
            yield Variant(
                variant_id=f"{rank_tag}_qw_a{a_tag}",
                hypothesis=(
                    f"Entry ranking: rank_mode={rank_mode}; entry_rank=abs_signal_x_pair_quality; "
                    f"alpha={alpha:.2f}"
                ),
                overrides={
                    "pair_selection.rank_mode": rank_mode,
                    "portfolio.entry_rank_mode": "abs_signal_x_pair_quality",
                    "portfolio.entry_pair_quality_alpha": float(alpha),
                },
            )


def _render_search_space_md(
    *,
    run_group: str,
    base_path: Path,
    dd_start: str,
    dd_end: str,
    variants: Sequence[Variant],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("# tailguard_r12 dd-focus rank/entry sweep50\n")
    lines.append(f"Date: {now}")
    lines.append(f"Run group: `{run_group}`")
    lines.append("Goal: проверить 2 рычага: pair ranking и entry ranking под maxpos.")
    lines.append("")
    lines.append("## DD-Focus window\n")
    lines.append(f"- walk_forward.start_date: `{dd_start}`")
    lines.append(f"- walk_forward.end_date: `{dd_end}`")
    lines.append("")
    lines.append("## Base config\n")
    lines.append(f"- Base (holdout): `{base_path.as_posix()}`")
    lines.append("")
    lines.append("## Variants\n")
    lines.append("")
    for v in variants:
        lines.append(f"- `{v.variant_id}`: {v.hypothesis}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-group",
        default="20260226_tailguard_r12_ddfocus_rank_entry_sweep50",
        help="Run group name (used in output dirs).",
    )
    ap.add_argument(
        "--base",
        default=(
            "configs/budget1000_autopilot/20260226_tailguard_r11_ddfocus_turnover_sweep100/"
            "holdout_tailguard_r11_ddfocus_v057_e1p65_ms1p60_var1p0065.yaml"
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

    variants: list[Variant] = []
    for v in _iter_variants():
        variants.append(v)
    # Expand entry-weight variants into v003..v050 while keeping stable ordering.
    controls = variants[:2]
    entry_weighted = variants[2:]
    entry_weighted = sorted(entry_weighted, key=lambda v: v.variant_id)
    variants = [controls[0], controls[1]]
    for i, v in enumerate(entry_weighted, start=3):
        variants.append(
            Variant(
                variant_id=f"v{i:03d}_{v.variant_id}",
                hypothesis=v.hypothesis,
                overrides=v.overrides,
            )
        )
    if len(variants) != 50:
        raise SystemExit(f"Expected 50 variants, got {len(variants)}")

    queue_entries: list[RunQueueEntry] = []
    space_rows: list[dict[str, Any]] = []

    if not args.dry_run:
        out_config_dir.mkdir(parents=True, exist_ok=True)
        out_queue_dir.mkdir(parents=True, exist_ok=True)

    for v in variants:
        holdout_cfg = copy.deepcopy(base_cfg)
        set_nested(holdout_cfg, "walk_forward.start_date", dd_start)
        set_nested(holdout_cfg, "walk_forward.end_date", dd_end)
        for k, val in v.overrides.items():
            set_nested(holdout_cfg, k, val)

        holdout_name = f"holdout_tailguard_r12_ddfocus_{v.variant_id}"
        holdout_cfg_path = out_config_dir / f"{holdout_name}.yaml"
        holdout_results = f"{runs_dir}/{run_group}/{holdout_name}"
        if not args.dry_run:
            holdout_cfg_path.write_text(
                yaml.safe_dump(holdout_cfg, sort_keys=False),
                encoding="utf-8",
            )
        queue_entries.append(
            RunQueueEntry(
                config_path=str(holdout_cfg_path.relative_to(app_root)),
                results_dir=holdout_results,
                status="planned",
            )
        )

        stress_cfg = copy.deepcopy(holdout_cfg)
        for k, val in STRESS_OVERRIDES.items():
            set_nested(stress_cfg, k, val)

        stress_name = f"stress_tailguard_r12_ddfocus_{v.variant_id}"
        stress_cfg_path = out_config_dir / f"{stress_name}.yaml"
        stress_results = f"{runs_dir}/{run_group}/{stress_name}"
        if not args.dry_run:
            stress_cfg_path.write_text(
                yaml.safe_dump(stress_cfg, sort_keys=False),
                encoding="utf-8",
            )
        queue_entries.append(
            RunQueueEntry(
                config_path=str(stress_cfg_path.relative_to(app_root)),
                results_dir=stress_results,
                status="planned",
            )
        )

        space_rows.append(
            {
                "variant_id": v.variant_id,
                "hypothesis": v.hypothesis,
                "dd_start": get_nested(holdout_cfg, "walk_forward.start_date"),
                "dd_end": get_nested(holdout_cfg, "walk_forward.end_date"),
                "rank_mode": get_nested(holdout_cfg, "pair_selection.rank_mode"),
                "entry_rank_mode": get_nested(holdout_cfg, "portfolio.entry_rank_mode"),
                "entry_pair_quality_alpha": get_nested(holdout_cfg, "portfolio.entry_pair_quality_alpha"),
            }
        )

    queue_path = out_queue_dir / "run_queue.csv"
    space_csv = out_queue_dir / "search_space.csv"
    space_md = out_queue_dir / "search_space.md"

    if args.dry_run:
        print(f"[dry-run] would write {len(queue_entries)} queue entries into: {queue_path}")
        print(f"[dry-run] would write configs into: {out_config_dir}")
        return 0

    write_run_queue(queue_path, queue_entries)
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
            variants=variants,
        ),
        encoding="utf-8",
    )

    print(f"Wrote configs: {out_config_dir.relative_to(app_root)}")
    print(f"Wrote queue:   {queue_path.relative_to(app_root)} ({len(queue_entries)} entries)")
    print(f"Wrote space:   {space_csv.relative_to(app_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

