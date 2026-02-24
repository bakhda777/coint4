#!/usr/bin/env python3
"""Generate a tiny r07b fullspan queue that denylists the top tail-offender symbol (AXSUSDT).

Motivation
----------
In r07 fullspan confirm, tail-loss concentration was dominated by AXSUSDT-DOTUSDT.
This experiment answers: does removing AXSUSDT from the universe materially improve
tail/DD/robust Sharpe without touching risk/stop logic?

Design goals
------------
- Base: r07 v02 (trade_balanced_B) holdout config.
- Only override `data_filters.exclude_symbols`.
- Produce 1 variant * holdout+stress (2 queue entries).

Run from app-root (coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/generate_tailguard_r07b_fullspan_denylist_axs.py
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
    return [
        Variant(
            "v01_deny_AXSUSDT",
            "Denylist AXSUSDT to remove the AXSUSDT-DOTUSDT tail offender (symbol-level exclude).",
            {"data_filters.exclude_symbols": ["AXSUSDT"]},
        )
    ]


def _render_search_space_md(*, run_group: str, base_path: Path, variants: list[Variant]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("# tailguard_r07b fullspan denylist AXSUSDT\n")
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
        default="20260224_tailguard_r07b_fullspan_denylist_axs",
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

        holdout_name = f"holdout_tailguard_r07b_fullspan_{variant.variant_id}"
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

        stress_name = f"stress_tailguard_r07b_fullspan_{variant.variant_id}"
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
                "exclude_symbols": ",".join(get_nested(holdout_cfg, "data_filters.exclude_symbols", []) or []),
                "start_date": get_nested(holdout_cfg, "walk_forward.start_date"),
                "end_date": get_nested(holdout_cfg, "walk_forward.end_date"),
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

