#!/usr/bin/env python3
"""Generate r11 dd-focus turnover/cost sweep (100 variants) around the current best fullspan baseline.

Goal
----
Filters-only sweeps (tradeability/quality) improved dd-focus but did not generalize to fullspan.
This run changes the strategy dynamics more radically by reducing churn/turnover and avoiding
high-vol regimes via adaptive thresholds.

We keep pair-selection and risk/stop structure intact and sweep only "turnover-first" knobs:
  - z-score entry threshold (zscore_threshold + zscore_entry_threshold)
  - min_spread_move_sigma (anti-churn entry gate)
  - max_var_multiplier (adaptive threshold cap; larger => fewer trades in high-vol)

We run these variants on the dd-focus window (worst DD regime), then confirm top candidates
on fullspan in a follow-up run-group.

Outputs
-------
  configs/budget1000_autopilot/<run_group>/*.yaml
  artifacts/wfa/aggregate/<run_group>/run_queue.csv
  artifacts/wfa/aggregate/<run_group>/search_space.(csv|md)

Run from app-root (coint4/):
  PYTHONPATH=src ./.venv/bin/python scripts/optimization/generate_tailguard_r11_ddfocus_turnover_sweep100.py
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


def _fmt_p(val: float, *, decimals: int) -> str:
    """Format float as a compact tag like 1p15 / 0p10 / 1p0065."""
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
class Axis:
    axis_id: str
    hypothesis: str
    overrides: dict[str, Any]


def _entry_axes() -> list[Axis]:
    # Entry threshold (zscore_threshold + zscore_entry_threshold must match).
    # Keep within a range that likely preserves min_trades>=200 on fullspan.
    vals = [1.15, 1.40, 1.65, 1.90, 2.15]
    out: list[Axis] = []
    for v in vals:
        tag = _fmt_p(v, decimals=2)
        out.append(
            Axis(
                axis_id=f"e{tag}",
                hypothesis=f"Entry threshold={v:.2f} (higher => fewer trades, higher SNR)",
                overrides={
                    "backtest.zscore_threshold": float(v),
                    "backtest.zscore_entry_threshold": float(v),
                },
            )
        )
    return out


def _minspread_axes() -> list[Axis]:
    # Entry gate: require spread to move since last flat by >= k * current_vol.
    vals = [0.10, 0.30, 0.60, 1.00, 1.60]
    out: list[Axis] = []
    for v in vals:
        tag = _fmt_p(v, decimals=2)
        out.append(
            Axis(
                axis_id=f"ms{tag}",
                hypothesis=f"min_spread_move_sigma={v:.2f} (anti-churn)",
                overrides={"backtest.min_spread_move_sigma": float(v)},
            )
        )
    return out


def _varcap_axes() -> list[Axis]:
    # Adaptive threshold cap (must be > 1.0; config validator enforces gt=1.0).
    vals = [1.0065, 1.20, 1.50, 2.00]
    out: list[Axis] = []
    for v in vals:
        decimals = 4 if v < 1.01 else 2
        tag = _fmt_p(v, decimals=decimals)
        out.append(
            Axis(
                axis_id=f"var{tag}",
                hypothesis=f"max_var_multiplier={v:.4f} (adaptive cap; higher => more risk-off in vol)",
                overrides={"backtest.max_var_multiplier": float(v)},
            )
        )
    return out


def _iter_variants(
    entries: list[Axis], mins: list[Axis], vars_: list[Axis]
) -> Iterable[tuple[int, Axis, Axis, Axis, str, dict[str, Any]]]:
    # Deterministic 5x5x4 grid -> 100 variants.
    n = 0
    for e in entries:
        for m in mins:
            for v in vars_:
                n += 1
                variant_id = f"v{n:03d}_{e.axis_id}_{m.axis_id}_{v.axis_id}"
                hypothesis = f"{e.hypothesis} + {m.hypothesis} + {v.hypothesis}"
                overrides = dict(e.overrides)
                overrides.update(m.overrides)
                overrides.update(v.overrides)
                yield n, e, m, v, hypothesis, overrides


def _render_search_space_md(
    *,
    run_group: str,
    base_path: Path,
    dd_start: str,
    dd_end: str,
    entry_axes: list[Axis],
    minspread_axes: list[Axis],
    varcap_axes: list[Axis],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("# tailguard_r11 dd-focus turnover sweep100\n")
    lines.append(f"Date: {now}")
    lines.append(f"Run group: `{run_group}`")
    lines.append("Goal: 100 вариантов (5x5x4) на dd-focus окне; меняем только turnover/cost-first параметры.")
    lines.append("")
    lines.append("## DD-Focus window\n")
    lines.append(f"- walk_forward.start_date: `{dd_start}`")
    lines.append(f"- walk_forward.end_date: `{dd_end}`")
    lines.append("")
    lines.append("## Base config\n")
    lines.append(f"- Base (holdout): `{base_path.as_posix()}`")
    lines.append("")
    lines.append("## Axes\n")
    lines.append("")
    lines.append("### Entry threshold (E)\n")
    for a in entry_axes:
        lines.append(f"- `{a.axis_id}`: {a.hypothesis}")
    lines.append("")
    lines.append("### min_spread_move_sigma (MS)\n")
    for a in minspread_axes:
        lines.append(f"- `{a.axis_id}`: {a.hypothesis}")
    lines.append("")
    lines.append("### max_var_multiplier (VAR)\n")
    for a in varcap_axes:
        lines.append(f"- `{a.axis_id}`: {a.hypothesis}")
    lines.append("")
    lines.append("## Variants\n")
    lines.append("")
    lines.append("Variants are the cartesian product E x MS x VAR (5 x 5 x 4 = 100).")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-group",
        default="20260226_tailguard_r11_ddfocus_turnover_sweep100",
        help="Run group name (used in output dirs).",
    )
    ap.add_argument(
        "--base",
        default=(
            "configs/budget1000_autopilot/20260226_tailguard_r10_fullspan_confirm_ddfocus_top8/"
            "holdout_tailguard_r10_fullspan_v01_control_r08v10.yaml"
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

    entry_axes = _entry_axes()
    minspread_axes = _minspread_axes()
    varcap_axes = _varcap_axes()
    if len(entry_axes) != 5 or len(minspread_axes) != 5 or len(varcap_axes) != 4:
        raise SystemExit("Expected 5 entry x 5 minspread x 4 varcap axes")

    queue_entries: list[RunQueueEntry] = []
    space_rows: list[dict[str, Any]] = []

    if not args.dry_run:
        out_config_dir.mkdir(parents=True, exist_ok=True)
        out_queue_dir.mkdir(parents=True, exist_ok=True)

    for idx, e_axis, ms_axis, var_axis, hypothesis, overrides in _iter_variants(
        entry_axes, minspread_axes, varcap_axes
    ):
        holdout_cfg = copy.deepcopy(base_cfg)
        # dd-focus window (test range); train starts at start_date - training_period_days.
        set_nested(holdout_cfg, "walk_forward.start_date", dd_start)
        set_nested(holdout_cfg, "walk_forward.end_date", dd_end)
        # Apply overrides (turnover/cost-first only).
        for k, v in overrides.items():
            set_nested(holdout_cfg, k, v)

        variant_id = f"v{idx:03d}_{e_axis.axis_id}_{ms_axis.axis_id}_{var_axis.axis_id}"
        holdout_name = f"holdout_tailguard_r11_ddfocus_{variant_id}"
        holdout_cfg_path = out_config_dir / f"{holdout_name}.yaml"
        holdout_results = f"{runs_dir}/{run_group}/{holdout_name}"

        if not args.dry_run:
            holdout_cfg_path.write_text(
                yaml.dump(holdout_cfg, default_flow_style=False, allow_unicode=True),
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
        for k, v in STRESS_OVERRIDES.items():
            set_nested(stress_cfg, k, v)

        stress_name = f"stress_tailguard_r11_ddfocus_{variant_id}"
        stress_cfg_path = out_config_dir / f"{stress_name}.yaml"
        stress_results = f"{runs_dir}/{run_group}/{stress_name}"

        if not args.dry_run:
            stress_cfg_path.write_text(
                yaml.dump(stress_cfg, default_flow_style=False, allow_unicode=True),
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
                "variant_id": variant_id,
                "hypothesis": hypothesis,
                "dd_start": get_nested(holdout_cfg, "walk_forward.start_date"),
                "dd_end": get_nested(holdout_cfg, "walk_forward.end_date"),
                "entry_threshold": get_nested(holdout_cfg, "backtest.zscore_entry_threshold"),
                "min_spread_move_sigma": get_nested(holdout_cfg, "backtest.min_spread_move_sigma"),
                "max_var_multiplier": get_nested(holdout_cfg, "backtest.max_var_multiplier"),
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
            entry_axes=entry_axes,
            minspread_axes=minspread_axes,
            varcap_axes=varcap_axes,
        ),
        encoding="utf-8",
    )

    print(f"Wrote configs: {out_config_dir.relative_to(app_root)}")
    print(f"Wrote queue:   {queue_path.relative_to(app_root)} ({len(queue_entries)} entries)")
    print(f"Wrote space:   {space_csv.relative_to(app_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

