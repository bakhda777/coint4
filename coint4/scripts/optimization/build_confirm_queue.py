#!/usr/bin/env python3
"""Build a small confirmatory queue (holdout + stress) from a shortlist queue.

This is intended to take an existing shortlist (typically a baseline queue of top-N)
and produce:
  - a confirm run queue with paired entries: holdout (original config) + stress
  - stress YAML copies with standardized friction overrides

The generated queue writes results under:
  artifacts/wfa/runs_clean/<cycle>/confirm/**

Run from app-root (coint4/):
  ./.venv/bin/python scripts/optimization/build_confirm_queue.py \
    --shortlist-queue artifacts/wfa/aggregate/clean_cycle_top10/20260216_clean_top10/baseline_run_queue.csv \
    --cycle 20260215_confirm_shortlist \
    --queue-dir artifacts/wfa/aggregate/clean_cycle_top10/20260215_confirm_shortlist \
    --stress-config-dir configs/clean_cycle_top10/confirm/stress \
    --limit 10
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from dataclasses import dataclass
from typing import Sequence


STRESS_OVERRIDES: Dict[str, Any] = {
    "backtest.commission_pct": 0.0006,
    "backtest.commission_rate_per_leg": 0.0006,
    "backtest.slippage_pct": 0.001,
    "backtest.slippage_stress_multiplier": 2.0,
}


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node: Dict[str, Any] = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


@dataclass
class RunQueueEntry:
    config_path: str
    results_dir: str
    status: str


def load_run_queue(path: Path) -> list[RunQueueEntry]:
    entries: list[RunQueueEntry] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                RunQueueEntry(
                    config_path=(row.get("config_path") or "").strip(),
                    results_dir=(row.get("results_dir") or "").strip(),
                    status=(row.get("status") or "").strip(),
                )
            )
    return entries


def write_run_queue(path: Path, entries: Sequence[RunQueueEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_path", "results_dir", "status"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "config_path": entry.config_path,
                    "results_dir": entry.results_dir,
                    "status": entry.status,
                }
            )


def _resolve_under_app_root(path: Path, app_root: Path) -> Path:
    if path.is_absolute():
        return path
    return app_root / path


def _rel_to_app_root(path: Path, app_root: Path) -> str:
    try:
        return str(path.relative_to(app_root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build confirmatory holdout+stress run queue from a shortlist queue"
    )
    parser.add_argument(
        "--shortlist-queue",
        required=True,
        help="Path to a shortlist/baseline queue CSV (config_path column is used)",
    )
    parser.add_argument(
        "--cycle",
        required=True,
        help="Cycle name used in results_dir: artifacts/wfa/runs_clean/<cycle>/confirm/**",
    )
    parser.add_argument(
        "--queue-dir",
        required=True,
        help="Directory where run_queue.csv will be written",
    )
    parser.add_argument(
        "--stress-config-dir",
        required=True,
        help="Directory where stress YAML copies will be written",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max number of shortlist configs to include (default: 20)",
    )
    args = parser.parse_args()

    app_root = Path(__file__).resolve().parents[2]

    shortlist_path = _resolve_under_app_root(Path(args.shortlist_queue), app_root)
    if not shortlist_path.exists():
        print(f"Error: shortlist queue not found: {shortlist_path}", file=sys.stderr)
        raise SystemExit(2)

    shortlist_entries = load_run_queue(shortlist_path)
    base_paths = [
        (Path(entry.config_path), entry)
        for entry in shortlist_entries
        if (entry.config_path or "").strip()
    ]

    if not base_paths:
        print(
            f"Error: no usable config_path entries in: {shortlist_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    base_paths = base_paths[: max(1, int(args.limit))]
    stress_dir = _resolve_under_app_root(Path(args.stress_config_dir), app_root)
    stress_dir.mkdir(parents=True, exist_ok=True)

    out_entries: list[RunQueueEntry] = []
    cycle = str(args.cycle).strip()
    runs_base = f"artifacts/wfa/runs_clean/{cycle}/confirm"

    for base_path, _entry in base_paths:
        base_path = _resolve_under_app_root(base_path, app_root)
        if not base_path.exists():
            print(f"Error: base config not found: {base_path}", file=sys.stderr)
            raise SystemExit(2)

        with base_path.open() as handle:
            base_cfg = yaml.safe_load(handle)
        if not isinstance(base_cfg, dict):
            print(f"Error: invalid YAML (expected dict): {base_path}", file=sys.stderr)
            raise SystemExit(2)

        base_stem = base_path.stem

        # Holdout entry uses the original shortlist config as-is.
        out_entries.append(
            RunQueueEntry(
                config_path=_rel_to_app_root(base_path, app_root),
                results_dir=f"{runs_base}/holdout/{base_stem}",
                status="planned",
            )
        )

        # Stress entry uses a copied YAML with standardized friction overrides.
        stress_cfg: Dict[str, Any] = copy.deepcopy(base_cfg)
        for key, value in STRESS_OVERRIDES.items():
            set_nested(stress_cfg, key, value)

        stress_name = base_path.name
        if not stress_name.startswith("stress_"):
            stress_name = f"stress_{stress_name}"
        stress_path = stress_dir / stress_name
        with stress_path.open("w") as handle:
            yaml.dump(stress_cfg, handle, default_flow_style=False, allow_unicode=True)

        out_entries.append(
            RunQueueEntry(
                config_path=_rel_to_app_root(stress_path, app_root),
                results_dir=f"{runs_base}/stress/{stress_path.stem}",
                status="planned",
            )
        )

    queue_dir = _resolve_under_app_root(Path(args.queue_dir), app_root)
    queue_path = queue_dir / "run_queue.csv"
    write_run_queue(queue_path, out_entries)
    print(f"Wrote {len(out_entries)} entries: {queue_path}")
    print(f"Stress configs: {stress_dir}")


if __name__ == "__main__":
    main()
